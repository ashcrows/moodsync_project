"""Training for the CNN mood classifier and the Bi-LSTM arc smoother.

Uses DEAM when configured, otherwise the synthetic demo dataset. Kept lightweight
so a full demo train completes in ~1 minute on a laptop CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config, dataset_tag
from .features.extract import load_audio, mel_chroma_windows, mel_windows
from .models.cnn import MoodCNN, mood_loss
from .models.lstm_smoother import ArcSmoother
from .platform_utils import Platform, artifacts_dir

# Fallback seed when neither `--seed` nor a config `seed` key is given. Changing
# it changes every published result, so it stays fixed.
DEFAULT_SEED = 42


def _log(msg: str) -> None:
    """Print immediately so progress is visible in logs and background shells.

    Python block-buffers stdout when it is not a TTY, so without flushing a long
    training run emits nothing until it exits and looks hung.
    """
    print(msg, flush=True)


def _progress(stage: str, i: int, total: int, t0: float, every: int = 100) -> None:
    """Emit a decode-progress line with a running ETA."""
    if i != total and (i % every or i == 0):
        return
    import time
    done = max(1, i)
    rate = done / max(1e-6, time.time() - t0)
    eta = (total - done) / max(1e-6, rate)
    _log(f"[{stage}] loading audio {i}/{total} ({100*i/total:.0f}%)  "
         f"{rate:.1f} songs/s  eta {eta/60:.1f} min")


def _cnn_input(y: np.ndarray, cfg: Config) -> np.ndarray:
    """Build the CNN input for a waveform, honouring `cnn.in_channels`.

    1 -> (W, n_mels, frames) mel only; 2 -> (W, 2, n_mels, frames) mel + chroma.
    """
    if int(getattr(cfg.cnn, "in_channels", 1)) >= 2:
        return mel_chroma_windows(y, cfg)
    return mel_windows(y, cfg)


def align_arc_to_windows(clip: dict, n_windows: int, cfg: Config):
    """Label each mel window using the arc's REAL timestamps.

    Returns (labels, keep) where `keep` marks windows that fall inside the
    annotated span. DEAM's dynamic ratings start at 15s, so naively stretching
    the arc over the whole file labels the 0s window with the 15s emotion --
    a systematic error of up to 15 seconds on every song.

    Clips without timing metadata (the synthetic demo) keep the original
    stretch-to-fit behaviour, where the arc does describe the whole clip.
    """
    arc = np.asarray(clip["arc"], dtype=np.float32).reshape(-1, 2)
    hop = float(cfg.audio.hop_seconds)
    win = float(cfg.audio.window_seconds)
    centres = np.arange(n_windows) * hop + win / 2.0

    start = clip.get("arc_start_s")
    step = clip.get("arc_hop_s")
    if start is None or step is None:
        xs = np.linspace(0, 1, len(arc))
        q = np.linspace(0, 1, n_windows)
        labels = np.stack([np.interp(q, xs, arc[:, 0]),
                           np.interp(q, xs, arc[:, 1])], axis=1)
        return labels.astype(np.float32), np.ones(n_windows, dtype=bool)

    times = float(start) + np.arange(len(arc)) * float(step)
    labels = np.stack([np.interp(centres, times, arc[:, 0]),
                       np.interp(centres, times, arc[:, 1])], axis=1)
    keep = (centres >= times[0]) & (centres <= times[-1])
    return labels.astype(np.float32), keep


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2) + 1e-8
    return float(1 - ss_res / ss_tot)


def _build_cnn_tensors(
    clips: List[dict], cfg: Config
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Turn clips into (mel_windows, per-window V/A labels, per-window song index).

    The song index is what makes an honest train/validation split possible: every
    window of a given song must land on the same side of the split.
    """
    import time
    X, Y, G = [], [], []
    dropped = 0
    t0 = time.time()
    _log(f"[cnn] decoding {len(clips)} songs ...")
    for ci, clip in enumerate(clips):
        _progress("cnn", ci, len(clips), t0)
        y = load_audio(clip["path"], cfg.audio.sample_rate)
        mels = _cnn_input(y, cfg)                        # (W, C, n_mels, frames)
        W = mels.shape[0]
        labels, keep = align_arc_to_windows(clip, W, cfg)
        if not keep.any():
            continue                                     # nothing annotated here
        X.append(mels[keep])
        Y.append(labels[keep])
        G.append(np.full(int(keep.sum()), ci, dtype=np.int64))
        dropped += int((~keep).sum())
    _progress("cnn", len(clips), len(clips), t0)
    if dropped:
        _log(f"[cnn] dropped {dropped} windows outside the annotated time span")
    Xt = torch.tensor(np.concatenate(X, axis=0))
    Yt = torch.tensor(np.concatenate(Y, axis=0))
    groups = np.concatenate(G, axis=0)
    return Xt, Yt, groups


def _group_split(groups: np.ndarray, train_frac: float = 0.8, seed: int = DEFAULT_SEED):
    """Split window indices by SONG, so no song appears in both sides.

    A plain per-window random split leaks: windows of the same song are highly
    correlated, so the model can memorise a song from its training windows and
    look good on its validation windows. That inflates R2 and makes the reported
    number meaningless as a generalisation estimate.
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(round(train_frac * len(shuffled))))
    if len(shuffled) > 1:
        n_train = min(n_train, len(shuffled) - 1)   # always keep a held-out song
    train_songs = set(shuffled[:n_train].tolist())
    is_train = np.array([g in train_songs for g in groups], dtype=bool)
    tr = torch.from_numpy(np.flatnonzero(is_train))
    va = torch.from_numpy(np.flatnonzero(~is_train))
    if len(va) == 0:                                 # single-song corpus
        va = tr
    return tr, va, len(train_songs), len(uniq) - len(train_songs)


def _group_split_3way(groups: np.ndarray, fracs=(0.7, 0.15, 0.15), seed: int = DEFAULT_SEED):
    """Split window indices by SONG into train / val / test.

    Early stopping needs a held-out signal, but selecting on the split that is
    later reported would be tuning on the test set. Val therefore drives early
    stopping and test is only ever measured once, at the end.
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_tr = max(1, int(round(fracs[0] * n)))
    n_va = max(1, int(round(fracs[1] * n))) if n > 2 else 0
    n_tr = min(n_tr, max(1, n - n_va - 1)) if n > 2 else n_tr
    sets = (set(shuffled[:n_tr].tolist()),
            set(shuffled[n_tr:n_tr + n_va].tolist()),
            set(shuffled[n_tr + n_va:].tolist()))
    out = []
    for s in sets:
        idx = np.flatnonzero(np.array([g in s for g in groups], dtype=bool))
        out.append(torch.from_numpy(idx))
    # Degenerate corpora (demo smoke runs): never hand back an empty split.
    for i in (1, 2):
        if len(out[i]) == 0:
            out[i] = out[0]
    return out[0], out[1], out[2], [len(s) for s in sets]


def spec_augment(x: torch.Tensor, time_mask: int, freq_mask: int,
                 channel: int = 0) -> torch.Tensor:
    """SpecAugment-style masking, applied to ONE channel of a batch.

    Only the mel channel is masked: blanking bands of the chroma channel would
    delete pitch classes outright, and those carry the harmony signal.
    """
    if time_mask <= 0 and freq_mask <= 0:
        return x
    x = x.clone()
    if x.dim() != 4 or channel >= x.shape[1]:
        return x
    B, _, H, W = x.shape
    if freq_mask > 0:
        f = torch.randint(0, max(1, freq_mask), (B,))
        f0 = (torch.rand(B) * torch.clamp(H - f, min=1).float()).long()
        for i in range(B):
            if f[i] > 0:
                x[i, channel, f0[i]:f0[i] + f[i], :] = 0.0
    if time_mask > 0:
        t = torch.randint(0, max(1, time_mask), (B,))
        t0 = (torch.rand(B) * torch.clamp(W - t, min=1).float()).long()
        for i in range(B):
            if t[i] > 0:
                x[i, channel, :, t0[i]:t0[i] + t[i]] = 0.0
    return x


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over valid timesteps only.

    `mask` is (B, T, 1) with 1.0 on real timesteps. Sequences are zero-padded to
    a common length for batching; without masking, padded slots dominate the loss
    whenever song lengths vary (on DEAM, ~92% of slots are padding).
    """
    denom = mask.sum() * pred.shape[-1]
    return (((pred - target) ** 2) * mask).sum() / torch.clamp(denom, min=1.0)


def length_mask(lengths: np.ndarray, max_T: int) -> torch.Tensor:
    """Build the (B, T, 1) validity mask for `masked_mse`."""
    m = torch.arange(max_T)[None, :] < torch.as_tensor(lengths, dtype=torch.long)[:, None]
    return m.unsqueeze(-1).float()


def _seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Make training deterministic/reproducible run-to-run."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_seed(cfg: Config) -> int:
    """The seed for this run: the config's `seed` key, else DEFAULT_SEED.

    `--seed` writes into the config before training starts, so both the split
    and the weight init follow one value. Keeping the fallback at DEFAULT_SEED
    means a config without the key reproduces earlier runs exactly.
    """
    try:
        raw = cfg["seed"]
    except (KeyError, TypeError):
        return DEFAULT_SEED
    if raw is None or str(raw).strip() == "":
        return DEFAULT_SEED
    return int(raw)


def train_cnn(clips: List[dict], cfg: Config, platform: Platform) -> Path:
    device = platform.device
    seed = resolve_seed(cfg)
    _log(f"[cnn] seed={seed}")
    _seed_everything(seed)
    Xt, Yt, groups = _build_cnn_tensors(clips, cfg)
    tr, va, te, n_songs = _group_split_3way(groups, seed=seed)
    _log(f"[cnn] song-level split: {n_songs[0]} train / {n_songs[1]} val / "
         f"{n_songs[2]} test songs ({len(tr)} / {len(va)} / {len(te)} windows)")
    _log("[cnn] val drives early stopping; test is measured once, at the end")

    ds = TensorDataset(Xt[tr], Yt[tr])
    dl = DataLoader(
        ds, batch_size=int(cfg.cnn.batch_size), shuffle=True,
        num_workers=platform.num_workers,
    )
    in_ch = int(getattr(cfg.cnn, "in_channels", 1))
    model = MoodCNN(n_mels=int(cfg.audio.n_mels), in_channels=in_ch,
                    dropout=float(getattr(cfg.cnn, "dropout", 0.0))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.cnn.lr),
                           weight_decay=float(getattr(cfg.cnn, "weight_decay", 0.0)))

    use_sa = bool(getattr(cfg.cnn, "spec_augment", False))
    t_mask = int(getattr(cfg.cnn, "spec_time_mask", 0))
    f_mask = int(getattr(cfg.cnn, "spec_freq_mask", 0))
    patience = int(getattr(cfg.cnn, "patience", 0)) or int(cfg.cnn.epochs)
    _log(f"[cnn] in_channels={in_ch} dropout={getattr(cfg.cnn,'dropout',0.0)} "
         f"weight_decay={getattr(cfg.cnn,'weight_decay',0.0)} "
         f"spec_augment={use_sa} patience={patience}")

    def _eval(idx):
        model.eval()
        with torch.no_grad():
            preds = []
            for b in range(0, len(idx), 256):
                preds.append(model(Xt[idx[b:b + 256]].to(device)).cpu().numpy())
        p = np.concatenate(preds, axis=0)
        t = Yt[idx].numpy()
        return _r2(p[:, 0], t[:, 0]), _r2(p[:, 1], t[:, 1])

    best_v, best_epoch, best_state, since = -1e9, 0, None, 0
    for epoch in range(int(cfg.cnn.epochs)):
        model.train()
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if use_sa:
                xb = spec_augment(xb, t_mask, f_mask, channel=0)
            opt.zero_grad()
            loss = mood_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        vv, va_r2 = _eval(va)
        flag = ""
        if vv > best_v:
            best_v, best_epoch, since = vv, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  *best*"
        else:
            since += 1
        _log(f"[cnn] epoch {epoch+1}/{cfg.cnn.epochs}  loss={total/len(ds):.4f}  "
             f"val R2 v={vv:.3f} a={va_r2:.3f}{flag}")
        if since >= patience:
            _log(f"[cnn] early stop: no val-valence gain for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        _log(f"[cnn] restored best epoch {best_epoch} (val valence R2={best_v:.3f})")

    # TEST split — reported once, never used for any decision above.
    r2v, r2a = _eval(te)
    tgt_v = float(getattr(cfg.cnn, "target_r2_valence", cfg.cnn.target_r2))
    tgt_a = float(getattr(cfg.cnn, "target_r2_arousal", cfg.cnn.target_r2))
    _log(f"[cnn] TEST R2  valence={r2v:.3f} (target>={tgt_v})  "
         f"arousal={r2a:.3f} (target>={tgt_a})  [held-out songs]")

    tag = dataset_tag(cfg)
    out = artifacts_dir(tag) / "cnn.pt"
    torch.save({"state_dict": model.state_dict(), "n_mels": int(cfg.audio.n_mels),
                "in_channels": in_ch, "dataset_tag": tag,
                "dropout": float(getattr(cfg.cnn, "dropout", 0.0))}, out)
    _log(f"[cnn] saved -> {out}  (dataset tag '{tag}')")
    return out


def train_lstm(clips: List[dict], cfg: Config, platform: Platform, cnn_path: Path) -> Path:
    """Train the Bi-LSTM to map noisy CNN arcs -> clean ground-truth arcs."""
    device = platform.device
    seed = resolve_seed(cfg)
    _log(f"[lstm] seed={seed}")
    _seed_everything(seed)
    ckpt = torch.load(cnn_path, map_location=device)
    cnn = MoodCNN(n_mels=ckpt["n_mels"],
                  in_channels=int(ckpt.get("in_channels", 1)),
                  dropout=float(ckpt.get("dropout", 0.0))).to(device)
    cnn.load_state_dict(ckpt["state_dict"])
    cnn.eval()

    import time
    seqs_noisy, seqs_clean = [], []
    max_T = 0
    t0 = time.time()
    _log(f"[lstm] re-encoding {len(clips)} songs through the CNN ...")
    with torch.no_grad():
        for li, clip in enumerate(clips):
            _progress("lstm", li, len(clips), t0)
            y = load_audio(clip["path"], cfg.audio.sample_rate)
            mel = _cnn_input(y, cfg)
            clean, keep = align_arc_to_windows(clip, mel.shape[0], cfg)
            if not keep.any():
                continue
            mels = torch.tensor(mel[keep]).to(device)
            noisy = cnn(mels).cpu().numpy()               # (W, 2)
            seqs_noisy.append(noisy.astype(np.float32))
            seqs_clean.append(clean[keep].astype(np.float32))
            max_T = max(max_T, len(noisy))

    # Pad sequences to equal length for batching, and keep a validity mask.
    # DEAM mixes 45s excerpts (60 annotation samples) with full-length songs
    # (>1200), so without a mask ~92% of the loss would be computed against
    # zero padding and the smoother would learn to predict silence.
    lengths = np.array([len(s) for s in seqs_noisy], dtype=np.int64)

    def _pad(seqs):
        out = np.zeros((len(seqs), max_T, 2), dtype=np.float32)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
        return torch.tensor(out)

    Xn, Xc = _pad(seqs_noisy), _pad(seqs_clean)
    mask = length_mask(lengths, max_T)                   # (B, T, 1)

    # Hold out whole songs here too, so the reported smoothing gain is a
    # generalisation estimate rather than a fit to the training songs.
    song_ids = np.arange(len(seqs_noisy))
    tr_idx, va_idx, n_tr, n_va = _group_split(song_ids, train_frac=0.8, seed=seed)
    pad_frac = 1.0 - float(lengths.sum()) / float(len(lengths) * max_T)
    _log(f"[lstm] song-level split: {n_tr} train / {n_va} val songs "
          f"(max_T={max_T}, {pad_frac*100:.1f}% padding masked out)")
    Xn_tr, Xc_tr, m_tr = Xn[tr_idx], Xc[tr_idx], mask[tr_idx]

    model = ArcSmoother(hidden=int(cfg.lstm.hidden), layers=int(cfg.lstm.layers)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lstm.lr))

    # Mini-batch over songs. A single full-batch step is fine for the ~50-clip
    # demo, but on DEAM it means a 1442 x 210 sequence batch whose backprop
    # graph exhausts unified memory and thrashes swap.
    bs = max(1, int(cfg.lstm.batch_size))
    n_tr_songs = len(tr_idx)
    n_batches = (n_tr_songs + bs - 1) // bs
    _log(f"[lstm] mini-batching: {n_tr_songs} songs, batch_size={bs} "
         f"({n_batches} steps/epoch)")

    model.train()
    for epoch in range(int(cfg.lstm.epochs)):
        perm = torch.randperm(n_tr_songs)
        total, seen = 0.0, 0
        for b in range(n_batches):
            sel = perm[b * bs:(b + 1) * bs]
            xb = Xn_tr[sel].to(device)
            yb = Xc_tr[sel].to(device)
            mb = m_tr[sel].to(device)
            opt.zero_grad()
            loss = masked_mse(model(xb), yb, mb)
            loss.backward()
            opt.step()
            total += loss.item() * len(sel)
            seen += len(sel)
        _log(f"[lstm] epoch {epoch+1}/{cfg.lstm.epochs}  loss={total/max(1,seen):.4f}")

    # Held-out R2 over REAL timesteps only, plus the same R2 for the raw CNN
    # arcs, so the smoother has to justify itself: if these are equal the
    # Bi-LSTM is adding nothing. Batched for the same memory reason.
    model.eval()
    with torch.no_grad():
        chunks = []
        for b in range(0, len(va_idx), bs):
            sel = va_idx[b:b + bs]
            chunks.append(model(Xn[sel].to(device)).cpu().numpy())
        pv_full = np.concatenate(chunks, axis=0)
    keep = mask[va_idx].squeeze(-1).numpy().astype(bool).reshape(-1)
    pv = pv_full.reshape(-1, 2)[keep]
    tv = Xc[va_idx].numpy().reshape(-1, 2)[keep]
    raw = Xn[va_idx].numpy().reshape(-1, 2)[keep]
    _log(f"[lstm] val R2  valence={_r2(pv[:, 0], tv[:, 0]):.3f}  "
          f"arousal={_r2(pv[:, 1], tv[:, 1]):.3f}  (held-out songs, padding excluded)")
    _log(f"[lstm] vs raw CNN  valence={_r2(raw[:, 0], tv[:, 0]):.3f}  "
          f"arousal={_r2(raw[:, 1], tv[:, 1]):.3f}  (smoother must beat this)")

    tag = dataset_tag(cfg)
    # Sit beside the CNN this smoother was fitted to, so the pair can never drift.
    out = Path(cnn_path).parent / "lstm.pt"
    torch.save({"state_dict": model.state_dict(), "dataset_tag": tag,
                "hidden": int(cfg.lstm.hidden), "layers": int(cfg.lstm.layers)}, out)
    _log(f"[lstm] saved -> {out}  (dataset tag '{tag}')")
    return out
