"""librosa feature extraction (Module M1 — Acoustic Analysis).

Two products are produced from a waveform:

1. `mel_windows`  -> a stack of (n_mels x mel_frames) mel-spectrograms, one per
   3-second window. This is the CNN input.
2. `extract_feature_sequence` -> a hand-crafted feature vector per window
   (MFCC + delta + delta2, chroma, spectral, tempo, ZCR, RMS). This is what the
   PySpark feature store persists as a ~128-dim Parquet vector.

Both are computed on non-overlapping windows so they line up frame-for-frame.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

FEATURE_NAMES = (
    "mfcc(40)+delta+delta2=120",
    "chroma(12)",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_bandwidth",
    "tempo",
    "zcr",
    "rms",
)


def _load_librosa():
    try:
        import librosa
        return librosa
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "librosa is required for feature extraction. Install with "
            "`pip install librosa soundfile`."
        ) from exc


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    librosa = _load_librosa()
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    return y.astype(np.float32)


def _frame_bounds(n_samples: int, win: int, hop: int) -> List[Tuple[int, int]]:
    bounds = []
    start = 0
    while start + win <= max(n_samples, win):
        bounds.append((start, start + win))
        start += hop
    if not bounds:
        bounds.append((0, n_samples))
    return bounds


def mel_windows(y: np.ndarray, cfg) -> np.ndarray:
    """Return array (n_windows, n_mels, mel_frames) of log-mel spectrograms."""
    librosa = _load_librosa()
    sr = cfg.audio.sample_rate
    win = int(cfg.audio.window_seconds * sr)
    hop = int(cfg.audio.hop_seconds * sr)
    n_mels = int(cfg.audio.n_mels)
    target_frames = int(cfg.audio.mel_frames)

    out = []
    for a, b in _frame_bounds(len(y), win, hop):
        seg = y[a:b]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        # hop_length chosen to land near `target_frames` columns.
        hop_length = max(1, win // target_frames)
        mel = librosa.feature.melspectrogram(
            y=seg, sr=sr, n_mels=n_mels, hop_length=hop_length
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        # Fix width to exactly target_frames.
        if logmel.shape[1] < target_frames:
            logmel = np.pad(
                logmel, ((0, 0), (0, target_frames - logmel.shape[1])), mode="edge"
            )
        else:
            logmel = logmel[:, :target_frames]
        # Normalize to roughly [-1, 1].
        logmel = logmel / 80.0
        out.append(logmel.astype(np.float32))
    return np.stack(out, axis=0)


def chroma_windows(y: np.ndarray, cfg) -> np.ndarray:
    """Return (n_windows, n_mels, mel_frames) chroma, aligned with `mel_windows`.

    Mel captures energy and timbre but is close to blind to *harmony* — and
    valence (major/minor, consonance) lives in harmony. Chroma folds the
    spectrum onto the 12 pitch classes, giving the CNN a channel where tonality
    is explicit.

    The CQT is computed once over the whole signal and then sliced per window:
    calling it per 3-second window costs ~4x more for the same result. The 12
    pitch classes are expanded to `n_mels` rows by nearest-neighbour repeat, not
    interpolation, so neighbouring pitch classes are not blended together.
    """
    librosa = _load_librosa()
    sr = cfg.audio.sample_rate
    win = int(cfg.audio.window_seconds * sr)
    hop = int(cfg.audio.hop_seconds * sr)
    n_mels = int(cfg.audio.n_mels)
    target_frames = int(cfg.audio.mel_frames)
    hop_length = max(1, win // target_frames)
    mode = str(getattr(cfg.audio, "chroma_mode", "cqt")).lower()

    try:
        if mode == "stft":
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        else:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    except Exception:
        # CQT can fail on pathologically short/silent input; fall back rather
        # than take the whole pipeline down.
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        except Exception:
            chroma = np.zeros((12, max(1, len(y) // hop_length)), dtype=np.float32)

    # 12 pitch classes -> n_mels rows, nearest-neighbour so classes stay discrete.
    rows = np.floor(np.arange(n_mels) * 12.0 / n_mels).astype(int)
    chroma = chroma[rows]                                    # (n_mels, T)

    out = []
    for a, b in _frame_bounds(len(y), win, hop):
        c0, c1 = a // hop_length, b // hop_length
        seg = chroma[:, c0:c1]
        if seg.shape[1] < target_frames:
            pad = target_frames - seg.shape[1]
            seg = np.pad(seg, ((0, 0), (0, pad)), mode="edge") if seg.shape[1] else \
                np.zeros((n_mels, target_frames), dtype=np.float32)
        else:
            seg = seg[:, :target_frames]
        out.append(seg.astype(np.float32))
    return np.stack(out, axis=0)


def mel_chroma_windows(y: np.ndarray, cfg) -> np.ndarray:
    """Return (n_windows, 2, n_mels, mel_frames): channel 0 mel, channel 1 chroma.

    This is the CNN input. Keeping both channels the same HxW lets a single
    conv stack see energy/timbre and harmony together.
    """
    mel = mel_windows(y, cfg)
    chroma = chroma_windows(y, cfg)
    n = min(len(mel), len(chroma))
    return np.stack([mel[:n], chroma[:n]], axis=1).astype(np.float32)


def extract_window_features(seg: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """Hand-crafted feature vector for a single window segment."""
    librosa = _load_librosa()
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=seg, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(seg)
    rms = librosa.feature.rms(y=seg)
    try:
        tempo = float(librosa.beat.tempo(y=seg, sr=sr)[0])
    except Exception:
        tempo = 0.0

    feats = np.concatenate([
        mfcc.mean(axis=1), d1.mean(axis=1), d2.mean(axis=1),   # 120
        chroma.mean(axis=1),                                    # 12
        centroid.mean(axis=1), rolloff.mean(axis=1),            # 2
        bandwidth.mean(axis=1),                                 # 1
        [tempo],                                                # 1
        zcr.mean(axis=1), rms.mean(axis=1),                     # 2
    ]).astype(np.float32)
    return feats


def extract_feature_sequence(y: np.ndarray, cfg) -> np.ndarray:
    """Return (n_windows, feat_dim) hand-crafted features for a waveform."""
    sr = cfg.audio.sample_rate
    win = int(cfg.audio.window_seconds * sr)
    hop = int(cfg.audio.hop_seconds * sr)
    n_mfcc = int(cfg.audio.n_mfcc)
    seqs = []
    for a, b in _frame_bounds(len(y), win, hop):
        seg = y[a:b]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        seqs.append(extract_window_features(seg, sr, n_mfcc))
    return np.stack(seqs, axis=0)
