"""End-to-end MoodSync pipeline: analyse -> narrate -> generate -> re-analyse -> score.

This is the round-trip loop that produces the Arc Match Score. Load a trained
CNN + LSTM (or train them first via train.py), then:

    ms = MoodSync(cfg, platform, full=False)
    result = ms.run_loop(target_arc)      # generate audio for a target arc & score it
    analysis = ms.analyze("song.wav")     # just extract the arc of an existing song

The two heavy subsystems can be enabled independently — MusicGen fits in 24 GB
while the 7B LoRA narrator does not:

    ms = MoodSync(cfg, platform, gen_full=True)        # MusicGen only
    ms = MoodSync(cfg, platform, narrator_full=True)   # LLM narrator only
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config, dataset_tag
from .features.extract import load_audio, mel_chroma_windows, mel_windows
from .generation.arc_match import arc_match_score, arc_shape_score
from .generation.musicgen import MusicGenerator, save_wav
from .models.cnn import MoodCNN
from .models.lstm_smoother import ArcSmoother, detect_sections
from .models.narrator import ArcNarrator
from .platform_utils import Platform, artifacts_dir


def generated_wav_path(cfg: Config) -> Path:
    """Locate the generated clip, preferring this config's dataset-scoped copy.

    Mirrors the checkpoint lookup: `artifacts/<tag>/generated.wav` first, then
    the flat `artifacts/generated.wav` of the unscoped layout. When neither
    exists the scoped path is returned, so callers report the location a fresh
    run will actually use.
    """
    scoped = artifacts_dir(dataset_tag(cfg)) / "generated.wav"
    if scoped.exists():
        return scoped
    legacy = artifacts_dir() / "generated.wav"
    return legacy if legacy.exists() else scoped


def config_path_from_env(env=None) -> Optional[str]:
    """The config path a server host should load, or None for the default.

    `serve-app`/`serve-api` export the resolved `--config` as MOODSYNC_CONFIG so
    the UI and API load the same dataset the CLI was pointed at. Unset or empty
    selects the default config.yaml.
    """
    import os
    env = os.environ if env is None else env
    return env.get("MOODSYNC_CONFIG") or None


def full_flags_from_env(env=None) -> tuple[bool, bool]:
    """Read (gen_full, narrator_full) from the environment, for the app/API hosts.

    `MOODSYNC_FULL` is honoured as a backward-compatible alias enabling both, so
    an older launcher that only sets it keeps working.
    """
    import os
    env = os.environ if env is None else env
    full = env.get("MOODSYNC_FULL", "0") == "1"
    return (full or env.get("MOODSYNC_GEN_FULL", "0") == "1",
            full or env.get("MOODSYNC_NARRATOR_FULL", "0") == "1")


class MoodSync:
    def __init__(
        self,
        cfg: Config,
        platform: Platform,
        full: bool = False,
        gen_full: Optional[bool] = None,
        narrator_full: Optional[bool] = None,
    ):
        """The two heavy subsystems are enabled independently.

        `gen_full` (MusicGen) and `narrator_full` (LoRA LLM) each default to
        `full` when left as None, so the original `MoodSync(cfg, platform,
        full=True)` call still turns on both.
        """
        self.cfg = cfg
        self.platform = platform
        self.gen_full = full if gen_full is None else bool(gen_full)
        self.narrator_full = full if narrator_full is None else bool(narrator_full)
        self.device = platform.device
        self.cnn = self._load_cnn()
        self.lstm = self._load_lstm()
        self.narrator = ArcNarrator(cfg, full=self.narrator_full, platform=platform)
        self.generator = MusicGenerator(cfg, full=self.gen_full, platform=platform)

    @property
    def full(self) -> bool:
        """True only when BOTH heavy subsystems are on (back-compat for callers)."""
        return self.gen_full and self.narrator_full

    # ---- model loading ---------------------------------------------------
    def _checkpoint(self, name: str) -> Optional[Path]:
        """Locate a checkpoint, preferring this config's dataset-scoped copy.

        Checkpoints live in `artifacts/<tag>/` so a synthetic `demo` run cannot
        overwrite DEAM-trained weights. Older flat `artifacts/<name>` files are
        still honoured, so existing checkouts keep working.
        """
        scoped = artifacts_dir(dataset_tag(self.cfg)) / name
        if scoped.exists():
            return scoped
        legacy = artifacts_dir() / name
        return legacy if legacy.exists() else None

    def _load_cnn(self) -> Optional[MoodCNN]:
        p = self._checkpoint("cnn.pt")
        if p is None:
            return None
        ckpt = torch.load(p, map_location=self.device)
        m = MoodCNN(n_mels=ckpt["n_mels"],
                    in_channels=int(ckpt.get("in_channels", 1)),
                    dropout=float(ckpt.get("dropout", 0.0))).to(self.device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        return m

    def _load_lstm(self) -> Optional[ArcSmoother]:
        p = self._checkpoint("lstm.pt")
        if p is None:
            return None
        ckpt = torch.load(p, map_location=self.device)
        m = ArcSmoother(hidden=ckpt["hidden"], layers=ckpt["layers"]).to(self.device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        return m

    def _cnn_input(self, y: np.ndarray) -> np.ndarray:
        """Build the CNN input matching the LOADED checkpoint's channel count.

        The checkpoint is authoritative, not config.yaml: a model trained on
        mel-only must keep receiving mel-only even if the config later asks for
        the 2-channel mel+chroma input.
        """
        want = int(getattr(self.cnn, "in_channels", 1)) if self.cnn is not None else 1
        if want >= 2:
            return mel_chroma_windows(y, self.cfg)
        return mel_windows(y, self.cfg)

    # ---- analysis (Modules M1 + M2) -------------------------------------
    def analyze(self, audio_path: str) -> dict:
        if self.cnn is None:
            raise RuntimeError("No trained CNN found. Run `moodsync demo` or `moodsync train` first.")
        y = load_audio(audio_path, self.cfg.audio.sample_rate)
        mels = torch.tensor(self._cnn_input(y)).to(self.device)
        with torch.no_grad():
            raw = self.cnn(mels).cpu().numpy()             # noisy per-window arc
            if self.lstm is not None:
                smoothed = self.lstm(
                    torch.tensor(raw).unsqueeze(0).to(self.device)
                ).squeeze(0).cpu().numpy()
            else:
                smoothed = raw
        sections = detect_sections(smoothed)
        prompt = self.narrator.narrate(smoothed, sections)
        return {
            "raw_arc": raw,
            "arc": smoothed,
            "sections": sections,
            "prompt": prompt,
        }

    # ---- re-analysis of generated audio ---------------------------------
    def _reanalyze_array(self, y: np.ndarray, sr: int) -> np.ndarray:
        # Resample to the model's sample rate if needed.
        if sr != self.cfg.audio.sample_rate:
            try:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg.audio.sample_rate)
            except Exception:
                pass
        mels = torch.tensor(self._cnn_input(y)).to(self.device)
        with torch.no_grad():
            raw = self.cnn(mels).cpu().numpy()
            if self.lstm is not None:
                raw = self.lstm(
                    torch.tensor(raw).unsqueeze(0).to(self.device)
                ).squeeze(0).cpu().numpy()
        return raw

    # ---- full round-trip loop -------------------------------------------
    def run_loop(self, target_arc: np.ndarray, out_wav: Optional[str] = None) -> dict:
        """Target arc -> prompt -> generate -> re-analyse -> Arc Match Score."""
        if self.cnn is None:
            raise RuntimeError("No trained CNN found. Run `moodsync demo` or `moodsync train` first.")
        target_arc = np.asarray(target_arc, dtype=np.float32).reshape(-1, 2)
        sections = detect_sections(target_arc)
        prompt = self.narrator.narrate(target_arc, sections)

        y, sr = self.generator.generate(prompt, target_arc)
        if out_wav is None:
            # Scope by dataset tag so a demo clip cannot overwrite a DEAM one.
            out_wav = str(artifacts_dir(dataset_tag(self.cfg)) / "generated.wav")
        save_wav(out_wav, y, sr)

        gen_arc = self._reanalyze_array(y, sr)
        n_points = int(self.cfg.arcmatch.resample_points)
        score = arc_match_score(target_arc, gen_arc, n_points)
        shape = arc_shape_score(target_arc, gen_arc, n_points)
        return {
            "prompt": prompt,
            "sections": sections,
            "target_arc": target_arc,
            "generated_arc": gen_arc,
            "arc_match_score": score,
            "arc_shape_score": shape,
            "wav_path": out_wav,
            "sample_rate": sr,
        }
