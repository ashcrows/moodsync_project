"""Synthetic demo dataset + arc-driven audio synthesizer.

Purpose: let the ENTIRE pipeline (train CNN, train LSTM, generate, score) run
end-to-end on a laptop with no external downloads. Each synthetic clip is built
from a known valence-arousal arc, which yields ground-truth labels for free and
gives the CNN something real to learn (arousal <-> brightness/energy/tempo,
valence <-> major/minor tonality).

The same synthesizer doubles as the light-mode music generator: given a target
arc, render audio whose emotion roughly follows it (see generation/musicgen.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), y.astype(np.float32), sr)


# Major and (natural) minor scale semitone offsets, used to make valence audible.
_MAJOR = [0, 2, 4, 5, 7, 9, 11, 12]
_MINOR = [0, 2, 3, 5, 7, 8, 10, 12]


def synth_audio_from_arc(
    arc: np.ndarray,
    sr: int = 22050,
    seconds: float = 15.0,
    seed: int = 0,
) -> np.ndarray:
    """Render a waveform whose emotion follows `arc` (shape (N, 2) = valence, arousal).

    valence in [-1,1] -> major (high) vs minor (low) tonality + consonance.
    arousal in [-1,1] -> tempo, note density, brightness (harmonics), loudness.
    """
    rng = np.random.default_rng(seed)
    arc = np.asarray(arc, dtype=np.float32).reshape(-1, 2)
    n_samples = int(sr * seconds)
    t = np.arange(n_samples) / sr
    y = np.zeros(n_samples, dtype=np.float32)

    # Interpolate the arc to per-sample control signals.
    idx = np.linspace(0, len(arc) - 1, n_samples)
    valence = np.interp(idx, np.arange(len(arc)), arc[:, 0])
    arousal = np.interp(idx, np.arange(len(arc)), arc[:, 1])

    base_freq = 220.0  # A3
    # Note rate (events/sec) grows with arousal.
    seg_len = max(1, int(sr * 0.25))
    n_events = n_samples // seg_len
    for e in range(n_events):
        s = e * seg_len
        en = min(n_samples, s + seg_len)
        v = float(valence[s])
        a = float(arousal[s])
        # Valence encodes BOTH tonality (major/minor) and register: positive
        # valence transposes up to ~+7 semitones, negative down. The register
        # shift moves spectral content in the mel-spectrogram in a way the CNN
        # can learn, and stays orthogonal to arousal (which drives energy/
        # brightness), so valence and arousal are separately recoverable.
        scale = _MAJOR if v >= 0 else _MINOR
        degree = int(rng.integers(0, len(scale)))
        transpose = v * 7.0
        semitone = scale[degree] + transpose
        freq = base_freq * (2 ** (semitone / 12.0))
        # Brightness: number of harmonics rises with arousal.
        n_harm = 1 + int(round((a + 1) * 2.5))
        seg_t = t[s:en] - t[s]
        wave = np.zeros(en - s, dtype=np.float32)
        for h in range(1, n_harm + 1):
            wave += (1.0 / h) * np.sin(2 * np.pi * freq * h * seg_t)
        # Valence pilot partial: a steady tone whose frequency rises monotonically
        # with valence (v=-1 -> ~400 Hz, v=+1 -> ~3200 Hz). This puts an
        # unambiguous, learnable mel peak position on valence that is independent
        # of the random melodic degree, so the CNN can recover valence cleanly.
        pilot_freq = 400.0 * (2 ** ((v + 1.0) * 1.5))
        wave += 0.35 * np.sin(2 * np.pi * pilot_freq * seg_t)
        # Amplitude envelope: sharper attack for high arousal.
        env = np.exp(-seg_t * (1.5 + (1 - a)))
        loud = 0.2 + 0.5 * (a + 1) / 2
        y[s:en] += (wave * env * loud).astype(np.float32)

    # Light noise for texture, more when arousal high.
    y += (0.02 * (arousal.mean() + 1) * rng.standard_normal(n_samples)).astype(np.float32)
    # Normalize.
    peak = np.max(np.abs(y)) + 1e-8
    y = 0.9 * y / peak
    return y.astype(np.float32)


def _random_arc(rng: np.random.Generator, n: int = 8) -> np.ndarray:
    """A smooth random valence-arousal arc with a clear shape (verse->chorus...)."""
    # Control points, then smooth.
    v = rng.uniform(-0.8, 0.8, size=4)
    a = rng.uniform(-0.8, 0.8, size=4)
    xs = np.linspace(0, 1, 4)
    q = np.linspace(0, 1, n)
    varc = np.interp(q, xs, v)
    aarc = np.interp(q, xs, a)
    return np.stack([varc, aarc], axis=1).astype(np.float32)


def generate_demo_dataset(
    out_dir: str | Path,
    n_clips: int = 24,
    clip_seconds: float = 15.0,
    sr: int = 22050,
    windows_per_clip: int = 5,
    seed: int = 42,
    signature: str | None = None,
) -> Dict:
    """Create synthetic audio clips + a manifest of per-window V/A labels.

    Returns the manifest dict; also writes <out_dir>/manifest.json and wavs.
    `signature` (if given) is stored in the manifest so callers can detect a
    stale cache when the config or synth source changes.
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    # Clear stale WAVs first: shrinking n_clips would otherwise leave orphans
    # (e.g. clip_056..063) that the features glob audio/*.wav would sweep into
    # the feature store even though the manifest no longer lists them.
    if audio_dir.exists():
        for old in audio_dir.glob("*.wav"):
            old.unlink()
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest: Dict = {"sample_rate": sr, "signature": signature, "clips": []}

    for i in range(n_clips):
        arc = _random_arc(rng, n=windows_per_clip)
        y = synth_audio_from_arc(arc, sr=sr, seconds=clip_seconds, seed=int(rng.integers(1e6)))
        wav_path = out_dir / "audio" / f"clip_{i:03d}.wav"
        _write_wav(wav_path, y, sr)
        manifest["clips"].append({
            "id": f"clip_{i:03d}",
            "path": str(wav_path),
            "arc": arc.tolist(),   # per-window [valence, arousal] ground truth
        })

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
