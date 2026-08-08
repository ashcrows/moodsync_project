"""Generation engine (Module M3).

* Light mode (default): render audio directly from the target arc using the
  arc-driven additive synthesizer in data/synth.py. This makes the whole loop
  runnable offline and, importantly, keeps a real relationship between the
  requested arc and the generated audio's emotion — so the Arc Match Score is
  meaningful even without MusicGen.

* Full mode (--gen-full, or --full): load MusicGen (facebook/musicgen-small by
  default) and condition on the narrator's text prompt. Falls back to light mode
  on error. Enabled independently of the LLM narrator, so MusicGen can run alone
  on a 24 GB machine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..data.synth import synth_audio_from_arc
from ..models.narrator import mood_word

# MusicGen conditions on a short text caption and degrades on long prose: a
# multi-sentence narration dilutes the musical descriptors it actually acts on.
MAX_CAPTION_CHARS = 200


def condense_caption(text: str, max_chars: int = MAX_CAPTION_CHARS) -> str:
    """Reduce a narration to one short single-line caption for MusicGen.

    Collapses whitespace, keeps at most the first two sentences, and truncates
    on a word boundary. Verbose LLM narrations are shortened here rather than at
    the narrator, so the full text stays available for display and reporting.
    """
    flat = " ".join(str(text or "").split())
    if not flat:
        return ""
    sentences, buf = [], ""
    for ch in flat:
        buf += ch
        if ch in ".!?":
            sentences.append(buf.strip())
            buf = ""
        if len(sentences) == 2:
            break
    if buf.strip() and len(sentences) < 2:
        sentences.append(buf.strip())
    out = " ".join(sentences).strip() or flat
    if len(out) <= max_chars:
        return out
    clipped = out[:max_chars]
    cut = clipped.rfind(" ")
    return (clipped[:cut] if cut > 0 else clipped).rstrip(" ,;:-").rstrip()


class MusicGenerator:
    def __init__(self, cfg, full: bool = False, platform=None):
        self.cfg = cfg
        self.full = full
        self.platform = platform
        self.sr = int(cfg.audio.sample_rate)
        self._model = None
        self._model_sr = None
        if full:
            self._try_load()

    @property
    def model_loaded(self) -> bool:
        """True when MusicGen weights are resident, so callers can report a fallback."""
        return self._model is not None

    def _try_load(self):
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            name = self.cfg.generation.musicgen_model
            device = self.platform.device if self.platform else "cpu"
            self._proc = AutoProcessor.from_pretrained(name)
            self._model = MusicgenForConditionalGeneration.from_pretrained(name).to(device)
            self._model_sr = self._model.config.audio_encoder.sampling_rate
            self._device = device
            print(f"[moodsync] MusicGen '{name}' loaded on {device}.")
        except Exception as exc:
            print(f"[moodsync] MusicGen unavailable ({exc}); using light synthesizer.")
            self._model = None

    def generate(
        self,
        prompt: str,
        target_arc: np.ndarray,
        seconds: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        """Return (waveform, sample_rate).

        `generation.segment_seconds` selects one-shot or segmented generation —
        see `_musicgen`.
        """
        seconds = seconds or float(self.cfg.generation.duration_seconds)
        if self.full and self._model is not None:
            try:
                return self._musicgen(prompt, target_arc, seconds)
            except Exception as exc:
                print(f"[moodsync] MusicGen generate failed ({exc}); using synth.")
        # Light path: arc-conditioned synthesizer.
        y = synth_audio_from_arc(target_arc, sr=self.sr, seconds=seconds, seed=7)
        return y, self.sr

    def _n_segments(self, target_arc: np.ndarray, seconds: float) -> int:
        """How many chunks to split the arc into for segmented generation.

        `segment_seconds <= 0` selects one-shot generation, the default.
        """
        seg_seconds = float(getattr(self.cfg.generation, "segment_seconds", 0.0))
        if seg_seconds <= 0:
            return 1
        arc_points = len(np.asarray(target_arc).reshape(-1, 2))
        return max(1, min(arc_points, int(seconds // seg_seconds)))

    def _musicgen(
        self,
        prompt: str,
        target_arc: np.ndarray,
        seconds: float,
    ) -> tuple[np.ndarray, int]:
        """Generate with MusicGen, one-shot by default.

        MusicGen is text-conditioned only — it never sees the arc. Two modes:

        * One-shot (`segment_seconds <= 0`, the default): one caption for the
          whole arc, one clip. Musically coherent, with no seams, at the cost of
          tracking the trajectory only loosely.
        * Segmented (`segment_seconds > 0`): the arc is split into chunks and a
          clip is generated per chunk, then crossfaded. Every chunk reuses the
          SAME base caption with only a short mood tag appended, so the chunks
          remain the same piece of music; captioning each chunk independently
          instead yields unrelated fragments.
        """
        target_arc = np.asarray(target_arc, dtype=np.float32).reshape(-1, 2)
        n_seg = self._n_segments(target_arc, seconds)
        if n_seg < 2:
            return self._musicgen_once(prompt, seconds)

        base = condense_caption(prompt)
        chunks = np.array_split(target_arc, n_seg)
        seg_seconds = seconds / n_seg
        pieces = []
        for i, chunk in enumerate(chunks):
            tag = mood_word(float(chunk[:, 0].mean()), float(chunk[:, 1].mean()))
            caption = f"{base} Section {i + 1}: {tag}."
            y, _ = self._musicgen_once(caption, seg_seconds)
            pieces.append(y)
            print(f"[moodsync] segment {i+1}/{n_seg}: {tag}")
        return self._crossfade(pieces, int(self._model_sr)), int(self._model_sr)

    def _musicgen_once(self, prompt: str, seconds: float) -> tuple[np.ndarray, int]:
        import torch
        # Single choke point for every MusicGen prompt, so both the rule-based
        # and the LLM narrator end up within the length the model handles well.
        text = condense_caption(prompt)
        max_new = max(16, int(seconds * 50))  # MusicGen ~50 tokens/sec
        inputs = self._proc(text=[text], padding=True, return_tensors="pt").to(self._device)
        with torch.no_grad():
            audio = self._model.generate(**inputs, max_new_tokens=max_new)
        y = audio[0, 0].cpu().numpy().astype(np.float32)
        return y, int(self._model_sr)

    @staticmethod
    def _trim_edges(y: np.ndarray, sr: int, thresh: float = 0.02,
                    max_trim_seconds: float = 0.75) -> np.ndarray:
        """Drop near-silent head/tail from one segment.

        MusicGen clips often fade in and out. Crossfading them untrimmed puts
        two quiet regions on top of each other, which is heard as a gap at the
        seam. The trim is capped so a genuinely quiet segment is not gutted.
        """
        y = np.asarray(y, dtype=np.float32)
        if y.size == 0:
            return y
        loud = np.flatnonzero(np.abs(y) > thresh)
        if loud.size == 0:
            return y
        limit = max(0, int(sr * max_trim_seconds))
        start = min(int(loud[0]), limit)
        end = max(int(loud[-1]) + 1, y.size - limit)
        return y[start:end] if end > start else y

    @classmethod
    def _crossfade(cls, pieces: list, sr: int, fade_seconds: float = 0.35) -> np.ndarray:
        """Concatenate segments with an overlapping equal-power crossfade.

        The blend consumes `fade` samples of the running output and the same
        number from the next segment, so the segments genuinely overlap rather
        than being butted together with a ramp.
        """
        pieces = [cls._trim_edges(p, sr) for p in pieces if np.asarray(p).size]
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        if len(pieces) == 1:
            return np.asarray(pieces[0], dtype=np.float32)
        fade = max(1, int(sr * fade_seconds))
        out = pieces[0]
        for nxt in pieces[1:]:
            n = min(fade, len(out), len(nxt))
            if n < 2:
                out = np.concatenate([out, nxt])
                continue
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            head, tail = out[:-n], out[-n:]
            blended = tail * np.cos(ramp * np.pi / 2) + nxt[:n] * np.sin(ramp * np.pi / 2)
            out = np.concatenate([head, blended, nxt[n:]])
        # Equal-power summing can exceed unity where segments are correlated;
        # rescale so the seam cannot clip on playback.
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32)


def save_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), np.asarray(y, dtype=np.float32), sr)
