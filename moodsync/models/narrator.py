"""Arc -> natural-language prompt narrator (Module M2, step 4).

Two backends:

* Rule-based (default, light): deterministic mapping from the sectioned arc to a
  MusicGen-ready caption. Runs everywhere, no downloads.
* LoRA LLM (--narrator-full, or --full): loads a PEFT/LoRA-adapted instruction
  model (Mistral-7B by default) and asks it to narrate the arc. Falls back to
  rule-based on any error. Enabled independently of MusicGen, since the 7B model
  is the memory-hungry half of the heavy path.

Both expose the same `.narrate(arc, sections) -> str` interface.
"""
from __future__ import annotations

from typing import List

import numpy as np


def mood_word(valence: float, arousal: float) -> str:
    """One-word mood label for a (valence, arousal) pair.

    Shared with the generator, which appends it to a base caption as a short
    per-segment tag rather than re-narrating each segment from scratch.
    """
    v = "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
    a = "high-energy" if arousal > 0.2 else "calm" if arousal < -0.2 else "moderate"
    mood = {
        ("positive", "high-energy"): "euphoric",
        ("positive", "calm"): "serene",
        ("positive", "moderate"): "warm",
        ("negative", "high-energy"): "tense",
        ("negative", "calm"): "melancholic",
        ("negative", "moderate"): "somber",
        ("neutral", "high-energy"): "driving",
        ("neutral", "calm"): "ambient",
        ("neutral", "moderate"): "steady",
    }[(v, a)]
    return mood


class ArcNarrator:
    def __init__(self, cfg, full: bool = False, platform=None):
        self.cfg = cfg
        self.full = full
        self.platform = platform
        self._llm = None
        if full:
            self._try_load_llm()

    # ---- rule-based backend ---------------------------------------------
    def _rule_narrate(self, arc: np.ndarray, sections: List[dict]) -> str:
        arc = np.asarray(arc, dtype=np.float32).reshape(-1, 2)
        phrases = []
        for seg in sections:
            s, e = seg["start"], seg["end"]
            v = float(arc[s:e, 0].mean())
            a = float(arc[s:e, 1].mean())
            phrases.append(f"{seg['label']}: {mood_word(v, a)}")
        # Overall descriptor.
        v_all, a_all = float(arc[:, 0].mean()), float(arc[:, 1].mean())
        overall = mood_word(v_all, a_all)
        trend = "rising" if arc[-1, 1] > arc[0, 1] else "resolving"
        return (
            f"A {overall}, instrumental piece with a {trend} emotional arc; "
            + "; ".join(phrases) + "."
        )

    # ---- LoRA LLM backend -----------------------------------------------
    def _try_load_llm(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            name = self.cfg.narrator.base_model
            device = self.platform.device if self.platform else "cpu"
            self._tok = AutoTokenizer.from_pretrained(name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float16 if device != "cpu" else torch.float32
            ).to(device)
            # Optional: attach a trained LoRA adapter if present in artifacts/lora/.
            try:
                from peft import PeftModel
                from ..platform_utils import artifacts_dir
                adapter = artifacts_dir() / "lora"
                if adapter.exists():
                    self._llm = PeftModel.from_pretrained(self._llm, str(adapter))
                    print(f"[moodsync] loaded LoRA adapter from {adapter}")
            except Exception:
                pass  # base model is fine for the demo
            self._device = device
        except Exception as exc:
            print(f"[moodsync] LLM narrator unavailable ({exc}); using rule-based.")
            self._llm = None

    def _llm_narrate(self, arc: np.ndarray, sections: List[dict]) -> str:
        arc = np.asarray(arc, dtype=np.float32).reshape(-1, 2)
        arc_str = "; ".join(
            f"{seg['label']}: valence={float(arc[seg['start']:seg['end'],0].mean()):+.2f}, "
            f"arousal={float(arc[seg['start']:seg['end'],1].mean()):+.2f}"
            for seg in sections
        )
        prompt = (
            "[INST] You caption music for a generator. Convert this emotional "
            "arc into ONE vivid instrumental music caption (no lyrics):\n"
            f"{arc_str} [/INST]"
        )
        import torch
        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._llm.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.8)
        text = self._tok.decode(out[0], skip_special_tokens=True)
        return text.split("[/INST]")[-1].strip() or self._rule_narrate(arc, sections)

    # ---- public ----------------------------------------------------------
    def narrate(self, arc, sections) -> str:
        if self.full and self._llm is not None:
            try:
                return self._llm_narrate(arc, sections)
            except Exception as exc:
                print(f"[moodsync] LLM narrate failed ({exc}); using rule-based.")
        return self._rule_narrate(arc, sections)
