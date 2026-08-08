"""Tiny YAML config loader with attribute-style access."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .platform_utils import project_root


class Config(dict):
    """dict that also supports cfg.audio.sample_rate style access."""

    def __getattr__(self, item: str) -> Any:
        try:
            val = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
            self[item] = val
        return val


def dataset_tag(cfg: Config) -> str:
    """Which dataset this config trains on: 'deam' when DEAM resolves, else 'demo'.

    Checkpoints are scoped by this tag so a quick `demo` run cannot clobber
    DEAM-trained weights. Override with an explicit `artifacts.tag` in the
    config to keep several real-data runs side by side.
    """
    try:
        explicit = str(cfg.artifacts.tag or "").strip()
    except AttributeError:
        explicit = ""
    if explicit:
        return explicit
    try:
        audio = str(cfg.datasets.deam_audio or "").strip()
        ann = str(cfg.datasets.deam_annotations or "").strip()
    except AttributeError:
        return "demo"
    if audio and ann and Path(audio).exists() and Path(ann).exists():
        return "deam"
    return "demo"


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = project_root() / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw)
