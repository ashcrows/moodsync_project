"""DEAM tests.

The loader/round-trip tests SKIP when DEAM is not configured, so `pytest -q`
stays green on a clean checkout. The config-rewriting helper is pure text
manipulation, so it is always exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from moodsync.config import load_config                      # noqa: E402
from moodsync.data.deam import load_deam                     # noqa: E402
from prepare_deam import _replace_value, update_config       # noqa: E402


def _deam_configured() -> bool:
    """True when some config on this machine points at a real DEAM copy.

    The committed config.yaml keeps the paths blank so demo/train stay on the
    synthetic sample, so the real paths live in the gitignored config.deam.yaml.
    Check that first, so these tests run whenever a real copy is configured.
    """
    for name in ("config.deam.yaml", "config.local.yaml", "config.yaml"):
        if not Path(name).exists():
            continue
        try:
            cfg = load_config(name)
        except Exception:
            continue
        audio = (cfg.datasets.deam_audio or "").strip()
        ann = (cfg.datasets.deam_annotations or "").strip()
        if audio and ann and Path(audio).exists() and Path(ann).exists():
            _DEAM_CONFIG.append(name)
            return True
    return False


_DEAM_CONFIG: list = []

requires_deam = pytest.mark.skipif(
    not _deam_configured(),
    reason="DEAM not configured; see README (config.deam.yaml) or run scripts/prepare_deam.py",
)


# ---- always-on: the config rewriter must never eat comments -----------------
def test_replace_value_preserves_comment_and_indent():
    text = 'datasets:\n  deam_audio: ""          # e.g. /path/to/DEAM/audio\n'
    out, changed = _replace_value(text, "deam_audio", "/data/DEAM/audio")
    assert changed
    assert '"/data/DEAM/audio"' in out
    assert "# e.g. /path/to/DEAM/audio" in out   # trailing comment survives
    assert out.startswith("datasets:\n  deam_audio:")  # indentation survives


def test_replace_value_is_idempotent():
    text = 'a:\n  deam_audio: "/x"   # c\n'
    once, changed_1 = _replace_value(text, "deam_audio", "/x")
    assert not changed_1 and once == text


def test_replace_value_missing_key_raises():
    with pytest.raises(KeyError):
        _replace_value("a:\n  other: 1\n", "deam_audio", "/x")


def test_update_config_roundtrip_preserves_other_lines(tmp_path):
    src = Path("config.yaml").read_text(encoding="utf-8")
    tmp = tmp_path / "config.yaml"
    tmp.write_text(src, encoding="utf-8")
    assert update_config(tmp, Path("/d/audio"), Path("/d/ann"))
    out = tmp.read_text(encoding="utf-8")
    # Only the two dataset lines may differ.
    diff = [(a, b) for a, b in zip(src.splitlines(), out.splitlines()) if a != b]
    assert len(diff) == 2
    assert all("deam_" in b for _, b in diff)
    assert not update_config(tmp, Path("/d/audio"), Path("/d/ann"))  # idempotent


# ---- skipped unless real DEAM is present -----------------------------------
@requires_deam
def test_load_deam_returns_clips():
    cfg = load_config(_DEAM_CONFIG[0])       # the config that actually has DEAM
    clips = load_deam(cfg, max_songs=3)
    assert clips, "DEAM configured but loader returned nothing"
    for c in clips:
        assert Path(c["path"]).exists()
        arc = c["arc"]
        assert arc.ndim == 2 and arc.shape[1] == 2
        assert float(arc.min()) >= -1.5 and float(arc.max()) <= 1.5  # mapped to [-1,1]
