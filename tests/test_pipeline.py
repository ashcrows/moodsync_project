"""Smoke tests: verify the core pieces work without heavy models or datasets."""
from __future__ import annotations

import numpy as np

from moodsync.config import load_config
from moodsync.generation.arc_match import arc_match_score, arc_shape_score, resample_arc
from moodsync.models.lstm_smoother import detect_sections
from moodsync.platform_utils import get_platform, resolve_os


def test_resolve_os():
    assert resolve_os("mac") == "mac"
    assert resolve_os("auto") in ("mac", "linux", "windows")


def test_arc_match_identity():
    arc = np.array([[0.2, 0.3], [0.5, -0.1], [-0.4, 0.8]], dtype=np.float32)
    assert arc_match_score(arc, arc) > 0.999


def test_arc_match_opposite():
    arc = np.array([[0.5, 0.5], [0.6, 0.6]], dtype=np.float32)
    assert arc_match_score(arc, -arc) < 0.0


def test_shape_score_identity_and_inverse():
    arc = np.array([[0.2, 0.3], [0.5, -0.1], [-0.4, 0.8]], dtype=np.float32)
    assert arc_shape_score(arc, arc) > 0.999
    assert arc_shape_score(arc, -arc) < -0.999


def test_shape_score_ignores_flat_arcs():
    """A trajectory-free arc must score ~0, not the ~0.42 raw cosine gives it."""
    target = np.array([[0.3, -0.5], [0.6, 0.6], [-0.3, 0.4], [0.2, -0.4]], dtype=np.float32)
    flat = np.tile(target.mean(axis=0), (len(target), 1))
    assert arc_match_score(target, flat) > 0.3        # uncentered cosine credits a flat arc
    assert abs(arc_shape_score(target, flat)) < 1e-6  # centering removes the credit


def test_masked_mse_ignores_padding():
    """Padded timesteps must not contribute — DEAM songs vary 60..1223 samples."""
    import torch

    from moodsync.train import length_mask, masked_mse

    max_T = 5
    mask = length_mask(np.array([5, 2]), max_T)
    pred = torch.zeros(2, max_T, 2)
    target = torch.zeros(2, max_T, 2)
    target[0, :, :] = 1.0        # 5 real steps, error 1 each
    target[1, :2, :] = 1.0       # 2 real steps, error 1 each
    target[1, 2:, :] = 99.0      # padding: must be ignored entirely
    assert abs(float(masked_mse(pred, target, mask)) - 1.0) < 1e-6
    assert float(torch.nn.functional.mse_loss(pred, target)) > 100  # unmasked loss is padding-dominated
    assert int(mask.sum()) == 7


def test_length_mask_all_equal_is_noop():
    """Equal-length sequences (the synthetic demo) must mask nothing."""
    from moodsync.train import length_mask

    mask = length_mask(np.array([4, 4, 4]), 4)
    assert int(mask.sum()) == 12 and mask.shape == (3, 4, 1)


def test_arc_alignment_respects_annotation_start():
    """DEAM ratings begin at 15s; windows before that must not be mislabelled."""
    from moodsync.train import align_arc_to_windows

    cfg = load_config()
    arc = np.stack([np.linspace(-1, 1, 60), np.linspace(-1, 1, 60)], axis=1).astype(np.float32)
    clip = {"arc": arc, "arc_start_s": 15.0, "arc_hop_s": 0.5}
    labels, keep = align_arc_to_windows(clip, 15, cfg)

    centres = np.arange(15) * float(cfg.audio.hop_seconds) + float(cfg.audio.window_seconds) / 2
    assert not keep[:5].any(), "windows before 15s must be dropped"
    assert keep[5:].all()
    times = 15.0 + np.arange(60) * 0.5
    assert abs(labels[5, 0] - np.interp(centres[5], times, arc[:, 0])) < 1e-5


def test_arc_alignment_synthetic_unchanged():
    """Clips without timing metadata keep the stretch-to-fit behaviour."""
    from moodsync.train import align_arc_to_windows

    cfg = load_config()
    arc = np.stack([np.linspace(-1, 1, 8), np.linspace(1, -1, 8)], axis=1).astype(np.float32)
    labels, keep = align_arc_to_windows({"arc": arc}, 6, cfg)
    assert keep.all() and labels.shape == (6, 2)


def _tone(seconds=9.0, sr=22050, hz=440.0):
    return np.sin(2 * np.pi * hz * np.arange(int(sr * seconds)) / sr).astype(np.float32)


def test_chroma_channel_matches_mel_geometry():
    """The harmony channel must be the same HxW as mel so they can be stacked."""
    from moodsync.features.extract import chroma_windows, mel_chroma_windows, mel_windows

    cfg = load_config()
    y = _tone()
    mel, chroma = mel_windows(y, cfg), chroma_windows(y, cfg)
    assert chroma.shape == mel.shape
    stacked = mel_chroma_windows(y, cfg)
    assert stacked.shape == (mel.shape[0], 2, mel.shape[1], mel.shape[2])


def test_cnn_accepts_two_channels():
    import torch

    from moodsync.models.cnn import MoodCNN

    net = MoodCNN(n_mels=128, in_channels=2, dropout=0.3)
    out = net(torch.zeros(3, 2, 128, 128))
    assert out.shape == (3, 2)
    assert float(out.abs().max()) <= 1.0          # tanh-bounded


def test_spec_augment_never_touches_chroma():
    """Masking pitch classes would delete the harmony signal chroma carries."""
    import torch

    from moodsync.train import spec_augment

    x = torch.ones(8, 2, 128, 128)
    out = spec_augment(x, time_mask=16, freq_mask=16, channel=0)
    assert int((out[:, 1] == 0).sum()) == 0, "chroma channel must be untouched"
    assert int((out[:, 0] == 0).sum()) > 0, "mel channel should be masked"


def test_three_way_split_has_no_song_overlap():
    from moodsync.train import _group_split_3way

    groups = np.repeat(np.arange(100), 5)
    tr, va, te, sizes = _group_split_3way(groups, seed=42)
    s = [set(groups[i.numpy()].tolist()) for i in (tr, va, te)]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
    assert len(s[0] | s[1] | s[2]) == 100
    assert sum(sizes) == 100


def test_dataset_tag_defaults_to_demo_when_deam_absent():
    """Blank/missing DEAM paths must scope checkpoints to the demo tag."""
    from moodsync.config import Config, dataset_tag

    blank = Config({"datasets": Config({"deam_audio": "", "deam_annotations": ""})})
    assert dataset_tag(blank) == "demo"
    missing = Config({"datasets": Config({"deam_audio": "/nope/a",
                                          "deam_annotations": "/nope/b"})})
    assert dataset_tag(missing) == "demo"


def test_dataset_tag_explicit_override_wins(tmp_path):
    from moodsync.config import Config, dataset_tag

    cfg = Config({"artifacts": Config({"tag": "run17"}),
                  "datasets": Config({"deam_audio": str(tmp_path),
                                      "deam_annotations": str(tmp_path)})})
    assert dataset_tag(cfg) == "run17"


def test_dataset_tag_detects_real_paths(tmp_path):
    from moodsync.config import Config, dataset_tag

    a, b = tmp_path / "audio", tmp_path / "ann"
    a.mkdir(); b.mkdir()
    cfg = Config({"datasets": Config({"deam_audio": str(a), "deam_annotations": str(b)})})
    assert dataset_tag(cfg) == "deam"


def test_artifacts_dir_is_scoped_by_tag():
    """demo and deam checkpoints must land in different directories."""
    from moodsync.platform_utils import artifacts_dir

    root, demo, deam = artifacts_dir(), artifacts_dir("demo"), artifacts_dir("deam")
    assert demo.name == "demo" and deam.name == "deam"
    assert demo.parent == root and deam.parent == root
    assert demo != deam


def test_resolve_seed_defaults_and_overrides():
    """A missing/blank seed must fall back to 42 so recorded results reproduce."""
    from moodsync.config import Config
    from moodsync.train import DEFAULT_SEED, resolve_seed

    assert DEFAULT_SEED == 42
    assert resolve_seed(Config({})) == 42               # key absent
    assert resolve_seed(Config({"seed": None})) == 42   # key present but empty
    assert resolve_seed(Config({"seed": 7})) == 7
    assert resolve_seed(Config({"seed": "123"})) == 123  # YAML may yield a string
    assert resolve_seed(load_config()) == 42            # committed config.yaml


def test_different_seeds_give_different_splits():
    """The seed must actually reach the split, or a multi-seed run is a no-op."""
    from moodsync.train import _group_split_3way

    groups = np.repeat(np.arange(100), 5)
    a = _group_split_3way(groups, seed=42)[0].numpy()
    b = _group_split_3way(groups, seed=7)[0].numpy()
    assert not np.array_equal(a, b), "seed 7 must not reproduce seed 42's split"
    assert np.array_equal(a, _group_split_3way(groups, seed=42)[0].numpy())


def test_generated_wav_prefers_scoped_copy_then_falls_back():
    """Demo and DEAM clips must not share one artifacts/generated.wav."""
    import shutil

    from moodsync.config import Config
    from moodsync.pipeline import generated_wav_path
    from moodsync.platform_utils import artifacts_dir

    tag = "pytest_tmp_tag"
    cfg = Config({"artifacts": Config({"tag": tag})})
    scoped = artifacts_dir(tag) / "generated.wav"
    try:
        scoped.write_bytes(b"RIFF")
        assert generated_wav_path(cfg) == scoped, "scoped copy must win"
        scoped.unlink()
        # No scoped copy: degrade to the flat path of the unscoped layout.
        fallback = generated_wav_path(cfg)
        assert fallback.name == "generated.wav"
        assert fallback.parent.name in ("artifacts", tag)
    finally:
        shutil.rmtree(artifacts_dir(tag), ignore_errors=True)


def test_config_path_from_env():
    """MOODSYNC_CONFIG lets the UI/API load the same dataset as the CLI."""
    from moodsync.pipeline import config_path_from_env

    assert config_path_from_env({}) is None                       # unset
    assert config_path_from_env({"MOODSYNC_CONFIG": ""}) is None   # no --config
    assert config_path_from_env({"MOODSYNC_CONFIG": "/x/c.yaml"}) == "/x/c.yaml"


def test_serve_config_env_is_absolute():
    """Server hosts may run from another cwd, so the path must be absolute."""
    from pathlib import Path

    from moodsync.cli import _config_env

    class A:
        config = "config.yaml"

    resolved = _config_env(A())
    assert Path(resolved).is_absolute() and Path(resolved).name == "config.yaml"

    class B:
        config = None

    assert _config_env(B()) == ""   # empty -> hosts keep the default config


def test_condense_caption_keeps_musicgen_prompts_short():
    """MusicGen degrades on long prose, so verbose narrations must be trimmed."""
    from moodsync.generation.musicgen import MAX_CAPTION_CHARS, condense_caption

    verbose = ("Imagine a sweeping cinematic journey.\n\n" + "Lush strings swell. " * 40)
    out = condense_caption(verbose)
    assert len(out) <= MAX_CAPTION_CHARS
    assert "\n" not in out
    assert not out.endswith(" ")

    short = "A warm, instrumental piece with a rising arc."
    assert condense_caption(short) == short          # already short: unchanged
    assert condense_caption("") == ""
    assert condense_caption(None) == ""


def test_one_shot_is_the_default_segment_mode():
    """segment_seconds <= 0 must collapse to a single un-chunked generation."""
    from moodsync.config import Config
    from moodsync.generation.musicgen import MusicGenerator

    arc = np.zeros((8, 2), dtype=np.float32)

    def gen(seg):
        cfg = Config({"audio": Config({"sample_rate": 22050}),
                      "generation": Config({"duration_seconds": 10,
                                            "segment_seconds": seg,
                                            "musicgen_model": "x"})})
        return MusicGenerator(cfg, full=False)._n_segments(arc, 10.0)

    assert gen(0) == 1 and gen(-1) == 1        # one-shot
    assert gen(2.5) > 1                        # segmented still chunks
    assert load_config().generation.segment_seconds == 0   # committed default


def test_crossfade_overlaps_and_trims_silence():
    """Segments must overlap, and near-silent edges must not create a gap."""
    from moodsync.generation.musicgen import MusicGenerator

    sr = 1000
    quiet = np.zeros(300, dtype=np.float32)
    loud = np.ones(1000, dtype=np.float32) * 0.5
    piece = np.concatenate([quiet, loud, quiet])       # silence-padded segment

    trimmed = MusicGenerator._trim_edges(piece, sr)
    assert len(trimmed) < len(piece) and abs(float(trimmed[0])) > 0.02

    out = MusicGenerator._crossfade([piece, piece], sr, fade_seconds=0.1)
    # Overlap: shorter than a plain concatenation of the two trimmed pieces.
    assert len(out) < 2 * len(trimmed)
    assert float(np.max(np.abs(out))) <= 1.0
    # No silent seam: every window in the joined region carries signal.
    mid = out[len(out) // 2 - 50: len(out) // 2 + 50]
    assert float(np.max(np.abs(mid))) > 0.02


def test_resample_shape():
    arc = np.random.uniform(-1, 1, size=(5, 2)).astype(np.float32)
    assert resample_arc(arc, 32).shape == (32, 2)


def test_sections():
    arc = np.array([[0.1, -0.6], [0.2, 0.7], [0.0, 0.0], [-0.3, -0.5]], dtype=np.float32)
    segs = detect_sections(arc)
    assert all("label" in s for s in segs)


def test_platform():
    p = get_platform("linux")
    assert p.os_name == "linux"
    assert p.device in ("cpu", "cuda", "mps")


def test_synth_and_config():
    cfg = load_config()
    assert cfg.audio.sample_rate == 22050
