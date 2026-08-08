"""MoodSync command-line interface.

Global `--os {auto,mac,linux,windows}` selects the target OS (default auto).
Testing target is macOS; the same commands run on Linux and Windows.

Commands:
  demo         Generate demo data, train CNN+LSTM, run one round-trip, print score.
  train        Train CNN + LSTM (uses DEAM if configured, else demo data).
  analyze      Extract the emotion arc of an audio file.
  generate     Generate audio for a target arc and report the Arc Match Score.
  features     Run the PySpark distributed feature pipeline over a folder.
  serve-api    Launch the FastAPI service.
  serve-app    Launch the Streamlit demo UI.
  info         Print resolved platform settings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import load_config
from .platform_utils import (
    VALID_OS,
    data_dir,
    get_platform,
)


def _add_common(p):
    p.add_argument("--os", choices=VALID_OS, default="auto",
                   help="Target OS (default: auto-detect). Testing target: mac.")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--full", action="store_true",
                   help="Enable BOTH heavy subsystems (MusicGen + LoRA LLM). "
                        "Convenience alias for --gen-full --narrator-full.")
    p.add_argument("--gen-full", dest="gen_full", action="store_true",
                   help="Enable MusicGen for generation (~2GB download). "
                        "Independent of the LLM narrator.")
    p.add_argument("--narrator-full", dest="narrator_full", action="store_true",
                   help="Enable the LoRA/Mistral LLM narrator (large download, "
                        "memory-hungry). Independent of MusicGen.")
    p.add_argument("--cpu", action="store_true", help="Force CPU (disable GPU/MPS).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for the song-level splits and weight init. "
                        "Overrides the config's `seed` key (default 42).")


def _load(args):
    cfg = load_config(args.config)
    # Fold --seed into the config so every consumer reads one resolved value.
    seed = getattr(args, "seed", None)
    if seed is not None:
        cfg["seed"] = int(seed)
    platform = get_platform(args.os, prefer_gpu=not getattr(args, "cpu", False))
    return cfg, platform


def _resolve_full(args) -> tuple[bool, bool]:
    """Resolve the heavy-model flags into (gen_full, narrator_full).

    `--full` stays a backward-compatible alias that turns on both subsystems, so
    each can also be enabled alone: MusicGen fits comfortably in 24 GB while the
    7B LoRA narrator does not.
    """
    full = bool(getattr(args, "full", False))
    return (full or bool(getattr(args, "gen_full", False)),
            full or bool(getattr(args, "narrator_full", False)))


def _demo_signature(cfg) -> str:
    """Fingerprint the demo dataset's inputs so a stale cache is detected.

    Includes the demo/audio config values AND the synth.py source, so editing the
    synthesizer or changing n_clips/clip_seconds/sample_rate invalidates the cached
    WAVs instead of silently training on old audio.
    """
    import hashlib
    from . import data as _data_pkg
    synth_src = (Path(_data_pkg.__file__).parent / "synth.py").read_bytes()
    payload = "|".join([
        str(cfg.demo.n_clips), str(cfg.demo.clip_seconds),
        str(cfg.audio.sample_rate),
        hashlib.sha256(synth_src).hexdigest(),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _get_clips(cfg, platform, regen: bool = False):
    """DEAM if configured, else generate/reuse the synthetic demo dataset.

    The cached demo dataset is reused only if its stored signature matches the
    current config + synth source; otherwise it is regenerated. Pass regen=True
    (CLI --regen) to force a rebuild.
    """
    from .data.deam import load_deam
    clips = load_deam(cfg)
    if clips:
        return clips
    from .data.synth import generate_demo_dataset
    demo_root = data_dir() / "demo"
    manifest_path = demo_root / "manifest.json"
    sig = _demo_signature(cfg)

    if manifest_path.exists() and not regen:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature") == sig:
            return manifest["clips"]
        print("[moodsync] demo config/synth changed — regenerating dataset...")
    elif manifest_path.exists():
        # Only reachable with regen=True (the branch above covers regen=False).
        # Gating on existence keeps a true first run on the "generating" message
        # instead of claiming to rebuild a dataset that isn't there yet.
        print("[moodsync] --regen: force-rebuilding demo dataset...")
    else:
        print("[moodsync] generating synthetic demo dataset...")

    manifest = generate_demo_dataset(
        demo_root,
        n_clips=int(cfg.demo.n_clips),
        clip_seconds=float(cfg.demo.clip_seconds),
        sr=int(cfg.audio.sample_rate),
        signature=sig,
    )
    return manifest["clips"]


def cmd_info(args):
    _, platform = _load(args)
    print(json.dumps({
        "target_os": platform.os_name,
        "running_on": platform.running_on,
        "device": platform.device,
        "num_workers": platform.num_workers,
    }, indent=2))


def cmd_demo(args):
    from .train import train_cnn, train_lstm
    from .pipeline import MoodSync
    cfg, platform = _load(args)
    print(f"[moodsync] target OS={platform.os_name} device={platform.device}")
    clips = _get_clips(cfg, platform, regen=getattr(args, "regen", False))
    cnn_path = train_cnn(clips, cfg, platform)
    train_lstm(clips, cfg, platform, cnn_path)

    gen_full, narrator_full = _resolve_full(args)
    ms = MoodSync(cfg, platform, gen_full=gen_full, narrator_full=narrator_full)
    # A designed target arc: calm verse -> euphoric chorus -> tense bridge -> resolve.
    target = np.array([
        [0.3, -0.5], [0.4, -0.2], [0.6, 0.6], [0.7, 0.7],
        [-0.3, 0.4], [-0.2, 0.1], [0.2, -0.4], [0.3, -0.6],
    ], dtype=np.float32)
    result = ms.run_loop(target)
    print("\n=== MoodSync round-trip ===")
    print("prompt:", result["prompt"])
    print("sections:", [s["label"] for s in result["sections"]])
    print(f"Arc Match Score:  {result['arc_match_score']:.4f}  (uncentered cosine)")
    print(f"Arc Shape Score:  {result['arc_shape_score']:.4f}  (trajectory only; "
          f"a flat arc scores 0.0)")
    print("generated wav:", result["wav_path"])


def cmd_train(args):
    from .train import train_cnn, train_lstm
    cfg, platform = _load(args)
    clips = _get_clips(cfg, platform, regen=getattr(args, "regen", False))
    cnn_path = train_cnn(clips, cfg, platform)
    train_lstm(clips, cfg, platform, cnn_path)


def cmd_analyze(args):
    from .pipeline import MoodSync
    cfg, platform = _load(args)
    gen_full, narrator_full = _resolve_full(args)
    ms = MoodSync(cfg, platform, gen_full=gen_full, narrator_full=narrator_full)
    res = ms.analyze(args.audio)
    print("prompt:", res["prompt"])
    print("sections:", [s["label"] for s in res["sections"]])
    arc = np.asarray(res["arc"])
    print("arc (valence,arousal) per window:")
    for i, (v, a) in enumerate(arc):
        print(f"  w{i:02d}: v={v:+.2f} a={a:+.2f}")


def cmd_generate(args):
    from .pipeline import MoodSync
    cfg, platform = _load(args)
    gen_full, narrator_full = _resolve_full(args)
    ms = MoodSync(cfg, platform, gen_full=gen_full, narrator_full=narrator_full)
    if args.arc:
        arc = np.array(json.loads(args.arc), dtype=np.float32)
    else:
        arc = np.array([[0.3, -0.5], [0.6, 0.6], [-0.2, 0.3], [0.2, -0.4]], dtype=np.float32)
    # out_wav=None lets the pipeline pick the dataset-scoped default.
    res = ms.run_loop(arc, out_wav=args.out)
    print("prompt:", res["prompt"])
    print(f"Arc Match Score:  {res['arc_match_score']:.4f}  (uncentered cosine)")
    print(f"Arc Shape Score:  {res['arc_shape_score']:.4f}  (trajectory only)")
    print("generated wav:", res["wav_path"])


def cmd_features(args):
    from .features.spark_pipeline import run_spark_feature_pipeline
    cfg, platform = _load(args)
    audio_glob = args.audio_glob or str(data_dir() / "demo" / "audio" / "*.wav")
    out = args.out or str(data_dir() / "feature_store.parquet")
    # Ensure demo data exists if using the default glob.
    if "demo" in audio_glob and not list(Path(audio_glob).parent.glob("*.wav")):
        _get_clips(cfg, platform)
    run_spark_feature_pipeline(audio_glob, out, cfg, platform)


def _config_env(args) -> str:
    """Absolute path of the config to hand a server process, '' when unset.

    The hosts resolve relative paths against their own working directory, which
    need not be the one the CLI was invoked from, so the path is absolutised
    here. An empty value means "no --config", and the hosts fall back to the
    default config.yaml.
    """
    path = getattr(args, "config", None)
    return str(Path(path).resolve()) if path else ""


def cmd_serve_api(args):
    import uvicorn
    # Set env so the API process picks up the same OS/config/flags.
    import os
    gen_full, narrator_full = _resolve_full(args)
    os.environ["MOODSYNC_OS"] = args.os
    os.environ["MOODSYNC_CONFIG"] = _config_env(args)
    os.environ["MOODSYNC_GEN_FULL"] = "1" if gen_full else "0"
    os.environ["MOODSYNC_NARRATOR_FULL"] = "1" if narrator_full else "0"
    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=False)


def cmd_serve_app(args):
    import subprocess
    import os
    root = Path(__file__).resolve().parent.parent
    gen_full, narrator_full = _resolve_full(args)
    env = dict(
        os.environ,
        MOODSYNC_OS=args.os,
        MOODSYNC_CONFIG=_config_env(args),
        MOODSYNC_GEN_FULL="1" if gen_full else "0",
        MOODSYNC_NARRATOR_FULL="1" if narrator_full else "0",
    )
    # Without a ~/.streamlit config, Streamlit's first-run onboarding prompt
    # waits on stdin and the server never binds its port. Headless mode skips
    # it; setdefault leaves an explicit environment setting untouched.
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(root / "app" / "streamlit_app.py")], env=env)


def build_parser():
    p = argparse.ArgumentParser(prog="moodsync", description="MoodSync — Emotion-Aware Music Intelligence")
    sub = p.add_subparsers(dest="command", required=True)

    for name, fn, extra in [
        ("info", cmd_info, None),
        ("demo", cmd_demo, None),
        ("train", cmd_train, None),
        ("analyze", cmd_analyze, "analyze"),
        ("generate", cmd_generate, "generate"),
        ("features", cmd_features, "features"),
        ("serve-api", cmd_serve_api, "api"),
        ("serve-app", cmd_serve_app, "app"),
    ]:
        sp = sub.add_parser(name)
        _add_common(sp)
        if name in ("demo", "train"):
            sp.add_argument("--regen", action="store_true",
                            help="Force-regenerate the synthetic demo dataset.")
        if extra == "analyze":
            sp.add_argument("audio", help="Path to an audio file (wav/mp3/...)")
        elif extra == "generate":
            sp.add_argument("--arc", default=None,
                            help='Target arc as JSON, e.g. "[[0.3,-0.5],[0.6,0.6]]"')
            sp.add_argument("--out", default=None, help="Output wav path")
        elif extra == "features":
            sp.add_argument("--audio-glob", dest="audio_glob", default=None)
            sp.add_argument("--out", default=None)
        elif extra == "api":
            sp.add_argument("--host", default="127.0.0.1")
            sp.add_argument("--port", type=int, default=8000)
        sp.set_defaults(func=fn)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
