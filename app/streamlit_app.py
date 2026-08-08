"""Streamlit demo UI (Module M3): upload -> view arc -> set target -> generate -> download.

Run: `moodsync serve-app --os mac`  (or `streamlit run app/streamlit_app.py`)
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the package importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from moodsync.config import Config, dataset_tag, load_config
from moodsync.pipeline import MoodSync, config_path_from_env, full_flags_from_env
from moodsync.platform_utils import get_platform


SEGMENT_SECONDS = 2.5          # chunk length when segmented mode is selected
MODE_ONESHOT = "One-shot (coherent)"
MODE_SEGMENTED = "Segmented (tracks arc, choppier)"
NARRATOR_RULE = "Rule-based (concise)"
NARRATOR_LLM = "LLM (verbose, heavy)"


@st.cache_resource(show_spinner="Loading models…")
def get_moodsync(use_musicgen: bool, segment_seconds: float, narrator_llm: bool,
                 musicgen_model: str, duration_seconds: int):
    """Build MoodSync for one combination of generation settings.

    Every setting is an argument so st.cache_resource keys on all of them: any
    change rebuilds the pipeline instead of silently reusing the previous one.
    The overrides are applied to a deep copy, leaving the on-disk config alone.
    """
    # MOODSYNC_CONFIG carries the CLI's --config, so the UI loads the same
    # dataset (and therefore the same checkpoints) the CLI was pointed at.
    cfg_path = config_path_from_env()
    cfg = copy.deepcopy(load_config(cfg_path))
    cfg["generation"] = Config({
        **dict(cfg.generation),
        "segment_seconds": float(segment_seconds),
        "musicgen_model": str(musicgen_model),
        "duration_seconds": int(duration_seconds),
    })
    platform = get_platform(os.environ.get("MOODSYNC_OS", "auto"))
    ms = MoodSync(cfg, platform, gen_full=bool(use_musicgen),
                  narrator_full=bool(narrator_llm))
    return ms, cfg, platform, cfg_path


st.set_page_config(page_title="MoodSync", layout="wide")
st.title("🎵 MoodSync — Emotion-Aware Music Intelligence")
st.caption("Temporal emotion arc modelling + emotion-conditioned generation · IIT Jodhpur")

st.sidebar.header("Generation settings")
_env_gen_full, _env_narrator_full = full_flags_from_env()
_defaults = load_config(config_path_from_env()).generation

use_musicgen = st.sidebar.toggle(
    "Use MusicGen", value=bool(_env_gen_full),
    help="Off uses the offline light synthesizer, which needs no downloads.")
mode = st.sidebar.selectbox("Generation mode", [MODE_ONESHOT, MODE_SEGMENTED], index=0)
narrator_choice = st.sidebar.selectbox("Narrator", [NARRATOR_RULE, NARRATOR_LLM],
                                       index=1 if _env_narrator_full else 0)
musicgen_model = st.sidebar.selectbox(
    "MusicGen model", ["facebook/musicgen-small", "facebook/musicgen-medium"], index=0)
_lengths = [8, 10, 12, 15]
_default_len = int(getattr(_defaults, "duration_seconds", 10))
clip_seconds = st.sidebar.selectbox(
    "Clip length (s)", _lengths,
    index=_lengths.index(_default_len) if _default_len in _lengths else 1)

segment_seconds = 0.0 if mode == MODE_ONESHOT else SEGMENT_SECONDS
st.sidebar.caption(
    "One-shot: one caption, one clip — musical and seamless. "
    "Segmented: chunked and crossfaded, tracks the arc more tightly but choppier."
)

try:
    ms, cfg, platform, cfg_path = get_moodsync(
        use_musicgen, segment_seconds, narrator_choice == NARRATOR_LLM,
        musicgen_model, clip_seconds)
except Exception as exc:
    st.error(f"Failed to initialise: {exc}")
    st.stop()

tag = dataset_tag(cfg)

st.sidebar.header("Resolved")
st.sidebar.write(
    f"Engine: **{'MusicGen' if ms.gen_full else 'light synth'}**"
    + (f" (`{cfg.generation.musicgen_model.split('/')[-1]}`)" if ms.gen_full else "")
)
st.sidebar.write(f"Mode: **{'one-shot' if segment_seconds <= 0 else f'segmented @ {segment_seconds}s'}**")
st.sidebar.write(f"Narrator: **{'LLM' if ms.narrator_full else 'rule-based'}**")
st.sidebar.write(f"Clip length: **{int(cfg.generation.duration_seconds)} s**")
if use_musicgen and not ms.generator.model_loaded:
    st.sidebar.warning(
        "MusicGen did not load — falling back to the light synthesizer. "
        "Install requirements-full.txt and check the server log."
    )

st.sidebar.header("Models")
# Which checkpoints are live matters most: 'demo' weights come from the tiny
# synthetic sample and will not give meaningful arcs for a real song.
st.sidebar.write(f"Dataset tag: **{tag}** (`artifacts/{tag}/`)")
st.sidebar.write(f"Config: **{Path(cfg_path).name if cfg_path else 'config.yaml'}**")
if tag == "demo":
    st.sidebar.info(
        "Synthetic demo weights. For real songs, start with "
        "`serve-app --config config.deam.yaml`."
    )

st.sidebar.header("Runtime")
st.sidebar.write(f"Target OS: **{platform.os_name}**")
st.sidebar.write(f"Device: **{platform.device}**")
if ms.cnn is None:
    st.warning("No trained model found. Run `moodsync demo` or `moodsync train` first.")

tab1, tab2 = st.tabs(["Analyse a song", "Draw a target arc → generate"])

with tab1:
    up = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg"])
    if up is not None and ms.cnn is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
            tmp.write(up.read())
            path = tmp.name
        res = ms.analyze(path)
        os.unlink(path)
        st.subheader("Emotional arc")
        arc = np.asarray(res["arc"])
        st.line_chart({"valence": arc[:, 0], "arousal": arc[:, 1]})
        st.write("**Sections:**", " → ".join(s["label"] for s in res["sections"]))
        st.write("**Narrated prompt:**", res["prompt"])

with tab2:
    st.write("Set a target emotional arc (valence & arousal at 8 points, -1..1):")
    cols = st.columns(2)
    n = 8
    valence, arousal = [], []
    with cols[0]:
        st.markdown("**Valence**")
        for i in range(n):
            valence.append(st.slider(f"v{i}", -1.0, 1.0, float(np.sin(i / n * np.pi)), 0.05, key=f"v{i}"))
    with cols[1]:
        st.markdown("**Arousal**")
        for i in range(n):
            arousal.append(st.slider(f"a{i}", -1.0, 1.0, float(-1 + 2 * i / n), 0.05, key=f"a{i}"))
    target = np.stack([valence, arousal], axis=1).astype(np.float32)
    st.line_chart({"valence": target[:, 0], "arousal": target[:, 1]})

    if st.button("Generate music for this arc", disabled=ms.cnn is None):
        with st.spinner("Generating and scoring..."):
            res = ms.run_loop(target)
        c1, c2 = st.columns(2)
        c1.metric("Arc Match Score", f"{res['arc_match_score']:.4f}",
                  help="Uncentered cosine — dominated by the overall emotional level.")
        c2.metric("Arc Shape Score", f"{res['arc_shape_score']:.4f}",
                  help="Trajectory only. A flat, trajectory-free arc scores 0.0.")
        st.write("**Prompt:**", res["prompt"])
        st.audio(res["wav_path"])
        gen = np.asarray(res["generated_arc"])
        st.write("Target vs generated arc:")
        st.line_chart({
            "target_valence": target[:, 0],
            "gen_valence": np.interp(np.linspace(0, 1, len(target)), np.linspace(0, 1, len(gen)), gen[:, 0]),
        })
