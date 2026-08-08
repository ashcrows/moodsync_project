# MoodSync — Setup & Run Guide (macOS · Apple Silicon)

**Tested target:** Apple **M4 Pro** · **24 GB** unified memory · latest macOS.
**Project:** MoodSync — Emotion-Aware Music Intelligence · Ashish Sinha · M25DE1047 · IIT Jodhpur.

This guide goes from a fresh Mac to a running demo, the interactive portal, and the
REST API. PyTorch uses the GPU through **MPS (Metal)** automatically — no CUDA and
no configuration needed.

For the concise overview, flag reference and DEAM results, see [README.md](README.md).

---

## 0. What you'll run

- **Light mode (default):** the whole pipeline — librosa features → CNN → Bi-LSTM
  arc → narrator → generation → **Arc Match + Arc Shape Score** — with **no
  downloads and no external datasets**. Start here. A full `demo` run takes well
  under a minute on the M4 Pro.
- **Heavy subsystems (opt-in, independent):** `--gen-full` swaps in **MusicGen**
  for real audio; `--narrator-full` swaps in a **LoRA LLM** narrator. `--full`
  enables both. Large downloads; see the memory notes in §8.

---

## 1. Prerequisites (one-time)

### 1a. Homebrew

```bash
brew --version
```

If not installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

On Apple Silicon, make sure Homebrew is on your PATH (the installer prints these
two lines):

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### 1b. Python 3.12

```bash
brew install python@3.12
python3.12 --version        # expect Python 3.12.x
```

### 1c. Audio libraries (for librosa / mp3 support)

```bash
brew install ffmpeg libsndfile
```

`ffmpeg` lets MoodSync read `.mp3` files; `libsndfile` backs `soundfile` for `.wav`.

### 1d. Java 17 — only for the PySpark feature pipeline

The `features` command needs a JDK. The CNN, Bi-LSTM, portal and API do **not**.
Skip this unless you are running the Spark step.

```bash
brew install temurin@17
export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
java -version               # expect openjdk version "17.x"
```

To make it permanent:

```bash
echo 'export JAVA_HOME="$(/usr/libexec/java_home -v 17)"' >> ~/.zprofile
```

---

## 2. Virtual environment

From inside the project folder:

```bash
cd moodsync_project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Your prompt should now start with `(.venv)`.

### A shell alias can silently override the venv

If `~/.zshrc` contains a line like:

```zsh
alias python="/opt/homebrew/bin/python3.12"
```

then `python` resolves through that alias **before** PATH is consulted, so an
activated virtualenv is bypassed. The symptom is confusing:

```
$ python -m moodsync.cli info --os mac
ModuleNotFoundError: No module named 'numpy'
$ pip install numpy
Requirement already satisfied: numpy in ./.venv/lib/python3.12/site-packages
```

`pip` is not aliased, so it correctly reports the package installed **in the venv**;
`python` is running a different interpreter entirely. Three ways to deal with it:

```bash
unalias python python3            # this shell only
.venv/bin/python -m moodsync.cli info --os mac    # always correct
make info                                          # uses .venv/bin/python by path
```

For a permanent fix, remove or comment out the alias in `~/.zshrc`.

---

## 3. Install dependencies

```bash
pip install -r requirements.txt      # light path
pip install -e ".[dev]"              # pytest, for `make test`
```

This installs numpy, scipy, librosa, soundfile, **torch** (with MPS support),
pyspark, fastapi, uvicorn, streamlit and pandas.

### Verify the GPU is visible to PyTorch

```bash
.venv/bin/python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Expect `MPS available: True`. Then confirm MoodSync picks it up:

```bash
make info            # or: .venv/bin/python -m moodsync.cli info --os mac
```

```json
{
  "target_os": "mac",
  "running_on": "mac",
  "device": "mps",
  "num_workers": 0
}
```

`device: mps` means training and inference use the GPU. `num_workers: 0` is
intentional on macOS — it is the reliable setting for PyTorch data loading here.

---

## 4. Run the demo (start here)

Generates synthetic data, trains the CNN + Bi-LSTM, runs one round-trip and prints
both scores:

```bash
make demo            # or: .venv/bin/python -m moodsync.cli demo --os mac
```

After the CNN and LSTM epochs:

```
=== MoodSync round-trip ===
prompt: A warm, instrumental piece with a resolving emotional arc; buildup: warm; chorus: euphoric; outro: steady.
sections: ['buildup', 'chorus', 'outro']
Arc Match Score:  0.7327  (uncentered cosine)
Arc Shape Score:  0.7990  (trajectory only; a flat arc scores 0.0)
generated wav: .../artifacts/demo/generated.wav
```

Play it with `afplay artifacts/demo/generated.wav`. Checkpoints and audio are
scoped by dataset (`artifacts/demo/` here, `artifacts/deam/` for real training), so
a demo run can never overwrite DEAM-trained weights.

Useful extras on `demo` and `train`:

```bash
.venv/bin/python -m moodsync.cli demo --os mac --regen       # rebuild the synthetic dataset
.venv/bin/python -m moodsync.cli demo --os mac --seed 7      # different split/init seed
```

> One-liner that creates the venv and installs for you:
> ```bash
> bash scripts/run_demo.sh mac
> ```

---

## 5. The portal (Streamlit)

```bash
make app                       # DEAM models  (--config config.deam.yaml)
# or, for the synthetic models:
.venv/bin/python -m moodsync.cli serve-app --os mac
```

Opens `http://localhost:8501`. Tab 1 analyses an uploaded song into an emotion arc;
Tab 2 sets a target arc with sliders, generates audio for it, and reports both
scores.

**Which models are loaded** follows `--config`: with `--config config.deam.yaml` the
DEAM-trained pair is used, otherwise the synthetic pair. The sidebar shows the
resolved dataset tag and config file, so this is visible at a glance. Synthetic
weights will not give meaningful arcs for a real song.

### Generation settings (sidebar)

Changing any of these rebuilds the pipeline for the session:

| Control | Options | Default |
|---|---|---|
| Use MusicGen | on / off (off = offline light synth) | off |
| Generation mode | One-shot (coherent) / Segmented (tracks arc, choppier) | One-shot |
| Narrator | Rule-based (concise) / LLM (verbose, heavy) | Rule-based |
| MusicGen model | `musicgen-small` / `musicgen-medium` | small |
| Clip length (s) | 8 / 10 / 12 / 15 | 10 |

**Recommended for a demonstration on 24 GB:** MusicGen **on**, mode **one-shot**,
narrator **rule-based**, model **musicgen-small**. That produces a listenable clip
without loading the 7B narrator. Requires `requirements-full.txt` (§8); if MusicGen
cannot load, the sidebar says so and the light synthesizer is used instead.

---

## 6. REST API

```bash
make api                       # DEAM models
# or:
.venv/bin/python -m moodsync.cli serve-api --os mac
```

Interactive docs at `http://127.0.0.1:8000/docs`. Endpoints: `GET /health`,
`POST /analyze` (upload audio), `POST /generate` (JSON target arc), `GET /download`
(the last generated wav).

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"arc": [[0.3,-0.6],[0.6,0.7],[-0.2,0.2],[0.1,-0.5]]}'
```

`/health` reports `dataset_tag` and `config`, so you can confirm which trained
models the service loaded.

---

## 7. Audio generation modes

The **default generator is an offline additive synthesizer** — controllable tones,
not music. It exists so the loop and the metric run without downloads. For real
music, install `requirements-full.txt` (§8) and enable MusicGen.

MusicGen is text-conditioned only — it never sees the arc — so
`generation.segment_seconds` picks the trade-off:

| Setting | Mode | Behaviour |
|---|---|---|
| `0` (default) | One-shot | One caption, one clip. Coherent, listenable, seamless; follows the arc loosely. |
| `> 0` | Segmented | Chunks generated separately and crossfaded. Tracks the arc more tightly; seams audible. |

A single text prompt cannot express tight time-varying control — that is a **known
limitation of text-conditioned generation**, not a defect. For a clip that will
actually be listened to, use one-shot. The portal exposes both under **Generation
settings**; from the CLI, set `generation.segment_seconds` in your config.

---

## 8. Heavy subsystems on 24 GB

Install into the same venv:

```bash
pip install -r requirements-full.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1     # unsupported ops fall back to CPU
```

The two subsystems are independent:

```bash
# MusicGen only — recommended on 24 GB (real audio, rule-based narrator)
.venv/bin/python -m moodsync.cli demo --os mac --gen-full

# LLM narrator only
.venv/bin/python -m moodsync.cli demo --os mac --narrator-full

# Both
.venv/bin/python -m moodsync.cli generate --os mac --full \
  --arc "[[0.3,-0.6],[0.6,0.7],[-0.2,0.2],[0.1,-0.5]]"
```

### Memory reality check for 24 GB

- **MusicGen-small** (the configured default, ~2.2 GB) runs comfortably. The first
  run downloads weights, so expect a one-time wait.
- **Mistral-7B narrator** needs roughly 14 GB in fp16. On 24 GB unified memory
  shared with macOS and a browser that is **tight** — it may swap, run slowly, or
  be killed. MoodSync then falls back to the rule-based narrator and keeps going.
- Recommendation: use **`--gen-full`**, not `--full`. Real MusicGen audio with the
  rule-based narrator stays well inside memory. Reach for `--narrator-full` only to
  exercise the LLM specifically, and close other heavy apps first.

---

## 9. Using real data (DEAM) — optional

The dataset is **DEAM — the MediaEval Database for Emotional Analysis in Music**,
downloaded from the University of Geneva CVML group:

**https://cvml.unige.ch/databases/DEAM/**

1802 excerpts with continuous per-second valence and arousal annotations, supplied
as `DEAM_audio.zip` (~1.3 GB) and `DEAM_Annotations.zip` (~4.7 MB). The audio is
Creative Commons licensed and served directly from that page, with no login or
terms form. It is not redistributed with this repository — download it separately
and record its location in the gitignored `config.deam.yaml`.

The committed `config.yaml` keeps the dataset paths **blank on purpose**, so `demo`
and `train` always fall back to the synthetic sample and no machine-specific paths
reach the repository. Real paths belong in a local, gitignored `config.deam.yaml`
selected with `--config`:

```bash
cp config.yaml config.deam.yaml
python scripts/prepare_deam.py --local-path /path/to/DEAM --config config.deam.yaml
```

The helper discovers the audio and annotation folders, validates them with the real
loader before writing, and edits only the two dataset lines (comments preserved).
Re-running it is a no-op; `--dry-run` previews. You can also edit the two values in
`config.deam.yaml` by hand.

```bash
# Real training on DEAM — checkpoints land in artifacts/deam/
.venv/bin/python -m moodsync.cli train   --os mac --config config.deam.yaml
.venv/bin/python -m moodsync.cli analyze --os mac --config config.deam.yaml song.mp3

# Spark feature store over FMA (needs Java, §1d)
.venv/bin/python -m moodsync.cli features --os mac --audio-glob "/path/to/fma_small/**/*.mp3"
```

`--config` **replaces** the whole config rather than merging it, so
`config.deam.yaml` must be a complete copy — hence the `cp` above. Without
`--config`, every command uses the blank paths and the synthetic dataset: plain
`train` and plain `serve-app` never touch DEAM.

---

## 10. Demo assets

```bash
make assets          # or: .venv/bin/python scripts/make_demo_assets.py
```

Requires DEAM paths in `config.deam.yaml` and trained `artifacts/deam/{cnn,lstm}.pt`.
Writes into `demo_outputs/` (gitignored): per-song emotion-arc plots with section
shading, a round-trip overlay of target vs re-analysed arc annotated with both
scores, and `demo_insights.md`.

---

## 11. Tests

```bash
pip install -e ".[dev]"
make test            # or: pytest -q
```

DEAM-specific tests skip cleanly when no dataset is configured.

---

## 12. Troubleshooting (Apple Silicon)

| Symptom | Fix |
|--------|-----|
| `ModuleNotFoundError` for a package `pip` says is installed | A shell alias is overriding the venv — see §2. Use `make …` or `.venv/bin/python …`. |
| `info` shows `device: cpu` not `mps` | Update PyTorch: `pip install -U torch`. Confirm with the check in §3. CPU still works, just slower. |
| MPS op error / `not implemented for MPS` | `export PYTORCH_ENABLE_MPS_FALLBACK=1` then re-run. |
| `features` fails with a Java/Spark error | Install Java 17 (§1d) and set `JAVA_HOME`. Only the Spark step needs it. |
| Can't read an `.mp3` | `brew install ffmpeg` (§1c). |
| `soundfile`/`libsndfile` load error | `brew install libsndfile`, then `pip install --force-reinstall soundfile`. |
| Streamlit/uvicorn "port already in use" | Add `--port 8010` (API); stop the other process for Streamlit. |
| Streamlit hangs at a "Welcome to Streamlit!" email prompt | Already handled — `serve-app` runs headless. If launching `streamlit run` directly, add `--server.headless true`. |
| Generated audio sounds choppy | Use one-shot mode (§7); segmented trades coherence for arc-following. |
| Generated audio is tones, not music | The light synthesizer is the default. Install `requirements-full.txt` and enable MusicGen (§8). |
| First heavy run seems stuck | It is downloading model weights. Let it finish once; later runs are cached. |
| Portal analyses a real song into a nonsense arc | Synthetic weights are loaded. Start with `--config config.deam.yaml` (§9). |
| `command not found: moodsync` | Use `python -m moodsync.cli …`, or `pip install -e .`. |

---

## Command reference

```
make info | make demo | make app | make api | make assets | make test

.venv/bin/python -m moodsync.cli info      --os mac
.venv/bin/python -m moodsync.cli demo      --os mac [--regen] [--seed 7]
.venv/bin/python -m moodsync.cli train     --os mac --config config.deam.yaml
.venv/bin/python -m moodsync.cli analyze   --os mac song.mp3
.venv/bin/python -m moodsync.cli generate  --os mac --arc "[[0.3,-0.5],[0.6,0.6]]"
.venv/bin/python -m moodsync.cli features  --os mac         # PySpark store (needs Java)
.venv/bin/python -m moodsync.cli serve-app --os mac
.venv/bin/python -m moodsync.cli serve-api --os mac
```

Every command accepts `--os {auto,mac,linux,windows}`, `--config PATH`, `--cpu`,
and the heavy-model flags `--gen-full` / `--narrator-full` / `--full`. `demo` and
`train` also accept `--regen` and `--seed N`.
