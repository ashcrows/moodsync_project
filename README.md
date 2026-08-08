# MoodSync — Emotion-Aware Music Intelligence

Temporal Emotion Arc Modelling and Emotion-Conditioned Music Generation.
**Ashish Sinha · M25DE1047 · M.Tech Data Engineering · IIT Jodhpur.**

A working, cross-platform (macOS / Linux / Windows) implementation of the MoodSync
pipeline: extract a song's continuous **valence-arousal emotion arc** over time,
narrate it into text, generate a mood-coherent audio clip that follows a target
arc, and score the match with the **Arc Match Score**.

---

## What runs, and how heavy it is

The default path is **light**: no external downloads, no dataset, no GPU. The two
heavy subsystems are opt-in and independent.

| Stage | Light (default) | Heavy (opt-in) |
|------|------------------|----------------|
| Feature extraction (M1) | librosa | librosa |
| Distributed feature store (M1) | PySpark `local[*]` | PySpark |
| Mood CNN + Bi-LSTM (M2) | real PyTorch models, trained locally | same |
| Arc narrator (M2) | deterministic rule-based | LoRA/PEFT Mistral-7B (`--narrator-full`) |
| Generation engine (M3) | arc-driven synthesizer | MusicGen (`--gen-full`) |
| Arc Match + Shape Score | ✅ | ✅ |
| Streamlit UI + FastAPI (M3) | ✅ | ✅ |

The light path trains on a bundled **synthetic demo dataset** and still produces a
meaningful Arc Match Score, because the demo synthesizer ties audio
brightness/energy/tonality to arousal/valence. Real data is selected with
`--config` (see [Using real data](#using-real-data-deam)).

---

## Quickstart (macOS — the test target)

```bash
cd moodsync_project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"                     # pytest, for `make test`

make demo                                   # data -> train CNN+LSTM -> round-trip -> scores
make app                                    # Streamlit UI   http://localhost:8501
make api                                    # FastAPI        http://127.0.0.1:8000/docs
```

The `make` targets call `.venv/bin/python` by path. On macOS a shell alias such as
`alias python="/opt/homebrew/bin/python3.12"` resolves **before** PATH and overrides
an activated virtualenv, which surfaces as `ModuleNotFoundError: No module named
'numpy'` even though `pip` reports the package installed. Use `make`, or invoke
`.venv/bin/python` explicitly, to avoid depending on shell configuration.

| Target | Command it runs |
|---|---|
| `make test` | `pytest -q` |
| `make info` | `moodsync.cli info --os mac` |
| `make demo` | `moodsync.cli demo --os mac` |
| `make app` | `moodsync.cli serve-app --os mac --config config.deam.yaml` |
| `make api` | `moodsync.cli serve-api --os mac --config config.deam.yaml` |
| `make assets` | `scripts/make_demo_assets.py` |

`PY` and `CONFIG` are overridable: `make app CONFIG=config.yaml`.

One-liner without make: `bash scripts/run_demo.sh mac` (Windows:
`scripts\run_demo.bat windows`).

---

## Commands and flags

```
moodsync info                     # resolved target OS, device, workers
moodsync demo                     # data -> train CNN+LSTM -> round-trip -> scores
moodsync train                    # train CNN + Bi-LSTM
moodsync analyze FILE.mp3         # emotion arc + sections + narrated prompt
moodsync generate --arc "[[0.3,-0.5],[0.6,0.6],[-0.2,0.3]]"
moodsync features                 # PySpark feature pipeline -> Parquet store
moodsync serve-api                # FastAPI service
moodsync serve-app                # Streamlit portal
```

Every command accepts:

| Flag | Meaning |
|---|---|
| `--os {auto,mac,linux,windows}` | Target OS conventions (DataLoader workers, multiprocessing start method, path handling). Default `auto`. Compute always runs on the real machine; a note is printed when `--os` differs from the detected host. |
| `--config PATH` | Config file to load. **Replaces** the whole config — it must be a complete copy, not a fragment. |
| `--cpu` | Force CPU, disabling GPU/MPS. |
| `--gen-full` | Enable MusicGen for generation (~2.2 GB download). |
| `--narrator-full` | Enable the LoRA/Mistral-7B narrator (~14 GB in fp16; tight on 24 GB). |
| `--full` | Alias enabling **both** heavy subsystems. |

`demo` and `train` additionally accept:

| Flag | Meaning |
|---|---|
| `--regen` | Force-regenerate the synthetic demo dataset instead of reusing the cache. |
| `--seed N` | Seed for the song-level splits and weight init. Overrides the `seed` config key (default 42). |

Both heavy subsystems require `pip install -r requirements-full.txt` and fall back
to the light path if their model cannot be loaded, so the pipeline never crashes on
a missing model.

---

## Using real data (DEAM)

The committed `config.yaml` keeps the dataset paths **blank on purpose**, so `demo`
and `train` always fall back to the synthetic sample and no machine-specific paths
reach the repository. Real paths live in a local, gitignored `config.deam.yaml`
selected with `--config`:

```bash
# One-time: create the local dataset config (gitignored)
cp config.yaml config.deam.yaml
python scripts/prepare_deam.py --local-path /path/to/DEAM --config config.deam.yaml
#   or download first:
#   python scripts/prepare_deam.py --download --dest data_store/deam --config config.deam.yaml

python -m moodsync.cli train   --os mac --config config.deam.yaml
python -m moodsync.cli analyze --os mac --config config.deam.yaml song.mp3
```

`--config` **replaces** the whole config rather than merging it, which is why
`config.deam.yaml` starts as a `cp` of `config.yaml`. Without `--config`, every
command uses the blank paths and the synthetic dataset — plain `train` and plain
`serve-app` never touch DEAM.

`prepare_deam.py` discovers the audio and annotation folders, validates them by
calling the real loader before writing anything, and edits only the two dataset
lines, preserving comments. Re-running it is a no-op; `--dry-run` previews the
change. DEAM comes from
[University of Geneva CVML](https://cvml.unige.ch/databases/DEAM/); the audio is
Creative Commons licensed and served without a login. `--download` fetches
`DEAM_audio.zip` (~1.3 GB) and `DEAM_Annotations.zip` (~4.7 MB); if that fails it
prints the manual download page and the exact `--local-path` command to run
afterwards. If the loader resolves 0 songs — usually because audio filenames do not
match the `song_id` column in `valence.csv` — the script refuses to write anything.

### Checkpoints are scoped by dataset

Checkpoints and generated audio live under `artifacts/<tag>/`, where the tag is
`deam` when the DEAM paths resolve and `demo` otherwise. A quick `make demo` can
therefore never overwrite DEAM-trained weights.

```
artifacts/demo/{cnn,lstm}.pt      artifacts/deam/{cnn,lstm}.pt
artifacts/<tag>/generated.wav
```

---

## Audio generation

**The default generator is the offline light synthesizer** — a controllable
additive synthesizer that produces tones, not music. It exists so the whole loop
and the metric run without downloads. For real music, install
`requirements-full.txt` and enable MusicGen with `--gen-full` or the portal's
"Use MusicGen" toggle.

MusicGen is **text-conditioned only** — it never sees the arc. Two modes trade
musical coherence against arc-following:

| `generation.segment_seconds` | Mode | Behaviour |
|---|---|---|
| `0` (default) | **One-shot** | The whole arc becomes one condensed caption and one clip. Coherent, listenable, seamless; follows the arc loosely. |
| `> 0` | **Segmented** | The arc is split into chunks of about that length, each generated and crossfaded. Tracks the arc more tightly; seams are audible. |

In segmented mode every chunk reuses one base caption with only a short mood tag
appended, so the chunks stay the same piece of music — captioning each chunk
independently produces unrelated fragments. Captions passed to MusicGen are
condensed to a single short line, because the model degrades on long prose. Segment
edges are trimmed of near-silence before an overlapping crossfade, so seams do not
fall into a gap.

**The honest trade-off.** One measured example (arc
`[[-0.5,-0.6],[0.0,0.0],[0.6,0.7],[0.3,0.2]]`, `musicgen-small`, rule-based
narrator, 10 s):

| Mode | Arc Match | Arc Shape |
|---|---|---|
| One-shot | −0.287 | −0.457 |
| Segmented (2.5 s) | +0.064 | +0.149 |

One-shot sounds like music but does not follow a hand-drawn arc; segmented follows
it better and sounds choppier. This is a **known limitation of text-conditioned
generation**, not a defect: a single text prompt cannot express tight time-varying
control. Treat the table as one illustrative run, not a benchmark. For a
demonstration where the clip is listened to, use one-shot.

---

## Portal and API

Both hosts take `--config`, which selects the trained models they load:

```bash
python -m moodsync.cli serve-app --os mac --config config.deam.yaml   # DEAM models
python -m moodsync.cli serve-app --os mac                             # synthetic models
```

The Streamlit sidebar shows the resolved dataset tag and config file; the API
reports the same via `GET /health` (`dataset_tag`, `config`).

The sidebar's **Generation settings** panel controls generation live — changing any
of these rebuilds the pipeline for the session:

| Control | Options | Default |
|---|---|---|
| Use MusicGen | on / off (off = light synth) | off |
| Generation mode | One-shot (coherent) / Segmented (tracks arc, choppier) | One-shot |
| Narrator | Rule-based (concise) / LLM (verbose, heavy) | Rule-based |
| MusicGen model | `musicgen-small` / `musicgen-medium` | small |
| Clip length (s) | 8 / 10 / 12 / 15 | 10 |

**Recommended demonstration combination:** MusicGen **on**, mode **one-shot**,
narrator **rule-based**, model **musicgen-small**. This gives a listenable clip on
a 24 GB machine without loading the 7B narrator.

The model is built at first use and cached per settings combination, so switching
`--config` requires restarting the service.

---

## Demo assets

```bash
make assets          # or: python scripts/make_demo_assets.py
```

Requires DEAM paths in `config.deam.yaml` and trained `artifacts/deam/{cnn,lstm}.pt`.
Writes to `demo_outputs/` (gitignored): per-song emotion-arc plots with section
shading, a round-trip overlay of target vs re-analysed arc annotated with both
scores, and `demo_insights.md` summarising sections, mean valence/arousal and the
narrated prompt per song.

---

## Architecture

```
Raw audio ─► [M1] librosa/PySpark features ─► [M2] CNN (v,a per 3s)
          ─► Bi-LSTM arc smoother + section detector ─► rule/LLM arc narrator
          ─► [M3] MusicGen / synth generation ─► re-analyse ─► Arc Match Score
```

- `moodsync/features/` — feature extraction + Spark ETL (M1)
- `moodsync/models/` — CNN, Bi-LSTM smoother, narrator (M2)
- `moodsync/generation/` — MusicGen wrapper + Arc Match Score (M3)
- `moodsync/pipeline.py` — the analyse → generate → re-analyse → score loop
- `app/` — Streamlit UI + FastAPI service (M3 deployment)

---

## Scoring: read both numbers

| Score | Definition | Answers |
|---|---|---|
| **Arc Match Score** | uncentered cosine of target vs re-analysed arc | "is the overall emotional *level* similar?" |
| **Arc Shape Score** | same, after removing each arc's per-dimension mean | "does the emotion *move* as instructed?" |

The Arc Match Score is dominated by the shared mean (DC) component: a completely
flat, trajectory-free arc still scores **≈ +0.42** against a typical target. Quote
it only alongside the Arc Shape Score, where a flat arc correctly scores **0.0**.
The two can rank the same pair of runs differently, as the generation table above
shows, which is exactly why both are always reported.

---

## Results on DEAM

Held-out test set, **song-level splits** so no song contributes windows to more than
one split, and per-second annotations aligned to their real 15 s start. Two-channel
(mel + chroma/CQT) CNN input.

| Metric | CNN | + Bi-LSTM | Target |
|---|---|---|---|
| Valence R² | 0.222 | **0.346** | ~0.40 |
| Arousal R² | 0.532 | **0.604** | ~0.65 |

The targets are **separate per dimension** on purpose: valence is genuinely harder
to predict from audio than arousal, because arousal maps onto energy, tempo and
brightness while valence depends on harmony and cultural context. A single 0.65 bar
for both is not defensible, and the low valence figure is a correctly-measured,
literature-consistent result rather than a failure.

The Bi-LSTM earns its place — on identical songs it lifts valence 0.213 → 0.346 and
arousal 0.534 → 0.604. One caveat: 89 of the smoother's 360 held-out songs were the
CNN's early-stopping validation set, so the Bi-LSTM row is very slightly optimistic.

Reproduce with `python -m moodsync.cli train --os mac --config config.deam.yaml`.
Vary `--seed` to check stability across splits.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q                # or: make test
```

DEAM-specific tests skip cleanly when no dataset is configured, so a fresh clone
stays green.

---

## Notes and honest limitations

- The **light generator** is a controllable additive synthesizer, not a learned
  music model; it exists so the loop and the metric run offline. Musical quality
  requires `--gen-full` (MusicGen).
- **Text-conditioned generation cannot follow an arc tightly.** See the trade-off
  table above.
- Demo training uses a small synthetic set purely so the code runs end-to-end. R²
  reported there is not a research result — use DEAM for real numbers.
- The LoRA narrator ships as a **hook**: `--narrator-full` loads the base
  instruction model; place a trained adapter in `artifacts/lora/` to use
  fine-tuned weights.
- The reported chroma-channel contribution to valence is a single-seed measurement
  and needs a multi-seed ablation before it is treated as a firm result.
