"""Generate MoodSync demo assets (emotion-arc plots + round-trip) from the DEAM model.

Run from the repository root:

    python scripts/make_demo_assets.py

Requires DEAM paths configured in config.deam.yaml and trained checkpoints at
artifacts/deam/{cnn,lstm}.pt, produced by:

    python -m moodsync.cli train --os mac --config config.deam.yaml

Outputs land in demo_outputs/ (arc PNGs, a generated .wav, and demo_insights.md).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from moodsync.config import load_config
from moodsync.platform_utils import get_platform
from moodsync.pipeline import MoodSync
from moodsync.generation.arc_match import resample_arc

OUT = ROOT / "demo_outputs"; OUT.mkdir(exist_ok=True)
TEAL, AMBER, INK, MUTE = "#1FA9A0", "#E8912B", "#1A2B3C", "#5A6B7B"
SEC = {"verse":"#DCE7EC","chorus":"#FBE3C4","bridge":"#E7E0FB","buildup":"#CDEDE9",
       "drop":"#F6D9D9","outro":"#E3E9ED"}
plt.rcParams.update({"font.family":"DejaVu Sans", "text.color":INK})

# Loading the DEAM config makes its dataset paths resolve, which scopes the
# checkpoint lookup to artifacts/deam/.
cfg = load_config(ROOT / "config.deam.yaml")
platform = get_platform("mac")
ms = MoodSync(cfg, platform)
assert ms.cnn is not None, ("No DEAM models found. First run:\n"
    "  python -m moodsync.cli train --os mac --config config.deam.yaml")
win = float(cfg.audio.window_seconds)
audio_dir = Path(str(cfg.datasets.deam_audio))

def plot_arc(sid, res, path):
    arc = np.asarray(res["arc"]); t = np.arange(len(arc))*win
    fig, ax = plt.subplots(figsize=(9,4.3))
    for seg in res["sections"]:
        ax.axvspan(seg["start"]*win, seg["end"]*win, color=SEC.get(seg["label"],"#EEE"), alpha=.6, lw=0)
        ax.text((seg["start"]+seg["end"])/2*win, 1.07, seg["label"], ha="center", fontsize=8, color=MUTE)
    ax.plot(t, arc[:,0], color=AMBER, lw=2.3, marker="o", ms=3, label="valence")
    ax.plot(t, arc[:,1], color=TEAL, lw=2.3, marker="o", ms=3, label="arousal")
    ax.axhline(0, color="#CCC", lw=1); ax.set_ylim(-1.05,1.2); ax.set_xlim(0, max(t[-1],1))
    ax.set_xlabel("time (s)"); ax.set_ylabel("valence / arousal")
    ax.set_title(f"DEAM song {sid} — emotion arc extracted by MoodSync", fontweight="bold", loc="left")
    ax.legend(loc="lower right", frameon=False); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200, facecolor="white"); plt.close(fig)

# Preferred song ids; any that are absent fall back to the first four on disk.
songs = ["1000","1001","1002","1003"]
avail = {p.stem: p for p in sorted(audio_dir.glob("*.mp3"))}
songs = [s for s in songs if s in avail] or list(avail)[:4]

lines = ["# MoodSync — demo insights (real DEAM model)\n"]
for sid in songs:
    res = ms.analyze(str(avail[sid]))
    plot_arc(sid, res, OUT/f"arc_song_{sid}.png")
    a = np.asarray(res["arc"])
    lines.append(f"## Song {sid}\n- sections: {' -> '.join(s['label'] for s in res['sections'])}\n"
                 f"- mean valence {a[:,0].mean():+.2f}, mean arousal {a[:,1].mean():+.2f}\n"
                 f"- narrated prompt: \"{res['prompt']}\"\n")

# Round-trip: target arc -> generate -> re-analyse -> score
target = np.array([[0.3,-0.5],[0.4,-0.2],[0.6,0.6],[0.7,0.7],
                   [-0.3,0.4],[-0.2,0.1],[0.2,-0.4],[0.3,-0.6]], dtype=np.float32)
g = ms.run_loop(target, out_wav=str(OUT/"generated_demo.wav"))
N=32; T=resample_arc(target,N); G=resample_arc(np.asarray(g["generated_arc"]),N); x=np.arange(N)
fig, axs = plt.subplots(1,2, figsize=(11,4))
for ax,d,name,c in [(axs[0],0,"valence",AMBER),(axs[1],1,"arousal",TEAL)]:
    ax.plot(x,T[:,d],color=c,lw=2.4,label="target arc")
    ax.plot(x,G[:,d],color=INK,lw=2,ls="--",label="generated (re-analysed)")
    ax.set_title(name,fontweight="bold"); ax.set_ylim(-1.05,1.05); ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle(f"Round-trip: target vs generated   |   Arc Match {g['arc_match_score']:.3f}  ·  "
             f"Arc Shape {g['arc_shape_score']:.3f}", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT/"roundtrip_overlay.png", dpi=200, facecolor="white"); plt.close(fig)

lines.append(f"\n## Round-trip generation\n- Arc Match Score: {g['arc_match_score']:.3f}\n"
             f"- Arc Shape Score: {g['arc_shape_score']:.3f}\n- prompt: \"{g['prompt']}\"\n"
             f"- audio: demo_outputs/generated_demo.wav\n")
(OUT/"demo_insights.md").write_text("\n".join(lines))
print(f"Done. Arc Match {g['arc_match_score']:.3f} / Shape {g['arc_shape_score']:.3f}")
print("Wrote:", ", ".join(sorted(p.name for p in OUT.iterdir())))
