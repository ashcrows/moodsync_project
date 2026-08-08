"""DEAM dataset loader (real-data hook).

DEAM ships per-song continuous valence/arousal annotations (2 Hz) plus audio.
This loader is intentionally defensive: if the configured paths are empty or
missing, it returns None so callers fall back to the synthetic demo data.

Expected layout (DEAM 2016 release):
  <deam_audio>/<song_id>.mp3
  <deam_annotations>/annotations averaged per song/dynamic (per second annotations)/
        valence.csv, arousal.csv   (columns: song_id, sample_15000ms, ...)

Because different DEAM mirrors reorganize files, the CSV parsing here is
best-effort; `_read_dynamic_csv` may need adjusting to match a given copy.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def _header_times(header) -> tuple[float, float]:
    """Return (start_seconds, hop_seconds) parsed from `sample_<ms>ms` columns.

    DEAM's dynamic annotations do NOT start at t=0 — they begin at 15000ms,
    because raters need lead-in before their ratings stabilise. Callers must
    align labels to these real timestamps rather than stretching them over the
    whole file.
    """
    ms = []
    for c in (header or [])[1:]:
        c = c.strip()
        if c.startswith("sample_") and c.endswith("ms"):
            try:
                ms.append(int(c[len("sample_"):-2]))
            except ValueError:
                pass
    if len(ms) < 2:
        return 15.0, 0.5                      # documented DEAM defaults
    return ms[0] / 1000.0, (ms[1] - ms[0]) / 1000.0


def _read_dynamic_csv(path: Path):
    import csv
    rows = {}
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows["__times__"] = _header_times(header)
        for row in reader:
            if not row:
                continue
            sid = row[0].strip()
            vals = []
            for c in row[1:]:
                try:
                    vals.append(float(c))
                except ValueError:
                    pass
            rows[sid] = np.asarray(vals, dtype=np.float32)
    return rows


def load_deam(cfg, max_songs: Optional[int] = None) -> Optional[List[dict]]:
    """Return a list of {id, path, arc} dicts, or None if DEAM isn't available.

    `arc` is (n_windows, 2) resampled to the same windowing the CNN uses.
    DEAM valence/arousal are on a 1..9 scale, mapped here to [-1, 1].
    """
    audio_dir = (cfg.datasets.deam_audio or "").strip()
    ann_dir = (cfg.datasets.deam_annotations or "").strip()
    if not audio_dir or not ann_dir:
        return None
    audio_dir = Path(audio_dir)
    ann_dir = Path(ann_dir)
    if not audio_dir.exists() or not ann_dir.exists():
        print(f"[moodsync] DEAM paths not found; using demo data instead.")
        return None

    val_csv = next(ann_dir.rglob("valence.csv"), None)
    aro_csv = next(ann_dir.rglob("arousal.csv"), None)
    if val_csv is None or aro_csv is None:
        print("[moodsync] DEAM valence/arousal CSVs not found; using demo data.")
        return None

    valence = _read_dynamic_csv(val_csv)
    arousal = _read_dynamic_csv(aro_csv)
    arc_start_s, arc_hop_s = valence.pop("__times__", (15.0, 0.5))
    arousal.pop("__times__", None)

    def _to_pm1(x):
        # DEAM dynamic ratings ~[1,9] -> [-1,1]. Robust even if already normalized.
        if np.nanmax(x) > 1.5:
            return (x - 5.0) / 4.0
        return x

    clips = []
    ids = sorted(set(valence) & set(arousal))
    if max_songs:
        ids = ids[:max_songs]
    for sid in ids:
        cand = list(audio_dir.glob(f"{sid}.*"))
        if not cand:
            continue
        v = _to_pm1(valence[sid])
        a = _to_pm1(arousal[sid])
        n = min(len(v), len(a))
        arc = np.stack([v[:n], a[:n]], axis=1).astype(np.float32)
        clips.append({
            "id": sid,
            "path": str(cand[0]),
            "arc": arc,
            # Real timestamps for arc[i]: arc_start_s + i * arc_hop_s. Without
            # these the trainer would stretch a 15s..45s annotation across the
            # full 0s..45s audio and mislabel every window.
            "arc_start_s": float(arc_start_s),
            "arc_hop_s": float(arc_hop_s),
        })
    print(f"[moodsync] Loaded {len(clips)} DEAM songs "
          f"(annotations start at {arc_start_s:.1f}s, {arc_hop_s:.1f}s apart).")
    return clips or None
