"""FastAPI service exposing MoodSync analyse/generate (Module M3).

Run: `moodsync serve-api --os mac` or `uvicorn app.api:app`.

Endpoints:
  GET  /health
  POST /analyze   (multipart file upload) -> arc + sections + prompt
  POST /generate  (json {"arc": [[v,a],...]}) -> arc match score + wav path
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from moodsync.config import dataset_tag, load_config
from moodsync.pipeline import MoodSync, config_path_from_env, full_flags_from_env
from moodsync.platform_utils import get_platform

app = FastAPI(title="MoodSync API", version="0.1.0")

_MS = None


def _get_ms() -> MoodSync:
    global _MS
    if _MS is None:
        cfg = load_config(config_path_from_env())
        platform = get_platform(os.environ.get("MOODSYNC_OS", "auto"))
        gen_full, narrator_full = full_flags_from_env()
        _MS = MoodSync(cfg, platform, gen_full=gen_full, narrator_full=narrator_full)
    return _MS


class ArcRequest(BaseModel):
    arc: List[List[float]]


@app.get("/health")
def health():
    ms = _get_ms()
    return {"status": "ok", "target_os": ms.platform.os_name,
            "device": ms.platform.device, "full": ms.full,
            "gen_full": ms.gen_full, "narrator_full": ms.narrator_full,
            "cnn_loaded": ms.cnn is not None,
            # Which checkpoints are live: 'deam' for real training, else 'demo'.
            "dataset_tag": dataset_tag(ms.cfg),
            "config": config_path_from_env() or "config.yaml"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ms = _get_ms()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    res = ms.analyze(tmp_path)
    os.unlink(tmp_path)
    return {
        "prompt": res["prompt"],
        "sections": res["sections"],
        "arc": np.asarray(res["arc"]).tolist(),
    }


@app.post("/generate")
def generate(req: ArcRequest):
    ms = _get_ms()
    arc = np.asarray(req.arc, dtype=np.float32)
    res = ms.run_loop(arc)
    return {
        "prompt": res["prompt"],
        "arc_match_score": res["arc_match_score"],
        "arc_shape_score": res["arc_shape_score"],
        "sections": res["sections"],
        "generated_arc": np.asarray(res["generated_arc"]).tolist(),
        "wav_path": res["wav_path"],
    }


@app.get("/download")
def download():
    ms = _get_ms()
    from moodsync.pipeline import generated_wav_path
    wav = generated_wav_path(ms.cfg)
    if not wav.exists():
        return {"error": "no generated audio yet"}
    return FileResponse(str(wav), media_type="audio/wav", filename="moodsync_generated.wav")
