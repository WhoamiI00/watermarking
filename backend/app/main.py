"""FastAPI entry point."""
from __future__ import annotations

import base64
import json
import os
import secrets
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .service import attack_and_recover, encode_pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
SESSION_DIR = BASE_DIR / ".sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Self-Recovery Reversible Image Watermarking",
              description="Implementation of Zhang et al. PLoS ONE 2018",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _b64(buf: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def _png_response(payload: dict) -> dict:
    """Replace any raw PNG bytes inside payload with base64 data URLs."""
    out = {}
    for k, v in payload.items():
        if isinstance(v, (bytes, bytearray)):
            out[k] = _b64(v)
        elif isinstance(v, dict):
            out[k] = _png_response(v)
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None if np.isnan(v) else 1e9
        else:
            out[k] = v
    return out


@app.post("/api/embed")
async def api_embed(
    image: UploadFile = File(...),
    key: int = Form(12345),
    gamma: float = Form(0.3),
    target_size: int = Form(512),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "empty image")
    try:
        result = encode_pipeline(image_bytes, key=key, gamma=gamma, target_size=target_size)
    except Exception as exc:
        raise HTTPException(500, f"embedding failed: {exc}")

    # store sidecar + watermarked in a session so the client doesn't have to
    # round-trip a fat blob on every attack request
    session_id = uuid.uuid4().hex[:12]
    sidecar_path = SESSION_DIR / f"{session_id}.sidecar.json"
    sidecar_path.write_text(json.dumps(result["sidecar"]))
    (SESSION_DIR / f"{session_id}.original.png").write_bytes(result["original"])
    (SESSION_DIR / f"{session_id}.watermarked.png").write_bytes(result["watermarked"])

    payload = {
        "session_id": session_id,
        "psnr_watermarked": result["psnr_watermarked"],
        "nc_reversibility": result["nc_reversibility"],
        "blocks_total": result["blocks_total"],
        "blocks_homogeneous": result["blocks_homogeneous"],
        "blocks_nonhomogeneous": result["blocks_nonhomogeneous"],
        "recovery_bits_total": result["recovery_bits_total"],
        "original": result["original"],
        "decomposition": result["decomposition"],
        "watermarked": result["watermarked"],
        "restored_no_attack": result["restored_no_attack"],
    }
    return JSONResponse(_png_response(payload))


@app.post("/api/attack")
async def api_attack(
    session_id: str = Form(...),
    attack: str = Form(...),
    attack_params: str = Form("{}"),
    donor: UploadFile = File(None),
):
    sidecar_path = SESSION_DIR / f"{session_id}.sidecar.json"
    wm_path = SESSION_DIR / f"{session_id}.watermarked.png"
    orig_path = SESSION_DIR / f"{session_id}.original.png"
    if not sidecar_path.exists() or not wm_path.exists():
        raise HTTPException(404, "session not found -- re-run /api/embed first")

    sidecar_json = json.loads(sidecar_path.read_text())
    wm_bytes = wm_path.read_bytes()
    orig_bytes = orig_path.read_bytes() if orig_path.exists() else None

    try:
        params = json.loads(attack_params or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"invalid attack_params JSON: {exc}")

    donor_bytes = await donor.read() if donor else None

    try:
        result = attack_and_recover(
            watermarked_bytes=wm_bytes,
            sidecar_json=sidecar_json,
            attack_name=attack,
            attack_params=params,
            donor_bytes=donor_bytes,
            original_bytes=orig_bytes,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"attack/recover failed: {exc}")

    return JSONResponse(_png_response(result))


@app.get("/api/health")
def health():
    return {"ok": True}


# --- static frontend ---------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
