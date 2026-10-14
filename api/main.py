"""
api/main.py -- FastAPI backend for SpeechLens (DESIGN §14, TASK_2 T01).

Endpoints:
  GET  /api/health                     Status, version, model load state.
  GET  /api/speeches                   List available speeches with metadata.
  GET  /api/media/{speech_id}/baseline.wav  16 kHz mono WAV of reference reader.
  POST /api/analyze                    Multipart audio upload -> Section 12 JSON.
  GET  /api/demo                       List available demo recordings and results.
  Static mounts:
    /demo                              Static demo audio and result files.
    /                                  Built frontend from web/out (if present).
"""
from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import numpy as np
import soundfile as sf

from speechlens.config import MAX_AUDIO_S, MAX_UPLOAD_MB, SR
from speechlens import audio
from speechlens.align.ctc import _model_cache, AlignmentError
from speechlens.pipeline import analyze

app = FastAPI(
    title="SpeechLens API",
    version="1.0.0",
    description="Contrastive speech analytics backend",
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Concurrency & Timeout Guards ─────────────────────────────────────────────
_SEMAPHORE = asyncio.Semaphore(1)
_TIMEOUT_SECONDS = 300

LIBRARY_DIR = Path("data/library")
CACHE_MEDIA_DIR = Path(".cache/media")
DATASET_DIR = Path("data/dataset")
DEMO_DIR = Path("data/demo")
WEB_PUBLIC_DEMO = Path("web/public/demo")
WEB_OUT_DIR = Path("web/out")


# ── Custom Error Response Helper ─────────────────────────────────────────────
def _err(status_code: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": code, "detail": detail},
    )


# ── Health Endpoint ──────────────────────────────────────────────────────────
@app.get("/api/health")
async def health() -> dict[str, Any]:
    from speechlens.align.ctc import _model_cache
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": _model_cache is not None,
    }


# ── Speeches Metadata Endpoint ───────────────────────────────────────────────
@app.get("/api/speeches")
async def list_speeches() -> list[dict[str, Any]]:
    speeches = []
    if not LIBRARY_DIR.exists():
        return speeches

    for speech_dir in sorted(LIBRARY_DIR.iterdir()):
        if not speech_dir.is_dir():
            continue
        meta_path = speech_dir / "meta.json"
        trans_path = speech_dir / "transcript.txt"
        if not meta_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            transcript = trans_path.read_text(encoding="utf-8").strip() if trans_path.exists() else ""
            sid = speech_dir.name
            readers = meta.get("readers", [])
            speeches.append({
                "id": sid,
                "title": meta.get("title", sid),
                "speaker": meta.get("speaker", "Unknown"),
                "duration_s": meta.get("duration_s", 0.0),
                "n_words": len(transcript.split()),
                "transcript": transcript,
                "baseline_audio_url": f"/api/media/{sid}/baseline.wav",
                "n_readers": len(readers),
            })
        except Exception:
            continue

    return speeches


# ── Baseline Audio Endpoint ──────────────────────────────────────────────────
@app.get("/api/media/{speech_id}/baseline.wav")
async def get_baseline_audio(speech_id: str):
    speech_dir = LIBRARY_DIR / speech_id
    if not speech_dir.exists():
        return _err(404, "not_found", f"Speech '{speech_id}' not found.")

    meta_path = speech_dir / "meta.json"
    if not meta_path.exists():
        return _err(404, "not_found", f"Metadata for speech '{speech_id}' not found.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mirror_source = meta.get("mirror_source")
    if not mirror_source and meta.get("readers"):
        mirror_source = meta["readers"][0]["id"]

    if not mirror_source:
        return _err(404, "not_found", "No baseline reader found for speech.")

    flac_path = speech_dir / "readers" / f"{mirror_source}.flac"
    if not flac_path.exists():
        return _err(404, "not_found", f"Audio file for reader '{mirror_source}' not found.")

    CACHE_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    wav_cache_path = CACHE_MEDIA_DIR / f"{speech_id}_baseline.wav"

    if not wav_cache_path.exists():
        samples = audio.load(flac_path)
        sf.write(str(wav_cache_path), samples, SR, subtype="PCM_16")

    return FileResponse(
        wav_cache_path,
        media_type="audio/wav",
        filename=f"{speech_id}_baseline.wav",
    )


# ── Analyze Endpoint ─────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze_audio(
    audio_file: UploadFile = File(..., alias="audio"),
    speech_id: str | None = Form(None),
    transcript: str | None = Form(None),
):
    # Guard 1: File size check (MAX_UPLOAD_MB)
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    content = await audio_file.read()

    if len(content) == 0:
        return _err(400, "empty_file", "Uploaded audio file is empty.")

    if len(content) > max_bytes:
        return _err(413, "payload_too_large", f"Audio file exceeds {MAX_UPLOAD_MB} MB limit.")

    # Guard 2: Norm mode transcript check
    is_norm_mode = speech_id is None or not (LIBRARY_DIR / speech_id / "baseline.json").exists()
    has_transcript = transcript is not None and transcript.strip() != ""
    if is_norm_mode and not has_transcript:
        return _err(422, "transcript_missing", "Transcript is required in norm mode.")

    # Guard 3: Convert/load audio
    # Save temporarily to parse format
    suffix = Path(audio_file.filename or "upload.wav").suffix.lower()
    allowed_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm", ".aac"}
    if suffix and suffix not in allowed_exts:
        return _err(400, "unsupported_format", f"Unsupported audio format '{suffix}'.")

    temp_path = CACHE_MEDIA_DIR / f"upload_{os.getpid()}_{asyncio.get_event_loop().time()}{suffix}"
    CACHE_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(content)

    try:
        try:
            samples = audio.load(temp_path)
        except Exception as exc:
            if "exceeds" in str(exc) or "too long" in str(exc).lower():
                return _err(400, "audio_too_long", str(exc))
            return _err(400, "unsupported_format", f"Failed to decode audio file: {exc}")

        # Guard 4: Duration check
        dur = len(samples) / SR
        if dur > MAX_AUDIO_S:
            return _err(400, "audio_too_long", f"Audio duration ({dur:.1f} s) exceeds {MAX_AUDIO_S} s limit.")

        # Guard 5: Silence check
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        if peak < 1e-4:
            return _err(400, "silent_audio", "Uploaded audio is silent or near-zero amplitude.")

        # Process through pipeline with semaphore and timeout
        async with _SEMAPHORE:
            def _run():
                return analyze(
                    temp_path,
                    speech_id=speech_id,
                    transcript_text=transcript,
                    use_cache=True,
                )

            res = await asyncio.wait_for(
                run_in_threadpool(_run),
                timeout=_TIMEOUT_SECONDS,
            )
            return res
    except asyncio.TimeoutError:
        return _err(504, "timeout", f"Analysis timed out after {_TIMEOUT_SECONDS} seconds.")
    except AlignmentError as exc:
        return _err(500, "alignment_failed", f"Forced alignment failed: {exc}")
    except ValueError as exc:
        if "Transcript missing" in str(exc):
            return _err(422, "transcript_missing", str(exc))
        return _err(400, "analysis_error", str(exc))
    except Exception as exc:
        return _err(500, "analysis_failed", f"Analysis error: {exc}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


# ── Demo Clips Endpoint ──────────────────────────────────────────────────────
@app.get("/api/demo")
async def list_demo_clips() -> list[dict[str, Any]]:
    # 1. If demo/index.json exists in data/demo or web/public/demo
    for idx_path in [DEMO_DIR / "index.json", Path("demo/index.json"), WEB_PUBLIC_DEMO / "index.json"]:
        if idx_path.exists():
            try:
                return json.loads(idx_path.read_text(encoding="utf-8"))
            except Exception:
                pass

    # 2. Fallback: discover available sample clips from data/dataset
    demos: list[dict[str, Any]] = []
    labels_file = DATASET_DIR / "speech-001" / "labels.json"
    if labels_file.exists():
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        for entry in labels[:8]:
            flaws = entry.get("flaws", [])
            fam = flaws[0]["family"] if flaws else "clean"
            lvl = entry.get("level", 0)
            fname = Path(entry["file"]).name
            demos.append({
                "id": fname.replace(".flac", ""),
                "label": f"{fam} (Level {lvl})",
                "level": lvl,
                "family": fam,
                "audio_url": f"/demo/{fname}",
                "result_url": f"/demo/{fname.replace('.flac', '.json')}",
            })

    return demos


# ── Static File Mounts ───────────────────────────────────────────────────────
# Mount /demo for demo audio and json results
demo_mount_dir = None
if DEMO_DIR.exists():
    demo_mount_dir = DEMO_DIR
elif WEB_PUBLIC_DEMO.exists():
    demo_mount_dir = WEB_PUBLIC_DEMO
elif (DATASET_DIR / "speech-001").exists():
    demo_mount_dir = DATASET_DIR / "speech-001"

if demo_mount_dir is not None:
    app.mount("/demo", StaticFiles(directory=str(demo_mount_dir)), name="demo")

# Mount / for built Next.js frontend (web/out) if present
if WEB_OUT_DIR.exists() and (WEB_OUT_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(WEB_OUT_DIR), html=True), name="frontend")
