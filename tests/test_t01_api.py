"""
test_t01_api.py -- TestClient tests for FastAPI backend (TASK_2 T01, DESIGN §14).

Covers:
  - GET  /api/health
  - GET  /api/speeches
  - GET  /api/media/{speech_id}/baseline.wav (200 and 404)
  - POST /api/analyze (reference mode, cache hit, schema validation)
  - POST /api/analyze (norm mode with custom transcript)
  - Error guards:
      * 413 payload_too_large (> 20 MB)
      * 400 empty_file (0 bytes)
      * 400 unsupported_format
      * 400 silent_audio
      * 400 audio_too_long (> 180 s)
      * 422 transcript_missing (norm mode without transcript)
  - GET  /api/demo
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from api.main import app
from speechlens.config import SR, MAX_UPLOAD_MB
from speechlens.models import AnalysisResult

client = TestClient(app)

SAMPLE_FLAC = Path("data/library/speech-001/readers/r4_jn.flac")
SAMPLE_SPEECH_ID = "speech-001"


# ── Health Endpoint ──────────────────────────────────────────────────────────

def test_api_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


# ── Speeches Endpoint ────────────────────────────────────────────────────────

def test_api_speeches():
    resp = client.get("/api/speeches")
    assert resp.status_code == 200
    speeches = resp.json()
    assert isinstance(speeches, list)
    assert len(speeches) >= 1
    s0 = speeches[0]
    required_fields = {
        "id", "title", "speaker", "duration_s",
        "n_words", "transcript", "baseline_audio_url", "n_readers"
    }
    assert required_fields.issubset(s0.keys())
    assert s0["id"] == "speech-001"
    assert s0["baseline_audio_url"] == "/api/media/speech-001/baseline.wav"


# ── Baseline Media Endpoint ──────────────────────────────────────────────────

def test_api_media_baseline():
    resp = client.get(f"/api/media/{SAMPLE_SPEECH_ID}/baseline.wav")
    assert resp.status_code == 200
    assert resp.headers["content-type"] in ("audio/wav", "audio/x-wav")
    assert len(resp.content) > 1000


def test_api_media_baseline_not_found():
    resp = client.get("/api/media/nonexistent_speech/baseline.wav")
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"] == "not_found"


# ── Analyze Reference Mode & Cache Hit ───────────────────────────────────────

def test_api_analyze_reference_mode():
    assert SAMPLE_FLAC.exists()
    audio_bytes = SAMPLE_FLAC.read_bytes()

    resp = client.post(
        "/api/analyze",
        data={"speech_id": SAMPLE_SPEECH_ID},
        files={"audio": ("r4_jn.flac", audio_bytes, "audio/flac")},
    )
    assert resp.status_code == 200
    result = resp.json()

    # Schema validation
    AnalysisResult.model_validate(result)

    assert result["schema_version"] == "1.0"
    assert result["meta"]["speech_id"] == SAMPLE_SPEECH_ID
    assert result["meta"]["mode"] == "reference"
    assert result["meta"]["duration_s"] > 0
    assert "scores" in result
    assert "words" in result
    assert "series" in result
    assert "regions" in result

    # Test cache hit on identical second call
    resp2 = client.post(
        "/api/analyze",
        data={"speech_id": SAMPLE_SPEECH_ID},
        files={"audio": ("r4_jn.flac", audio_bytes, "audio/flac")},
    )
    assert resp2.status_code == 200
    assert resp2.json()["meta"]["input_sha256"] == result["meta"]["input_sha256"]


# ── Analyze Norm Mode ────────────────────────────────────────────────────────

def test_api_analyze_norm_mode():
    # Create a 3-second speech audio clip
    t = np.linspace(0, 3.0, 3 * SR, endpoint=False)
    # Synthetic speech-like harmonic signal
    signal = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.sin(2 * np.pi * 300 * t)
    buf = io.BytesIO()
    sf.write(buf, signal, SR, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    resp = client.post(
        "/api/analyze",
        data={"transcript": "testing normal speech mode"},
        files={"audio": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["meta"]["mode"] == "norm"
    assert any("norm mode" in w.lower() for w in result["warnings"])
    AnalysisResult.model_validate(result)


# ── Error Guards (400, 413, 422) ─────────────────────────────────────────────

def test_guard_empty_file():
    resp = client.post(
        "/api/analyze",
        data={"speech_id": SAMPLE_SPEECH_ID},
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "empty_file"


def test_guard_payload_too_large():
    # > 20 MB (21 MB dummy payload)
    huge_bytes = b"0" * (21 * 1024 * 1024)
    resp = client.post(
        "/api/analyze",
        data={"speech_id": SAMPLE_SPEECH_ID},
        files={"audio": ("huge.wav", huge_bytes, "audio/wav")},
    )
    assert resp.status_code == 413
    data = resp.json()
    assert data["error"] == "payload_too_large"


def test_guard_unsupported_format():
    resp = client.post(
        "/api/analyze",
        data={"speech_id": SAMPLE_SPEECH_ID},
        files={"audio": ("malicious.exe", b"MZ\x90\x00not an audio file", "application/octet-stream")},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "unsupported_format"


def test_guard_silent_audio():
    # 2 seconds of pure zeros
    zeros = np.zeros(2 * SR, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, zeros, SR, format="WAV", subtype="PCM_16")
    resp = client.post(
        "/api/analyze",
        data={"speech_id": SAMPLE_SPEECH_ID},
        files={"audio": ("silent.wav", buf.getvalue(), "audio/wav")},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "silent_audio"


def test_guard_audio_too_long():
    # Audio length > 180s (182s of dummy tone)
    # Using 1000 Hz sample rate in header or 182s mono to keep test bytes reasonable
    t = np.linspace(0, 182.0, int(182.0 * SR), endpoint=False)
    # Just need sf header duration to be > 180
    signal = 0.1 * np.sin(2 * np.pi * 200 * t[:SR*182])
    buf = io.BytesIO()
    sf.write(buf, signal, SR, format="WAV", subtype="PCM_16")
    resp = client.post(
        "/api/analyze",
        data={"speech_id": SAMPLE_SPEECH_ID},
        files={"audio": ("too_long.wav", buf.getvalue(), "audio/wav")},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "audio_too_long"


def test_guard_transcript_missing_in_norm_mode():
    # No speech_id and no transcript provided
    t = np.linspace(0, 1.0, SR, endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 200 * t)
    buf = io.BytesIO()
    sf.write(buf, signal, SR, format="WAV", subtype="PCM_16")
    resp = client.post(
        "/api/analyze",
        files={"audio": ("sample.wav", buf.getvalue(), "audio/wav")},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "transcript_missing"


# ── Demo Endpoint ────────────────────────────────────────────────────────────

def test_api_demo():
    resp = client.get("/api/demo")
    assert resp.status_code == 200
    demos = resp.json()
    assert isinstance(demos, list)
    if demos:
        d0 = demos[0]
        assert "id" in d0
        assert "label" in d0
        assert "audio_url" in d0
        assert "result_url" in d0
