"""tests/test_t05_demo_assets.py -- Test demo assets and limits per T05."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DEMO_DIR = REPO_ROOT / "data" / "demo"
WEB_DEMO_DIR = REPO_ROOT / "web" / "public" / "demo"
MAX_TOTAL_AUDIO_MB = 15.0


def test_demo_index_exists():
    """Index JSON exists in data/demo and web/public/demo with >= 8 clips."""
    data_idx = DATA_DEMO_DIR / "index.json"
    web_idx = WEB_DEMO_DIR / "index.json"
    assert data_idx.exists(), "data/demo/index.json missing"
    assert web_idx.exists(), "web/public/demo/index.json missing"

    clips = json.loads(data_idx.read_text(encoding="utf-8"))
    assert len(clips) >= 8, f"Expected >= 8 clips, found {len(clips)}"

    # Check required families and levels
    levels = {c["level"] for c in clips}
    families = {c["family"] for c in clips}
    assert 0 in levels, "L0 clip missing"
    assert {1, 2, 3}.issubset(levels), "L1, L2, L3 levels missing"
    assert "clipping" in families, "clipping example missing"
    assert "monotone" in families, "monotone example missing"


def test_demo_total_audio_size_under_limit():
    """Total demo audio files are <= 15 MB."""
    audio_files = list(DATA_DEMO_DIR.glob("*.flac")) + list(DATA_DEMO_DIR.glob("*.wav"))
    assert len(audio_files) >= 8, f"Found {len(audio_files)} audio files in demo dir"

    total_bytes = sum(f.stat().st_size for f in audio_files)
    total_mb = total_bytes / (1024 * 1024)
    assert total_mb <= MAX_TOTAL_AUDIO_MB, (
        f"Demo audio size {total_mb:.2f} MB exceeds {MAX_TOTAL_AUDIO_MB} MB limit"
    )


def test_demo_precomputed_results_valid():
    """Every demo clip has a corresponding valid JSON result file."""
    data_idx = DATA_DEMO_DIR / "index.json"
    clips = json.loads(data_idx.read_text(encoding="utf-8"))

    for clip in clips:
        json_file = DATA_DEMO_DIR / f"{clip['id']}.json"
        assert json_file.exists(), f"Missing precomputed JSON: {json_file}"
        res = json.loads(json_file.read_text(encoding="utf-8"))
        assert "scores" in res
        assert "total" in res["scores"]
        assert "dimensions" in res["scores"]
        assert "regions" in res


def test_api_demo_endpoint_lists_clips():
    """GET /api/demo returns the list of demo clips matching index.json."""
    client = TestClient(app)
    resp = client.get("/api/demo")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 8
    assert all("id" in d and "audio_url" in d and "result_url" in d for d in data)
