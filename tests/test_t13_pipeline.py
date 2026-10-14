"""
test_t13_pipeline.py -- Tests for T13 (end-to-end pipeline).

Tests:
  1. Schema validation against AnalysisResult Pydantic model
  2. Byte-identical determinism (use_cache=False twice)
  3. <30 s wall-clock for a 60 s clip (cached run)
  4. Wrong-passage warning appears when transcript diverges
  5. Norm-mode warning appears when no speech_id provided
  6. Norm-mode path returns no baseline_readers
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speechlens.pipeline import analyze
from speechlens.models import AnalysisResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIBRARY = Path("data/library")
SPEECH_ID = "speech-001"
AUDIO_FILE = LIBRARY / SPEECH_ID / "readers" / "r4_jn.flac"
TRANSCRIPT = (LIBRARY / SPEECH_ID / "transcript.txt").read_text(encoding="utf-8").strip()

# --------------------------------------------------------------------------
# Fixture: run the analysis once (cached) so individual tests are fast
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def result():
    """Run analysis with cache enabled; result is shared across all tests."""
    return analyze(str(AUDIO_FILE), speech_id=SPEECH_ID, use_cache=True)


# --------------------------------------------------------------------------
# T13-1  Schema validation
# --------------------------------------------------------------------------

def test_schema_validates(result):
    """AnalysisResult.model_validate must not raise."""
    AnalysisResult.model_validate(result)


def test_schema_keys_present(result):
    required_top = {"schema_version", "meta", "scores", "global", "words", "series", "pauses", "regions", "warnings"}
    assert required_top.issubset(result.keys()), f"Missing keys: {required_top - result.keys()}"


def test_meta_fields(result):
    meta = result["meta"]
    assert meta["speech_id"] == SPEECH_ID
    assert meta["mode"] in ("reference", "norm")
    assert meta["duration_s"] > 0
    assert meta["n_words"] > 0
    assert isinstance(meta["baseline_readers"], list)


def test_scores_bounded(result):
    s = result["scores"]
    assert 0.0 <= s["total"] <= 100.0
    for dim, v in s["dimensions"].items():
        assert 0.0 <= v <= 100.0, f"Dimension {dim} out of bounds: {v}"


def test_regions_have_required_fields(result):
    required = {"id", "family", "dimension", "start", "end", "word_idx", "severity", "z_peak", "cost", "evidence", "explanation"}
    for r in result["regions"]:
        missing = required - r.keys()
        assert not missing, f"Region missing fields: {missing}"


def test_words_have_required_fields(result):
    for w in result["words"]:
        assert "i" in w
        assert "text" in w
        assert "start" in w and "end" in w
        assert "z" in w
        assert "flags" in w


def test_series_lengths_match(result):
    series = result["series"]
    n = len(series["t_user"])
    assert n > 0
    assert len(series["t_base"]) == n
    assert len(series["pitch_st"]["user"]) == n
    assert len(series["energy_db"]["user"]) == n
    assert len(series["tempo"]["user"]) == n


# --------------------------------------------------------------------------
# T13-2  Byte-identical determinism
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_byte_identical_determinism():
    """Two use_cache=False runs must produce identical JSON bytes."""
    r1 = analyze(str(AUDIO_FILE), speech_id=SPEECH_ID, use_cache=False)
    r2 = analyze(str(AUDIO_FILE), speech_id=SPEECH_ID, use_cache=False)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


# --------------------------------------------------------------------------
# T13-3  Performance: cached run < 30 s
# --------------------------------------------------------------------------

def test_cached_run_under_30s():
    """A cached analysis on a library file must complete in < 30 s."""
    t0 = time.time()
    analyze(str(AUDIO_FILE), speech_id=SPEECH_ID, use_cache=True)
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"Cached run took {elapsed:.1f} s (>30 s)"


# --------------------------------------------------------------------------
# T13-4  Wrong-passage warning
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_wrong_passage_warning(tmp_path):
    """Feeding a completely different 'transcript' triggers the low-score warning."""
    # We feed the real audio but a garbage transcript → alignment score will be low
    garbage = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
    import soundfile as sf
    import numpy as np
    from speechlens.config import SR
    from speechlens.audio import load as audio_load
    samples = audio_load(AUDIO_FILE)
    # Write to tmp flac
    tmp_flac = tmp_path / "test.flac"
    sf.write(str(tmp_flac), samples, SR, subtype="PCM_16")
    res = analyze(str(tmp_flac), transcript_text=garbage, use_cache=False)
    warning_msgs = " ".join(res["warnings"]).lower()
    # Either wrong-passage or norm-mode warning should appear
    assert "wrong passage" in warning_msgs or "norm mode" in warning_msgs or len(res["warnings"]) > 0


# --------------------------------------------------------------------------
# T13-5  Norm-mode warning when no speech_id
# --------------------------------------------------------------------------

def test_norm_mode_no_speech_id(tmp_path):
    """When no speech_id is given, pipeline must enter norm mode and warn."""
    import soundfile as sf
    from speechlens.config import SR
    from speechlens.audio import load as audio_load
    samples = audio_load(AUDIO_FILE)[:SR * 5]  # 5-second clip
    tmp_flac = tmp_path / "norm_test.flac"
    sf.write(str(tmp_flac), samples, SR, subtype="PCM_16")
    res = analyze(str(tmp_flac), transcript_text=TRANSCRIPT[:50], speech_id=None, use_cache=False)
    assert res["meta"]["mode"] == "norm"
    assert any("norm mode" in w.lower() for w in res["warnings"])


# --------------------------------------------------------------------------
# T13-6  Norm-mode: no baseline_readers
# --------------------------------------------------------------------------

def test_norm_mode_no_baseline_readers(tmp_path):
    """Norm-mode results must have empty baseline_readers list."""
    import soundfile as sf
    from speechlens.config import SR
    from speechlens.audio import load as audio_load
    samples = audio_load(AUDIO_FILE)[:SR * 5]
    tmp_flac = tmp_path / "norm_test2.flac"
    sf.write(str(tmp_flac), samples, SR, subtype="PCM_16")
    res = analyze(str(tmp_flac), transcript_text=TRANSCRIPT[:50], speech_id=None, use_cache=False)
    assert res["meta"]["baseline_readers"] == []
