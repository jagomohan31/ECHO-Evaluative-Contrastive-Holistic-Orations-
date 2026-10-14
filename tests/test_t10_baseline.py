"""
tests/test_t10_baseline.py -- T10: baseline model schema tests.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import pytest

LIBRARY_DIR = Path("data/library")
WORD_FEATURES = [
    "share", "pause_before", "f0_mean_st", "f0_slope",
    "en_mean_rel", "en_slope", "prominence", "hnr_rel", "tilt_rel",
]


def _load_baseline(speech_id: str) -> dict:
    p = LIBRARY_DIR / speech_id / "baseline.json"
    if not p.exists():
        pytest.skip(f"baseline.json not found for {speech_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_norms() -> dict:
    p = LIBRARY_DIR / "norms.json"
    if not p.exists():
        pytest.skip("norms.json not found")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# T10-01  baseline.json has words list with mu for each DESIGN 8.1 feature
# ---------------------------------------------------------------------------
def test_baseline_word_mu_fields():
    baseline = _load_baseline("speech-001")
    words = baseline.get("words", [])
    assert len(words) > 0, "no words in baseline"
    for feat in WORD_FEATURES:
        key = f"mu_{feat}"
        assert key in words[0], f"missing {key} in baseline words[0]"


# ---------------------------------------------------------------------------
# T10-02  sigma is 0 for single-reader speeches (DESIGN 8.1)
# ---------------------------------------------------------------------------
def test_baseline_sigma_zero_for_single_reader():
    for speech_id in ("speech-002", "speech-003", "speech-004"):
        p = LIBRARY_DIR / speech_id / "baseline.json"
        if not p.exists():
            continue
        baseline = json.loads(p.read_text(encoding="utf-8"))
        if baseline.get("n_baseline_readers", 1) >= 3:
            continue
        for feat in WORD_FEATURES:
            key = f"sigma_{feat}"
            if words := baseline.get("words"):
                assert words[0].get(key, 0.0) == 0.0, (
                    f"sigma should be 0 for single reader, got {words[0].get(key)}"
                )
        break


# ---------------------------------------------------------------------------
# T10-03  sigma > 0 for speech-001 which has 3 baseline readers
# ---------------------------------------------------------------------------
def test_baseline_sigma_nonzero_for_multi_reader():
    baseline = _load_baseline("speech-001")
    if baseline.get("n_baseline_readers", 0) < 3:
        pytest.skip("speech-001 needs 3+ readers")
    # At least one word should have a non-zero sigma for share
    nonzero = [
        w for w in baseline["words"]
        if w.get("sigma_share", 0.0) > 0.0
    ]
    assert len(nonzero) > 0, "expected some non-zero sigma_share in multi-reader baseline"


# ---------------------------------------------------------------------------
# T10-04  norms.json has required keys
# ---------------------------------------------------------------------------
def test_norms_keys():
    norms = _load_norms()
    required = {"conservative_defaults", "note", "sigma_D", "sigma_win",
                "artic_rate_mean", "artic_rate_sd"}
    missing = required - set(norms.keys())
    assert not missing, f"norms.json missing keys: {missing}"


# ---------------------------------------------------------------------------
# T10-05  norms.json has floored sigma_D > 0
# ---------------------------------------------------------------------------
def test_norms_sigma_D_positive():
    norms = _load_norms()
    assert norms.get("sigma_D", 0.0) >= 0.10, (
        f"sigma_D should be >= 0.10 (floor), got {norms.get('sigma_D')}"
    )


# ---------------------------------------------------------------------------
# T10-06  baseline articulation rate is within speech-typical range
# ---------------------------------------------------------------------------
def test_baseline_artic_rate_plausible():
    for speech_id in ("speech-001", "speech-002", "speech-003", "speech-004"):
        p = LIBRARY_DIR / speech_id / "baseline.json"
        if not p.exists():
            continue
        baseline = json.loads(p.read_text(encoding="utf-8"))
        ar = baseline.get("artic_rate", 0.0)
        assert 0.5 <= ar <= 10.0, (
            f"{speech_id}: artic_rate {ar:.2f} outside typical range 0.5-10 syl/s"
        )


# ---------------------------------------------------------------------------
# T10-07  norms conservative_defaults flag is consistent with multi-reader count
# ---------------------------------------------------------------------------
def test_norms_conservative_flag_consistent():
    norms = _load_norms()
    # We have only speech-001 with 3 readers; norms should be conservative
    # (< 2 multi-reader speeches means conservative_defaults=True)
    # This is expected given the current library
    assert isinstance(norms["conservative_defaults"], bool)
    if norms["conservative_defaults"]:
        # Conservative defaults should have all expected keys
        required_keys = {
            "ln_share_sd", "pause_sd", "f0_mean_sd", "f0_slope_sd",
            "en_mean_sd", "en_slope_sd", "prominence_sd", "hnr_sd", "tilt_sd"
        }
        missing = required_keys - set(norms.keys())
        assert not missing, f"conservative norms missing: {missing}"