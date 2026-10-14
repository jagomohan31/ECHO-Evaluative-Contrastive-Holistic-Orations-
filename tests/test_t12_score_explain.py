"""
tests/test_t12_score_explain.py -- T12 tests for scoring rubric and causal explanations (DESIGN 10, 11).
"""
from __future__ import annotations
import pytest
from speechlens.config import WEIGHTS, K
from speechlens.detect import Region
from speechlens.score import (
    compute_region_deduction, compute_scores, compute_region_costs, DIMENSIONS,
)
from speechlens.explain import explain_region, explain_all

ALL_FAMILIES = [
    "pace_fast", "pace_slow", "pace_global", "pause_missing",
    "pause_excess", "hesitation_filled", "monotone", "terminal_contour",
    "stress_missed", "volume_low", "clipping", "trailing_off",
    "muffled", "word_swallowed",
]

FAMILY_DIM = {
    "pace_fast": "pacing", "pace_slow": "pacing", "pace_global": "pacing",
    "pause_missing": "pacing", "monotone": "intonation", "terminal_contour": "intonation",
    "stress_missed": "intonation", "pause_excess": "fluency", "hesitation_filled": "fluency",
    "volume_low": "volume", "clipping": "volume", "trailing_off": "volume",
    "muffled": "clarity", "word_swallowed": "clarity",
}


# ── Scoring Tests (DESIGN 10) ────────────────────────────────────────────────

def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100
    for d in DIMENSIONS:
        assert d in WEIGHTS


def test_empty_regions_perfect_score():
    res = compute_scores([], total_duration=60.0)
    assert res["total"] == 100.0
    for d in DIMENSIONS:
        assert res["dimensions"][d] == 100.0


def test_scores_bounded_0_to_100():
    # Massive deductions
    huge_regions = [
        Region(family="volume_low", dimension="volume", start=0.0, end=60.0,
               word_start=0, word_end=10, severity=1.0, z_peak=-4.0)
        for _ in range(10)
    ]
    res = compute_scores(huge_regions, total_duration=60.0)
    assert 0.0 <= res["total"] <= 100.0
    assert res["dimensions"]["volume"] == 0.0
    # Other untouched dimensions stay 100
    assert res["dimensions"]["pacing"] == 100.0


def test_severity_monotonicity():
    """Higher severity never raises the score."""
    r_mild = [Region(family="monotone", dimension="intonation", start=0.0, end=10.0,
                     word_start=0, word_end=5, severity=0.3, z_peak=-1.8)]
    r_severe = [Region(family="monotone", dimension="intonation", start=0.0, end=10.0,
                       word_start=0, word_end=5, severity=0.9, z_peak=-3.5)]

    score_mild = compute_scores(r_mild, total_duration=60.0)["total"]
    score_severe = compute_scores(r_severe, total_duration=60.0)["total"]

    assert score_severe <= score_mild


def test_duration_monotonicity():
    """Longer duration flaw never raises the score."""
    r_short = [Region(family="volume_low", dimension="volume", start=0.0, end=5.0,
                      word_start=0, word_end=2, severity=0.8, z_peak=-2.5)]
    r_long = [Region(family="volume_low", dimension="volume", start=0.0, end=30.0,
                     word_start=0, word_end=15, severity=0.8, z_peak=-2.5)]

    score_short = compute_scores(r_short, total_duration=60.0)["total"]
    score_long = compute_scores(r_long, total_duration=60.0)["total"]

    assert score_long <= score_short


def test_region_cost_equals_difference():
    """Cost of each region equals total without region minus current total."""
    r1 = Region(id="r1", family="volume_low", dimension="volume", start=0.0, end=5.0,
                word_start=0, word_end=2, severity=0.6, z_peak=-2.0)
    r2 = Region(id="r2", family="monotone", dimension="intonation", start=10.0, end=20.0,
                word_start=3, word_end=7, severity=0.7, z_peak=-2.5)
    regs = [r1, r2]

    base_total = compute_scores(regs, total_duration=60.0)["total"]
    compute_region_costs(regs, total_duration=60.0)

    total_without_r1 = compute_scores([r2], total_duration=60.0)["total"]
    total_without_r2 = compute_scores([r1], total_duration=60.0)["total"]

    assert r1.cost == pytest.approx(total_without_r1 - base_total, abs=1e-4)
    assert r2.cost == pytest.approx(total_without_r2 - base_total, abs=1e-4)


# ── Explanation Tests (DESIGN 11) ─────────────────────────────────────────────

def test_every_family_has_template():
    """Ensure all 14 families generate all 5 required explanation keys."""
    sample_evidence = {
        "pace_fast": {"t0": 1.0, "t1": 3.0, "text": "quick words", "pct": 25.0, "z": -2.1, "formula": "z = -2.1"},
        "pace_slow": {"t0": 2.0, "t1": 5.0, "text": "slow speech", "pct": 30.0, "z": 2.3, "formula": "z = 2.3"},
        "pace_global": {"obs": 6.2, "exp": 4.1, "pct": 51.2, "z": 0.41, "formula": "g = 0.41"},
        "pause_missing": {"word": "and", "t": 4.5, "obs_ms": 50, "exp_ms": 350, "sig_ms": 80, "prev": "before", "punct": ",", "z": -3.7, "formula": "z = -3.7"},
        "pause_excess": {"word": "nation", "t": 12.0, "obs_ms": 950, "z": 2.8, "formula": "z = 2.8"},
        "hesitation_filled": {"ms": 450, "t": 7.2, "word": "we", "formula": "dur = 450 ms"},
        "monotone": {"t0": 5.0, "t1": 10.0, "obs": 0.6, "exp": 2.4, "sig": 0.4, "z": -4.5, "formula": "z = -4.5"},
        "terminal_contour": {"word": "liberty", "t": 15.0, "obs": 4.2, "exp": -1.5, "z": 3.1, "formula": "z = 3.1"},
        "stress_missed": {"word": "honour", "obs": 0.1, "exp": 1.5, "z": -2.8, "formula": "z = -2.8"},
        "volume_low": {"t0": 1.0, "t1": 3.0, "obs": -18.0, "exp": -5.0, "sig": 2.5, "z": -5.2, "formula": "z = -5.2"},
        "clipping": {"pct": 4.5, "t0": 8.0, "t1": 9.5, "z": 0.45, "formula": "clip = 4.5%"},
        "trailing_off": {"word": "earth", "obs": -14.0, "exp": -2.0, "z": -3.2, "formula": "z = -3.2"},
        "muffled": {"t0": 2.0, "t1": 6.0, "obs": -8.0, "exp": -2.0, "sig": 1.5, "obs2": -12.0, "exp2": -4.0, "z": -3.0, "formula": "z = -3.0"},
        "word_swallowed": {"word": "proposition", "t": 11.2, "score": 0.18, "ms": 45, "formula": "score = 0.18"},
    }

    required_keys = {"summary", "context", "effect", "fix", "math"}

    for fam in ALL_FAMILIES:
        ev = sample_evidence.get(fam, {"z": -2.0})
        reg = Region(
            family=fam,
            dimension=FAMILY_DIM[fam],
            start=1.0,
            end=3.0,
            word_start=0,
            word_end=2,
            severity=0.5,
            z_peak=-2.0,
            evidence=ev,
        )
        exp = explain_region(reg)
        assert set(exp.keys()) == required_keys, f"Missing keys in {fam}"
        for k, v in exp.items():
            assert isinstance(v, str) and len(v.strip()) > 0, f"Empty field {k} in {fam}"


def test_explain_all_in_place():
    r = Region(family="clipping", dimension="volume", start=1.0, end=2.0,
               word_start=1, word_end=2, severity=0.5, z_peak=0.5,
               evidence={"pct": 3.0, "t0": 1.0, "t1": 2.0, "formula": "clip = 3%"})
    assert not r.explanation
    explain_all([r])
    assert bool(r.explanation)
    assert r.explanation["fix"] == "Move back from the microphone or lower the input gain."
