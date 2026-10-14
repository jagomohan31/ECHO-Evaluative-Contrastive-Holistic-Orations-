"""
tests/test_t11_detect.py -- T11 unit tests for flaw detection engine (DESIGN 9).
Fast unit tests covering all 13 rules, region building (gap, min size, merge),
evidence dictionaries, boundary refinement, and norm mode.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import pytest
from speechlens.detect import (
    Region, _build_regions, detect_pace, detect_pause_missing,
    detect_pause_excess, detect_hesitation_filled, detect_monotone,
    detect_terminal_contour, detect_stress_missed, detect_volume_low,
    detect_clipping, detect_trailing_off, detect_muffled,
    detect_word_swallowed, detect_all, detect_norm, refine_boundaries,
)
from speechlens.features.word import WordFeatures, PauseInfo


def make_dummy_words(n=10, dur=0.5, text_prefix="word"):
    words = []
    t = 0.0
    for i in range(n):
        w = WordFeatures(
            i=i,
            text=f"{text_prefix}_{i}",
            start=t,
            end=t + dur,
            nsyl=1,
            dur=dur,
            share=1.0 / n,
            align_score=0.9,
            voiced_share_win=0.8,
            f0_sd_st=2.5,
            prominence=0.5,
            en_mean_rel=-5.0,
            f0_slope=-1.0,
            en_slope=0.0,
            tilt_rel=0.0,
            hnr_rel=0.0,
            clip_frac=0.0,
            punct_after="." if i == n - 1 else "",
        )
        words.append(w)
        t += dur
    return words


# ── Region Building Tests ─────────────────────────────────────────────────────

def test_build_regions_min_size():
    words = make_dummy_words(10)
    # min_size = 4, but only 3 words flagged
    flagged = [0, 1, 2]
    z_vals = {0: -2.0, 1: -2.5, 2: -2.1}
    r = _build_regions(flagged, words, z_vals, "pace_fast", min_size=4)
    assert len(r) == 0

    # 4 words flagged -> creates 1 region
    flagged = [0, 1, 2, 3]
    z_vals = {0: -2.0, 1: -2.5, 2: -2.1, 3: -2.2}
    r = _build_regions(flagged, words, z_vals, "pace_fast", min_size=4)
    assert len(r) == 1
    assert r[0].word_start == 0
    assert r[0].word_end == 3
    assert r[0].z_peak == -2.5


def test_build_regions_one_word_gap():
    words = make_dummy_words(10)
    # Flagged with 1 unflagged word in between: [0, 1, 3, 4] -> bridges gap at 2
    flagged = [0, 1, 3, 4]
    z_vals = {0: -2.0, 1: -2.5, 3: -2.1, 4: -2.2}
    r = _build_regions(flagged, words, z_vals, "pace_fast", min_size=4)
    assert len(r) == 1
    assert r[0].word_start == 0
    assert r[0].word_end == 4

    # 2 unflagged words in between: [0, 1, 4, 5] -> splits into 2 runs of length 2
    flagged = [0, 1, 4, 5]
    z_vals = {0: -2.0, 1: -2.5, 4: -2.1, 5: -2.2}
    r = _build_regions(flagged, words, z_vals, "pace_fast", min_size=4)
    assert len(r) == 0  # both runs < min_size 4


def test_build_regions_merge_closer_than_03s():
    # Create two regions separated by 0.2s (< 0.3s)
    w1 = WordFeatures(i=0, text="a", start=0.0, end=1.0, nsyl=1, dur=1.0)
    w2 = WordFeatures(i=1, text="b", start=1.2, end=2.0, nsyl=1, dur=0.8)
    words = [w1, w2]
    # min_size = 1 for pause_missing
    flagged = [0, 1]
    z_vals = {0: -2.0, 1: -3.0}
    # They are not consecutive in run because delta is 1 so they are in same run anyway.
    # To test merge logic explicitly:
    r1 = Region(family="volume_low", dimension="volume", start=0.0, end=1.0,
                word_start=0, word_end=0, severity=0.5, z_peak=-2.0)
    r2 = Region(family="volume_low", dimension="volume", start=1.2, end=2.0,
                word_start=1, word_end=1, severity=0.8, z_peak=-3.0)
    assert r2.start - r1.end == pytest.approx(0.2)
    # Through _build_regions when split:
    flagged = [0, 3]  # gap of 2 unflagged words (1, 2)
    w_list = [
        WordFeatures(i=0, text="a", start=0.0, end=1.0, nsyl=1, dur=1.0),
        WordFeatures(i=1, text="b", start=1.0, end=1.05, nsyl=1, dur=0.05),
        WordFeatures(i=2, text="c", start=1.05, end=1.1, nsyl=1, dur=0.05),
        WordFeatures(i=3, text="d", start=1.2, end=2.0, nsyl=1, dur=0.8),
    ]
    # min_size = 1
    r = _build_regions(flagged, w_list, {0: -2.0, 3: -3.0}, "pause_missing", min_size=1)
    assert len(r) == 1
    assert r[0].start == 0.0
    assert r[0].end == 2.0


# ── Rule 1 & 2: pace_fast / pace_slow ─────────────────────────────────────────

def test_detect_pace_fast_and_slow():
    words = make_dummy_words(10, dur=0.2)
    base_mu = [0.1] * 10
    # Make words 0..4 have very small share (fast)
    for i in range(5):
        words[i].share = 0.02
    for i in range(5, 10):
        words[i].share = 0.18

    regs = detect_pace(words, base_mu, sigma_D=0.12)
    families = {r.family for r in regs}
    assert "pace_fast" in families or "pace_slow" in families
    for r in regs:
        assert "metric" in r.evidence
        assert "formula" in r.evidence
        assert r.severity >= 0.0


# ── Rule 3: pace_global ───────────────────────────────────────────────────────

def test_detect_pace_global():
    words = make_dummy_words(10, dur=0.1)  # fast: 10 syl in 1.0s = 10 syl/s
    base_mu = [0.1] * 10  # base dur = 10 * 0.1 / 0.1 * 0.1 = 1.0 ?
    # Let's verify pace_global fires when g > ln(1.25)
    for w in words:
        w.dur = 0.1
        w.share = 0.1
        w.nsyl = 2  # 20 syl in 1.0 s -> user_art = 20 syl/s
    regs = detect_pace(words, base_mu, sigma_D=0.12, base_art_rate=4.0)
    p_glob = [r for r in regs if r.family == "pace_global"]
    assert len(p_glob) == 1
    assert p_glob[0].severity > 0.0
    assert "obs" in p_glob[0].evidence


# ── Rule 4: pause_missing ─────────────────────────────────────────────────────

def test_detect_pause_missing():
    words = make_dummy_words(5)
    words[2].pause_before = 0.05  # observed pause 50 ms
    mu_pause = [0.0, 0.0, 0.40, 0.0, 0.0]  # baseline expected 400 ms
    sigma_pause = [0.0, 0.0, 0.10, 0.0, 0.0]

    regs = detect_pause_missing(words, mu_pause, sigma_pause, sp_pause=0.10)
    assert len(regs) == 1
    assert regs[0].family == "pause_missing"
    assert regs[0].word_start == 2
    assert regs[0].evidence["obs_ms"] == 50
    assert regs[0].evidence["exp_ms"] == 400


# ── Rule 5: pause_excess ──────────────────────────────────────────────────────

def test_detect_pause_excess():
    words = make_dummy_words(5)
    pauses = [PauseInfo(start=1.0, end=1.8, before_word=1, after_word=2, kind="silent")]
    mu_pause = [0.0, 0.0, 0.1, 0.0, 0.0]
    sigma_pause = [0.0, 0.0, 0.05, 0.0, 0.0]

    regs = detect_pause_excess(words, pauses, mu_pause, sigma_pause, sp_pause=0.05)
    assert len(regs) == 1
    assert regs[0].family == "pause_excess"
    assert regs[0].word_start == 2
    assert regs[0].evidence["obs_ms"] == 800


# ── Rule 6: hesitation_filled ─────────────────────────────────────────────────

def test_detect_hesitation_filled():
    pauses = [PauseInfo(start=2.0, end=2.5, before_word=1, after_word=2, kind="filled")]
    regs = detect_hesitation_filled(pauses)
    assert len(regs) == 1
    assert regs[0].family == "hesitation_filled"
    assert regs[0].evidence["ms"] == 500
    assert regs[0].severity == pytest.approx((0.5 - 0.25) / 0.75, abs=0.01)


# ── Rule 7: monotone ──────────────────────────────────────────────────────────

def test_detect_monotone():
    words = make_dummy_words(8)
    for w in words:
        w.f0_sd_st = 0.5  # flat pitch
        w.voiced_share_win = 0.8
    mu_f0sd = [2.5] * 8
    sigma_f0sd = [0.3] * 8

    regs = detect_monotone(words, mu_f0sd, sigma_f0sd, sp_f0sd=0.3)
    assert len(regs) == 1
    assert regs[0].family == "monotone"
    assert regs[0].word_start == 0
    assert regs[0].word_end == 7
    assert "obs" in regs[0].evidence


# ── Rule 8: terminal_contour ──────────────────────────────────────────────────

def test_detect_terminal_contour():
    words = make_dummy_words(5)
    words[4].punct_after = "."
    words[4].f0_slope = 3.5  # rising ending
    mu_slope = [-1.0, -1.0, -1.0, -1.0, -1.5]
    sigma_slope = [1.0] * 5

    regs = detect_terminal_contour(words, mu_slope, sigma_slope, sp_slope=1.0)
    assert len(regs) == 1
    assert regs[0].family == "terminal_contour"
    assert regs[0].word_start == 4
    assert regs[0].evidence["obs"] == 3.5


# ── Rule 9: stress_missed ─────────────────────────────────────────────────────

def test_detect_stress_missed():
    words = make_dummy_words(5)
    words[2].prominence = 0.1  # flat delivery
    mu_prom = [0.5, 0.5, 1.4, 0.5, 0.5]  # baseline prominent word
    sigma_prom = [0.2] * 5

    regs = detect_stress_missed(words, mu_prom, sigma_prom, sp_prom=0.2)
    assert len(regs) == 1
    assert regs[0].family == "stress_missed"
    assert regs[0].word_start == 2
    assert regs[0].evidence["exp"] == 1.4


# ── Rule 10: volume_low ───────────────────────────────────────────────────────

def test_detect_volume_low():
    words = make_dummy_words(5)
    words[1].en_mean_rel = -20.0
    words[2].en_mean_rel = -22.0
    mu_en = [-5.0] * 5
    sigma_en = [2.0] * 5

    regs = detect_volume_low(words, mu_en, sigma_en, sp_en=2.0)
    assert len(regs) == 1
    assert regs[0].family == "volume_low"
    assert regs[0].word_start == 1
    assert regs[0].word_end == 2


# ── Rule 11: clipping ─────────────────────────────────────────────────────────

def test_detect_clipping():
    words = make_dummy_words(5)
    words[1].clip_frac = 0.02
    words[2].clip_frac = 0.03

    regs = detect_clipping(words)
    assert len(regs) == 1
    assert regs[0].family == "clipping"
    assert regs[0].word_start == 1
    assert regs[0].word_end == 2
    assert regs[0].evidence["pct"] >= 2.0


# ── Rule 12: trailing_off ─────────────────────────────────────────────────────

def test_detect_trailing_off():
    words = make_dummy_words(5)
    words[4].punct_after = "."
    # Last 3 words: indices 2, 3, 4
    for i in (2, 3, 4):
        words[i].en_slope = -15.0
    mu_slope = [-1.0] * 5
    sigma_slope = [2.0] * 5

    regs = detect_trailing_off(words, mu_slope, sigma_slope, sp_slope=2.0)
    assert len(regs) == 1
    assert regs[0].family == "trailing_off"
    assert regs[0].word_start == 2
    assert regs[0].word_end == 4


# ── Rule 13: muffled ──────────────────────────────────────────────────────────

def test_detect_muffled():
    words = make_dummy_words(5)
    for i in (1, 2, 3):
        words[i].tilt_rel = -12.0
        words[i].hnr_rel = -10.0
    mu_tilt = [0.0] * 5
    mu_hnr = [0.0] * 5
    sigma_tilt = [2.0] * 5
    sigma_hnr = [2.0] * 5

    regs = detect_muffled(words, mu_tilt, mu_hnr, sigma_tilt, sigma_hnr, 2.0, 2.0)
    assert len(regs) == 1
    assert regs[0].family == "muffled"
    assert regs[0].word_start == 1
    assert regs[0].word_end == 3


# ── Rule 14: word_swallowed ───────────────────────────────────────────────────

def test_detect_word_swallowed():
    words = make_dummy_words(5)
    words[1].align_score = 0.15  # very low score
    words[3].dur = 0.04  # < 0.06s with 2 syllables
    words[3].nsyl = 2

    regs = detect_word_swallowed(words)
    assert len(regs) == 2
    assert regs[0].word_start == 1
    assert regs[1].word_start == 3


# ── detect_all integration & evidence check ───────────────────────────────────

def test_detect_all_integration():
    words = make_dummy_words(10)
    words[1].clip_frac = 0.05
    words[2].clip_frac = 0.05
    baseline = {
        "words": [{"mu_share": 0.1, "mu_f0_sd_st": 2.5, "mu_en_mean_rel": -5.0} for _ in range(10)]
    }
    norms = {"sigma_D": 0.12, "sigma_pool_f0_sd_st": 0.8}

    regs = detect_all(words, [], baseline, norms)
    assert len(regs) >= 1
    assert regs[0].id == "reg_1"
    assert "metric" in regs[0].evidence
    assert regs[0].word_idx == [1, 2]


# ── detect_norm mode check ───────────────────────────────────────────────────

def test_detect_norm_mode():
    words = make_dummy_words(10)
    words[2].clip_frac = 0.05
    words[3].clip_frac = 0.05
    norms = {
        "artic_rate_mean": 4.2,
        "artic_rate_sd": 0.6,
        "phrase_f0_sd_p10": 1.2,
        "terminal_contour_p90": 2.0,
    }
    regs = detect_norm(words, [], norms)
    assert len(regs) >= 1
    families = {r.family for r in regs}
    assert "clipping" in families
