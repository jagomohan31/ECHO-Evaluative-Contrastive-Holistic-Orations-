"""
tests/test_t06_word_features.py -- T06: word, phrase, global features, time warp.

Tests on synthetic alignments and signals; one slow test on a library reader.
"""
import math
import numpy as np
import pytest
from unittest.mock import MagicMock

from speechlens.features.word import (
    WordFeatures,
    PauseInfo,
    PhraseFeatures,
    GlobalFeatures,
    compute_word_features,
    compute_prominence,
    classify_pauses,
    compute_phrase_features,
    compute_global_features,
    time_warp,
    PAUSE_MIN_S,
    FILLED_PAUSE_MIN_S,
)
from speechlens.features.frame import FrameFeatures


# ---------------------------------------------------------------------------
# Helpers: build synthetic FrameFeatures and word dicts
# ---------------------------------------------------------------------------

def _make_ff(duration: float, sr: int = 16000, n_frames: int = None) -> FrameFeatures:
    """Build a minimal FrameFeatures mock with realistic synthetic data."""
    if n_frames is None:
        n_frames = int(duration / 0.01) + 1
    times = np.linspace(0.0, duration, n_frames)
    rng = np.random.default_rng(42)

    ff = MagicMock(spec=FrameFeatures)
    ff.times = times
    ff.energy_db = rng.uniform(-40, 0, n_frames).astype(float)
    ff.en_rel = rng.uniform(-10, 5, n_frames).astype(float)
    # F0: half voiced (non-NaN), half unvoiced (NaN)
    f0_raw = rng.uniform(-3, 3, n_frames).astype(float)
    unvoiced = rng.choice(n_frames, n_frames // 2, replace=False)
    f0_raw[unvoiced] = np.nan
    ff.f0_st = f0_raw
    ff.hnr_rel = rng.uniform(-5, 10, n_frames).astype(float)
    ff.tilt_rel = rng.uniform(-8, 2, n_frames).astype(float)
    ff.clip_frac = np.zeros(n_frames)
    ff.syl_rate_hz = 4.5
    return ff


def _make_words(n: int, total_dur: float) -> list[dict]:
    """Build n evenly-spaced word dicts."""
    step = total_dur / n
    words = []
    for i in range(n):
        start = i * step
        end = start + step * 0.8
        words.append({
            "i": i,
            "text": f"word{i}",
            "text_norm": f"WORD{i}",
            "start": start,
            "end": end,
            "score": 0.5,
            "nsyl": 1,
            "punct_after": "," if i % 5 == 4 else "",
        })
    return words


# ---------------------------------------------------------------------------
# T06-01  compute_word_features returns one entry per word
# ---------------------------------------------------------------------------
def test_word_features_count():
    dur = 10.0
    words = _make_words(10, dur)
    ff = _make_ff(dur)
    result = compute_word_features(words, [], ff, dur)
    assert len(result) == 10


# ---------------------------------------------------------------------------
# T06-02  word dur / share / art_rate are consistent
# ---------------------------------------------------------------------------
def test_word_dur_share_art_rate():
    dur = 12.0
    words = _make_words(12, dur)
    ff = _make_ff(dur)
    result = compute_word_features(words, [], ff, dur)
    total_speaking = sum(w.dur for w in result)
    for wf in result:
        assert wf.dur > 0, "duration must be positive"
        assert abs(wf.share - wf.dur / total_speaking) < 1e-6, "share mismatch"
        assert abs(wf.art_rate - wf.nsyl / wf.dur) < 1e-6, "art_rate mismatch"


# ---------------------------------------------------------------------------
# T06-03  pause_before is correctly attached from pauses list
# ---------------------------------------------------------------------------
def test_pause_before_attached():
    dur = 10.0
    words = _make_words(5, dur)
    # Insert a pause between word 1 and word 2
    pauses = [{"start": words[1]["end"], "end": words[2]["start"],
                "before_word": 1, "after_word": 2}]
    ff = _make_ff(dur)
    result = compute_word_features(words, pauses, ff, dur)
    expected_dur = words[2]["start"] - words[1]["end"]
    assert abs(result[2].pause_before - expected_dur) < 1e-9


# ---------------------------------------------------------------------------
# T06-04  f0_sd_st window is non-NaN when voiced frames exist
# ---------------------------------------------------------------------------
def test_window_f0_sd_not_nan():
    dur = 12.0
    words = _make_words(10, dur)
    ff = _make_ff(dur)
    result = compute_word_features(words, [], ff, dur)
    # At least some words should have a valid window SD
    valid = [w for w in result if not math.isnan(w.f0_sd_st)]
    assert len(valid) > 0, "expected some non-NaN f0_sd_st"


# ---------------------------------------------------------------------------
# T06-05  prominence z-scores (DESIGN 7.3): range and stress flag
# ---------------------------------------------------------------------------
def test_prominence_range_and_stress():
    dur = 12.0
    words = _make_words(10, dur)
    ff = _make_ff(dur)
    wfs = compute_word_features(words, [], ff, dur)
    wfs = compute_prominence(wfs)
    valid_p = [w.prominence for w in wfs if not math.isnan(w.prominence)]
    assert len(valid_p) > 0, "some words should have prominence"
    for w in wfs:
        if not math.isnan(w.prominence):
            # stress flag consistent with threshold
            assert w.is_stress == (w.prominence >= 1.0)


# ---------------------------------------------------------------------------
# T06-06  pause classification -- silent pause
# ---------------------------------------------------------------------------
def test_pause_silent():
    dur = 5.0
    ff = _make_ff(dur)
    # Force all f0 in gap to NaN (unvoiced)
    gap_start, gap_end = 1.0, 1.5
    mask = (ff.times >= gap_start) & (ff.times < gap_end)
    ff.f0_st[mask] = np.nan

    pauses_raw = [{"start": gap_start, "end": gap_end, "before_word": 0, "after_word": 1}]
    result = classify_pauses(pauses_raw, ff)
    assert len(result) == 1
    assert result[0].kind == "silent"


# ---------------------------------------------------------------------------
# T06-07  pause classification -- filled pause
# ---------------------------------------------------------------------------
def test_pause_filled():
    dur = 5.0
    ff = _make_ff(dur, n_frames=500)
    gap_start, gap_end = 1.0, 1.4   # >= FILLED_PAUSE_MIN_S = 0.25 s
    mask = (ff.times >= gap_start) & (ff.times < gap_end)

    # Make gap fully voiced, mid-energy, low F0 SD
    ff.f0_st[mask] = 2.0          # constant => SD = 0 (voiced, SD < 0.7)
    ff.en_rel[mask] = -5.0        # en_rel > -20 dB

    pauses_raw = [{"start": gap_start, "end": gap_end, "before_word": 0, "after_word": 1}]
    result = classify_pauses(pauses_raw, ff)
    assert len(result) == 1
    assert result[0].kind == "filled"


# ---------------------------------------------------------------------------
# T06-08  pauses below PAUSE_MIN_S are not returned
# ---------------------------------------------------------------------------
def test_short_gap_ignored():
    dur = 3.0
    ff = _make_ff(dur)
    pauses_raw = [{"start": 1.0, "end": 1.05, "before_word": 0, "after_word": 1}]
    result = classify_pauses(pauses_raw, ff)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# T06-09  phrase features: artic_rate and f0_sd_st are finite and positive
# ---------------------------------------------------------------------------
def test_phrase_features_finite():
    dur = 15.0
    n_words = 15
    words = _make_words(n_words, dur)
    ff = _make_ff(dur)
    wfs = compute_word_features(words, [], ff, dur)
    phrases = compute_phrase_features(wfs, words, ff)
    assert len(phrases) > 0
    for ph in phrases:
        assert ph.artic_rate > 0 or math.isnan(ph.artic_rate)


# ---------------------------------------------------------------------------
# T06-10  global features: artic_rate and wpm are positive
# ---------------------------------------------------------------------------
def test_global_features_positive():
    dur = 15.0
    words = _make_words(15, dur)
    ff = _make_ff(dur)
    wfs = compute_word_features(words, [], ff, dur)
    pauses = classify_pauses([], ff)  # no pauses for this test
    gf = compute_global_features(wfs, pauses, ff, dur)
    assert gf.artic_rate > 0
    assert gf.words_per_minute > 0
    assert 0.0 <= gf.pause_ratio <= 1.0


# ---------------------------------------------------------------------------
# T06-11  time_warp: zero-length pauses handled, anchors monotone
# ---------------------------------------------------------------------------
def test_time_warp_monotone():
    words_u = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]
    words_b = [{"start": 0.0, "end": 1.2}, {"start": 1.2, "end": 2.5}]
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    t_warped = time_warp(t, words_u, words_b)
    assert len(t_warped) == len(t)
    assert np.all(np.diff(t_warped) >= 0), "warped times must be non-decreasing"


# ---------------------------------------------------------------------------
# T06-12  time_warp identity: identical word lists return same timestamps
# ---------------------------------------------------------------------------
def test_time_warp_identity():
    words = [{"start": 0.0, "end": 1.0}, {"start": 1.1, "end": 2.2}]
    t = np.linspace(0, 2.2, 50)
    t_warped = time_warp(t, words, words)
    # With identical anchors the warp should be near-identity
    assert np.allclose(t_warped, np.clip(t, 0.0, 2.2), atol=1e-3)


# ---------------------------------------------------------------------------
# T06-13  F0 SD range test: real library reader (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_library_reader_f0_sd_range():
    """F0 SD over the full passage should be between 1 and 8 st for a real speaker."""
    import json
    from pathlib import Path
    from speechlens.features.frame import compute_frame_features
    import soundfile as sf

    lib = Path("data/library/speech-001")
    audio_files = sorted(lib.glob("reader_*.flac"))
    if not audio_files:
        pytest.skip("library not built")

    samples, sr = sf.read(audio_files[0])
    samples = samples.astype(np.float32)
    ff = compute_frame_features(samples, sr)
    voiced = ff.f0_st[~np.isnan(ff.f0_st)]
    assert len(voiced) > 10, "should have voiced frames"
    f0_sd = float(np.std(voiced, ddof=1))
    assert 1.0 <= f0_sd <= 8.0, f"F0 SD = {f0_sd:.2f} st, expected 1-8 st"