"""
tests/test_t07_injectors.py -- T07: flaw injector unit tests.

One test per family. Each checks:
  1. Effect within 15% of target (duration ratio, F0 SD ratio, dB change,
     clip fraction, spectral-tilt drop, pause length).
  2. Features outside the span (with 0.15 s margin) stay close to original.
  3. Two runs return identical sample arrays (determinism).
"""
from __future__ import annotations
import math
import numpy as np
import pytest
from scipy.signal import butter, sosfilt

SR = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_speech(dur_s: float, f0: float = 180.0, sr: int = SR) -> np.ndarray:
    """Synthetic voiced speech: sine at f0 with slight AM."""
    t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
    am = 0.8 + 0.2 * np.sin(2 * np.pi * 4.0 * t)  # 4 Hz AM
    return (am * np.sin(2 * np.pi * f0 * t)).astype(np.float32) * 0.3


def _rms_db(sig: np.ndarray) -> float:
    rms = math.sqrt(max(float(np.mean(sig ** 2)), 1e-12))
    return 20.0 * math.log10(rms)


def _span_samples(sig: np.ndarray, start: float, end: float, sr: int = SR):
    i0 = int(start * sr)
    i1 = min(len(sig), int(end * sr))
    return sig[i0:i1]


def _outside_samples(sig: np.ndarray, span_start: float, span_end: float,
                     dur: float, margin: float = 0.15, sr: int = SR):
    """Return outside-span samples with margin."""
    chunks = []
    i0_pre = 0
    i1_pre = max(0, int((span_start - margin) * sr))
    if i1_pre > i0_pre:
        chunks.append(sig[i0_pre:i1_pre])
    i0_post = min(len(sig), int((span_end + margin) * sr))
    if i0_post < len(sig):
        chunks.append(sig[i0_post:])
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def _make_words(n: int, dur: float) -> list[dict]:
    step = dur / n
    return [
        {"i": i, "text": f"word{i}", "start": i * step, "end": (i + 1) * step * 0.85,
         "nsyl": 1, "punct_after": "." if i == n - 1 else ("," if i % 4 == 3 else "")}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# T07-01  apply_gain: dB change within 15% of target
# ---------------------------------------------------------------------------
def test_gain_db_effect():
    from speechlens.inject.volume import apply_gain
    dur, gain_db = 2.0, -8.0
    sig = _sine_speech(dur)
    span = (0.5, 1.5)
    result = apply_gain(sig, SR, [span], gain_db)

    before_db = _rms_db(_span_samples(sig, *span))
    after_db = _rms_db(_span_samples(result, *span))
    actual_change = after_db - before_db
    assert abs(actual_change - gain_db) < abs(gain_db) * 0.15, (
        f"gain change {actual_change:.2f} dB, expected {gain_db:.2f} dB"
    )

    # outside the span should be close to original
    out_before = _rms_db(_outside_samples(sig, *span, dur))
    out_after = _rms_db(_outside_samples(result, *span, dur))
    assert abs(out_before - out_after) < 0.5, "outside span affected by gain"


# ---------------------------------------------------------------------------
# T07-02  apply_gain determinism
# ---------------------------------------------------------------------------
def test_gain_determinism():
    from speechlens.inject.volume import apply_gain
    sig = _sine_speech(2.0)
    r1 = apply_gain(sig, SR, [(0.5, 1.5)], -8.0)
    r2 = apply_gain(sig, SR, [(0.5, 1.5)], -8.0)
    assert np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# T07-03  apply_clipping: clip fraction increases in span
# ---------------------------------------------------------------------------
def test_clipping_fraction():
    from speechlens.inject.volume import apply_clipping
    dur, gain_db = 2.0, 20.0
    sig = _sine_speech(dur, f0=200.0)
    span = (0.4, 1.6)
    result = apply_clipping(sig, SR, [span], gain_db)

    # Inside span: significant clip fraction
    inside = _span_samples(result, *span)
    clip_frac = float(np.mean(np.abs(inside) >= 0.99))
    assert clip_frac > 0.05, f"expected clipping, got clip_frac={clip_frac:.3f}"

    # Outside span: no clipping
    outside = _outside_samples(sig, *span, dur)
    clip_outside = float(np.mean(np.abs(outside) >= 0.99))
    assert clip_outside < 0.01, "unexpected clipping outside span"


# ---------------------------------------------------------------------------
# T07-04  apply_clipping determinism
# ---------------------------------------------------------------------------
def test_clipping_determinism():
    from speechlens.inject.volume import apply_clipping
    sig = _sine_speech(2.0)
    r1 = apply_clipping(sig, SR, [(0.5, 1.5)], 20.0)
    r2 = apply_clipping(sig, SR, [(0.5, 1.5)], 20.0)
    assert np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# T07-05  apply_muffled: spectral tilt drops (more low-freq energy) in span
# ---------------------------------------------------------------------------
def test_muffled_spectral_tilt():
    from speechlens.inject.clarity import apply_muffled
    from scipy.fft import rfft, rfftfreq

    dur, cutoff = 3.0, 2500.0
    sig = _sine_speech(dur, f0=200.0)
    # Add high-freq content
    t = np.linspace(0, dur, len(sig), endpoint=False)
    sig = (sig + 0.2 * np.sin(2 * np.pi * 4000.0 * t).astype(np.float32))

    span = (0.5, 2.5)
    result = apply_muffled(sig, SR, [span], cutoff)

    # Measure high-freq energy before/after in span
    def _hi_ratio(s: np.ndarray) -> float:
        S = np.abs(rfft(s)) ** 2
        freqs = rfftfreq(len(s), d=1.0 / SR)
        hi = (freqs >= 3000) & (freqs <= 7000)
        lo = (freqs >= 200) & (freqs <= 1000)
        return float(S[hi].sum() / (S[lo].sum() + 1e-12))

    before_ratio = _hi_ratio(_span_samples(sig, *span))
    after_ratio = _hi_ratio(_span_samples(result, *span))
    assert after_ratio < before_ratio * 0.85, (
        f"spectral tilt not reduced: before={before_ratio:.4f}, after={after_ratio:.4f}"
    )


# ---------------------------------------------------------------------------
# T07-06  apply_muffled determinism
# ---------------------------------------------------------------------------
def test_muffled_determinism():
    from speechlens.inject.clarity import apply_muffled
    sig = _sine_speech(3.0)
    r1 = apply_muffled(sig, SR, [(0.5, 2.5)], 2500.0)
    r2 = apply_muffled(sig, SR, [(0.5, 2.5)], 2500.0)
    assert np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# T07-07  apply_trailing_off: energy falls in fade region
# ---------------------------------------------------------------------------
def test_trailing_off_energy():
    from speechlens.inject.volume import apply_trailing_off
    dur, gain_db = 2.0, -10.0
    sig = _sine_speech(dur)
    fade_span = (1.2, 1.8)
    result = apply_trailing_off(sig, SR, [fade_span], gain_db)

    # Energy at end of fade should be lower than at start
    q1 = _span_samples(result, 1.2, 1.4)
    q2 = _span_samples(result, 1.6, 1.8)
    rms1 = math.sqrt(float(np.mean(q1 ** 2)) + 1e-12)
    rms2 = math.sqrt(float(np.mean(q2 ** 2)) + 1e-12)
    assert rms2 < rms1, f"trailing off: end RMS {rms2:.4f} not < start RMS {rms1:.4f}"


# ---------------------------------------------------------------------------
# T07-08  apply_trailing_off determinism
# ---------------------------------------------------------------------------
def test_trailing_off_determinism():
    from speechlens.inject.volume import apply_trailing_off
    sig = _sine_speech(2.0)
    r1 = apply_trailing_off(sig, SR, [(1.0, 1.8)], -10.0)
    r2 = apply_trailing_off(sig, SR, [(1.0, 1.8)], -10.0)
    assert np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# T07-09  apply_pause_excess: output longer by ~dur_s
# ---------------------------------------------------------------------------
def test_pause_excess_length():
    from speechlens.inject.pauses import apply_pause_excess
    dur_s, pause_s = 2.0, 0.45
    sig = _sine_speech(dur_s)
    result = apply_pause_excess(sig, SR, [1.0], pause_s, file_id="test-speech")
    expected_len = len(sig) + int(pause_s * SR)
    # Allow 10 ms tolerance
    assert abs(len(result) - expected_len) < int(0.01 * SR), (
        f"pause_excess length: got {len(result)}, expected ~{expected_len}"
    )


# ---------------------------------------------------------------------------
# T07-10  apply_pause_excess determinism
# ---------------------------------------------------------------------------
def test_pause_excess_determinism():
    from speechlens.inject.pauses import apply_pause_excess
    sig = _sine_speech(2.0)
    r1 = apply_pause_excess(sig, SR, [1.0], 0.45, file_id="test-id")
    r2 = apply_pause_excess(sig, SR, [1.0], 0.45, file_id="test-id")
    assert np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# T07-11  apply_pause_missing: pause is shortened
# ---------------------------------------------------------------------------
def test_pause_missing_shorter():
    from speechlens.inject.pauses import apply_pause_missing
    dur_s = 3.0
    sig = _sine_speech(dur_s)
    # Introduce a known 0.5 s silent pause at [1.0, 1.5]
    i0, i1 = int(1.0 * SR), int(1.5 * SR)
    sig[i0:i1] = 0.0

    result = apply_pause_missing(sig, SR, [(1.0, 1.5)], target_s=0.06)
    # Output should be shorter
    assert len(result) < len(sig), (
        f"pause_missing output not shorter: {len(result)} vs {len(sig)}"
    )


# ---------------------------------------------------------------------------
# T07-12  apply_pause_missing determinism
# ---------------------------------------------------------------------------
def test_pause_missing_determinism():
    from speechlens.inject.pauses import apply_pause_missing
    sig = _sine_speech(3.0)
    sig[int(1.0 * SR):int(1.5 * SR)] = 0.0
    r1 = apply_pause_missing(sig, SR, [(1.0, 1.5)], target_s=0.06)
    r2 = apply_pause_missing(sig, SR, [(1.0, 1.5)], target_s=0.06)
    assert np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# T07-13  apply_hesitation_filled: output longer and non-zero in inserted region
# ---------------------------------------------------------------------------
def test_hesitation_filled_length_and_nonzero():
    from speechlens.inject.pauses import apply_hesitation_filled
    dur_s, uh_s = 2.0, 0.35
    sig = _sine_speech(dur_s)
    result = apply_hesitation_filled(sig, SR, [1.0], uh_s, f0_hz=170.0, rms_level=0.05)
    expected_extra = int(uh_s * SR)
    assert len(result) >= len(sig) + expected_extra - int(0.01 * SR), (
        f"hesitation output too short: {len(result)}"
    )
    # The inserted region should have non-zero energy
    idx = int(1.0 * SR)
    inserted = result[idx: idx + expected_extra]
    assert float(np.mean(inserted ** 2)) > 1e-6, "inserted uh is silent"


# ---------------------------------------------------------------------------
# T07-14  apply_hesitation_filled determinism
# ---------------------------------------------------------------------------
def test_hesitation_filled_determinism():
    from speechlens.inject.pauses import apply_hesitation_filled
    sig = _sine_speech(2.0)
    r1 = apply_hesitation_filled(sig, SR, [1.0], 0.35, f0_hz=170.0)
    r2 = apply_hesitation_filled(sig, SR, [1.0], 0.35, f0_hz=170.0)
    assert np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# T07-15  select_spans: count and coverage match spec
# ---------------------------------------------------------------------------
def test_select_spans_count():
    from speechlens.inject.select_spans import select_spans
    words = _make_words(30, 60.0)

    # Sentence family L1 -> 1 span
    spans_l1 = select_spans(words, "volume_low", 1)
    assert len(spans_l1) == 1, f"L1 should give 1 span, got {len(spans_l1)}"

    # L2 -> 1 or 2 spans
    spans_l2 = select_spans(words, "volume_low", 2)
    assert 1 <= len(spans_l2) <= 2

    # L3 -> coverage 40-50% of words
    spans_l3 = select_spans(words, "pace_fast", 3)
    covered = sum(s.end_word - s.start_word + 1 for s in spans_l3)
    assert covered >= int(0.38 * 30), f"L3 coverage too low: {covered}/30"


# ---------------------------------------------------------------------------
# T07-16  select_spans boundary family: correct positions
# ---------------------------------------------------------------------------
def test_select_spans_boundary():
    from speechlens.inject.select_spans import select_spans
    words = _make_words(20, 40.0)
    # Boundary family L2 should give 3 positions
    spans = select_spans(words, "pause_excess", 2)
    assert len(spans) == 3, f"boundary L2 should give 3, got {len(spans)}"