"""
tests/test_t02_aligner.py
T02 fast tests: trellis, backtrack, word grouping, refinement, AlignmentError,
cache key stability. No model downloads.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── trellis ───────────────────────────────────────────────────────────────────

def test_ctc_trellis_shape():
    from speechlens.align.ctc import ctc_trellis
    T, V, S = 20, 10, 5
    log_probs = np.log(np.ones((T, V)) / V)
    tokens = [1, 2, 3, 4, 5]
    trellis = ctc_trellis(log_probs, tokens, blank_id=0)
    assert trellis.shape == (T, S)


def test_ctc_trellis_recovers_known_path():
    """
    Synthetic: push high probability on a specific token at each frame.
    Backtrack returns one (frame, token_index) per FRAME, not per token.
    We verify the path covers all token indices.
    """
    from speechlens.align.ctc import ctc_trellis, ctc_backtrack

    V = 5
    tokens = [1, 2, 3]
    T = 10
    log_probs = np.log(np.ones((T, V)) / V)
    for t, tok in [(2, 1), (5, 2), (8, 3)]:
        log_probs[t, tok] += 5.0  # strong signal

    trellis = ctc_trellis(log_probs, tokens, blank_id=0)
    path = ctc_backtrack(trellis, log_probs, tokens, blank_id=0)

    # Path has one entry per frame (up to T).
    assert len(path) >= len(tokens)
    # Frames are non-decreasing.
    frames = [f for f, _ in path]
    assert frames == sorted(frames)
    # All token indices appear somewhere in the path.
    seen_tokens = {s for _, s in path}
    assert seen_tokens == {0, 1, 2}  # token indices 0..S-1


def test_ctc_trellis_too_short():
    from speechlens.align.ctc import ctc_trellis, AlignmentError
    T, V = 3, 5
    log_probs = np.log(np.ones((T, V)) / V)
    tokens = [1, 2, 3, 4, 5]  # more tokens than frames
    with pytest.raises(AlignmentError):
        ctc_trellis(log_probs, tokens, blank_id=0)


# ── backtrack ─────────────────────────────────────────────────────────────────

def test_ctc_backtrack_covers_all_tokens():
    """
    Backtrack returns (frame, token_index) for each step along the path.
    The set of token_index values must include every token position.
    """
    from speechlens.align.ctc import ctc_trellis, ctc_backtrack
    T, V, S = 15, 6, 4
    log_probs = np.log(np.ones((T, V)) / V)
    tokens = [1, 2, 3, 4]
    trellis = ctc_trellis(log_probs, tokens, blank_id=0)
    path = ctc_backtrack(trellis, log_probs, tokens, blank_id=0)
    # Must cover all 4 token positions.
    seen = {s for _, s in path}
    assert seen == {0, 1, 2, 3}
    # Frames non-decreasing.
    frames = [f for f, _ in path]
    assert frames == sorted(frames)


# ── energy-based boundary refinement ─────────────────────────────────────────

def test_refinement_does_not_cross_midpoint():
    """Refinement must not push word end past the midpoint of a gap."""
    from speechlens.align.ctc import WordAlign, _refine_boundaries

    sr = 16000
    duration = 1.0
    samples = np.ones(int(duration * sr), dtype=np.float32) * 0.5
    w1 = WordAlign(i=0, text="hello", text_norm="HELLO", punct_after="",
                   start=0.0, end=0.3, score=0.9, nsyl=2)
    w2 = WordAlign(i=1, text="world", text_norm="WORLD", punct_after="",
                   start=0.5, end=0.8, score=0.9, nsyl=1)
    refined = _refine_boundaries([w1, w2], samples, sr=sr)
    midpoint = (0.3 + 0.5) / 2  # 0.4 s
    assert refined[0].end <= midpoint + 1e-3
    assert refined[1].start >= midpoint - 1e-3


def test_refinement_silent_gap():
    """In a silent gap, refinement should not extend into it."""
    from speechlens.align.ctc import WordAlign, _refine_boundaries

    sr = 16000
    samples = np.zeros(sr, dtype=np.float32)
    samples[:int(0.4 * sr)] = 0.5
    samples[int(0.8 * sr):] = 0.5

    w1 = WordAlign(i=0, text="hello", text_norm="HELLO", punct_after="",
                   start=0.0, end=0.35, score=0.9, nsyl=2)
    w2 = WordAlign(i=1, text="world", text_norm="WORLD", punct_after="",
                   start=0.85, end=1.0, score=0.9, nsyl=1)
    refined = _refine_boundaries([w1, w2], samples, sr=sr)
    assert refined[0].end < 0.65


# ── pause extraction ──────────────────────────────────────────────────────────

def test_pause_extraction():
    from speechlens.align.ctc import WordAlign, PauseAlign, _extract_pauses

    words = [
        WordAlign(i=0, text="a", text_norm="A", punct_after="", start=0.0, end=0.3, score=1.0, nsyl=1),
        WordAlign(i=1, text="b", text_norm="B", punct_after="", start=0.6, end=0.9, score=1.0, nsyl=1),
        WordAlign(i=2, text="c", text_norm="C", punct_after="", start=1.0, end=1.1, score=1.0, nsyl=1),
    ]
    pauses = _extract_pauses(words, min_pause_s=0.15)
    assert len(pauses) == 1
    assert pauses[0].start == pytest.approx(0.3, abs=1e-3)
    assert pauses[0].end == pytest.approx(0.6, abs=1e-3)
    assert pauses[0].before_word == 1
    assert pauses[0].after_word == 0


def test_no_pauses_when_short_gap():
    from speechlens.align.ctc import WordAlign, _extract_pauses

    words = [
        WordAlign(i=0, text="a", text_norm="A", punct_after="", start=0.0, end=0.3, score=1.0, nsyl=1),
        WordAlign(i=1, text="b", text_norm="B", punct_after="", start=0.35, end=0.6, score=1.0, nsyl=1),
    ]
    pauses = _extract_pauses(words, min_pause_s=0.15)
    assert len(pauses) == 0


# ── cache key stability ───────────────────────────────────────────────────────

def test_cache_key_stable():
    from speechlens.align.ctc import _cache_key
    samples = np.ones(100, dtype=np.float32)
    k1 = _cache_key(samples, "HELLO WORLD")
    k2 = _cache_key(samples, "HELLO WORLD")
    assert k1 == k2


def test_cache_key_different_transcript():
    from speechlens.align.ctc import _cache_key
    samples = np.ones(100, dtype=np.float32)
    k1 = _cache_key(samples, "HELLO WORLD")
    k2 = _cache_key(samples, "GOODBYE WORLD")
    assert k1 != k2


# ── sanity checks ─────────────────────────────────────────────────────────────

def test_sanity_check_warns_wrong_passage():
    from speechlens.align.ctc import WordAlign, _sanity_check

    words = [
        WordAlign(i=0, text="hello", text_norm="HELLO", punct_after="",
                  start=0.0, end=0.5, score=0.2, nsyl=2),
        WordAlign(i=1, text="world", text_norm="WORLD", punct_after="",
                  start=0.5, end=1.0, score=0.25, nsyl=1),
    ]
    warnings = _sanity_check(words, duration_s=1.5)
    assert len(warnings) > 0  # should warn about low alignment score


def test_sanity_check_clean():
    from speechlens.align.ctc import WordAlign, _sanity_check

    words = [
        WordAlign(i=0, text="hello", text_norm="HELLO", punct_after="",
                  start=0.0, end=0.5, score=0.9, nsyl=2),
        WordAlign(i=1, text="world", text_norm="WORLD", punct_after="",
                  start=0.5, end=1.0, score=0.85, nsyl=1),
    ]
    warnings = _sanity_check(words, duration_s=1.5)
    assert len(warnings) == 0