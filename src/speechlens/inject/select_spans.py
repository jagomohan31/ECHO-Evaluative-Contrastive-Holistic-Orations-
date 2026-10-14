"""
inject/select_spans.py -- Deterministic span selection (DESIGN 5.6).

select_spans(words, family, level) -> list of (word_start_idx, word_end_idx) pairs
representing the target spans for each flaw injection.

No randomness: positions are derived from text length fractions.
Noise-slice and filler seeds are derived from crc32(file_id).
"""
from __future__ import annotations
import re
import zlib
from typing import NamedTuple


# Families that select at phrase *boundaries* (gap insertion)
BOUNDARY_FAMILIES = {"pause_excess", "hesitation_filled", "terminal_contour"}

# Families that select *sentences* (span replacement)
SENTENCE_FAMILIES = {
    "pace_fast", "pace_slow", "pause_missing",
    "monotone", "stress_missed",
    "volume_low", "clipping", "trailing_off", "muffled",
}

_SENTENCE_END = re.compile(r"[.!?]")
_PHRASE_PUNCT = re.compile(r"[,;:.!?\u2014\-]")

# L1: 1 boundary at 40%; L2: 3 at 25/50/75%; L3: 6 evenly spaced
_BOUNDARY_FRACS = {1: [0.40], 2: [0.25, 0.50, 0.75], 3: [i/7 for i in range(1, 7)]}
# Number of spans/pauses per level
_LEVEL_COUNT = {1: 1, 2: 2, 3: 6}  # overridden per family below


class Span(NamedTuple):
    start_word: int   # inclusive
    end_word: int     # inclusive


def _sentence_boundaries(words: list[dict]) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each sentence in the word list."""
    sentences = []
    start = 0
    for i, w in enumerate(words):
        if _SENTENCE_END.search(w.get("punct_after", "")):
            sentences.append((start, i))
            start = i + 1
    if start < len(words):
        sentences.append((start, len(words) - 1))
    return sentences


def _phrase_boundaries(words: list[dict]) -> list[int]:
    """Return word indices immediately BEFORE each phrase boundary (gap position)."""
    boundaries = []
    for i, w in enumerate(words):
        if i < len(words) - 1 and _PHRASE_PUNCT.search(w.get("punct_after", "")):
            boundaries.append(i)
    return boundaries


def _sentence_at_frac(sentences: list[tuple[int, int]], frac: float) -> tuple[int, int]:
    """Pick the sentence whose mid-word is closest to frac * total_words."""
    if not sentences:
        return (0, 0)
    n = sentences[-1][1] + 1
    target = max(0, min(n - 1, int(frac * n)))
    best = min(sentences, key=lambda s: abs((s[0] + s[1]) // 2 - target))
    return best


def _boundary_at_frac(boundaries: list[int], frac: float, n_words: int) -> int:
    """Pick the boundary position closest to frac * n_words."""
    if not boundaries:
        return n_words // 2
    target = int(frac * n_words)
    return min(boundaries, key=lambda b: abs(b - target))


def select_spans(
    words: list[dict],
    family: str,
    level: int,  # 1, 2, or 3
    coverage_rule: bool = True,
) -> list[Span]:
    """
    Return a list of Span(start_word, end_word) pairs for the injection.

    For boundary families the span is the single word index BEFORE the
    boundary (the gap follows it), with start_word == end_word.

    For L3 sentence families using the coverage rule, contiguous sentences
    starting at 20% until ~40-50% of words are covered are used.
    """
    n = len(words)
    if n == 0:
        return []

    sentences = _sentence_boundaries(words)
    boundaries = _phrase_boundaries(words)

    # -------- Boundary families --------
    if family in BOUNDARY_FAMILIES:
        fracs = _BOUNDARY_FRACS.get(level, [0.4])
        spans = []
        for frac in fracs:
            pos = _boundary_at_frac(boundaries, frac, n)
            spans.append(Span(pos, pos))
        return spans

    # -------- Sentence families --------
    if level == 1:
        s = _sentence_at_frac(sentences, 0.40)
        return [Span(s[0], s[1])]

    if level == 2:
        s1 = _sentence_at_frac(sentences, 0.25)
        s2 = _sentence_at_frac(sentences, 0.65)
        seen: set[tuple[int, int]] = set()
        result = []
        for s in (s1, s2):
            if s not in seen:
                seen.add(s)
                result.append(Span(s[0], s[1]))
        return result

    # Level 3 -- coverage rule: contiguous sentences from ~20% until 40-50% covered
    target_lo = int(0.40 * n)
    target_hi = int(0.50 * n)
    start_frac = 0.20

    # Find the sentence closest to 20%
    start_sent_idx = 0
    target_start = int(start_frac * n)
    best_dist = abs(sentences[0][0] - target_start)
    for si, s in enumerate(sentences):
        d = abs(s[0] - target_start)
        if d < best_dist:
            best_dist = d
            start_sent_idx = si

    result = []
    covered = 0
    for si in range(start_sent_idx, len(sentences)):
        s = sentences[si]
        result.append(Span(s[0], s[1]))
        covered += s[1] - s[0] + 1
        if covered >= target_lo:
            break

    # If coverage overshoot is extreme (> 50%), trim the last sentence
    if coverage_rule and covered > target_hi and len(result) > 1:
        result.pop()

    return result if result else [Span(0, n - 1)]


def file_seed(file_id: str) -> int:
    """Deterministic seed for noise/filler generation, from crc32 of the file id."""
    return zlib.crc32(file_id.encode()) & 0xFFFFFFFF