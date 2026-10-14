"""
speechlens/text.py
Transcript normalisation (DESIGN §6 text rules), punctuation splitting,
and syllable counting.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from num2words import num2words
from unidecode import unidecode


# ── normalisation ─────────────────────────────────────────────────────────────

_DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_KEEP_RE = re.compile(r"[^A-Z\']")          # after upper-casing


def normalise_text(text: str) -> str:
    """
    Normalise *text* per DESIGN §6:
      1. Spell out numbers (num2words).
      2. Transliterate accents (unidecode).
      3. Uppercase.
      4. Keep only A-Z and apostrophe.
      5. Collapse whitespace.
    """
    # Spell out numbers first.
    text = _DIGIT_RE.sub(lambda m: num2words(float(m.group())), text)
    # Transliterate accented characters.
    text = unidecode(text)
    # Uppercase.
    text = text.upper()
    # Replace hyphens and dashes with spaces.
    text = re.sub(r"[-\u2013\u2014]", " ", text)
    # Remove everything except A-Z and apostrophe.
    text = _KEEP_RE.sub(" ", text)
    # Collapse whitespace.
    text = " ".join(text.split())
    return text


# ── punctuation splitting ──────────────────────────────────────────────────────

_SENT_END_RE = re.compile(r"[.?!;]")
_PHRASE_END_RE = re.compile(r"[,;:.\-?!]")


@dataclass
class WordToken:
    """One word from the display transcript."""
    i: int
    text: str           # display text
    text_norm: str      # normalised text (A-Z + apostrophe, uppercase)
    punct_after: str    # punctuation that follows this word, or ""
    nsyl: int = 0


def tokenise(display_text: str) -> list[WordToken]:
    """
    Split *display_text* into WordToken objects.
    Preserves punctuation in ``punct_after``.
    Digits are allowed in *display_text* but normalised away in *text_norm*.
    """
    # Scan character by character to collect words and their trailing punctuation.
    tokens: list[WordToken] = []
    pattern = re.compile(r"([A-Za-z0-9\'\-]+)([^A-Za-z0-9\'\-]*)", re.UNICODE)
    idx = 0
    for m in pattern.finditer(display_text):
        word = m.group(1)
        trailing = m.group(2)
        # Extract punctuation characters from trailing.
        punct = "".join(c for c in trailing if c in ".,;:?!-\u2013\u2014")
        norm = normalise_text(word)
        tok = WordToken(
            i=idx,
            text=word,
            text_norm=norm,
            punct_after=punct,
            nsyl=count_syllables(norm),
        )
        tokens.append(tok)
        idx += 1
    return tokens


# ── syllable counting ─────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def count_syllables(word_norm: str) -> int:
    """
    Count syllables in *word_norm* (uppercase, A-Z + apostrophe only).
    Priority: cmudict pronunciation -> vowel-group heuristic.
    """
    if not word_norm:
        return 0

    # Try cmudict first.
    try:
        import cmudict as _cmudict  # type: ignore
        d = _get_cmudict()
        prons = d.get(word_norm.lower(), [])
        if prons:
            # Count phones ending in a digit (stress marker).
            return sum(1 for ph in prons[0] if ph[-1].isdigit())
    except Exception:
        pass

    # Vowel-group heuristic.
    return _vowel_heuristic(word_norm)


@lru_cache(maxsize=1)
def _get_cmudict() -> dict:
    import cmudict as _cmudict  # type: ignore
    return _cmudict.dict()


_VOWEL_RE = re.compile(r"[AEIOUY]+", re.IGNORECASE)
_SILENT_E_RE = re.compile(r"[^AEIOUY]E$", re.IGNORECASE)


def _vowel_heuristic(word: str) -> int:
    """Rough syllable count via vowel groups."""
    n = len(_VOWEL_RE.findall(word))
    if _SILENT_E_RE.search(word) and n > 1:
        n -= 1
    return max(1, n)


# ── sentence / phrase boundaries ──────────────────────────────────────────────

def sentence_ends(tokens: list[WordToken]) -> list[int]:
    """Return indices of tokens that end a sentence (. ? ! ;)."""
    return [t.i for t in tokens if any(c in t.punct_after for c in ".?!;")]


def phrase_ends(tokens: list[WordToken]) -> list[int]:
    """Return indices of tokens that end a phrase (, ; : . - ? !)."""
    return [t.i for t in tokens if t.punct_after]