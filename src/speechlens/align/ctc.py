"""
speechlens/align/ctc.py
CTC forced alignment with wav2vec2-base-960h (DESIGN §6).

Fast path (no model): trellis + backtrack over supplied log-probs.
Slow path: loads the HF model, encodes audio, runs the fast path.

Public surface:
    load_model()                  -> (model, processor)
    align(samples, transcript)    -> AlignResult
    align_file(audio_path, ...)   -> dict  (JSON-serialisable)
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from speechlens.config import SR
from speechlens.text import WordToken, normalise_text, tokenise

MODEL_ID = "facebook/wav2vec2-base-960h"
_CACHE_DIR = Path(".cache") / "align"

# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class WordAlign:
    i: int
    text: str          # display text
    text_norm: str
    punct_after: str
    start: float       # seconds
    end: float
    score: float       # mean posterior of token frames
    nsyl: int


@dataclass
class PauseAlign:
    start: float
    end: float
    before_word: int   # index of next word
    after_word: int    # index of prev word


@dataclass
class AlignResult:
    model: str
    stride_s: float
    duration_s: float
    words: list[WordAlign]
    pauses: list[PauseAlign]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "stride_s": round(self.stride_s, 6),
            "duration_s": round(self.duration_s, 4),
            "words": [
                {
                    "i": w.i,
                    "text": w.text,
                    "text_norm": w.text_norm,
                    "punct_after": w.punct_after,
                    "start": round(w.start, 4),
                    "end": round(w.end, 4),
                    "score": round(w.score, 4),
                    "nsyl": w.nsyl,
                }
                for w in self.words
            ],
            "pauses": [
                {
                    "start": round(p.start, 4),
                    "end": round(p.end, 4),
                    "before_word": p.before_word,
                    "after_word": p.after_word,
                }
                for p in self.pauses
            ],
        }


class AlignmentError(Exception):
    pass


# ── CTC trellis (numpy, no model dependency) ──────────────────────────────────

def _build_token_sequence(words: list[WordToken], processor) -> tuple[list[int], list[int]]:
    """
    Build the CTC token sequence for the transcript.
    Returns (tokens, word_boundaries) where word_boundaries[k] = word index for token k.
    """
    vocab = processor.tokenizer.get_vocab()
    blank_id = vocab.get("<pad>", 0)
    sep_id = vocab.get("|", None)

    tokens: list[int] = []
    word_of_token: list[int] = []

    for w in words:
        if not w.text_norm:
            continue
        word_tokens = []
        for ch in w.text_norm:
            tid = vocab.get(ch, None)
            if tid is None:
                continue
            word_tokens.append(tid)
        if not word_tokens:
            continue
        tokens.extend(word_tokens)
        word_of_token.extend([w.i] * len(word_tokens))
        if sep_id is not None:
            tokens.append(sep_id)
            word_of_token.append(w.i)

    return tokens, word_of_token


def ctc_trellis(log_probs: np.ndarray, tokens: list[int], blank_id: int = 0) -> np.ndarray:
    """
    Build a CTC trellis.
    log_probs: (T, V) log-softmax emissions.
    tokens: target token id sequence of length S.
    Returns trellis of shape (T, S) with log-probabilities.
    """
    T, V = log_probs.shape
    S = len(tokens)
    if T < S:
        raise AlignmentError(
            f"Audio too short ({T} frames) for transcript ({S} tokens). "
            "Check that the transcript matches the audio."
        )

    trellis = np.full((T, S), -np.inf, dtype=np.float64)
    # Initialise
    trellis[0, 0] = log_probs[0, blank_id] + log_probs[0, tokens[0]]
    if S > 1:
        trellis[0, 1] = log_probs[0, tokens[0]] + log_probs[0, tokens[1]]

    for t in range(1, T):
        for s in range(S):
            stay = trellis[t - 1, s]
            move = trellis[t - 1, s - 1] if s > 0 else -np.inf
            emit = log_probs[t, tokens[s]]
            trellis[t, s] = max(stay, move) + emit

    return trellis


def ctc_backtrack(
    trellis: np.ndarray, log_probs: np.ndarray, tokens: list[int], blank_id: int = 0
) -> list[tuple[int, int]]:
    """
    Backtrack through the trellis to find the best path.
    Returns a list of (frame, token_index) pairs (one per token).
    """
    T, S = trellis.shape
    path: list[tuple[int, int]] = []
    t = T - 1
    s = S - 1
    while s >= 0:
        path.append((t, s))
        if s == 0:
            break
        if t == 0:
            s -= 1
            continue
        if trellis[t - 1, s] >= trellis[t - 1, s - 1]:
            t -= 1
        else:
            t -= 1
            s -= 1
    path.reverse()
    # Guarantee every token index 0..S-1 appears at least once.
    seen = {si for _, si in path}
    if len(seen) < S:
        frames_by_si = {si: f for f, si in path}
        full_path: list[tuple[int, int]] = []
        for si in range(S):
            if si in frames_by_si:
                full_path.append((frames_by_si[si], si))
            else:
                prev_f = full_path[-1][0] if full_path else 0
                next_f = next((frames_by_si[sj] for sj in range(si + 1, S) if sj in frames_by_si), T - 1)
                full_path.append(((prev_f + next_f) // 2, si))
        path = full_path
    return path


def _path_to_words(
    path: list[tuple[int, int]],
    tokens: list[int],
    word_of_token: list[int],
    log_probs: np.ndarray,
    stride_s: float,
    display_words: list[WordToken],
) -> list[WordAlign]:
    """
    Group the CTC path into per-word alignments.
    """
    if not path:
        return []

    # Group frames by word index.
    from collections import defaultdict
    word_frames: dict[int, list[int]] = defaultdict(list)
    word_scores: dict[int, list[float]] = defaultdict(list)
    for frame, tok_idx in path:
        if tok_idx >= len(word_of_token):
            continue
        wi = word_of_token[tok_idx]
        word_frames[wi].append(frame)
        tok_id = tokens[tok_idx]
        word_scores[wi].append(float(np.exp(log_probs[frame, tok_id])))

    result: list[WordAlign] = []
    for dw in display_words:
        frames = word_frames.get(dw.i, [])
        scores = word_scores.get(dw.i, [])
        if not frames:
            # Fallback: estimate from position
            pos = dw.i / max(len(display_words), 1)
            T = log_probs.shape[0]
            f = int(pos * T)
            wa = WordAlign(
                i=dw.i, text=dw.text, text_norm=dw.text_norm,
                punct_after=dw.punct_after,
                start=round(f * stride_s, 4),
                end=round((f + 1) * stride_s, 4),
                score=0.0, nsyl=dw.nsyl,
            )
        else:
            start_f = min(frames)
            end_f = max(frames) + 1
            wa = WordAlign(
                i=dw.i, text=dw.text, text_norm=dw.text_norm,
                punct_after=dw.punct_after,
                start=round(start_f * stride_s, 4),
                end=round(end_f * stride_s, 4),
                score=round(float(np.mean(scores)), 4) if scores else 0.0,
                nsyl=dw.nsyl,
            )
        result.append(wa)
    return result


# ── energy-based boundary refinement (DESIGN §6) ──────────────────────────────

def _refine_boundaries(
    words: list[WordAlign],
    samples: np.ndarray,
    sr: int = SR,
    hop_s: float = 0.010,
    speech_threshold_db: float = 12.0,
) -> list[WordAlign]:
    """
    Extend word boundaries into adjacent gaps using frame energy (DESIGN §6).
    """
    hop = int(hop_s * sr)
    n_frames = max(1, len(samples) // hop)
    energy_db = np.array([
        20 * np.log10(np.sqrt(np.mean(samples[i * hop: (i + 1) * hop] ** 2)) + 1e-8)
        for i in range(n_frames)
    ])
    noise_floor = np.percentile(energy_db, 10)
    threshold = noise_floor + speech_threshold_db

    refined = list(words)
    for k in range(len(refined) - 1):
        w1 = refined[k]
        w2 = refined[k + 1]
        gap_start = w1.end
        gap_end = w2.start
        midpoint = (gap_start + gap_end) / 2.0

        # Extend w1 forward while energy above threshold.
        f = int(gap_start / hop_s)
        while f * hop_s < midpoint and f < n_frames and energy_db[f] > threshold:
            f += 1
        new_end = min(f * hop_s, midpoint)
        refined[k] = WordAlign(
            i=w1.i, text=w1.text, text_norm=w1.text_norm, punct_after=w1.punct_after,
            start=w1.start, end=round(new_end, 4), score=w1.score, nsyl=w1.nsyl,
        )

        # Extend w2 backward while energy above threshold.
        f = int(gap_end / hop_s)
        while f * hop_s > midpoint and f > 0 and energy_db[min(f, n_frames - 1)] > threshold:
            f -= 1
        new_start = max(f * hop_s, midpoint)
        refined[k + 1] = WordAlign(
            i=w2.i, text=w2.text, text_norm=w2.text_norm, punct_after=w2.punct_after,
            start=round(new_start, 4), end=w2.end, score=w2.score, nsyl=w2.nsyl,
        )

    return refined


# ── pause extraction ──────────────────────────────────────────────────────────

def _extract_pauses(words: list[WordAlign], min_pause_s: float) -> list[PauseAlign]:
    pauses: list[PauseAlign] = []
    for k in range(len(words) - 1):
        gap = words[k + 1].start - words[k].end
        if gap >= min_pause_s:
            pauses.append(PauseAlign(
                start=words[k].end,
                end=words[k + 1].start,
                before_word=words[k + 1].i,
                after_word=words[k].i,
            ))
    return pauses


# ── sanity checks ─────────────────────────────────────────────────────────────

def _sanity_check(words: list[WordAlign], duration_s: float) -> list[str]:
    warnings: list[str] = []
    for i, w in enumerate(words):
        if w.start >= w.end:
            warnings.append(f"Word {i} '{w.text}': zero or negative duration ({w.start}-{w.end})")
        if w.end > duration_s + 0.05:
            warnings.append(f"Word {i} '{w.text}': end {w.end} > audio duration {duration_s}")
        if w.end - w.start > 2.5:
            warnings.append(f"Word {i} '{w.text}': very long ({w.end - w.start:.2f} s)")

    if words:
        mean_score = np.mean([w.score for w in words])
        if mean_score < 0.35:
            warnings.append(
                f"Mean word alignment score {mean_score:.3f} < 0.35 — "
                "check that the transcript matches the audio (wrong passage?)"
            )
    return warnings


# ── model loading ─────────────────────────────────────────────────────────────

_model_cache: tuple | None = None


def load_model():
    """Load wav2vec2-base-960h. Cached in-process."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    torch.set_num_threads(1)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    _model_cache = (model, processor)
    return _model_cache


# ── cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(samples: np.ndarray, transcript_norm: str) -> str:
    h = hashlib.sha256(samples.tobytes())
    h.update(transcript_norm.encode())
    h.update(MODEL_ID.encode())
    return h.hexdigest()


def _cache_path(key: str, base: Path = _CACHE_DIR) -> Path:
    return base / f"{key}.json"


def _load_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(key: str, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(key)
    p.write_text(json.dumps(data), encoding="utf-8")


# ── public API ────────────────────────────────────────────────────────────────

def align(
    samples: np.ndarray,
    transcript: str,
    use_cache: bool = True,
) -> AlignResult:
    """
    Force-align *samples* (mono 16 kHz float32) against *transcript*.
    Loads the wav2vec2 model on first call.
    """
    import torch

    tokens_text = tokenise(transcript)
    transcript_norm = " ".join(t.text_norm for t in tokens_text if t.text_norm)

    # Cache lookup.
    cache_key = _cache_key(samples, transcript_norm)
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return _dict_to_result(cached)

    model, processor = load_model()

    # Get emissions.
    inputs = processor(samples, sampling_rate=SR, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits  # (1, T, V)
    log_probs = torch.log_softmax(logits[0], dim=-1).numpy()  # (T, V)

    stride_s = len(samples) / SR / log_probs.shape[0]

    # Build token sequence.
    vocab = processor.tokenizer.get_vocab()
    blank_id = vocab.get("<pad>", 0)
    token_ids, word_of_token = _build_token_sequence(tokens_text, processor)

    if not token_ids:
        raise AlignmentError("Transcript produced no tokens after normalisation.")

    # CTC trellis + backtrack.
    trellis = ctc_trellis(log_probs, token_ids, blank_id=blank_id)
    path = ctc_backtrack(trellis, log_probs, token_ids, blank_id=blank_id)

    # Path → words.
    words = _path_to_words(path, token_ids, word_of_token, log_probs, stride_s, tokens_text)

    # Energy refinement.
    words = _refine_boundaries(words, samples)

    # Pauses.
    from speechlens.config import PAUSE_MIN_S
    pauses = _extract_pauses(words, PAUSE_MIN_S)

    duration_s = len(samples) / SR
    result = AlignResult(
        model=MODEL_ID,
        stride_s=stride_s,
        duration_s=duration_s,
        words=words,
        pauses=pauses,
    )

    # Save cache.
    if use_cache:
        _save_cache(cache_key, result.to_dict())

    return result


def align_file(
    audio_path: str | Path,
    transcript: str | None = None,
    speech_id: str | None = None,
    library_root: Path | None = None,
) -> dict:
    """
    Align an audio file. If *transcript* is a path, read it.
    If *speech_id* is given, look up the library transcript.
    Returns the JSON-serialisable align result dict.
    """
    from speechlens.audio import load as load_audio

    samples = load_audio(audio_path)

    if transcript is None and speech_id is not None:
        lib = library_root or Path("data/library")
        txt_path = lib / speech_id / "transcript.txt"
        if txt_path.exists():
            transcript = txt_path.read_text(encoding="utf-8").strip()

    if transcript is None:
        raise ValueError("transcript or speech_id is required for alignment.")

    # If transcript is a file path, read it.
    tp = Path(transcript)
    if tp.exists() and tp.is_file():
        transcript = tp.read_text(encoding="utf-8").strip()

    result = align(samples, transcript)
    return result.to_dict()


def _dict_to_result(d: dict) -> AlignResult:
    words = [
        WordAlign(
            i=w["i"], text=w["text"], text_norm=w["text_norm"],
            punct_after=w["punct_after"], start=w["start"], end=w["end"],
            score=w["score"], nsyl=w["nsyl"],
        )
        for w in d["words"]
    ]
    pauses = [
        PauseAlign(start=p["start"], end=p["end"],
                   before_word=p["before_word"], after_word=p["after_word"])
        for p in d["pauses"]
    ]
    return AlignResult(
        model=d["model"],
        stride_s=d["stride_s"],
        duration_s=d["duration_s"],
        words=words,
        pauses=pauses,
    )