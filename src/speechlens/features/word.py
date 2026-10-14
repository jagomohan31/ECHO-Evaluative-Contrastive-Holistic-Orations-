"""
features/word.py -- Word, phrase, global features and time-warp overlay.

Implements DESIGN sections 7.2-7.5:
  7.2  Per-word features, 6-word window f0_sd_st, phrase features, global features.
  7.3  Prominence and stress points.
  7.4  Pause classification (silent / filled / other).
  7.5  Monotone time-warp for overlay rendering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .frame import FrameFeatures

# ---------------------------------------------------------------------------
# Constants (mirrors config defaults; override via config dict if needed)
# ---------------------------------------------------------------------------
PAUSE_MIN_S: float = 0.10          # minimum gap to count as a pause
FILLED_PAUSE_MIN_S: float = 0.25   # minimum gap to be considered filled
EXPECTED_PAUSE_S: float = 0.25     # mu threshold that marks a pause "expected"
WINDOW_SIZE: int = 6               # words in the f0_sd_st window


# ---------------------------------------------------------------------------
# Per-word feature dataclass
# ---------------------------------------------------------------------------
@dataclass
class WordFeatures:
    """Features derived for a single aligned word."""
    i: int
    text: str
    start: float
    end: float
    nsyl: int

    dur: float = 0.0
    art_rate: float = 0.0        # nsyl / dur
    share: float = 0.0           # dur / total speaking time

    pause_before: float = 0.0    # gap in seconds (0 if none)
    pause_after: float = 0.0

    f0_mean_st: float = float("nan")
    f0_max_st: float = float("nan")
    f0_range_st: float = float("nan")   # P90 - P10 of voiced frames
    f0_slope: float = float("nan")      # st/s, linear fit (needs >= 3 voiced frames)
    en_mean_rel: float = float("nan")
    en_max_rel: float = float("nan")
    en_slope: float = float("nan")      # dB/s
    hnr_rel: float = float("nan")       # median
    tilt_rel: float = float("nan")      # median
    voiced_ratio: float = 0.0
    align_score: float = 0.0
    clip_frac: float = 0.0

    # window features (filled later by compute_word_features)
    f0_sd_st: float = float("nan")      # SD over 6-word window
    voiced_share_win: float = float("nan")

    # prominence (filled later)
    prominence: float = float("nan")
    is_stress: bool = False
    punct_after: str = ""


@dataclass
class PauseInfo:
    """A detected pause between consecutive words."""
    start: float
    end: float
    before_word: int
    after_word: int
    kind: str = "other"    # "silent" | "filled" | "other"
    is_expected: bool = False


@dataclass
class PhraseFeatures:
    """Features over a phrase (split at ,;:.!? and dashes)."""
    words: list[int] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    artic_rate: float = float("nan")    # syl / speaking time
    speech_rate: float = float("nan")   # syl / (speaking + pauses)
    f0_sd_st: float = float("nan")
    f0_range_st: float = float("nan")
    energy_range: float = float("nan")


@dataclass
class GlobalFeatures:
    """Passage-level aggregates."""
    artic_rate: float = float("nan")    # syl / speaking time
    words_per_minute: float = float("nan")
    pause_ratio: float = float("nan")   # total pause / total duration
    mean_pause: float = float("nan")
    f0_sd_st: float = float("nan")
    syl_rate_hz: float = float("nan")   # from frame-level modulation spectrum


# ---------------------------------------------------------------------------
# Sentence boundary detection
# ---------------------------------------------------------------------------
_PHRASE_PUNCT = re.compile(r"[,;:.!?\u2014-]")


def _is_sentence_end(punct: str) -> bool:
    return bool(re.search(r"[.!?]", punct))


def _split_phrases(
    words: list[dict],
) -> list[list[int]]:
    """Return lists of word indices that form each phrase, split at phrase punctuation."""
    phrases: list[list[int]] = [[]]
    for w in words:
        phrases[-1].append(w["i"])
        if _PHRASE_PUNCT.search(w.get("punct_after", "")):
            phrases.append([])
    # Drop trailing empty phrase
    if phrases and not phrases[-1]:
        phrases.pop()
    return phrases


# ---------------------------------------------------------------------------
# Per-word feature extraction
# ---------------------------------------------------------------------------
def _slice_frames(
    times: np.ndarray, start: float, end: float
) -> np.ndarray:
    """Return boolean mask for frames inside [start, end)."""
    return (times >= start) & (times < end)


def compute_word_features(
    words: list[dict],
    pauses: list[dict],
    ff: FrameFeatures,
    total_dur: float,
) -> list[WordFeatures]:
    """
    Compute per-word features for all aligned words.

    Parameters
    ----------
    words:   list of dicts from *.align.json  (keys: i, text, start, end, score, nsyl, punct_after)
    pauses:  list of dicts from *.align.json  (keys: start, end, before_word, after_word)
    ff:      FrameFeatures for the recording
    total_dur: audio duration in seconds
    """
    if not words:
        return []

    total_speaking = sum(w["end"] - w["start"] for w in words)
    if total_speaking <= 0:
        total_speaking = total_dur

    # Build pause lookup: word_i -> pause_duration
    pause_before: dict[int, float] = {}
    pause_after: dict[int, float] = {}
    for p in pauses:
        dur = p["end"] - p["start"]
        if p.get("after_word") is not None:
            pause_before[p["after_word"]] = dur
        if p.get("before_word") is not None:
            pause_after[p["before_word"]] = dur

    times = ff.times  # 1-D array of frame centre times

    result: list[WordFeatures] = []
    for w in words:
        wf = WordFeatures(
            i=w["i"],
            text=w["text"],
            start=w["start"],
            end=w["end"],
            nsyl=max(1, w.get("nsyl", 1)),
            dur=w["end"] - w["start"],
            align_score=w.get("score", 0.0),
            punct_after=w.get("punct_after", ""),
        )
        if wf.dur > 0:
            wf.art_rate = wf.nsyl / wf.dur
        wf.share = wf.dur / total_speaking
        wf.pause_before = pause_before.get(w["i"], 0.0)
        wf.pause_after = pause_after.get(w["i"], 0.0)

        mask = _slice_frames(times, wf.start, wf.end)
        if not np.any(mask):
            result.append(wf)
            continue

        # F0
        f0 = ff.f0_st[mask] if ff.f0_st is not None else np.array([])
        voiced = f0[~np.isnan(f0)]
        wf.voiced_ratio = len(voiced) / np.sum(mask) if np.sum(mask) > 0 else 0.0

        if len(voiced) > 0:
            wf.f0_mean_st = float(np.mean(voiced))
            wf.f0_max_st = float(np.max(voiced))
            p10, p90 = np.nanpercentile(voiced, [10, 90])
            wf.f0_range_st = float(p90 - p10)

        if len(voiced) >= 3:
            frame_times = times[mask][~np.isnan(f0)]
            slope, _ = np.polyfit(frame_times, voiced, 1)
            wf.f0_slope = float(slope)

        # Energy
        en = ff.energy_db[mask] if ff.energy_db is not None else np.array([])
        en_rel_arr = ff.en_rel[mask] if ff.en_rel is not None else np.array([])
        if len(en_rel_arr) > 0:
            wf.en_mean_rel = float(np.nanmean(en_rel_arr))
            wf.en_max_rel = float(np.nanmax(en_rel_arr))
            if wf.dur > 0 and len(en) >= 2:
                slope_e, _ = np.polyfit(times[mask], en, 1)
                wf.en_slope = float(slope_e)

        # HNR
        if ff.hnr_rel is not None:
            hnr = ff.hnr_rel[mask]
            valid_hnr = hnr[~np.isnan(hnr)]
            if len(valid_hnr) > 0:
                wf.hnr_rel = float(np.median(valid_hnr))

        # Spectral tilt
        if ff.tilt_rel is not None:
            tilt = ff.tilt_rel[mask]
            valid_tilt = tilt[~np.isnan(tilt)]
            if len(valid_tilt) > 0:
                wf.tilt_rel = float(np.median(valid_tilt))

        # Clip fraction
        if ff.clip_frac is not None:
            wf.clip_frac = float(np.nanmean(ff.clip_frac[mask]))

        result.append(wf)

    # ----- Window features (f0_sd_st over 6-word sliding windows) -----
    half = WINDOW_SIZE // 2
    for idx, wf in enumerate(result):
        lo = max(0, idx - half)
        hi = min(len(result), idx + half + 1)
        window = result[lo:hi]
        # Collect all voiced f0 frames for the window span
        w_start = window[0].start
        w_end = window[-1].end
        mask_w = _slice_frames(times, w_start, w_end)
        f0_w = ff.f0_st[mask_w] if ff.f0_st is not None else np.array([])
        voiced_w = f0_w[~np.isnan(f0_w)]
        if len(voiced_w) >= 2:
            wf.f0_sd_st = float(np.std(voiced_w, ddof=1))
        total_frames = np.sum(mask_w)
        wf.voiced_share_win = len(voiced_w) / total_frames if total_frames > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Prominence (DESIGN 7.3)
# ---------------------------------------------------------------------------
def compute_prominence(word_features: list[WordFeatures]) -> list[WordFeatures]:
    """
    P_i = mean of up to 3 within-file z-scores:
          f0_max_st, en_max_rel, ln(dur / nsyl).
    Requires at least 2 of the 3 to be non-NaN.
    A stress point is P_i >= 1.0.
    """
    n = len(word_features)
    if n == 0:
        return word_features

    f0_vals = np.array([w.f0_max_st for w in word_features])
    en_vals = np.array([w.en_max_rel for w in word_features])
    ln_dur_vals = np.array([np.log(w.dur / w.nsyl) if w.dur > 0 else np.nan for w in word_features])

    def _zscore(arr: np.ndarray) -> np.ndarray:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return np.full_like(arr, np.nan)
        mu, sd = np.nanmean(arr), np.nanstd(arr, ddof=1)
        if sd < 1e-9:
            return np.zeros_like(arr)
        return (arr - mu) / sd

    z_f0 = _zscore(f0_vals)
    z_en = _zscore(en_vals)
    z_dur = _zscore(ln_dur_vals)

    for i, wf in enumerate(word_features):
        components = []
        for z in (z_f0[i], z_en[i], z_dur[i]):
            if not np.isnan(z):
                components.append(z)
        if len(components) >= 2:
            wf.prominence = float(np.mean(components))
            wf.is_stress = wf.prominence >= 1.0
        else:
            wf.prominence = float("nan")
            wf.is_stress = False

    return word_features


# ---------------------------------------------------------------------------
# Pause classification (DESIGN 7.4)
# ---------------------------------------------------------------------------
def classify_pauses(
    pauses_raw: list[dict],
    ff: FrameFeatures,
    baseline_pause_mu: Optional[dict[int, float]] = None,
) -> list[PauseInfo]:
    """
    Classify each pause as silent, filled, or other.
    baseline_pause_mu: optional {after_word_idx: expected_pause_s}.
    """
    result: list[PauseInfo] = []
    times = ff.times

    for p in pauses_raw:
        dur = p["end"] - p["start"]
        if dur < PAUSE_MIN_S:
            continue  # not a real pause

        after_word = p.get("after_word", -1)
        pi = PauseInfo(
            start=p["start"],
            end=p["end"],
            before_word=p.get("before_word", -1),
            after_word=after_word,
        )

        # Characterise the gap frames
        mask = _slice_frames(times, p["start"], p["end"])
        n_frames = int(np.sum(mask))

        if n_frames > 0 and ff.f0_st is not None:
            f0_gap = ff.f0_st[mask]
            voiced = f0_gap[~np.isnan(f0_gap)]
            voiced_frac = len(voiced) / n_frames

            en_gap = ff.en_rel[mask] if ff.en_rel is not None else np.array([])
            mean_en = float(np.nanmean(en_gap)) if len(en_gap) > 0 else -99.0

            f0_sd = float(np.nanstd(voiced, ddof=1)) if len(voiced) >= 2 else 0.0
        else:
            voiced_frac = 0.0
            mean_en = -99.0
            f0_sd = 0.0

        # Classification rules (DESIGN 7.4)
        if (
            dur >= FILLED_PAUSE_MIN_S
            and voiced_frac >= 0.6
            and mean_en > -20.0
            and f0_sd < 0.7
        ):
            pi.kind = "filled"
        elif voiced_frac < 0.2:
            pi.kind = "silent"
        else:
            pi.kind = "other"

        # Expected?
        if baseline_pause_mu and after_word in baseline_pause_mu:
            pi.is_expected = baseline_pause_mu[after_word] >= EXPECTED_PAUSE_S

        result.append(pi)

    return result


# ---------------------------------------------------------------------------
# Phrase features (DESIGN 7.2)
# ---------------------------------------------------------------------------
def compute_phrase_features(
    words: list[WordFeatures],
    align_words: list[dict],
    ff: FrameFeatures,
) -> list[PhraseFeatures]:
    """Compute phrase-level features split at ,;:.!? and dashes."""
    phrase_groups = _split_phrases(align_words)
    times = ff.times
    result = []

    for group in phrase_groups:
        if not group:
            continue
        pf = PhraseFeatures(words=group)
        wfs = [w for w in words if w.i in set(group)]
        if not wfs:
            continue

        pf.start = wfs[0].start
        pf.end = wfs[-1].end
        total_dur_phrase = pf.end - pf.start
        if total_dur_phrase <= 0:
            continue

        total_syl = sum(w.nsyl for w in wfs)
        speaking_time = sum(w.dur for w in wfs)

        if speaking_time > 0:
            pf.artic_rate = total_syl / speaking_time
        if total_dur_phrase > 0:
            pf.speech_rate = total_syl / total_dur_phrase

        mask = _slice_frames(times, pf.start, pf.end)
        f0_arr = ff.f0_st[mask] if ff.f0_st is not None else np.array([])
        voiced = f0_arr[~np.isnan(f0_arr)]
        if len(voiced) >= 2:
            pf.f0_sd_st = float(np.std(voiced, ddof=1))
            pf.f0_range_st = float(np.nanpercentile(voiced, 90) - np.nanpercentile(voiced, 10))

        en_arr = ff.en_rel[mask] if ff.en_rel is not None else np.array([])
        valid_en = en_arr[~np.isnan(en_arr)]
        if len(valid_en) >= 2:
            pf.energy_range = float(np.nanmax(valid_en) - np.nanmin(valid_en))

        result.append(pf)

    return result


# ---------------------------------------------------------------------------
# Global features (DESIGN 7.2)
# ---------------------------------------------------------------------------
def compute_global_features(
    words: list[WordFeatures],
    pauses: list[PauseInfo],
    ff: FrameFeatures,
    total_dur: float,
) -> GlobalFeatures:
    """Passage-level aggregate features."""
    gf = GlobalFeatures()
    if not words:
        return gf

    total_syl = sum(w.nsyl for w in words)
    speaking_time = sum(w.dur for w in words)

    if speaking_time > 0:
        gf.artic_rate = total_syl / speaking_time
    if total_dur > 0:
        gf.words_per_minute = len(words) / total_dur * 60.0

    real_pauses = [p for p in pauses if p.kind in ("silent", "filled")]
    total_pause = sum(p.end - p.start for p in real_pauses)
    if total_dur > 0:
        gf.pause_ratio = total_pause / total_dur
    if real_pauses:
        gf.mean_pause = total_pause / len(real_pauses)

    # F0 SD over the whole passage
    if ff.f0_st is not None:
        voiced = ff.f0_st[~np.isnan(ff.f0_st)]
        if len(voiced) >= 2:
            gf.f0_sd_st = float(np.std(voiced, ddof=1))

    gf.syl_rate_hz = ff.syl_rate_hz if hasattr(ff, "syl_rate_hz") else float("nan")

    return gf


# ---------------------------------------------------------------------------
# Time warp for overlay (DESIGN 7.5)
# ---------------------------------------------------------------------------
def time_warp(
    t_user: np.ndarray,
    words_user: list[dict],
    words_base: list[dict],
) -> np.ndarray:
    """
    Map user timestamps to baseline timestamps using word-boundary anchors.

    Anchors = interleaved (word_start, word_end) pairs for both recordings.
    Adds 1e-4 * anchor_index to keep the user anchor array strictly increasing.

    Parameters
    ----------
    t_user      : 1-D array of times to warp (in the user recording's time base)
    words_user  : list of dicts with 'start'/'end' (same word count as words_base)
    words_base  : list of dicts with 'start'/'end'
    """
    if len(words_user) == 0 or len(words_base) == 0:
        return t_user.copy()

    n = min(len(words_user), len(words_base))
    anchors_u: list[float] = []
    anchors_b: list[float] = []
    for k in range(n):
        anchors_u.append(words_user[k]["start"])
        anchors_b.append(words_base[k]["start"])
        anchors_u.append(words_user[k]["end"])
        anchors_b.append(words_base[k]["end"])

    # Add epsilon offsets to ensure strict monotonicity
    au = np.array(anchors_u, dtype=float)
    ab = np.array(anchors_b, dtype=float)
    eps = np.arange(len(au)) * 1e-4
    au = au + eps

    # Clip t_user to the valid anchor range
    t_clipped = np.clip(t_user, au[0], au[-1])
    return np.interp(t_clipped, au, ab)