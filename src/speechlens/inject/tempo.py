"""
inject/tempo.py -- Tempo and pitch injectors using parselmouth Manipulation.

Families: pace_fast, pace_slow, monotone, terminal_contour, stress_missed.
All functions are pure (no side effects) and deterministic.
"""
from __future__ import annotations
import numpy as np


def _get_parselmouth():
    import parselmouth
    return parselmouth


def _sound(samples: np.ndarray, sr: int):
    pm = _get_parselmouth()
    return pm.Sound(samples.astype(np.float64), sampling_frequency=float(sr))


def _seed_praat(seed: int = 42) -> None:
    pm = _get_parselmouth()
    pm.praat.run(f"random_initializeWithSeedUnsafelyButPredictably({int(seed)})")


# ---------------------------------------------------------------------------
# Pace (tempo scaling via DurationTier + PSOLA)
# ---------------------------------------------------------------------------

def apply_pace(
    samples: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    factor: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Scale speech duration over `spans` by `factor` using Praat DurationTier.
    factor > 1 slows down; factor < 1 speeds up.
    DurationTier value = 1/factor (Praat convention: relative duration).

    Parameters
    ----------
    spans : list of (start_s, end_s)
    factor: speed factor (1.15 = 15% faster than source -> DurationTier = 1/1.15)
    """
    pm = _get_parselmouth()
    snd = _sound(samples, sr)
    manip = pm.praat.call(snd, "To Manipulation", 0.01, 75.0, 600.0)
    dur_tier = pm.praat.call(manip, "Create DurationTier", "dur", 0.0, snd.duration)

    rel_dur = 1.0 / factor
    eps = 1e-4
    for start, end in spans:
        # Add points just outside the span to anchor neutral (1.0) regions
        pm.praat.call(dur_tier, "Add point", max(0.0, start - eps), 1.0)
        pm.praat.call(dur_tier, "Add point", start + eps, rel_dur)
        pm.praat.call(dur_tier, "Add point", end - eps, rel_dur)
        pm.praat.call(dur_tier, "Add point", min(snd.duration, end + eps), 1.0)

    pm.praat.call([manip, dur_tier], "Replace duration tier")
    _seed_praat(seed)
    result = pm.praat.call(manip, "Get resynthesis (overlap-add)")
    return result.values[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Monotone (pitch compression via PitchTier)
# ---------------------------------------------------------------------------

def apply_monotone(
    samples: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    alpha: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Replace PitchTier over spans with f_ref * (f / f_ref)^alpha.
    alpha=1 is identity; alpha=0 is fully flat; alpha=0.1 is near-monotone.

    f_ref = geometric mean of voiced F0 in the span.
    """
    pm = _get_parselmouth()
    snd = _sound(samples, sr)
    manip = pm.praat.call(snd, "To Manipulation", 0.01, 75.0, 600.0)
    pitch_tier = pm.praat.call(manip, "Extract pitch tier")

    # Read existing pitch-tier points via index API (List not available in all versions)
    n_pts = int(pm.praat.call(pitch_tier, "Get number of points"))
    pts: list[tuple[float, float]] = []
    for idx in range(1, n_pts + 1):
        t = float(pm.praat.call(pitch_tier, "Get time from index", idx))
        f = float(pm.praat.call(pitch_tier, "Get value at index", idx))
        pts.append((t, f))

    if not pts:
        return samples.copy()

    # Build new pitch tier
    new_tier = pm.praat.call(
        manip, "Create PitchTier", "pitch", 0.0, snd.duration
    )
    for t, f in pts:
        in_span = any(start <= t <= end for start, end in spans)
        if in_span and f > 0:
            # Find f_ref for the containing span
            span_freqs = [
                fq for tt, fq in pts
                if any(s <= tt <= e for s, e in spans) and fq > 0
            ]
            f_ref = float(np.exp(np.mean(np.log(span_freqs)))) if span_freqs else f
            f_new = f_ref * ((f / f_ref) ** alpha)
            pm.praat.call(new_tier, "Add point", t, max(50.0, f_new))
        else:
            pm.praat.call(new_tier, "Add point", t, f)

    pm.praat.call([manip, new_tier], "Replace pitch tier")
    _seed_praat(seed)
    result = pm.praat.call(manip, "Get resynthesis (overlap-add)")
    return result.values[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Terminal contour (raise pitch over last 0.35 s of sentence-final word)
# ---------------------------------------------------------------------------

def apply_terminal_contour(
    samples: np.ndarray,
    sr: int,
    word_ends: list[float],
    rise_st: float,
    window_s: float = 0.35,
    seed: int = 42,
) -> np.ndarray:
    """
    Raise PitchTier at the end of each sentence-final word by `rise_st` semitones
    over the last `window_s` seconds of the word.

    word_ends: list of sentence-final word end times in seconds.
    """
    pm = _get_parselmouth()
    snd = _sound(samples, sr)
    manip = pm.praat.call(snd, "To Manipulation", 0.01, 75.0, 600.0)
    pitch_tier = pm.praat.call(manip, "Extract pitch tier")

    # Read existing pitch-tier points via index API
    n_pts = int(pm.praat.call(pitch_tier, "Get number of points"))
    pts: list[tuple[float, float]] = []
    for idx in range(1, n_pts + 1):
        t = float(pm.praat.call(pitch_tier, "Get time from index", idx))
        f = float(pm.praat.call(pitch_tier, "Get value at index", idx))
        pts.append((t, f))

    if not pts:
        return samples.copy()

    new_tier = pm.praat.call(
        manip, "Create PitchTier", "pitch", 0.0, snd.duration
    )
    for t, f in pts:
        new_f = f
        for we in word_ends:
            window_start = max(0.0, we - window_s)
            if window_start <= t <= we and f > 0:
                # Linear ramp: 0 -> rise_st over the window
                progress = (t - window_start) / max(window_s, 1e-6)
                delta_st = progress * rise_st
                new_f = f * (2.0 ** (delta_st / 12.0))
                break
        pm.praat.call(new_tier, "Add point", t, max(50.0, new_f))

    pm.praat.call([manip, new_tier], "Replace pitch tier")
    _seed_praat(seed)
    result = pm.praat.call(manip, "Get resynthesis (overlap-add)")
    return result.values[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Stress missed (pitch + level + duration on prominent words)
# ---------------------------------------------------------------------------

def apply_stress_missed(
    samples: np.ndarray,
    sr: int,
    word_spans: list[tuple[float, float]],
    alpha: float = 0.2,
    gain_db: float = -5.0,
    dur_ratio: float = 0.85,
    seed: int = 42,
) -> np.ndarray:
    """
    On prominent words: compress pitch (alpha), reduce level (-5 dB),
    and shorten duration (relative_duration = dur_ratio).
    """
    if not word_spans:
        return samples.copy()

    # First apply tempo change
    result = apply_pace(samples, sr, word_spans, 1.0 / dur_ratio, seed=seed)

    # Then pitch compression (monotone-style but per-word)
    result = apply_monotone(result, sr, word_spans, alpha, seed=seed)

    # Then gain
    result = apply_gain(result, sr, word_spans, gain_db)
    return result

    # Then pitch compression (monotone-style but per-word)
    result = apply_monotone(result, sr, word_spans, alpha)

    # Then gain
    result = apply_gain(result, sr, word_spans, gain_db)
    return result


# Imported here to avoid circular; gain is in volume.py -- use a local fallback
def apply_gain(
    samples: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    gain_db: float,
) -> np.ndarray:
    """Apply gain_db to the given time spans with 50 ms ramps."""
    out = samples.astype(np.float32).copy()
    ramp_n = int(0.050 * sr)
    factor = 10.0 ** (gain_db / 20.0)
    for start, end in spans:
        i0 = int(start * sr)
        i1 = min(len(out), int(end * sr))
        if i0 >= i1:
            continue
        ramp_up = np.linspace(1.0, factor, min(ramp_n, i1 - i0))
        ramp_dn = np.linspace(factor, 1.0, min(ramp_n, i1 - i0))
        n_span = i1 - i0
        env = np.full(n_span, factor)
        env[: len(ramp_up)] = ramp_up
        env[max(0, n_span - len(ramp_dn)):] = ramp_dn[max(0, len(ramp_dn) - n_span):]
        out[i0:i1] *= env
    return out