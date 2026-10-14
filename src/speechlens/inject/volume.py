"""
inject/volume.py -- Volume and clipping injectors.

Families: volume_low, clipping, trailing_off.
All pure functions, deterministic.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Gain with ramps (volume_low / clipping base)
# ---------------------------------------------------------------------------

def apply_gain(
    samples: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    gain_db: float,
    ramp_s: float = 0.050,
) -> np.ndarray:
    """
    Apply `gain_db` inside each span with linear ramps of `ramp_s` seconds.
    Returns a new float32 array.
    """
    out = samples.astype(np.float32).copy()
    ramp_n = max(1, int(ramp_s * sr))
    factor = 10.0 ** (gain_db / 20.0)

    for start, end in spans:
        i0 = max(0, int(start * sr))
        i1 = min(len(out), int(end * sr))
        if i0 >= i1:
            continue
        n_span = i1 - i0
        env = np.full(n_span, factor, dtype=np.float32)

        up_n = min(ramp_n, n_span // 2)
        dn_n = min(ramp_n, n_span - up_n)
        env[:up_n] = np.linspace(1.0, factor, up_n)
        env[n_span - dn_n:] = np.linspace(factor, 1.0, dn_n)
        out[i0:i1] *= env

    return out


# ---------------------------------------------------------------------------
# Clipping: gain then hard-clip to [-1, 1]
# ---------------------------------------------------------------------------

def apply_clipping(
    samples: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    gain_db: float,
) -> np.ndarray:
    """Amplify then hard-clip to [-1, 1] within spans."""
    out = apply_gain(samples, sr, spans, gain_db)
    for start, end in spans:
        i0 = max(0, int(start * sr))
        i1 = min(len(out), int(end * sr))
        out[i0:i1] = np.clip(out[i0:i1], -1.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Trailing off: linear fade over last 3 words before sentence end
# ---------------------------------------------------------------------------

def apply_trailing_off(
    samples: np.ndarray,
    sr: int,
    fade_spans: list[tuple[float, float]],
    gain_db: float,
) -> np.ndarray:
    """
    Apply a linear gain fade from 0 dB down to `gain_db` over each fade_span.
    fade_spans should be (start_of_last_3_words, end_of_sentence).
    """
    out = samples.astype(np.float32).copy()
    for start, end in fade_spans:
        i0 = max(0, int(start * sr))
        i1 = min(len(out), int(end * sr))
        if i0 >= i1:
            continue
        n = i1 - i0
        # Ramp from 1.0 to factor (factor < 1 for negative gain_db)
        factor = 10.0 ** (gain_db / 20.0)
        ramp = np.linspace(1.0, factor, n, dtype=np.float32)
        out[i0:i1] *= ramp
    return out