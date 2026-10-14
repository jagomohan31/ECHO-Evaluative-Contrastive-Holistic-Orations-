"""
inject/clarity.py -- Spectral clarity injectors.

Families: muffled.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt


def apply_muffled(
    samples: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    cutoff_hz: float,
) -> np.ndarray:
    """
    Apply 6th-order Butterworth low-pass filter (zero phase) within spans.
    Outside spans the signal is unchanged.

    cutoff_hz: filter cutoff, e.g. 5000, 3500, 2500 Hz per DESIGN 5.5.
    """
    out = samples.astype(np.float32).copy()
    nyq = sr / 2.0
    if cutoff_hz >= nyq:
        return out  # no-op if cutoff above Nyquist

    sos = butter(6, cutoff_hz / nyq, btype="low", output="sos")

    for start, end in spans:
        i0 = max(0, int(start * sr))
        i1 = min(len(out), int(end * sr))
        if i0 >= i1:
            continue
        chunk = out[i0:i1].astype(np.float64)
        filtered = sosfiltfilt(sos, chunk).astype(np.float32)
        out[i0:i1] = filtered

    return out