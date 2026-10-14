"""
inject/__init__.py -- Public API for flaw injection.

All injectors are pure functions that take (samples, sr, ...) and return
a new float32 ndarray. Nothing is written to disk here.
"""
from .select_spans import select_spans, file_seed, Span
from .volume import apply_gain, apply_clipping, apply_trailing_off
from .clarity import apply_muffled
from .pauses import apply_pause_missing, apply_pause_excess, apply_hesitation_filled
from .tempo import (
    apply_pace,
    apply_monotone,
    apply_terminal_contour,
    apply_stress_missed,
)

__all__ = [
    "select_spans",
    "file_seed",
    "Span",
    "apply_gain",
    "apply_clipping",
    "apply_trailing_off",
    "apply_muffled",
    "apply_pause_missing",
    "apply_pause_excess",
    "apply_hesitation_filled",
    "apply_pace",
    "apply_monotone",
    "apply_terminal_contour",
    "apply_stress_missed",
]