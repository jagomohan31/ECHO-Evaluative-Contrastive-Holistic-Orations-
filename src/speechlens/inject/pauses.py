"""
inject/pauses.py -- Pause injectors (pause_missing, pause_excess, hesitation_filled).

All pure functions, deterministic (seed from crc32).
"""
from __future__ import annotations
import numpy as np
from speechlens.inject.select_spans import file_seed


# ---------------------------------------------------------------------------
# pause_missing: shorten an existing pause with 10 ms crossfades
# ---------------------------------------------------------------------------

def apply_pause_missing(
    samples: np.ndarray,
    sr: int,
    pause_spans: list[tuple[float, float]],
    target_s: float,
    min_s: float = 0.060,
) -> np.ndarray:
    """
    Shorten each pause in pause_spans to target_s (but not below min_s).
    Uses 10 ms crossfades at both splice points.

    pause_spans: (pause_start, pause_end) in seconds.
    target_s: desired pause duration (e.g. 0.50 * original, or 0.06 s absolute).
    """
    fade_n = max(1, int(0.010 * sr))
    out_chunks: list[np.ndarray] = []
    prev_end = 0

    # Sort spans so we can splice in order
    for start, end in sorted(pause_spans):
        orig_dur = end - start
        new_dur = max(min_s, min(target_s, orig_dur))
        i0 = int(start * sr)
        i1 = int(end * sr)

        # Copy audio before this pause
        out_chunks.append(samples[prev_end:i0].astype(np.float32))

        # Insert shortened pause (first new_dur seconds of the pause)
        new_n = int(new_dur * sr)
        pause_chunk = samples[i0: i0 + new_n].astype(np.float32).copy()

        # Crossfade at start
        f_n = min(fade_n, len(pause_chunk))
        pause_chunk[:f_n] *= np.linspace(0.0, 1.0, f_n)
        # Crossfade at end
        pause_chunk[-f_n:] *= np.linspace(1.0, 0.0, f_n)
        out_chunks.append(pause_chunk)
        prev_end = i1

    out_chunks.append(samples[prev_end:].astype(np.float32))
    return np.concatenate(out_chunks)


# ---------------------------------------------------------------------------
# pause_excess: insert room-noise slice (seed from file_id)
# ---------------------------------------------------------------------------

def _pick_noise_slice(samples: np.ndarray, sr: int, dur_s: float, seed: int) -> np.ndarray:
    """Pick the quietest 0.3 s slice and loop/tile to dur_s."""
    slice_n = max(1, int(0.3 * sr))
    target_n = max(1, int(dur_s * sr))
    rng = np.random.default_rng(seed)

    n = len(samples)
    if n < slice_n:
        return np.zeros(target_n, dtype=np.float32)

    # Find the quietest 0.3 s window
    step = max(1, slice_n // 4)
    best_rms = float("inf")
    best_i = 0
    for i in range(0, n - slice_n, step):
        rms = float(np.sqrt(np.mean(samples[i: i + slice_n] ** 2)))
        if rms < best_rms:
            best_rms = rms
            best_i = i

    noise_src = samples[best_i: best_i + slice_n].astype(np.float32)
    # Tile to target length
    repeats = (target_n // slice_n) + 2
    noise = np.tile(noise_src, repeats)[:target_n]
    return noise


def apply_pause_excess(
    samples: np.ndarray,
    sr: int,
    insert_positions: list[float],
    dur_s: float,
    file_id: str = "default",
) -> np.ndarray:
    """
    Insert a room-noise slice of dur_s at each position (word-boundary time).
    Positions are in seconds; inserts happen in reverse order to preserve indices.
    """
    seed = file_seed(file_id)
    noise_slice = _pick_noise_slice(samples, sr, dur_s, seed)
    target_n = len(noise_slice)

    # Work in sample indices, insert from back to front
    positions_sorted = sorted(insert_positions, reverse=True)
    out = samples.astype(np.float32).copy()
    for pos in positions_sorted:
        idx = max(0, min(len(out), int(pos * sr)))
        out = np.concatenate([out[:idx], noise_slice, out[idx:]])
    return out


# ---------------------------------------------------------------------------
# hesitation_filled: insert synthetic "uh" at mid-phrase boundaries
# ---------------------------------------------------------------------------

def _synth_uh(
    dur_s: float,
    sr: int,
    f0_hz: float,
    fade_s: float = 0.030,
) -> np.ndarray:
    """
    Source-filter "uh": pulse train at f0_hz, filtered with formants ~600/1200/2500 Hz.
    Level = median voiced RMS of caller (applied externally).
    """
    from scipy.signal import lfilter

    n = max(1, int(dur_s * sr))
    fade_n = max(1, int(fade_s * sr))

    # Pulse train
    period = max(1, int(sr / max(f0_hz, 50.0)))
    glottal = np.zeros(n, dtype=np.float32)
    glottal[::period] = 1.0

    # Simple LP formant filter cascade (very rough "uh")
    # F1=600, F2=1200, F3=2500 -- two-pole resonators
    def _resonator(sig: np.ndarray, f: float, bw: float) -> np.ndarray:
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        b = [1 - r]
        a = [1, -2 * r * np.cos(theta), r * r]
        return lfilter(b, a, sig).astype(np.float32)

    sig = _resonator(glottal, 600.0, 80.0)
    sig = _resonator(sig, 1200.0, 120.0)
    sig = _resonator(sig, 2500.0, 200.0)

    # Normalise to unit RMS, then apply fades
    rms = np.sqrt(np.mean(sig ** 2))
    if rms > 1e-9:
        sig /= rms

    fade_in = np.linspace(0.0, 1.0, min(fade_n, n))
    fade_out = np.linspace(1.0, 0.0, min(fade_n, n))
    sig[:len(fade_in)] *= fade_in
    sig[-len(fade_out):] *= fade_out
    return sig


def apply_hesitation_filled(
    samples: np.ndarray,
    sr: int,
    insert_positions: list[float],
    dur_s: float,
    f0_hz: float,
    rms_level: float = 0.05,
    file_id: str = "default",
) -> np.ndarray:
    """
    Insert synthetic filled pauses ("uh") at each position.
    """
    uh = _synth_uh(dur_s, sr, f0_hz) * rms_level
    positions_sorted = sorted(insert_positions, reverse=True)
    out = samples.astype(np.float32).copy()
    for pos in positions_sorted:
        idx = max(0, min(len(out), int(pos * sr)))
        out = np.concatenate([out[:idx], uh, out[idx:]])
    return out