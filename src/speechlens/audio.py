"""
speechlens/audio.py
Load or convert any input to mono 16 kHz float32 via imageio-ffmpeg.
Peak-normalise, trim leading/trailing silence, reject audio over MAX_AUDIO_S.
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from speechlens.config import MAX_AUDIO_S, SR


class AudioError(Exception):
    """Raised for unsupported, silent, empty or too-long audio."""


def load(path: str | Path) -> np.ndarray:
    """
    Load *path* to mono 16 kHz float32 ndarray.

    Accepted: any format ffmpeg understands (wav, flac, mp3, opus, webm, …).
    Returns a 1-D float32 array, peak-normalised, silence-trimmed.
    Raises AudioError on empty, silent or over-length audio.
    """
    path = Path(path)
    if not path.exists():
        raise AudioError(f"File not found: {path}")

    samples = _ffmpeg_decode(path)

    if samples.size == 0:
        raise AudioError("Audio is empty.")

    duration = samples.size / SR
    if duration > MAX_AUDIO_S:
        raise AudioError(
            f"Audio is {duration:.1f} s, which exceeds the {MAX_AUDIO_S} s limit."
        )

    samples = _trim_silence(samples)

    if samples.size < SR * 0.5:
        raise AudioError("Audio is silent or too short after trimming.")

    # Peak-normalise.
    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples / peak

    return samples.astype(np.float32)


def load_bytes(data: bytes, suffix: str = ".wav") -> np.ndarray:
    """Load from raw bytes (e.g. an upload buffer)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return load(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def sha256(samples: np.ndarray) -> str:
    """Hex SHA-256 of the raw float32 sample bytes."""
    return hashlib.sha256(samples.tobytes()).hexdigest()


def _ffmpeg_decode(path: Path) -> np.ndarray:
    """Decode *path* to mono 16 kHz float32 PCM via ffmpeg."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    cmd = [
        ffmpeg_exe,
        "-y", "-i", str(path),
        "-ac", "1",
        "-ar", str(SR),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise AudioError(
            f"ffmpeg failed on {path.name}: {result.stderr.decode(errors='replace')[:200]}"
        )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _trim_silence(
    samples: np.ndarray,
    top_db: float = 40.0,
    frame_len: int = 512,
    hop_len: int = 128,
) -> np.ndarray:
    """Trim leading and trailing silence using energy threshold."""
    rms = np.array([
        np.sqrt(np.mean(samples[i: i + frame_len] ** 2))
        for i in range(0, len(samples) - frame_len, hop_len)
    ])
    if rms.size == 0:
        return samples
    ref = rms.max()
    threshold = ref * 10 ** (-top_db / 20)
    voiced = rms > threshold
    if not voiced.any():
        return samples

    start_frame = np.argmax(voiced)
    end_frame = len(voiced) - np.argmax(voiced[::-1]) - 1
    start_sample = start_frame * hop_len
    end_sample = min((end_frame + 1) * hop_len + frame_len, len(samples))
    return samples[start_sample:end_sample]