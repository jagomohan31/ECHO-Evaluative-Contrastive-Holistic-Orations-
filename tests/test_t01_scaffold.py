"""
tests/test_t01_scaffold.py
T01 fast-suite: audio round-trip, text normalisation, syllables, CLI help,
schema validation of a hand-written example.
No model downloads, no external network calls.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_wav(path: Path, duration_s: float = 1.0, sr: int = 16000, freq: float = 440.0) -> None:
    """Write a minimal mono 16-bit WAV sine at *freq* Hz."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ── text normalisation ─────────────────────────────────────────────────────────

def test_normalise_digits():
    from speechlens.text import normalise_text
    result = normalise_text("I have 4 cats and 12 dogs.")
    # digits must be gone
    assert not any(c.isdigit() for c in result)
    # "four" should appear
    assert "FOUR" in result


def test_normalise_accents():
    from speechlens.text import normalise_text
    result = normalise_text("café naïve résumé")
    assert result == "CAFE NAIVE RESUME"


def test_normalise_keeps_apostrophe():
    from speechlens.text import normalise_text
    result = normalise_text("don't can't")
    assert "DON'T" in result
    assert "CAN'T" in result


def test_normalise_removes_special():
    from speechlens.text import normalise_text
    result = normalise_text("Hello, world! (test) #1.")
    # Only A-Z, apostrophe and spaces
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ' " for c in result)


# ── syllables ─────────────────────────────────────────────────────────────────

def test_syllables_simple():
    from speechlens.text import count_syllables
    assert count_syllables("HELLO") >= 2


def test_syllables_one():
    from speechlens.text import count_syllables
    assert count_syllables("CAT") == 1


def test_syllables_empty():
    from speechlens.text import count_syllables
    assert count_syllables("") == 0


# ── tokenise ──────────────────────────────────────────────────────────────────

def test_tokenise_punct_after():
    from speechlens.text import tokenise
    tokens = tokenise("Hello, world!")
    assert tokens[0].text == "Hello"
    assert "," in tokens[0].punct_after
    assert tokens[1].text == "world"
    assert "!" in tokens[1].punct_after


def test_tokenise_indices():
    from speechlens.text import tokenise
    tokens = tokenise("one two three")
    assert [t.i for t in tokens] == [0, 1, 2]


# ── CLI help ──────────────────────────────────────────────────────────────────

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "speechlens", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "analyze" in result.stdout


@pytest.mark.parametrize("cmd", [
    "fetch", "align", "dataset", "baseline", "calibrate",
    "analyze", "evaluate", "warmup", "serve", "reproduce",
])
def test_cli_subcommand_help(cmd):
    result = subprocess.run(
        [sys.executable, "-m", "speechlens", cmd, "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"'{cmd} --help' failed:\n{result.stderr}"


# ── schema validation ─────────────────────────────────────────────────────────

def _minimal_result() -> dict:
    """Hand-written schema-valid AnalysisResult."""
    return {
        "schema_version": "1.0",
        "meta": {
            "speech_id": None,
            "mode": "norm",
            "duration_s": 10.0,
            "n_words": 5,
            "baseline_readers": [],
            "software": {"speechlens": "0.1.0", "model": "facebook/wav2vec2-base-960h"},
            "input_sha256": "abc123",
        },
        "scores": {
            "total": 85.0,
            "dimensions": {
                "pacing": 90.0, "intonation": 80.0, "fluency": 85.0,
                "volume": 88.0, "clarity": 82.0,
            },
        },
        "global": {
            "articulation_rate_syl_s": 4.5,
            "baseline_articulation_rate_syl_s": None,
            "wpm": 120.0,
            "pause_ratio": 0.15,
            "f0_sd_st": 2.3,
        },
        "words": [
            {
                "i": 0, "text": "hello", "start": 0.0, "end": 0.5,
                "base_start": None, "base_end": None,
                "z": {"tempo": None, "pitch": None, "energy": None, "pause": None},
                "prominence": {"user": None, "base": None},
                "align_score": 0.9, "flags": [],
            }
        ],
        "series": {
            "hop_s": 0.02,
            "t_user": [0.0, 0.02],
            "t_base": [0.0, 0.02],
            "pitch_st": {"user": [None, 1.0], "base": [None, 0.5], "lo": [None, -1.0], "hi": [None, 2.0]},
            "energy_db": {"user": [-10.0, -9.0], "base": [-10.5, -9.5], "lo": [-15.0, -14.0], "hi": [-5.0, -4.0]},
            "tempo": {"user": [4.0, 4.5], "base": [4.2, 4.2]},
        },
        "pauses": [{"start": 0.5, "end": 0.7, "kind": "silent", "expected": True}],
        "regions": [],
        "warnings": [],
    }


def test_schema_valid():
    import jsonschema
    schema_path = Path(__file__).parent.parent / "schema" / "analysis_result.schema.json"
    schema = json.loads(schema_path.read_text())
    instance = _minimal_result()
    jsonschema.validate(instance, schema)  # raises on failure


def test_schema_invalid_mode():
    import jsonschema
    schema_path = Path(__file__).parent.parent / "schema" / "analysis_result.schema.json"
    schema = json.loads(schema_path.read_text())
    instance = _minimal_result()
    instance["meta"]["mode"] = "bad_mode"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance, schema)


def test_pydantic_model():
    from speechlens.models import AnalysisResult
    obj = AnalysisResult.model_validate(_minimal_result())
    assert obj.scores.total == 85.0
    assert obj.meta.mode == "norm"


# ── config constants ──────────────────────────────────────────────────────────

def test_config_constants():
    from speechlens import config
    assert config.SR == 16000
    assert config.FRAME_S == 0.025
    assert config.HOP_S == 0.010
    assert config.Z_TOL in (1.5, 1.75)
    assert config.K in (0.8, 0.5)
    assert sum(config.WEIGHTS.values()) == 100