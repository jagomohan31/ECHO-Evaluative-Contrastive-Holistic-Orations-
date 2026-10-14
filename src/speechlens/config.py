"""
speechlens/config.py
All design constants (DESIGN §4). Overrides are loaded from
config/calibrated.json when present — those values win over these defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

# ── audio ────────────────────────────────────────────────────────────────────
SR: int = 16000
FRAME_S: float = 0.025
HOP_S: float = 0.010

# ── pause / hesitation ───────────────────────────────────────────────────────
PAUSE_MIN_S: float = 0.15
EXPECTED_PAUSE_S: float = 0.25
FILLED_PAUSE_MIN_S: float = 0.25

# ── detection ────────────────────────────────────────────────────────────────
Z_TOL: float = 1.5
Z_MAX: float = 4.0
SIGMA_FLOOR_MULT: float = 1.0

# ── window sizes ─────────────────────────────────────────────────────────────
WINDOW_WORDS: dict[str, int] = {"tempo": 5, "monotone": 6}
MIN_WORDS: dict[str, int] = {
    "pace": 4,
    "monotone": 6,
    "volume": 2,
    "trailing_off": 3,
    "muffled": 3,
}

# ── scoring ──────────────────────────────────────────────────────────────────
K: float = 0.8
WEIGHTS: dict[str, int] = {
    "pacing": 25,
    "intonation": 25,
    "fluency": 20,
    "volume": 15,
    "clarity": 15,
}

# ── quality guards ───────────────────────────────────────────────────────────
CLIP_FRAC_MIN: float = 0.001
SWALLOW_SCORE_MAX: float = 0.30
SWALLOW_DUR_MAX_S: float = 0.06

# ── upload / analysis limits ─────────────────────────────────────────────────
MAX_AUDIO_S: int = 180
MAX_UPLOAD_MB: int = 20
ROUND_DECIMALS: int = 4


def _load_overrides() -> None:
    """Apply config/calibrated.json overrides when the file exists."""
    here = Path(__file__).resolve()
    # Walk up to find the repo root (contains config/ directory).
    for parent in [here.parent.parent.parent, here.parent.parent, Path.cwd()]:
        cfg = parent / "config" / "calibrated.json"
        if cfg.exists():
            data = json.loads(cfg.read_text(encoding="utf-8"))
            params = data.get("best", data)
            g = globals()
            for key, val in params.items():
                k_upper = key.upper()
                if k_upper in g:
                    g[k_upper] = val
                elif key in g:
                    g[key] = val
            return


_load_overrides()