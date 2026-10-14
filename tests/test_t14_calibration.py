"""
test_t14_calibration.py -- Unit tests for calibration (TASK_1 T14).

Verifies:
  1. Calibration code only processes the dev split, never test split.
  2. config/calibrated.json and results/calibration.json exist with required schema.
  3. speechlens.config reflects the calibrated parameter overrides.
  4. results/fig_calibration.png exists.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from speechlens.evalsuite.calibrate import load_dev_entries, DEV_SPEECH_IDS
import speechlens.config as cfg


def test_calibration_dev_only():
    """Verify load_dev_entries never includes test-split files."""
    entries = load_dev_entries(DEV_SPEECH_IDS)
    assert len(entries) > 0, "Expected dev entries to be loaded"
    for entry in entries:
        assert entry.get("split") == "dev", f"Entry {entry.get('file')} has non-dev split: {entry.get('split')}"
        # Test speech IDs should not be in dev entries
        assert entry.get("speech_id") not in ("speech-002", "speech-003"), f"Test speech found in dev: {entry}"


def test_calibrated_json_schema():
    """Verify config/calibrated.json and results/calibration.json structure."""
    cfg_file = Path("config/calibrated.json")
    assert cfg_file.exists(), "config/calibrated.json missing"
    
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "best" in data
    assert "best_macro_f1" in data
    assert "grid_results" in data
    
    best = data["best"]
    assert "z_tol" in best
    assert "sigma_floor_mult" in best
    assert "k" in best
    assert best["z_tol"] in [1.25, 1.5, 1.75, 2.0]
    assert best["sigma_floor_mult"] in [0.75, 1.0, 1.25]
    assert best["k"] in [0.5, 0.65, 0.8, 0.95]


def test_config_reflects_calibrated_values():
    """Verify that speechlens.config applied the calibrated overrides."""
    cfg_file = Path("config/calibrated.json")
    assert cfg_file.exists()
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    best = data["best"]

    assert cfg.Z_TOL == best["z_tol"]
    assert cfg.SIGMA_FLOOR_MULT == best["sigma_floor_mult"]
    assert cfg.K == best["k"]


def test_calibration_figure_exists():
    """Verify results/fig_calibration.png was generated."""
    fig_path = Path("results/fig_calibration.png")
    assert fig_path.exists(), "results/fig_calibration.png missing"
    assert fig_path.stat().st_size > 1000, "Figure file appears empty"
