"""
test_t15_evaluation.py -- Unit and integration tests for evaluation suite (TASK_1 T15).

Verifies:
  1. results/metrics.json has all required fields including SHA256 checksums and software versions.
  2. results/REPORT.md exists and is formatted correctly.
  3. results/tables/*.csv exist.
  4. results/fig_*.png figures exist.
  5. Slow test verifies --quick mode runs to completion in under 10 minutes.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from speechlens.evalsuite.evaluate import run


def test_metrics_json_schema():
    """Verify results/metrics.json structure per DESIGN 13."""
    metrics_path = Path("results/metrics.json")
    assert metrics_path.exists(), "results/metrics.json missing"
    
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    
    # Metadata checksums
    assert "sha256_calibrated_json" in data
    assert "dataset_manifest_sha256" in data
    assert "software_versions" in data
    assert data["software_versions"]["speechlens"] == "1.0.0"

    # Evaluations E1-E8
    assert "e1_detection" in data
    assert "e2_monotonicity" in data
    assert "e3_severity" in data
    assert "e4_loo" in data
    assert "e5_latency" in data
    assert "e6_determinism" in data
    assert "e7_bounds" in data
    assert "e8_explanations" in data


def test_report_exists():
    """Verify results/REPORT.md exists and contains summary."""
    report_path = Path("results/REPORT.md")
    assert report_path.exists(), "results/REPORT.md missing"
    content = report_path.read_text(encoding="utf-8")
    assert "SpeechLens — Evaluation Report" in content
    assert "E1 Region detection" in content
    assert "E2 Monotonicity" in content
    assert "E6 Determinism" in content


def test_tables_exist():
    """Verify generated CSV tables in results/tables/."""
    tables_dir = Path("results/tables")
    assert tables_dir.exists(), "results/tables/ missing"
    assert (tables_dir / "per_family_metrics.csv").exists()
    assert (tables_dir / "eval_summary.csv").exists()


def test_figures_exist():
    """Verify generated figures in results/."""
    results_dir = Path("results")
    assert (results_dir / "fig_confusion_matrix.png").exists()
    assert (results_dir / "fig_severity_curve.png").exists()
    assert (results_dir / "fig_example_overlay.png").exists()


@pytest.mark.slow
def test_evaluate_quick_completes_under_10min():
    """Verify that quick evaluation completes in well under 10 minutes (600s)."""
    t0 = time.time()
    res = run(quick=True)
    dur = time.time() - t0
    assert dur < 600.0, f"Quick evaluation took too long: {dur:.1f}s (> 600s)"
    assert res.get("metrics") is not None or "e1_detection" in res
