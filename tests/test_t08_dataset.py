"""
tests/test_t08_dataset.py -- T08: contrastive dataset build tests.

Tests (fast): manifest schema, labels schema, validation_report zero errors.
Slow test: full rebuild is deterministic (sha256 matches on second run).
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import pytest

DATASET_DIR = Path("data/dataset")
LIBRARY_DIR = Path("data/library")


def _manifest_rows() -> list[dict]:
    p = DATASET_DIR / "manifest.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _val_report() -> dict:
    p = DATASET_DIR / "validation_report.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _all_labels() -> list[dict]:
    results = []
    for labels_file in DATASET_DIR.glob("*/labels.json"):
        results.extend(json.loads(labels_file.read_text(encoding="utf-8")))
    return results


# ---------------------------------------------------------------------------
# T08-01  Dataset dir exists (build has run)
# ---------------------------------------------------------------------------
def test_dataset_dir_exists():
    if not DATASET_DIR.exists():
        pytest.skip("dataset not built yet -- run `python -m speechlens dataset build`")
    assert (DATASET_DIR / "manifest.csv").exists(), "manifest.csv missing"
    assert (DATASET_DIR / "validation_report.json").exists(), "validation_report.json missing"


# ---------------------------------------------------------------------------
# T08-02  Validation report shows zero errors
# ---------------------------------------------------------------------------
def test_validation_zero_errors():
    report = _val_report()
    if not report:
        pytest.skip("validation_report.json not found")
    assert report.get("n_errors", 0) == 0, (
        f"dataset build had {report['n_errors']} errors"
    )


# ---------------------------------------------------------------------------
# T08-03  At least 100 flawed files exist
# ---------------------------------------------------------------------------
def test_at_least_100_injected():
    rows = _manifest_rows()
    if not rows:
        pytest.skip("manifest.csv not found")
    injected = [r for r in rows if r.get("role") == "injected"]
    assert len(injected) >= 100, (
        f"expected >= 100 injected files, got {len(injected)}"
    )


# ---------------------------------------------------------------------------
# T08-04  manifest.csv columns match schema
# ---------------------------------------------------------------------------
def test_manifest_columns():
    p = DATASET_DIR / "manifest.csv"
    if not p.exists():
        pytest.skip("manifest.csv not found")
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"file", "speech_id", "reader_id", "role", "level",
                    "families", "n_flaws", "duration_s", "split", "sha256", "leaky_baseline"}
        assert expected.issubset(set(reader.fieldnames or [])), (
            f"missing columns: {expected - set(reader.fieldnames or [])}"
        )


# ---------------------------------------------------------------------------
# T08-05  labels.json entries have required fields and synthetic_flaw flag
# ---------------------------------------------------------------------------
def test_labels_schema():
    labels = _all_labels()
    if not labels:
        pytest.skip("labels.json not found")
    required = {"file", "speech_id", "reader_id", "level", "split",
                "leaky_baseline", "synthetic_flaw", "flaws"}
    for entry in labels:
        missing = required - set(entry.keys())
        assert not missing, f"labels entry missing fields: {missing}"
        if entry["synthetic_flaw"]:
            assert isinstance(entry["flaws"], list) and len(entry["flaws"]) > 0, (
                "synthetic_flaw=True but flaws list is empty"
            )


# ---------------------------------------------------------------------------
# T08-06  FLAC files referenced in manifest actually exist
# ---------------------------------------------------------------------------
def test_flac_files_exist():
    rows = _manifest_rows()
    if not rows:
        pytest.skip("manifest.csv not found")
    missing = []
    for r in rows:
        p = Path(r["file"])
        if not p.exists():
            missing.append(r["file"])
    assert len(missing) == 0, (
        f"{len(missing)} files in manifest.csv do not exist: {missing[:5]}..."
    )


# ---------------------------------------------------------------------------
# T08-07  Determinism: rebuild one speech gives identical sha256 (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_rebuild_determinism():
    """Rebuild speech-001 and verify sha256 of injected files matches first run."""
    import hashlib
    from speechlens.dataset.build import build

    rows = _manifest_rows()
    if not rows:
        pytest.skip("dataset not built; run build first")

    # Record sha256s for speech-001 injected files from first build
    first_sha = {
        r["file"]: r["sha256"]
        for r in rows
        if r["speech_id"] == "speech-001" and r["role"] == "injected"
    }
    if not first_sha:
        pytest.skip("no speech-001 injected files in manifest")

    # Rebuild speech-001 only
    build(
        library_dir=LIBRARY_DIR,
        dataset_dir=DATASET_DIR,
        speeches=["speech-001"],
        skip_align=True,
    )

    # Check sha256s match
    def _sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    mismatches = []
    for filepath, sha in first_sha.items():
        if Path(filepath).exists():
            new_sha = _sha256(filepath)
            if new_sha != sha:
                mismatches.append(filepath)

    assert len(mismatches) == 0, (
        f"Non-deterministic files: {mismatches}"
    )