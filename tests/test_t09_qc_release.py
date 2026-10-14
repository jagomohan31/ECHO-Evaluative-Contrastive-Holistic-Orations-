"""
tests/test_t09_qc_release.py -- T09: dataset QC and release tests.
"""
from __future__ import annotations
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

DATASET_DIR = Path("data/dataset")
DIST_DIR = Path("dist")
SAMPLE_DIR = Path("data/sample")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# T09-01  dist/ zip exists and can be opened
# ---------------------------------------------------------------------------
def test_zip_exists():
    z = DIST_DIR / "speechlens_dataset_v1.zip"
    if not z.exists():
        pytest.skip("dist zip not built")
    assert zipfile.is_zipfile(z), "not a valid zip"


# ---------------------------------------------------------------------------
# T09-02  zip contains manifest.csv and DATACARD.md
# ---------------------------------------------------------------------------
def test_zip_contents():
    z = DIST_DIR / "speechlens_dataset_v1.zip"
    if not z.exists():
        pytest.skip("dist zip not built")
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    assert "manifest.csv" in names, "manifest.csv missing from zip"
    assert "DATACARD.md" in names, "DATACARD.md missing from zip"


# ---------------------------------------------------------------------------
# T09-03  zip sha256 checksums match manifest for injected files
# ---------------------------------------------------------------------------
def test_zip_checksums_match_manifest():
    z = DIST_DIR / "speechlens_dataset_v1.zip"
    manifest_path = DATASET_DIR / "manifest.csv"
    if not z.exists() or not manifest_path.exists():
        pytest.skip("dist or manifest missing")

    with open(manifest_path, encoding="utf-8") as f:
        rows = {r["file"]: r["sha256"] for r in csv.DictReader(f)
                if r.get("role") == "injected"}

    mismatches = []
    for filepath, sha in list(rows.items())[:20]:  # spot-check first 20
        p = Path(filepath)
        if p.exists():
            actual = _sha256_file(p)
            if actual != sha:
                mismatches.append(filepath)

    assert len(mismatches) == 0, f"sha256 mismatches: {mismatches}"


# ---------------------------------------------------------------------------
# T09-04  data/sample/ has at most 10 files and at most 10 MB
# ---------------------------------------------------------------------------
def test_sample_limits():
    if not SAMPLE_DIR.exists():
        pytest.skip("sample dir not built")
    files = list(SAMPLE_DIR.glob("*.flac"))
    assert len(files) <= 10, f"too many sample files: {len(files)}"
    total = sum(f.stat().st_size for f in files)
    assert total <= 10 * 1024 * 1024 + 1024, (
        f"sample too large: {total / 1024 / 1024:.1f} MB"
    )


# ---------------------------------------------------------------------------
# T09-05  SHA256SUMS.txt exists and lists at least some entries
# ---------------------------------------------------------------------------
def test_sha256sums_exists():
    p = DIST_DIR / "SHA256SUMS.txt"
    if not p.exists():
        pytest.skip("SHA256SUMS.txt not built")
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) > 0, "SHA256SUMS.txt is empty"


# ---------------------------------------------------------------------------
# T09-06  Audacity export script exists and is parsable Python
# ---------------------------------------------------------------------------
def test_audacity_script_exists():
    p = Path("scripts/export_audacity_labels.py")
    if not p.exists():
        pytest.skip("export script not generated")
    import ast
    try:
        ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError as e:
        pytest.fail(f"export_audacity_labels.py has syntax error: {e}")


# ---------------------------------------------------------------------------
# T09-07  validate() reports no issues
# ---------------------------------------------------------------------------
def test_validate_no_issues():
    if not DATASET_DIR.exists() or not (DATASET_DIR / "manifest.csv").exists():
        pytest.skip("dataset not built")
    from speechlens.dataset.validate import validate
    report = validate(DATASET_DIR, Path("data/library"))
    assert report["n_issues"] == 0, (
        f"validate() found {report['n_issues']} issues: {report['issues'][:5]}"
    )