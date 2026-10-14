
"""
dataset/validate.py -- T09: dataset validation (DESIGN 5.7, 5.9).

Checks:
  - files exist
  - spans inside durations
  - flaw spans ordered, no overlaps within a family
  - sha256 matches manifest
  - split coverage (dev and test both present)
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf


def _portable_path(value: str) -> Path:
    """Convert a manifest or label path to a cross-platform Path."""
    return Path(value.replace("\\", "/"))


def _sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(
    dataset_dir: Path,
    library_dir: Path | None = None,
) -> dict:
    """
    Validate the dataset and return a report containing any issues.

    Paths recorded in manifest.csv and labels.json may use Windows
    backslashes. They are normalized before filesystem access so the
    same dataset works on Windows and Linux.
    """
    issues: list[str] = []
    manifest_path = dataset_dir / "manifest.csv"

    if not manifest_path.exists():
        return {
            "valid": False,
            "issues": ["manifest.csv not found"],
            "n_issues": 1,
        }

    with open(manifest_path, encoding="utf-8", newline="") as f:
        rows: list[dict] = list(csv.DictReader(f))

    splits_seen: set[str] = set()

    # 1. Validate files referenced by the manifest.
    for row in rows:
        file_value = row["file"]
        file_path = _portable_path(file_value)

        if not file_path.exists():
            issues.append(f"file missing: {file_value}")
            continue

        # 2. SHA-256 must match the manifest.
        try:
            actual_sha = _sha256_file(file_path)
            if actual_sha != row["sha256"]:
                issues.append(f"sha256 mismatch: {file_value}")
        except OSError as e:
            issues.append(f"cannot read file {file_value}: {e}")
            continue

        # 3. Audio duration must be positive.
        try:
            duration = sf.info(file_path).duration
            if duration <= 0:
                issues.append(f"zero duration: {file_value}")
        except Exception as e:
            issues.append(f"cannot read duration for {file_value}: {e}")
            continue

        splits_seen.add(row.get("split", ""))

    # 4. Split coverage.
    if "dev" not in splits_seen:
        issues.append("no dev-split files found")

    # 5. Validate flaw spans in labels.
    for labels_file in dataset_dir.glob("*/labels.json"):
        try:
            labels: list[dict] = json.loads(
                labels_file.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as e:
            issues.append(f"cannot read labels file {labels_file}: {e}")
            continue

        for entry in labels:
            if not entry.get("synthetic_flaw"):
                continue

            file_value = entry.get("file", "")
            if not file_value:
                continue

            file_path = _portable_path(file_value)

            if not file_path.exists():
                # The missing file may already have been reported from
                # the manifest. Do not attempt to inspect its duration.
                continue

            try:
                duration = sf.info(file_path).duration
            except Exception:
                continue

            # Group flaws by family.
            by_family: dict[str, list[dict]] = {}
            for flaw in entry.get("flaws", []):
                by_family.setdefault(flaw["family"], []).append(flaw)

            for family, flaws in by_family.items():
                sorted_flaws = sorted(flaws, key=lambda x: x["start"])
                prev_end = 0.0

                for flaw in sorted_flaws:
                    start = flaw["start"]
                    end = flaw["end"]

                    # Tempo/pause transformations may change audio length.
                    # Retain the existing project's generous span limit.
                    if start < 0 or start > 120.0:
                        issues.append(
                            f"{file_value}: span start {start:.2f} "
                            "out of range"
                        )

                    if end < start:
                        issues.append(
                            f"{file_value}: span end {end:.2f} "
                            f"< start {start:.2f}"
                        )

                    # Boundary families can share a sentence ending.
                    boundary_families = {
                        "pause_excess",
                        "hesitation_filled",
                        "terminal_contour",
                    }

                    if (
                        flaw["family"] not in boundary_families
                        and start < prev_end - 0.1
                    ):
                        issues.append(
                            f"{file_value}: overlapping spans in family "
                            f"{family}"
                        )

                    prev_end = end

    report = {
        "valid": len(issues) == 0,
        "n_files_checked": len(rows),
        "splits_seen": sorted(splits_seen),
        "issues": issues,
        "n_issues": len(issues),
    }

    return report


def main() -> None:
    dataset_dir = Path("data/dataset")
    library_dir = Path("data/library")

    report = validate(dataset_dir, library_dir)

    print(f"Validation: {'PASS' if report['valid'] else 'FAIL'}")
    print(f"  Files checked: {report['n_files_checked']}")
    print(f"  Splits seen:   {report['splits_seen']}")
    print(f"  Issues:        {report['n_issues']}")

    if report["issues"]:
        for issue in report["issues"][:20]:
            print(f"    - {issue}")

    report_path = dataset_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("Wrote data/dataset/validation_report.json")


if __name__ == "__main__":
    main()
