"""
dataset/release.py -- T09: dataset release package (DESIGN 5.9).

Creates:
  dist/speechlens_dataset_v1.zip
  dist/SHA256SUMS.txt
  data/sample/  (at most 10 files, 10 MB total)
  scripts/export_audacity_labels.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import soundfile as sf


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


DATACARD_TEMPLATE = """\
# SpeechLens Dataset v1 -- DATACARD

## Summary
Contrastive speech dataset with injected prosodic flaws at three severity levels,
grounded to time spans for evaluation of spoken-language analysis systems.

## Licence
All source recordings are LibriVox public-domain (CC0). Synthetic modifications
are licensed CC0 for research use.

## Ethics
Flawed files are synthetic modifications of real people's recordings, made only
for evaluation. They are flagged `synthetic_flaw: true`, keep source attribution,
and **must not be redistributed as authentic speech** (DESIGN 5.8).

## Speeches
{speeches}

## Flaw families
{families}

## Statistics
- Injected files: {n_injected}
- L0 ideal recordings: {n_l0}
- Total duration: {total_dur_min:.1f} min

## Splits
- Dev: {n_dev} files
- Test: {n_test} files

## Columns in manifest.csv
file, speech_id, reader_id, role, level, families, n_flaws, duration_s, split, sha256, leaky_baseline

## Leakage policy
For speeches with fewer than 3 readers, mirrors are built from the same recording
as the baseline (leaky_baseline=true). These results are reported separately in
the evaluation as optimistic upper bounds.
"""


def build_datacard(dataset_dir: Path, library_dir: Path) -> str:
    """Generate DATACARD.md content."""
    manifest_path = dataset_dir / "manifest.csv"
    rows: list[dict] = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    speech_ids = sorted(set(r["speech_id"] for r in rows))
    speeches_text = ""
    for sid in speech_ids:
        meta_path = library_dir / sid / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            speeches_text += f"- **{sid}**: {meta.get('title', sid)}\n"

    from speechlens.dataset.build import FAMILY_DIM
    families_text = "\n".join(f"- {f} ({d})" for f, d in FAMILY_DIM.items())

    injected = [r for r in rows if r.get("role") == "injected"]
    l0 = [r for r in rows if r.get("level") == "0"]
    dev = [r for r in rows if r.get("split") == "dev"]
    test = [r for r in rows if r.get("split") == "test"]
    total_dur = sum(float(r.get("duration_s", 0)) for r in rows)

    return DATACARD_TEMPLATE.format(
        speeches=speeches_text,
        families=families_text,
        n_injected=len(injected),
        n_l0=len(l0),
        total_dur_min=total_dur / 60.0,
        n_dev=len(dev),
        n_test=len(test),
    )


def build_sample(dataset_dir: Path, sample_dir: Path) -> None:
    """Copy up to 10 injected files (max 10 MB total) to data/sample/."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing
    for f in sample_dir.glob("*.flac"):
        f.unlink()

    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        return

    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Pick up to 10 injected files, spread across speeches and levels
    injected = [r for r in rows if r.get("role") == "injected"]
    # Sort for determinism
    injected_sorted = sorted(injected, key=lambda r: (r["speech_id"], r["level"], r["families"]))

    selected: list[dict] = []
    total_bytes = 0
    limit_bytes = 10 * 1024 * 1024  # 10 MB

    seen_speech_level: set[tuple] = set()
    for r in injected_sorted:
        key = (r["speech_id"], r["level"])
        if key in seen_speech_level:
            continue
        p = Path(r["file"])
        if not p.exists():
            continue
        size = p.stat().st_size
        if total_bytes + size > limit_bytes:
            continue
        selected.append(r)
        seen_speech_level.add(key)
        total_bytes += size
        if len(selected) >= 10:
            break

    for r in selected:
        src = Path(r["file"])
        dst = sample_dir / src.name
        shutil.copy2(src, dst)

    print(f"Sample: {len(selected)} files, {total_bytes / 1024 / 1024:.1f} MB")


def build_zip(
    dataset_dir: Path,
    library_dir: Path,
    dist_dir: Path,
) -> Path:
    """Build dist/speechlens_dataset_v1.zip and SHA256SUMS.txt."""
    dist_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dist_dir / "speechlens_dataset_v1.zip"

    manifest_path = dataset_dir / "manifest.csv"
    rows: list[dict] = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    sha_lines: list[str] = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add manifest.csv
        if manifest_path.exists():
            zf.write(manifest_path, "manifest.csv")
        # Add labels.json files
        for labels_file in sorted(dataset_dir.glob("*/labels.json")):
            arc = labels_file.relative_to(dataset_dir.parent)
            zf.write(labels_file, str(arc))
        # Add validation_report.json
        val = dataset_dir / "validation_report.json"
        if val.exists():
            zf.write(val, "validation_report.json")
        # Add DATACARD.md
        datacard = build_datacard(dataset_dir, library_dir)
        zf.writestr("DATACARD.md", datacard)
        # Add all injected FLAC files
        for row in rows:
            if row.get("role") != "injected":
                continue
            p = Path(row["file"])
            if not p.exists():
                continue
            arc = p.relative_to(dataset_dir.parent) if p.is_relative_to(dataset_dir.parent) else p.name
            zf.write(p, str(arc))
            sha_lines.append(f"{row['sha256']}  {arc}")

    sha_path = dist_dir / "SHA256SUMS.txt"
    sha_path.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    zip_sha = _sha256_file(zip_path)
    print(f"Zip: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"SHA256: {zip_sha}")
    return zip_path


def write_audacity_script(scripts_dir: Path) -> None:
    """Write scripts/export_audacity_labels.py."""
    scripts_dir.mkdir(parents=True, exist_ok=True)
    code = '''\
#!/usr/bin/env python
"""
export_audacity_labels.py -- Export flaw labels as Audacity label tracks.

Usage:
    python scripts/export_audacity_labels.py [--speech speech-001] [--out ./labels]

Writes one tab-separated .txt file per audio file, which Audacity can
import via File -> Import -> Labels.

Format: start_s <TAB> end_s <TAB> family:level
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def export_labels(dataset_dir: Path, out_dir: Path, speech_filter: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for labels_file in sorted(dataset_dir.glob("*/labels.json")):
        speech_id = labels_file.parent.name
        if speech_filter and speech_id != speech_filter:
            continue
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        for entry in labels:
            if not entry.get("synthetic_flaw") or not entry.get("flaws"):
                continue
            stem = Path(entry["file"]).stem
            out_path = out_dir / f"{stem}.txt"
            lines = []
            for flaw in entry["flaws"]:
                label = f"{flaw['family']}:L{entry['level']}"
                lines.append(f"{flaw['start']:.6f}\\t{flaw['end']:.6f}\\t{label}")
            out_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    print(f"Audacity labels written to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", default=None)
    ap.add_argument("--out", default="./audacity_labels")
    args = ap.parse_args()
    export_labels(Path("data/dataset"), Path(args.out), args.speech)
'''
    (scripts_dir / "export_audacity_labels.py").write_text(code, encoding="utf-8")
    print("Wrote scripts/export_audacity_labels.py")


def main() -> None:
    dataset_dir = Path("data/dataset")
    library_dir = Path("data/library")
    dist_dir = Path("dist")
    sample_dir = Path("data/sample")
    scripts_dir = Path("scripts")

    # Datacard
    datacard = build_datacard(dataset_dir, library_dir)
    (dataset_dir / "DATACARD.md").write_text(datacard, encoding="utf-8")
    print("Wrote DATACARD.md")

    # Sample
    build_sample(dataset_dir, sample_dir)

    # Zip + checksums
    build_zip(dataset_dir, library_dir, dist_dir)

    # Audacity script
    write_audacity_script(scripts_dir)

    print("Release package complete.")


if __name__ == "__main__":
    main()