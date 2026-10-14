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
                lines.append(f"{flaw['start']:.6f}\t{flaw['end']:.6f}\t{label}")
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audacity labels written to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", default=None)
    ap.add_argument("--out", default="./audacity_labels")
    args = ap.parse_args()
    export_labels(Path("data/dataset"), Path(args.out), args.speech)
