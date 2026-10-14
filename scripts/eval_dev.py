"""
scripts/eval_dev.py -- Run evaluation on the dev split and output results/dev_pre_calibration.json (T11).
"""
from __future__ import annotations
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
import numpy as np
import soundfile as sf

from speechlens.config import SR
from speechlens.features.frame import compute_frame_features
from speechlens.features.word import compute_word_features
from speechlens.detect import detect_all
from speechlens.dataset.remap import remap_words_and_pauses
from speechlens.evalsuite.grounding import match_file_regions, compute_metrics


def _process_single_file(args: tuple) -> dict[str, Any]:
    file_path_str, entry, base_words, base_pauses, baseline, norms = args
    file_path = Path(file_path_str)
    if not file_path.exists():
        return {"tp": 0, "fp": 0, "fn": len(entry.get("flaws", [])), "ious": [], "by_family": {}}

    y, sr = sf.read(file_path, dtype="float32")
    if sr != SR:
        # Resample if needed
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    dur = len(y) / SR
    ff = compute_frame_features(y, SR)

    flaws = entry.get("flaws", [])
    remapped_words, remapped_pauses = remap_words_and_pauses(base_words, base_pauses, flaws)
    pause_dicts = [{"start": p.start, "end": p.end, "before_word": p.before_word,
                    "after_word": p.after_word, "kind": p.kind} for p in remapped_pauses]

    word_feats = compute_word_features(remapped_words, pause_dicts, ff, dur)
    predicted_regions = detect_all(word_feats, remapped_pauses, baseline, norms, ff=ff)

    match_result = match_file_regions(predicted_regions, flaws, tau=0.3)
    return match_result


def evaluate_dev_split(
    library_dir: Path = Path("data/library"),
    dataset_dir: Path = Path("data/dataset"),
    out_path: Path = Path("results/dev_pre_calibration.json"),
    max_workers: int = 8,
) -> dict[str, Any]:
    splits_path = library_dir / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else {}
    dev_speeches = [sid for sid, split in splits.items() if split == "dev"]
    if not dev_speeches:
        dev_speeches = ["speech-001", "speech-004"]

    norms_path = library_dir / "norms.json"
    norms = json.loads(norms_path.read_text(encoding="utf-8"))

    tasks = []
    file_entries = []

    for sid in dev_speeches:
        speech_dir = library_dir / sid
        meta_path = speech_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mirror_id = meta.get("mirror_source") or meta["readers"][0]["id"]

        align_path = speech_dir / "readers" / f"{mirror_id}.align.json"
        align_data = json.loads(align_path.read_text(encoding="utf-8"))
        base_words = align_data.get("words", [])
        base_pauses = align_data.get("pauses", [])

        baseline_path = speech_dir / "baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        labels_path = dataset_dir / sid / "labels.json"
        if not labels_path.exists():
            continue
        labels = json.loads(labels_path.read_text(encoding="utf-8"))

        for entry in labels:
            fpath = entry["file"]
            tasks.append((fpath, entry, base_words, base_pauses, baseline, norms))
            file_entries.append(entry)

    print(f"Evaluating {len(tasks)} dev split files across {max_workers} workers...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_single_file, tasks))

    elapsed = time.time() - t0
    print(f"Completed processing {len(tasks)} files in {elapsed:.1f}s.")

    metrics = compute_metrics(results)
    metrics["dev_speeches"] = dev_speeches
    metrics["n_files"] = len(tasks)
    metrics["elapsed_s"] = round(elapsed, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print("Dev Split Pre-Calibration Metrics (IoU >= 0.3):")
    print("=" * 65)
    print(f"{'Family':<22} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'Mean IoU':<9}")
    print("-" * 65)
    for fam, d in sorted(metrics.get("per_family", {}).items()):
        print(f"{fam:<22} | {d['precision']:<9.3f} | {d['recall']:<9.3f} | {d['f1']:<9.3f} | {d['mean_iou']:<9.3f}")
    print("-" * 65)
    ov = metrics.get("overall", {})
    print(f"{'OVERALL':<22} | {ov['precision']:<9.3f} | {ov['recall']:<9.3f} | {ov['f1']:<9.3f} | {ov['mean_iou']:<9.3f}")
    print("=" * 65 + "\n")

    return metrics


if __name__ == "__main__":
    evaluate_dev_split()
