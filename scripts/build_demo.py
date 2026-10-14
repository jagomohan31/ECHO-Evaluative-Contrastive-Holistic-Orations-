"""scripts/build_demo.py -- Build offline demo assets for SpeechLens dashboard.

Picks at least 8 clips with leaky_baseline: false (L0, L1, L2, L3 of two families,
plus clipping and monotone), copies audio (<= 15 MB total) and precomputed analysis
result JSON to data/demo/ and web/public/demo/, and writes demo/index.json.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "dataset"
LIBRARY_DIR = REPO_ROOT / "data" / "library"
DATA_DEMO_DIR = REPO_ROOT / "data" / "demo"
WEB_DEMO_DIR = REPO_ROOT / "web" / "public" / "demo"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
MAX_TOTAL_AUDIO_MB = 15.0


def select_demo_clips() -> list[dict]:
    """Deterministically select demo clips satisfying T05 criteria."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # All non-leaky clips
    non_leaky = [r for r in rows if r["leaky_baseline"] == "False"]

    selected: list[dict] = []

    # 1. L0 clean clip (mirror_source)
    l0_clips = [r for r in non_leaky if r["role"] == "mirror_source" and r["level"] == "0"]
    if not l0_clips:
        l0_clips = [r for r in non_leaky if r["level"] == "0"]
    if l0_clips:
        c = l0_clips[0]
        selected.append({
            "id": "speech-001_clean_L0",
            "file": c["file"],
            "speech_id": c["speech_id"],
            "family": "ideal",
            "level": 0,
            "label": "Ideal Baseline (Level 0)",
        })

    # 2. Family 1: pace_fast (L1, L2, L3)
    for lvl in ["1", "2", "3"]:
        matching = [r for r in non_leaky if r["families"] == "pace_fast" and r["level"] == lvl]
        if matching:
            c = matching[0]
            selected.append({
                "id": f"speech-001_pace_fast_L{lvl}",
                "file": c["file"],
                "speech_id": c["speech_id"],
                "family": "pace_fast",
                "level": int(lvl),
                "label": f"Fast Pacing (Level {lvl})",
            })

    # 3. Family 2: pace_slow (L1, L2, L3)
    for lvl in ["1", "2", "3"]:
        matching = [r for r in non_leaky if r["families"] == "pace_slow" and r["level"] == lvl]
        if matching:
            c = matching[0]
            selected.append({
                "id": f"speech-001_pace_slow_L{lvl}",
                "file": c["file"],
                "speech_id": c["speech_id"],
                "family": "pace_slow",
                "level": int(lvl),
                "label": f"Slow Pacing (Level {lvl})",
            })

    # 4. Clipping example (L2 or first available)
    clip_matching = [r for r in non_leaky if r["families"] == "clipping" and r["level"] == "2"]
    if not clip_matching:
        clip_matching = [r for r in non_leaky if r["families"] == "clipping"]
    if clip_matching:
        c = clip_matching[0]
        selected.append({
            "id": f"speech-001_clipping_L{c['level']}",
            "file": c["file"],
            "speech_id": c["speech_id"],
            "family": "clipping",
            "level": int(c["level"]),
            "label": f"Audio Clipping (Level {c['level']})",
        })

    # 5. Monotone example (L2 or first available)
    mono_matching = [r for r in non_leaky if r["families"] == "monotone" and r["level"] == "2"]
    if not mono_matching:
        mono_matching = [r for r in non_leaky if r["families"] == "monotone"]
    if mono_matching:
        c = mono_matching[0]
        selected.append({
            "id": f"speech-001_monotone_L{c['level']}",
            "file": c["file"],
            "speech_id": c["speech_id"],
            "family": "monotone",
            "level": int(c["level"]),
            "label": f"Monotone Pitch (Level {c['level']})",
        })

    return selected


def build_demo_assets():
    """Build assets and precompute pipeline outputs."""
    from speechlens.pipeline import analyze

    DATA_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DEMO_DIR.mkdir(parents=True, exist_ok=True)

    clips = select_demo_clips()
    print(f"Selected {len(clips)} demo clips.")

    total_audio_bytes = 0
    index_entries = []

    for clip in clips:
        src_path = REPO_ROOT / clip["file"]
        if not src_path.exists():
            print(f"Warning: audio file missing: {src_path}")
            continue

        clip_id = clip["id"]
        dst_audio_name = f"{clip_id}.flac"
        dst_json_name = f"{clip_id}.json"

        # Copy audio
        data_audio_path = DATA_DEMO_DIR / dst_audio_name
        web_audio_path = WEB_DEMO_DIR / dst_audio_name
        shutil.copy2(src_path, data_audio_path)
        shutil.copy2(src_path, web_audio_path)

        audio_size = src_path.stat().st_size
        total_audio_bytes += audio_size

        # Precompute analysis result
        data_json_path = DATA_DEMO_DIR / dst_json_name
        web_json_path = WEB_DEMO_DIR / dst_json_name

        print(f"Analyzing {clip_id}...")
        result_dict = analyze(
            audio_path=src_path,
            speech_id=clip["speech_id"],
            transcript_text=None,
        )

        with open(data_json_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        with open(web_json_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

        index_entries.append({
            "id": clip_id,
            "label": clip["label"],
            "level": clip["level"],
            "family": clip["family"],
            "audio_url": f"/demo/{dst_audio_name}",
            "result_url": f"/demo/{dst_json_name}",
        })

    total_mb = total_audio_bytes / (1024 * 1024)
    print(f"Total audio size: {total_mb:.2f} MB (limit: {MAX_TOTAL_AUDIO_MB} MB)")
    if total_mb > MAX_TOTAL_AUDIO_MB:
        raise ValueError(f"Audio size {total_mb:.2f} MB exceeds {MAX_TOTAL_AUDIO_MB} MB limit!")

    # Write index.json in data/demo, web/public/demo, and demo/
    index_json_str = json.dumps(index_entries, indent=2)
    (DATA_DEMO_DIR / "index.json").write_text(index_json_str, encoding="utf-8")
    (WEB_DEMO_DIR / "index.json").write_text(index_json_str, encoding="utf-8")
    (REPO_ROOT / "demo").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "demo" / "index.json").write_text(index_json_str, encoding="utf-8")

    print(f"Demo assets built successfully: {len(index_entries)} clips.")


if __name__ == "__main__":
    build_demo_assets()
