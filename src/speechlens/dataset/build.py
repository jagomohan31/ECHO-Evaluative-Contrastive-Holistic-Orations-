"""
dataset/build.py -- Build the contrastive dataset (DESIGN 5.5-5.7).

`python -m speechlens dataset build` loops speech x family x level,
injects into the mirror_source reader, writes FLAC, re-aligns every output,
derives labels, runs the analytic remap cross-check, and writes:
  data/dataset/<speech_id>/labels.json
  data/dataset/manifest.csv
  data/dataset/validation_report.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from speechlens.config import SR
from speechlens.inject import (
    apply_gain, apply_clipping, apply_trailing_off, apply_muffled,
    apply_pause_missing, apply_pause_excess, apply_hesitation_filled,
    apply_pace, apply_monotone, apply_terminal_contour, select_spans,
)
from speechlens.features.word import compute_word_features, classify_pauses
from speechlens.features.frame import compute_frame_features

# Families and their dimensions (DESIGN 5.5 / 10.1)
FAMILY_DIM: dict[str, str] = {
    "pace_fast": "pacing", "pace_slow": "pacing",
    "pause_missing": "pacing", "monotone": "intonation",
    "terminal_contour": "intonation", "stress_missed": "intonation",
    "pause_excess": "fluency", "hesitation_filled": "fluency",
    "volume_low": "volume", "clipping": "volume", "trailing_off": "volume",
    "muffled": "clarity",
}

# DESIGN 5.5 parameters per (family, level)
_PARAMS: dict[tuple[str, int], dict] = {
    ("pace_fast",  1): {"factor": 1.15}, ("pace_fast",  2): {"factor": 1.40},
    ("pace_fast",  3): {"factor": 1.70},
    ("pace_slow",  1): {"factor": 0.85}, ("pace_slow",  2): {"factor": 0.70},
    ("pace_slow",  3): {"factor": 0.55},
    ("pause_missing", 1): {"target_s": None, "ratio": 0.50},
    ("pause_missing", 2): {"target_s": 0.060, "ratio": None},
    ("pause_missing", 3): {"target_s": 0.060, "ratio": None},
    ("pause_excess",  1): {"dur_s": 0.45},
    ("pause_excess",  2): {"dur_s": 0.80},
    ("pause_excess",  3): {"dur_s": 1.20},
    ("hesitation_filled", 1): {"dur_s": 0.35},
    ("hesitation_filled", 2): {"dur_s": 0.45},
    ("hesitation_filled", 3): {"dur_s": 0.60},
    ("monotone",    1): {"alpha": 0.60}, ("monotone",    2): {"alpha": 0.35},
    ("monotone",    3): {"alpha": 0.10},
    ("terminal_contour", 1): {"rise_st": 2.5}, ("terminal_contour", 2): {"rise_st": 4.0},
    ("terminal_contour", 3): {"rise_st": 6.0},
    ("stress_missed", 1): {"n_words": 1}, ("stress_missed", 2): {"n_words": 3},
    ("stress_missed", 3): {"n_words": 6},
    ("volume_low",  1): {"gain_db": -4.0}, ("volume_low",  2): {"gain_db": -8.0},
    ("volume_low",  3): {"gain_db": -14.0},
    ("clipping",    1): {"gain_db": 6.0},  ("clipping",    2): {"gain_db": 10.0},
    ("clipping",    3): {"gain_db": 16.0},
    ("trailing_off", 1): {"gain_db": -6.0}, ("trailing_off", 2): {"gain_db": -10.0},
    ("trailing_off", 3): {"gain_db": -16.0},
    ("muffled",     1): {"cutoff_hz": 5000}, ("muffled",     2): {"cutoff_hz": 3500},
    ("muffled",     3): {"cutoff_hz": 2500},
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_align(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _median_f0(samples: np.ndarray) -> float:
    ff = compute_frame_features(samples)
    voiced = ff.f0_hz[ff.f0_hz > 0]
    return float(np.median(voiced)) if len(voiced) > 0 else 150.0


def _rms_level(samples: np.ndarray, sr: int) -> float:
    ff = compute_frame_features(samples, sr)
    voiced_mask = ff.f0_hz > 0
    if voiced_mask.any():
        # voiced frames
        hop = int(0.010 * sr)
        voiced_rms = []
        for i in np.where(voiced_mask)[0]:
            chunk = samples[i * hop: (i + 1) * hop]
            if len(chunk) > 0:
                voiced_rms.append(float(np.sqrt(np.mean(chunk ** 2))))
        if voiced_rms:
            return float(np.median(voiced_rms))
    return float(np.sqrt(np.mean(samples ** 2))) + 1e-9


def _inject(
    samples: np.ndarray,
    sr: int,
    family: str,
    level: int,
    words_align: list[dict],
    file_id: str,
) -> tuple[np.ndarray, list[dict]]:
    """
    Apply the flaw and return (flawed_samples, flaw_records).
    flaw_record keys: family, dimension, word_idx, start, end, params.
    """
    params = _PARAMS.get((family, level), {})
    spans_obj = select_spans(words_align, family, level)
    if not spans_obj:
        return samples.copy(), []

    # Convert Span objects -> time spans
    def _word_span(span):
        ws = words_align[span.start_word]["start"]
        we = words_align[span.end_word]["end"]
        return ws, we

    time_spans = [_word_span(s) for s in spans_obj]
    word_idx_pairs = [(s.start_word, s.end_word) for s in spans_obj]

    result = samples.copy()
    flaw_records = []

    if family in ("pace_fast", "pace_slow"):
        factor = params["factor"]
        result = apply_pace(result, sr, time_spans, factor)
        for (wi_s, wi_e), (ts, te) in zip(word_idx_pairs, time_spans):
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [wi_s, wi_e], "start": ts, "end": te,
                                  "params": params})

    elif family == "pause_missing":
        # Find pauses at span positions
        pauses_in_span = []
        for ts, te in time_spans:
            for w in words_align:
                if ts <= w["start"] <= te:
                    pb = w["start"] - (words_align[w["i"] - 1]["end"] if w["i"] > 0 else 0)
                    if pb >= 0.25:
                        pauses_in_span.append((w["start"] - pb, w["start"]))
        if not pauses_in_span:
            pauses_in_span = [(ts, ts + 0.3) for ts, te in time_spans[:1]]
        ratio = params.get("ratio")
        target_s_abs = params.get("target_s")
        for ps, pe in pauses_in_span:
            orig_dur = pe - ps
            if ratio is not None:
                tgt = max(0.060, orig_dur * ratio)
            else:
                tgt = target_s_abs
            result = apply_pause_missing(result, sr, [(ps, pe)], tgt)
        flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                              "word_idx": [spans_obj[0].start_word, spans_obj[-1].end_word],
                              "start": time_spans[0][0], "end": time_spans[-1][1],
                              "params": params})

    elif family == "pause_excess":
        dur_s = params["dur_s"]
        positions = [te for ts, te in time_spans]
        result = apply_pause_excess(result, sr, positions, dur_s, file_id=file_id)
        for pos, (ts, te) in zip(positions, time_spans):
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [spans_obj[0].start_word, spans_obj[0].end_word],
                                  "start": pos, "end": pos + dur_s, "params": params})

    elif family == "hesitation_filled":
        dur_s = params["dur_s"]
        f0_hz = _median_f0(samples)
        rms = _rms_level(samples, sr)
        positions = [te for ts, te in time_spans]
        result = apply_hesitation_filled(result, sr, positions, dur_s, f0_hz, rms, file_id)
        for pos in positions:
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [spans_obj[0].start_word, spans_obj[0].end_word],
                                  "start": pos, "end": pos + dur_s, "params": params})

    elif family == "monotone":
        alpha = params["alpha"]
        result = apply_monotone(result, sr, time_spans, alpha)
        for (wi_s, wi_e), (ts, te) in zip(word_idx_pairs, time_spans):
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [wi_s, wi_e], "start": ts, "end": te,
                                  "params": params})

    elif family == "terminal_contour":
        rise_st = params["rise_st"]
        word_ends = [te for ts, te in time_spans]
        result = apply_terminal_contour(result, sr, word_ends, rise_st)
        for (wi_s, wi_e), (ts, te) in zip(word_idx_pairs, time_spans):
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [wi_s, wi_e], "start": ts, "end": te,
                                  "params": params})

    elif family == "stress_missed":
        from speechlens.inject.tempo import apply_stress_missed
        n_words_target = params.get("n_words", 1)
        # Use first n_words_target spans
        used = time_spans[:n_words_target]
        used_wi = word_idx_pairs[:n_words_target]
        result = apply_stress_missed(result, sr, used)
        for (wi_s, wi_e), (ts, te) in zip(used_wi, used):
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [wi_s, wi_e], "start": ts, "end": te,
                                  "params": params})

    elif family == "volume_low":
        gain_db = params["gain_db"]
        result = apply_gain(result, sr, time_spans, gain_db)
        for (wi_s, wi_e), (ts, te) in zip(word_idx_pairs, time_spans):
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [wi_s, wi_e], "start": ts, "end": te,
                                  "params": params})

    elif family == "clipping":
        gain_db = params["gain_db"]
        result = apply_clipping(result, sr, time_spans, gain_db)
        for (wi_s, wi_e), (ts, te) in zip(word_idx_pairs, time_spans):
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [wi_s, wi_e], "start": ts, "end": te,
                                  "params": params})

    elif family == "trailing_off":
        gain_db = params["gain_db"]
        result = apply_trailing_off(result, sr, time_spans, gain_db)
        for (wi_s, wi_e), (ts, te) in zip(word_idx_pairs, time_spans):
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [wi_s, wi_e], "start": ts, "end": te,
                                  "params": params})

    elif family == "muffled":
        cutoff_hz = params["cutoff_hz"]
        result = apply_muffled(result, sr, time_spans, float(cutoff_hz))
        for (wi_s, wi_e), (ts, te) in zip(word_idx_pairs, time_spans):
            flaw_records.append({"family": family, "dimension": FAMILY_DIM[family],
                                  "word_idx": [wi_s, wi_e], "start": ts, "end": te,
                                  "params": params})

    return result, flaw_records


def _align_output(samples: np.ndarray, sr: int, transcript_path: Path) -> dict | None:
    """Re-align flawed audio. Returns align dict or None on failure."""
    try:
        from speechlens.align.ctc import align
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        return align(samples, sr, transcript)
    except Exception as e:
        print(f"    [align error] {e}")
        return None


def build(
    library_dir: Path,
    dataset_dir: Path,
    families: list[str] | None = None,
    levels: list[int] | None = None,
    speeches: list[str] | None = None,
    skip_align: bool = False,
) -> dict:
    """
    Build the contrastive dataset.
    Returns stats dict with n_files, n_errors.
    """
    if families is None:
        families = list(FAMILY_DIM.keys())
    if levels is None:
        levels = [1, 2, 3]

    dataset_dir.mkdir(parents=True, exist_ok=True)

    all_labels: dict[str, list] = {}
    manifest_rows: list[dict] = []
    validation_issues: list[dict] = []
    n_files = 0
    n_errors = 0

    speech_dirs = sorted(library_dir.glob("speech-*"))
    if speeches:
        speech_dirs = [d for d in speech_dirs if d.name in speeches]

    for speech_dir in speech_dirs:
        speech_id = speech_dir.name
        meta_path = speech_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        mirror_id = meta.get("mirror_source") or meta["readers"][0]["id"]
        leaky = meta.get("leaky_baseline", False)
        split = meta.get("split", "dev")

        mirror_flac = speech_dir / "readers" / f"{mirror_id}.flac"
        mirror_align_path = speech_dir / "readers" / f"{mirror_id}.align.json"
        transcript_path = speech_dir / "transcript.txt"

        if not mirror_flac.exists():
            print(f"[skip] {speech_id}: mirror {mirror_id} not found")
            continue

        samples, sr_file = sf.read(mirror_flac)
        samples = samples.astype(np.float32)
        if sr_file != SR:
            import librosa
            samples = librosa.resample(samples, orig_sr=sr_file, target_sr=SR)
        sr_use = SR

        align_data = _load_align(mirror_align_path) if mirror_align_path.exists() else None
        words_align = align_data["words"] if align_data else []

        out_dir = dataset_dir / speech_id
        out_dir.mkdir(parents=True, exist_ok=True)
        labels_list: list[dict] = all_labels.setdefault(speech_id, [])

        # L0 baseline rows (all readers unmodified)
        for reader in meta["readers"]:
            rid = reader["id"]
            src_flac = speech_dir / "readers" / f"{rid}.flac"
            if not src_flac.exists():
                continue
            role = "mirror_source" if rid == mirror_id else "baseline"
            row = {
                "file": str(src_flac),
                "speech_id": speech_id,
                "reader_id": rid,
                "role": role,
                "level": 0,
                "families": "",
                "n_flaws": 0,
                "duration_s": round(sf.info(src_flac).duration, 3),
                "split": split,
                "sha256": _sha256_file(src_flac),
                "leaky_baseline": leaky,
            }
            manifest_rows.append(row)
            labels_list.append({
                "file": str(src_flac),
                "speech_id": speech_id,
                "reader_id": rid,
                "level": 0,
                "split": split,
                "leaky_baseline": leaky,
                "synthetic_flaw": False,
                "flaws": [],
            })

        # Injected files
        for family in families:
            for level in levels:
                file_id = f"{speech_id}_{mirror_id}_{family}_L{level}"
                out_flac = out_dir / f"{file_id}.flac"

                try:
                    flawed, flaw_records = _inject(
                        samples, sr_use, family, level, words_align, file_id
                    )
                except Exception as e:
                    print(f"  [inject error] {file_id}: {e}")
                    n_errors += 1
                    continue

                sf.write(out_flac, flawed, SR, subtype="PCM_16")
                sha = _sha256_file(out_flac)

                # Re-align
                new_align = None
                if not skip_align and transcript_path.exists():
                    print(f"  aligning {file_id}...", end=" ", flush=True)
                    new_align = _align_output(flawed, SR, transcript_path)
                    if new_align:
                        align_out = out_flac.with_suffix(".align.json")
                        align_out.write_text(json.dumps(new_align, indent=2), encoding="utf-8")
                        print("ok")
                    else:
                        print("failed")

                # Analytic time remap cross-check (DESIGN 5.7.4)
                remap_disagreement_ms = _remap_cross_check(
                    flaw_records, new_align, words_align
                )
                if remap_disagreement_ms > 60.0:
                    validation_issues.append({
                        "file": str(out_flac),
                        "remap_disagreement_ms": remap_disagreement_ms,
                    })

                dur = len(flawed) / SR
                labels_list.append({
                    "file": str(out_flac),
                    "speech_id": speech_id,
                    "reader_id": mirror_id,
                    "level": level,
                    "split": split,
                    "leaky_baseline": leaky,
                    "synthetic_flaw": True,
                    "flaws": flaw_records,
                })

                manifest_rows.append({
                    "file": str(out_flac),
                    "speech_id": speech_id,
                    "reader_id": mirror_id,
                    "role": "injected",
                    "level": level,
                    "families": family,
                    "n_flaws": len(flaw_records),
                    "duration_s": round(dur, 3),
                    "split": split,
                    "sha256": sha,
                    "leaky_baseline": leaky,
                })
                n_files += 1
                print(f"  {file_id}: {len(flaw_records)} flaws, {dur:.1f}s, remap={remap_disagreement_ms:.0f}ms")

        # Write per-speech labels.json
        labels_path = out_dir / "labels.json"
        labels_path.write_text(json.dumps(labels_list, indent=2), encoding="utf-8")

    # Write manifest.csv
    manifest_path = dataset_dir / "manifest.csv"
    if speeches and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            other_rows = [r for r in reader if r.get("speech_id") not in speeches]
        manifest_rows = other_rows + manifest_rows
        manifest_rows.sort(key=lambda r: (r["speech_id"], r["role"], int(r.get("level", 0)), r.get("families", "")))

    fieldnames = ["file", "speech_id", "reader_id", "role", "level", "families",
                  "n_flaws", "duration_s", "split", "sha256", "leaky_baseline"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    # Write validation_report.json
    val_report = {
        "n_files": len(manifest_rows),
        "n_injected": len([r for r in manifest_rows if r.get("role") == "injected"]),
        "n_errors": n_errors,
        "remap_flags": validation_issues,
        "error_count": len(validation_issues),
    }
    (dataset_dir / "validation_report.json").write_text(
        json.dumps(val_report, indent=2), encoding="utf-8"
    )

    print(f"\nDataset build complete: {n_files} injected files, {n_errors} errors, "
          f"{len(validation_issues)} remap flags.")
    return val_report


def _remap_cross_check(
    flaw_records: list[dict],
    new_align: dict | None,
    original_words: list[dict],
) -> float:
    """
    Compute mean disagreement in ms between analytic remap and realigned timestamps.
    Returns 0.0 if alignment unavailable.
    """
    if not new_align or not flaw_records or not original_words:
        return 0.0
    new_words = new_align.get("words", [])
    if len(new_words) == 0:
        return 0.0

    diffs: list[float] = []
    for rec in flaw_records:
        wi_s, wi_e = rec["word_idx"]
        # Compare start of first word and end of last word
        if wi_s < len(new_words):
            diffs.append(abs(new_words[wi_s]["start"] - rec["start"]) * 1000)
        if wi_e < len(new_words):
            diffs.append(abs(new_words[wi_e]["end"] - rec["end"]) * 1000)

    return float(np.mean(diffs)) if diffs else 0.0

def main() -> None:
    """Entry point for `python -m speechlens dataset build`."""
    from pathlib import Path
    library_dir = Path("data/library")
    dataset_dir = Path("data/dataset")
    stats = build(library_dir, dataset_dir)
    print(f"n_injected={stats['n_injected']}, n_errors={stats['n_errors']}")