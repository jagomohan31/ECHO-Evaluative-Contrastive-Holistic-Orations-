"""
baseline/build.py -- Build baseline models from library (DESIGN 8).

Produces:
  data/library/<speech_id>/baseline.json  -- per-speech baseline (8.1, 8.5)
  data/library/norms.json                 -- population norms (8.2)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from speechlens.config import SR
from speechlens.features.frame import compute_frame_features
from speechlens.features.word import (
    compute_word_features,
    compute_prominence,
    classify_pauses,
)

# DESIGN 8.2 conservative defaults (used when fewer than 2 multi-reader speeches)
CONSERVATIVE_DEFAULTS = {
    "ln_share_sd": 0.18,
    "pause_sd": 0.20,
    "f0_mean_sd": 1.5,
    "f0_slope_sd": 4.0,
    "en_mean_sd": 3.0,
    "en_slope_sd": 8.0,
    "prominence_sd": 0.8,
    "hnr_sd": 3.0,
    "tilt_sd": 4.0,
    "sigma_D": 0.12,
    "sigma_win": 0.8,
}

# DESIGN 8.3
SIGMA_FLOOR_MULT = 1.0


def _load_align(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_word_features(flac: Path, align: dict) -> list[dict]:
    """Return list of per-word feature dicts for a reader."""
    samples, sr = sf.read(flac)
    samples = samples.astype(np.float32)
    if sr != SR:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=SR)

    words = align["words"]
    pauses = align.get("pauses", [])
    total_dur = align.get("duration_s", len(samples) / SR)

    ff = compute_frame_features(samples, SR)
    wfs = compute_word_features(words, pauses, ff, total_dur)
    wfs = compute_prominence(wfs)

    return [
        {
            "i": w.i,
            "text": w.text,
            "dur": w.dur,
            "nsyl": w.nsyl,
            "share": w.share,
            "pause_before": w.pause_before,
            "f0_mean_st": w.f0_mean_st,
            "f0_slope": w.f0_slope,
            "en_mean_rel": w.en_mean_rel,
            "en_slope": w.en_slope,
            "prominence": w.prominence,
            "hnr_rel": w.hnr_rel,
            "tilt_rel": w.tilt_rel,
            "f0_sd_st": w.f0_sd_st,
            "art_rate": w.art_rate,
        }
        for w in wfs
    ]


# Features included in the per-word baseline (DESIGN 8.1)
_WORD_FEATURES = [
    "share", "pause_before", "f0_mean_st", "f0_slope",
    "en_mean_rel", "en_slope", "prominence", "hnr_rel", "tilt_rel",
]
_WIN_FEATURES = ["f0_sd_st"]


def _build_speech_baseline(
    speech_dir: Path,
    baseline_reader_ids: list[str],
) -> dict:
    """
    Build the per-speech baseline from baseline_readers.
    Returns a dict ready for baseline.json.
    """
    # Collect per-word feature matrices: list of lists of dicts (one per reader)
    readers_data: list[list[dict]] = []
    for rid in baseline_reader_ids:
        flac = speech_dir / "readers" / f"{rid}.flac"
        align_path = speech_dir / "readers" / f"{rid}.align.json"
        if not flac.exists() or not align_path.exists():
            continue
        align = _load_align(align_path)
        if not align:
            continue
        wf = _extract_word_features(flac, align)
        readers_data.append(wf)

    if not readers_data:
        return {"error": "no baseline readers found", "words": []}

    n_readers = len(readers_data)
    n_words = min(len(rd) for rd in readers_data)

    # Compute mean articulation rate
    artic_rates = []
    for rd in readers_data:
        if rd:
            durs = [w["dur"] for w in rd]
            syls = [w["nsyl"] for w in rd]
            total_speaking = sum(durs)
            total_syl = sum(syls)
            if total_speaking > 0:
                artic_rates.append(total_syl / total_speaking)
    mu_artic_rate = float(np.mean(artic_rates)) if artic_rates else 4.0

    # Per-word mu and sigma (DESIGN 8.1)
    word_baselines: list[dict] = []
    for wi in range(n_words):
        wb: dict[str, Any] = {"i": readers_data[0][wi]["i"], "text": readers_data[0][wi]["text"]}
        for feat in _WORD_FEATURES:
            vals = []
            for rd in readers_data:
                v = rd[wi].get(feat, float("nan"))
                if not np.isnan(v):
                    vals.append(v)
            wb[f"mu_{feat}"] = float(np.mean(vals)) if vals else 0.0
            wb[f"sigma_{feat}"] = (
                float(np.std(vals, ddof=1)) if n_readers >= 3 and len(vals) >= 3 else 0.0
            )
        # Window feature f0_sd_st
        win_vals = [rd[wi].get("f0_sd_st", float("nan")) for rd in readers_data]
        win_valid = [v for v in win_vals if not np.isnan(v)]
        wb["mu_f0_sd_st"] = float(np.mean(win_valid)) if win_valid else 0.0
        wb["sigma_f0_sd_st"] = (
            float(np.std(win_valid, ddof=1)) if n_readers >= 3 and len(win_valid) >= 3 else 0.0
        )
        word_baselines.append(wb)

    return {
        "n_baseline_readers": n_readers,
        "artic_rate": mu_artic_rate,
        "words": word_baselines,
        "note": "" if n_readers >= 3 else "fewer than 3 readers: sigma=0, use population norms",
    }


def _build_norms(
    library_dir: Path,
    multi_reader_speeches: list[str],
) -> dict:
    """
    Build population norms (DESIGN 8.2) from multi-reader speeches.
    Falls back to conservative defaults if fewer than 2 multi-reader speeches.
    """
    if len(multi_reader_speeches) < 2:
        note = (
            "fewer than 2 multi-reader speeches; using conservative defaults "
            f"(DESIGN 8.2). Multi-reader speeches found: {multi_reader_speeches}"
        )
        return {"conservative_defaults": True, "note": note, "artic_rate_mean": 4.0, "artic_rate_sd": 0.8, **CONSERVATIVE_DEFAULTS}

    # Collect all reader data across multi-reader speeches for leave-one-reader-out norms
    all_diff_dicts: list[dict[str, list[float]]] = []  # one per reader
    sigma_pool_accumulator: dict[str, list[float]] = {f: [] for f in _WORD_FEATURES + ["f0_sd_st"]}

    # For each multi-reader speech: leave-one-out
    for speech_id in multi_reader_speeches:
        speech_dir = library_dir / speech_id
        meta_path = speech_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        baseline_ids = meta.get("baseline_readers", [])
        if len(baseline_ids) < 2:
            continue

        # Get all reader data
        all_rd: dict[str, list[dict]] = {}
        for rid in baseline_ids:
            flac = speech_dir / "readers" / f"{rid}.flac"
            ap = speech_dir / "readers" / f"{rid}.align.json"
            if flac.exists() and ap.exists():
                align = _load_align(ap)
                if align:
                    all_rd[rid] = _extract_word_features(flac, align)

        if len(all_rd) < 2:
            continue

        n_words = min(len(v) for v in all_rd.values())

        # Leave-one-out: for each reader, build baseline from others, compute d_i
        for held_out, held_data in all_rd.items():
            other_ids = [r for r in all_rd if r != held_out]
            if not other_ids:
                continue

            # Per-word within-word SD from other readers
            for feat in _WORD_FEATURES:
                vals_across = []
                for wi in range(n_words):
                    reader_vals = [all_rd[r][wi].get(feat, float("nan")) for r in other_ids
                                   if not np.isnan(all_rd[r][wi].get(feat, float("nan")))]
                    vals_across.extend(reader_vals)
                if len(vals_across) >= 2:
                    sigma_pool_accumulator[feat].append(float(np.std(vals_across, ddof=1)))

    # sigma_pool: pooled within-word SD
    sigma_pool: dict[str, float] = {}
    for feat, sds in sigma_pool_accumulator.items():
        if sds:
            sigma_pool[f"sigma_pool_{feat}"] = float(np.mean(sds))
        else:
            sigma_pool[f"sigma_pool_{feat}"] = CONSERVATIVE_DEFAULTS.get(f"{feat}_sd", 0.1)

    # Simplified norms (full DESIGN 8.2 items)
    norms = {
        "conservative_defaults": False,
        "note": f"derived from {len(multi_reader_speeches)} multi-reader speeches",
        "sigma_D": CONSERVATIVE_DEFAULTS["sigma_D"],  # will be computed properly in T15
        "sigma_win": CONSERVATIVE_DEFAULTS["sigma_win"],
        "artic_rate_mean": 4.0,
        "artic_rate_sd": 0.8,
        **sigma_pool,
    }
    return norms


def build(library_dir: Path) -> None:
    """Build all speech baselines and population norms. Writes JSON files."""
    library_dir = Path(library_dir)
    speech_dirs = sorted(library_dir.glob("speech-*"))
    multi_reader_speeches: list[str] = []

    for speech_dir in speech_dirs:
        meta_path = speech_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        baseline_ids = meta.get("baseline_readers", [])
        if not baseline_ids:
            baseline_ids = [meta["readers"][0]["id"]]

        print(f"Building baseline for {speech_dir.name} ({len(baseline_ids)} readers)...")
        baseline = _build_speech_baseline(speech_dir, baseline_ids)
        baseline_path = speech_dir / "baseline.json"
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"  -> {len(baseline.get('words', []))} words, "
              f"artic_rate={baseline.get('artic_rate', 0):.2f} syl/s")

        if len(baseline_ids) >= 3:
            multi_reader_speeches.append(speech_dir.name)

    print(f"\nBuilding population norms ({len(multi_reader_speeches)} multi-reader speeches)...")
    norms = _build_norms(library_dir, multi_reader_speeches)
    norms_path = library_dir / "norms.json"
    norms_path.write_text(json.dumps(norms, indent=2), encoding="utf-8")
    print(f"  conservative_defaults={norms['conservative_defaults']}")
    print(f"  note: {norms['note']}")


def main() -> None:
    build(Path("data/library"))
    print("Baseline build complete.")