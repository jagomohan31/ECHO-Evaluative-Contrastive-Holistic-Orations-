"""
evalsuite/calibrate.py -- T14 Grid-search calibration (DESIGN §13, TASK_1 T14).

Grid over:
  Z_TOL            ∈ {1.25, 1.5, 1.75, 2.0}
  SIGMA_FLOOR_MULT ∈ {0.75, 1.0, 1.25}
  K                ∈ {0.5, 0.65, 0.8, 0.95}

Strategy (single-process, fast):
  Phase 1 – extract features once per file (remap for synthetic, CTC for 4 clean readers).
  Phase 2 – for each (z_tol, sigma_floor), re-run detect_all on cached objects (~10ms/file).
  Phase 3 – evaluate macro-F1; pick best config + K.

Saves: config/calibrated.json

Usage:
  python -m speechlens.evalsuite.calibrate            # full 4×3×4 grid
  python -m speechlens.evalsuite.calibrate --dry-run  # 2×2×2 smoke test
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from speechlens.evalsuite.grounding import match_file_regions, compute_metrics
from speechlens.dataset.remap import remap_words_and_pauses

GRID_Z_TOL       = [1.25, 1.5, 1.75, 2.0]
GRID_SIGMA_FLOOR = [0.75, 1.0, 1.25]
GRID_K           = [0.5, 0.65, 0.8, 0.95]
IOU_TAU          = 0.3
DEV_SPEECH_IDS   = ["speech-001", "speech-004"]

LIBRARY_DIR = Path("data/library")
DATASET_DIR = Path("data/dataset")
CONFIG_DIR  = Path("config")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _macro_f1(metrics: dict) -> float:
    """Compute macro-F1 as unweighted mean of per-family F1 scores."""
    per_fam = metrics.get("per_family", {})
    if not per_fam:
        return 0.0
    return sum(v["f1"] for v in per_fam.values()) / len(per_fam)


def eval_pred_gt(pred_lists: list[list[dict]], gt_lists: list[list[dict]], dur_list: list | None = None) -> dict:
    """Run match_file_regions per file, then aggregate with compute_metrics."""
    if dur_list is None:
        dur_list = [None] * len(pred_lists)
    file_results = [
        match_file_regions(pred, gt, tau=IOU_TAU, clip_duration=dur)
        for pred, gt, dur in zip(pred_lists, gt_lists, dur_list)
    ]
    metrics = compute_metrics(file_results)
    metrics["macro_f1"] = round(_macro_f1(metrics), 4)
    return metrics


def load_dev_entries(speech_ids: list[str]) -> list[dict]:
    entries = []
    for sid in speech_ids:
        p = DATASET_DIR / sid / "labels.json"
        if not p.exists():
            print(f"  [WARN] labels.json not found for {sid}")
            continue
        for e in json.loads(p.read_text(encoding="utf-8")):
            if e.get("split") == "dev":
                entries.append(e)
    return entries


def _get_mirror_id(speech_id: str) -> str:
    meta_path = LIBRARY_DIR / speech_id / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mid = meta.get("mirror_source")
        if mid:
            return mid
        readers = meta.get("readers", [])
        if readers:
            return readers[0]["id"]
    return "r1"


def _load_mirror_align(speech_id: str) -> tuple[list[dict], list[dict]]:
    mirror_id = _get_mirror_id(speech_id)
    align_path = LIBRARY_DIR / speech_id / "readers" / f"{mirror_id}.align.json"
    if not align_path.exists():
        return [], []
    data = json.loads(align_path.read_text(encoding="utf-8"))
    return data.get("words", []), data.get("pauses", [])


# ---------------------------------------------------------------------------
# Phase 1: feature extraction (once per file)
# ---------------------------------------------------------------------------

def extract_features(entries: list[dict]) -> list[dict]:
    import speechlens.config as cfg
    from speechlens import audio
    from speechlens.align.ctc import align
    from speechlens.features.frame import compute_frame_features
    from speechlens.features.word import compute_word_features, classify_pauses

    norms_path = LIBRARY_DIR / "norms.json"
    norms = json.loads(norms_path.read_text(encoding="utf-8")) if norms_path.exists() else {}
    mirror_cache: dict[str, tuple[list[dict], list[dict]]] = {}

    feature_cache: list[dict] = []
    for idx, entry in enumerate(entries):
        file_path = Path(entry["file"])
        speech_id = entry["speech_id"]
        flaws     = entry.get("flaws", [])
        synthetic = entry.get("synthetic_flaw", entry.get("level", 0) > 0)

        print(f"  [{idx+1}/{len(entries)}] {file_path.name}  ", end="", flush=True)
        t0 = time.time()

        try:
            samples = audio.load(file_path)
            total_dur = len(samples) / cfg.SR

            baseline_path = LIBRARY_DIR / speech_id / "baseline.json"
            baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
            base_words_stat = baseline.get("words", [])
            base_pause_dict = {
                i: bw.get("mu_pause_before", 0.0)
                for i, bw in enumerate(base_words_stat)
            } if base_words_stat else None

            ff = compute_frame_features(samples, cfg.SR)

            if synthetic and flaws:
                # Analytic remap — no CTC needed
                if speech_id not in mirror_cache:
                    mirror_cache[speech_id] = _load_mirror_align(speech_id)
                mirror_words, mirror_pauses = mirror_cache[speech_id]

                if mirror_words:
                    remapped_words, remapped_pauses = remap_words_and_pauses(
                        mirror_words, mirror_pauses, flaws
                    )
                    align_words_d = remapped_words
                    align_pauses_d = [
                        {"start": p.start, "end": p.end,
                         "before_word": p.before_word, "after_word": p.after_word}
                        for p in remapped_pauses
                    ]
                else:
                    align_words_d, align_pauses_d = _ctc_align(
                        samples, speech_id, cfg.SR
                    )
            else:
                align_words_d, align_pauses_d = _ctc_align(
                    samples, speech_id, cfg.SR
                )

            word_feats = compute_word_features(align_words_d, align_pauses_d, ff, total_dur)
            classified_pauses = classify_pauses(
                align_pauses_d, ff, baseline_pause_mu=base_pause_dict
            )

            feature_cache.append({
                "entry": entry,
                "word_feats": word_feats,
                "classified_pauses": classified_pauses,
                "baseline": baseline,
                "norms": norms,
                "ff": ff,
                "total_dur": total_dur,
                "ok": True,
            })
            print(f"{time.time()-t0:.1f}s", flush=True)

        except Exception as exc:
            import traceback
            print(f"ERROR: {exc}", flush=True)
            traceback.print_exc()
            feature_cache.append({"entry": entry, "ok": False})

    return feature_cache


def _ctc_align(samples, speech_id: str, SR: int) -> tuple[list[dict], list[dict]]:
    from speechlens.align.ctc import align
    transcript_path = LIBRARY_DIR / speech_id / "transcript.txt"
    transcript = transcript_path.read_text(encoding="utf-8").strip()
    align_res = align(samples, transcript, use_cache=True)
    align_words_d = [
        {"i": w.i, "text": w.text, "text_norm": w.text_norm,
         "punct_after": w.punct_after, "start": w.start,
         "end": w.end, "score": w.score, "nsyl": w.nsyl}
        for w in align_res.words
    ]
    align_pauses_d = [
        {"start": p.start, "end": p.end,
         "before_word": p.before_word, "after_word": p.after_word}
        for p in align_res.pauses
    ]
    return align_words_d, align_pauses_d


# ---------------------------------------------------------------------------
# Phase 2: detection with patched config
# ---------------------------------------------------------------------------

def detect_with_config(
    feature_cache: list[dict],
    z_tol: float,
    sigma_floor: float,
) -> list[list[dict]]:
    import speechlens.config as cfg
    cfg.Z_TOL = z_tol
    cfg.SIGMA_FLOOR_MULT = sigma_floor
    import speechlens.detect as det_mod
    importlib.reload(det_mod)

    results: list[list[dict]] = []
    for item in feature_cache:
        if not item.get("ok"):
            results.append([])
            continue
        try:
            if item["baseline"]:
                regions = det_mod.detect_all(
                    item["word_feats"], item["classified_pauses"],
                    item["baseline"], item["norms"], ff=item["ff"],
                )
            else:
                regions = det_mod.detect_norm(
                    item["word_feats"], item["classified_pauses"],
                    item["norms"], ff=item["ff"],
                )
            results.append([r.to_dict() for r in regions])
        except Exception as exc:
            print(f"  [WARN] detect error: {exc}")
            results.append([])
    return results


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def calibrate(
    dry_run: bool = False,
    output_path: Path = CONFIG_DIR / "calibrated.json",
) -> dict[str, Any]:
    entries = load_dev_entries(DEV_SPEECH_IDS)
    print(f"Dev entries: {len(entries)}", flush=True)
    if not entries:
        raise RuntimeError("No dev entries found.")

    z_tol_vals  = GRID_Z_TOL[:2]       if dry_run else GRID_Z_TOL
    sigma_vals  = GRID_SIGMA_FLOOR[:2] if dry_run else GRID_SIGMA_FLOOR
    k_vals      = GRID_K[:2]           if dry_run else GRID_K
    configs_zs  = list(product(z_tol_vals, sigma_vals))

    print(f"Detection configs: {len(configs_zs)} | K values: {len(k_vals)}", flush=True)
    gt_lists = [e.get("flaws", []) for e in entries]

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    print("\nPhase 1: Feature extraction…", flush=True)
    t0 = time.time()
    feature_cache = extract_features(entries)
    ok_count = sum(1 for f in feature_cache if f.get("ok"))
    print(f"\nFeature extraction: {ok_count}/{len(entries)} ok  {time.time()-t0:.1f}s\n", flush=True)

    # ── Phase 2: detection grid ──────────────────────────────────────────────
    print("Phase 2: Detection grid…", flush=True)
    best_f1   = -1.0
    best_cfg: dict[str, Any] = {}
    best_metrics: dict = {}
    grid_results: list[dict] = []

    for z_tol, sigma_floor in configs_zs:
        t1 = time.time()
        pred_lists = detect_with_config(feature_cache, z_tol, sigma_floor)
        det_time = time.time() - t1
        metrics = eval_pred_gt(pred_lists, gt_lists)
        macro_f1 = metrics["macro_f1"]
        overall_f1 = metrics.get("overall", {}).get("f1", 0.0)
        print(
            f"  z_tol={z_tol}  sigma_floor={sigma_floor}"
            f"  macro_F1={macro_f1:.4f}  overall_F1={overall_f1:.4f}  ({det_time:.2f}s)",
            flush=True,
        )

        for k in k_vals:
            grid_results.append({
                "z_tol": z_tol,
                "sigma_floor_mult": sigma_floor,
                "k": k,
                "macro_f1": macro_f1,
                "overall_f1": overall_f1,
                "per_family": metrics.get("per_family", {}),
                "overall": metrics.get("overall", {}),
            })
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_cfg = {"z_tol": z_tol, "sigma_floor_mult": sigma_floor, "k": k}
                best_metrics = metrics

    output = {
        "best": best_cfg,
        "best_macro_f1": best_f1,
        "best_metrics": best_metrics,
        "grid_results": grid_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Best config: {best_cfg}  macro_F1={best_f1:.4f}")
    print(f"[OK] Saved -> {output_path}")

    # Also save to results/calibration.json per DESIGN 10.4
    results_json = Path("results/calibration.json")
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Saved -> {results_json}")

    # Generate calibration plot
    plot_calibration(output, Path("results/fig_calibration.png"))
    return output


def plot_calibration(data: dict, out_path: Path) -> None:
    """Generate calibration grid search plot."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        grid = data.get("grid_results", [])
        if not grid:
            return

        # Unique z_tol and sigma_floor values
        z_tols = sorted(list({entry["z_tol"] for entry in grid}))
        sigmas = sorted(list({entry["sigma_floor_mult"] for entry in grid}))

        f1_matrix = np.zeros((len(sigmas), len(z_tols)))
        for entry in grid:
            zi = z_tols.index(entry["z_tol"])
            si = sigmas.index(entry["sigma_floor_mult"])
            f1_matrix[si, zi] = max(f1_matrix[si, zi], entry["macro_f1"])

        fig, ax = plt.subplots(figsize=(7, 5))
        cax = ax.imshow(f1_matrix, cmap="viridis", aspect="auto", origin="lower")
        fig.colorbar(cax, label="Macro F1 @ IoU 0.3")

        ax.set_xticks(range(len(z_tols)))
        ax.set_xticklabels([str(z) for z in z_tols])
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(s) for s in sigmas])

        ax.set_xlabel("Z_TOL")
        ax.set_ylabel("SIGMA_FLOOR_MULT")
        best = data.get("best", {})
        ax.set_title(f"SpeechLens Calibration Grid (Best: Z_TOL={best.get('z_tol')}, SIGMA={best.get('sigma_floor_mult')})")

        # Add text annotations in each cell
        for i in range(len(sigmas)):
            for j in range(len(z_tols)):
                val = f1_matrix[i, j]
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color="white" if val < np.mean(f1_matrix) else "black")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved -> {out_path}")
    except Exception as exc:
        print(f"[WARN] Failed to plot calibration figure: {exc}")


def run(dry_run: bool = False, output_path: Path = CONFIG_DIR / "calibrated.json") -> dict[str, Any]:
    return calibrate(dry_run=dry_run, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=CONFIG_DIR / "calibrated.json")
    args = parser.parse_args()
    calibrate(dry_run=args.dry_run, output_path=args.output)
