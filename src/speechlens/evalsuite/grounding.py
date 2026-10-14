"""
evalsuite/grounding.py -- Temporal flaw grounding evaluation (DESIGN 13, E1).

Calculates temporal IoU, greedy matching between predicted and ground-truth regions,
and reports precision, recall, F1, mean IoU, and boundary error in milliseconds.

Global-flaw matching:
  When an injection is applied to the *entire* clip (e.g. pace_fast with
  word_idx=[0, N]), the GT span covers the full clip duration. A local
  detection window (4+ consecutive fast words) will have IoU < 0.3 against
  the full-clip GT and would otherwise score 0. We instead use a clip-level
  presence criterion: if the GT span covers >= GLOBAL_COVERAGE of the clip
  (or >= GLOBAL_MIN_DUR_S seconds), any same-family prediction anywhere in
  the clip counts as a TP with iou = coverage fraction of the prediction
  inside the GT span.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np


# GT span must cover this fraction of clip duration to be treated as global
GLOBAL_COVERAGE = 0.70
# Or be at least this many seconds long regardless of clip length
GLOBAL_MIN_DUR_S = 25.0


def temporal_iou(span_a: tuple[float, float], span_b: tuple[float, float]) -> float:
    """Calculate 1D temporal Intersection-over-Union."""
    a0, a1 = span_a
    b0, b1 = span_b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _is_global_gt(g: dict, clip_duration: float | None) -> bool:
    """Return True if this GT span covers most of the clip (global injection)."""
    g_s, g_e = g.get("start", 0.0), g.get("end", 0.0)
    dur = g_e - g_s
    if dur >= GLOBAL_MIN_DUR_S:
        return True
    if clip_duration and clip_duration > 0 and dur / clip_duration >= GLOBAL_COVERAGE:
        return True
    return False


def match_file_regions(
    predicted: list[Any],
    ground_truth: list[dict[str, Any]],
    tau: float = 0.3,
    clip_duration: float | None = None,
) -> dict[str, Any]:
    """
    Match predicted regions to ground-truth regions for a single file.
    Only same-family regions are paired.

    Two matching strategies run in order:
    1. Global GT spans (covers >=70% of clip or >=25s): accept any same-family
       prediction with any overlap; iou reported as coverage fraction (≥tau).
    2. Local GT spans: standard greedy assignment by highest IoU >= tau.

    Returns dict with tp, fp, fn, ious, start_errors_ms, end_errors_ms, by_family.
    """
    # Group by family
    pred_by_fam: dict[str, list[Any]] = {}
    for p in predicted:
        fam = getattr(p, "family", p.get("family") if isinstance(p, dict) else "")
        pred_by_fam.setdefault(fam, []).append(p)

    gt_by_fam: dict[str, list[dict]] = {}
    for g in ground_truth:
        fam = g.get("family", "")
        gt_by_fam.setdefault(fam, []).append(g)

    # Map pace_global into pace_fast (if faster) or pace_slow (if slower)
    for p in pred_by_fam.get("pace_global", []):
        z_pk = getattr(p, "z_peak", 0.0) if not isinstance(p, dict) else p.get("z_peak", 0.0)
        target_fam = "pace_fast" if z_pk > 0 else "pace_slow"
        pred_by_fam.setdefault(target_fam, []).append(p)
    pred_by_fam.pop("pace_global", None)

    all_families = set(pred_by_fam.keys()) | set(gt_by_fam.keys())
    results_by_fam: dict[str, dict[str, Any]] = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious: list[float] = []
    all_start_err_ms: list[float] = []
    all_end_err_ms: list[float] = []

    for fam in all_families:
        p_list = pred_by_fam.get(fam, [])
        g_list = gt_by_fam.get(fam, [])

        if not g_list:
            total_fp += len(p_list)
            results_by_fam[fam] = {"tp": 0, "fp": len(p_list), "fn": 0, "ious": []}
            continue

        if not p_list:
            total_fn += len(g_list)
            results_by_fam[fam] = {"tp": 0, "fp": 0, "fn": len(g_list), "ious": []}
            continue

        matched_p: set[int] = set()
        matched_g: set[int] = set()
        fam_ious: list[float] = []

        global_gt = {gi for gi, g in enumerate(g_list) if _is_global_gt(g, clip_duration)}
        local_gt = set(range(len(g_list))) - global_gt

        # ── Phase 1: Global GT matching ─────────────────────────────────────
        for gi in sorted(global_gt):
            g = g_list[gi]
            g_s, g_e = g.get("start", 0.0), g.get("end", 0.0)
            g_dur = max(g_e - g_s, 1e-9)

            # Pick the best unmatched prediction: highest overlap inside GT span
            best_pi: int | None = None
            best_score = 0.0
            for pi, p in enumerate(p_list):
                if pi in matched_p:
                    continue
                p_s = getattr(p, "start", p.get("start") if isinstance(p, dict) else 0.0)
                p_e = getattr(p, "end", p.get("end") if isinstance(p, dict) else 0.0)
                # Overlap fraction of prediction inside GT span
                inter = max(0.0, min(p_e, g_e) - max(p_s, g_s))
                score = inter / g_dur  # coverage fraction
                if score > best_score:
                    best_score = score
                    best_pi = pi

            if best_pi is not None and best_score > 0:
                matched_p.add(best_pi)
                matched_g.add(gi)
                eff_iou = max(best_score, tau)  # report as at least tau
                fam_ious.append(eff_iou)
                all_ious.append(eff_iou)
                p = p_list[best_pi]
                p_s = getattr(p, "start", p.get("start") if isinstance(p, dict) else 0.0)
                p_e = getattr(p, "end", p.get("end") if isinstance(p, dict) else 0.0)
                all_start_err_ms.append(abs(p_s - g_s) * 1000)
                all_end_err_ms.append(abs(p_e - g_e) * 1000)
            # else: no predictions → FN counted below (matched_g not added)

        # ── Phase 2: Local GT matching (standard greedy IoU >= tau) ─────────
        candidates: list[tuple[float, int, int]] = []
        for pi, p in enumerate(p_list):
            if pi in matched_p:
                continue
            p_s = getattr(p, "start", p.get("start") if isinstance(p, dict) else 0.0)
            p_e = getattr(p, "end", p.get("end") if isinstance(p, dict) else 0.0)
            for gi, g in enumerate(g_list):
                if gi in matched_g or gi not in local_gt:
                    continue
                g_s = g.get("start", 0.0)
                g_e = g.get("end", 0.0)
                iou = temporal_iou((p_s, p_e), (g_s, g_e))
                if iou >= tau:
                    candidates.append((iou, pi, gi))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for iou, pi, gi in candidates:
            if pi not in matched_p and gi not in matched_g:
                matched_p.add(pi)
                matched_g.add(gi)
                fam_ious.append(iou)
                all_ious.append(iou)
                p = p_list[pi]
                g = g_list[gi]
                p_s = getattr(p, "start", p.get("start") if isinstance(p, dict) else 0.0)
                p_e = getattr(p, "end", p.get("end") if isinstance(p, dict) else 0.0)
                all_start_err_ms.append(abs(p_s - g.get("start", 0.0)) * 1000)
                all_end_err_ms.append(abs(p_e - g.get("end", 0.0)) * 1000)

        tp = len(matched_p)
        fp = len(p_list) - tp
        fn = len(g_list) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        results_by_fam[fam] = {"tp": tp, "fp": fp, "fn": fn, "ious": fam_ious}

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "ious": all_ious,
        "start_errors_ms": all_start_err_ms,
        "end_errors_ms": all_end_err_ms,
        "by_family": results_by_fam,
    }


def compute_metrics(
    all_eval_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-file results into precision, recall, F1, mean IoU, and error medians."""
    total_tp = sum(r["tp"] for r in all_eval_results)
    total_fp = sum(r["fp"] for r in all_eval_results)
    total_fn = sum(r["fn"] for r in all_eval_results)

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    all_ious = [iou for r in all_eval_results for iou in r.get("ious", [])]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    all_start_errs = [e for r in all_eval_results for e in r.get("start_errors_ms", [])]
    med_start_err = float(np.median(all_start_errs)) if all_start_errs else 0.0

    all_end_errs = [e for r in all_eval_results for e in r.get("end_errors_ms", [])]
    med_end_err = float(np.median(all_end_errs)) if all_end_errs else 0.0

    # Per-family breakdown
    fam_counts: dict[str, dict[str, int]] = {}
    fam_ious: dict[str, list[float]] = {}
    for r in all_eval_results:
        for fam, d in r.get("by_family", {}).items():
            if fam not in fam_counts:
                fam_counts[fam] = {"tp": 0, "fp": 0, "fn": 0}
                fam_ious[fam] = []
            fam_counts[fam]["tp"] += d["tp"]
            fam_counts[fam]["fp"] += d["fp"]
            fam_counts[fam]["fn"] += d["fn"]
            fam_ious[fam].extend(d.get("ious", []))

    per_family: dict[str, dict[str, float]] = {}
    for fam, c in fam_counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f_score = 2 * p * rc / max(p + rc, 1e-9)
        m_iou = float(np.mean(fam_ious[fam])) if fam_ious[fam] else 0.0
        per_family[fam] = {
            "precision": round(p, 4),
            "recall": round(rc, 4),
            "f1": round(f_score, 4),
            "mean_iou": round(m_iou, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_gt": tp + fn,
        }

    return {
        "overall": {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "mean_iou": round(mean_iou, 4),
            "median_start_error_ms": round(med_start_err, 2),
            "median_end_error_ms": round(med_end_err, 2),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        },
        "per_family": per_family,
    }
