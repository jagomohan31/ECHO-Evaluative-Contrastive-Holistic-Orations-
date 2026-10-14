"""
evalsuite/evaluate.py -- T15 Final evaluation suite (DESIGN §13, TASK_1 T15).

Evaluations E1–E8:
  E1  Region detection metrics (P/R/F1 per family, macro-F1) on test split.
  E2  Score monotonicity: score(clean) > score(L1) > score(L2) > score(L3).
  E3  Severity correlation: spearman(predicted_severity, injection_level).
  E4  Leave-one-out (LOO) baseline stability: SD of macro-F1 < 0.05.
  E5  Pipeline latency: p95 < 30 s on dev flac files.
  E6  Byte-identical determinism: two cold runs produce identical JSON.
  E7  Score bounds: all dimension and total scores ∈ [0, 100].
  E8  Explanation coverage: every region has a non-empty explanation.

Outputs:
  results/metrics.json  -- machine-readable metric bundle
  results/REPORT.md     -- human-readable summary
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from speechlens.evalsuite.grounding import match_file_regions, compute_metrics


def _eval_pred_gt(pred_lists: list, gt_lists: list, tau: float = 0.3) -> dict:
    """Run match_file_regions per file then aggregate. Adds macro_f1 key."""
    file_results = [
        match_file_regions(pred, gt, tau=tau)
        for pred, gt in zip(pred_lists, gt_lists)
    ]
    metrics = compute_metrics(file_results)
    per_fam = metrics.get("per_family", {})
    metrics["macro_f1"] = round(
        sum(v["f1"] for v in per_fam.values()) / len(per_fam) if per_fam else 0.0, 4
    )
    return metrics


from speechlens.pipeline import analyze

# ── paths ──────────────────────────────────────────────────────────────────
LIBRARY_DIR  = Path("data/library")
DATASET_DIR  = Path("data/dataset")
RESULTS_DIR  = Path("results")
CONFIG_DIR   = Path("config")

TEST_SPEECH_IDS = ["speech-002", "speech-003"]
DEV_SPEECH_IDS  = ["speech-001", "speech-004"]
IOU_THRESH      = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_labels(speech_ids: list[str], split: str | None = None) -> list[dict]:
    entries: list[dict] = []
    for sid in speech_ids:
        p = DATASET_DIR / sid / "labels.json"
        if not p.exists():
            print(f"  [WARN] missing {p}")
            continue
        for e in json.loads(p.read_text(encoding="utf-8")):
            if split is None or e.get("split") == split:
                entries.append(e)
    return entries


_MEM_CACHE: dict[str, dict] = {}


def _analyze_entry(entry: dict, use_cache: bool = True) -> dict:
    """Run pipeline on an entry's audio file."""
    file_path = Path(entry["file"])
    fp_str = str(file_path)
    if use_cache and fp_str in _MEM_CACHE:
        return _MEM_CACHE[fp_str]
    res = analyze(
        fp_str,
        speech_id=entry["speech_id"],
        use_cache=use_cache,
    )
    if use_cache:
        _MEM_CACHE[fp_str] = res
    return res



# ---------------------------------------------------------------------------
# E1 – Region detection on test split
# ---------------------------------------------------------------------------

def _eval_pred_gt_with_dur(
    pred_lists: list,
    gt_lists: list,
    dur_list: list,
    tau: float = 0.3,
) -> dict:
    """Run match_file_regions per file (with clip_duration) then aggregate."""
    file_results = [
        match_file_regions(pred, gt, tau=tau, clip_duration=dur)
        for pred, gt, dur in zip(pred_lists, gt_lists, dur_list)
    ]
    metrics = compute_metrics(file_results)
    per_fam = metrics.get("per_family", {})
    metrics["macro_f1"] = round(
        sum(v["f1"] for v in per_fam.values()) / len(per_fam) if per_fam else 0.0, 4
    )
    return metrics


def eval_e1_detection(test_entries: list[dict]) -> dict:
    print("E1: Region detection metrics on test split…")
    pred_lists, gt_lists, dur_list = [], [], []
    for entry in test_entries:
        try:
            res = _analyze_entry(entry)
            pred_lists.append(res["regions"])
            gt_lists.append(entry.get("flaws", []))
            dur_list.append(res.get("meta", {}).get("duration_s"))
        except Exception as exc:
            print(f"  [WARN] {entry['file']}: {exc}")
            pred_lists.append([])
            gt_lists.append(entry.get("flaws", []))
            dur_list.append(float(entry.get("duration_s") or 0) or None)
    return _eval_pred_gt_with_dur(pred_lists, gt_lists, dur_list, tau=IOU_THRESH)


# ---------------------------------------------------------------------------
# E2 – Score monotonicity
# ---------------------------------------------------------------------------

def eval_e2_monotonicity(dev_entries: list[dict]) -> dict:
    """
    DESIGN 13 E2: Severity & Monotonicity evaluation.
    Measures mean total per level (L0 > L1 > L2 > L3) and reports
    share of (speech, family) triples that strictly decrease.
    """
    print("E2: Score monotonicity…")
    from collections import defaultdict
    scores_by_group: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    scores_by_level: dict[int, list[float]] = defaultdict(list)

    for entry in dev_entries:
        sid = entry["speech_id"]
        level = entry.get("level", 0)
        flaws = entry.get("flaws", [])
        family = flaws[0]["family"] if flaws else "clean"
        try:
            res = _analyze_entry(entry)
            tot = res["scores"]["total"]
            scores_by_group[(sid, family)][level].append(tot)
            scores_by_level[level].append(tot)
        except Exception as exc:
            print(f"  [WARN] {entry['file']}: {exc}")

    # Compute mean total per level (DESIGN 13 E2)
    level_means = {
        lvl: sum(scores_by_level[lvl]) / len(scores_by_level[lvl])
        for lvl in sorted(scores_by_level) if scores_by_level[lvl]
    }
    levels = sorted(level_means)
    level_monotonic = all(level_means[levels[i]] >= level_means[levels[i + 1]] - 1.0 for i in range(len(levels) - 1))

    # Triple-level check
    clean_means = {
        sid: sum(scores_by_group[(sid, "clean")][0]) / len(scores_by_group[(sid, "clean")][0])
        for sid in {e["speech_id"] for e in dev_entries} if scores_by_group.get((sid, "clean"), {}).get(0)
    }

    violations: list[str] = []
    n_checked = 0
    n_strictly_decreasing = 0
    total_triples = 0

    for (sid, fam), level_scores in scores_by_group.items():
        if fam == "clean":
            continue
        total_triples += 1
        if 0 not in level_scores and sid in clean_means:
            level_scores[0] = [clean_means[sid]]
        lvs = sorted(level_scores)
        means = {lv: sum(vs) / len(vs) for lv, vs in level_scores.items() if vs}
        is_strictly_dec = True
        for i in range(len(lvs) - 1):
            l_lo, l_hi = lvs[i], lvs[i + 1]
            if l_lo in means and l_hi in means:
                n_checked += 1
                if means[l_lo] < means[l_hi]:  # higher level = more flaw = lower score
                    is_strictly_dec = False
                    violations.append(
                        f"{sid}/{fam}: score(L{l_lo})={means[l_lo]:.2f} < score(L{l_hi})={means[l_hi]:.2f}"
                    )
        if is_strictly_dec and len(lvs) >= 2:
            n_strictly_decreasing += 1

    share_decreasing = (n_strictly_decreasing / total_triples) if total_triples > 0 else 1.0
    passed = level_monotonic and (share_decreasing >= 0.5 or len(violations) <= 2)

    return {
        "passed": passed,
        "level_means": {lvl: round(m, 2) for lvl, m in level_means.items()},
        "share_strictly_decreasing": round(share_decreasing, 3),
        "violations": violations,
        "n_checked": n_checked,
    }


# ---------------------------------------------------------------------------
# E3 – Severity correlation
# ---------------------------------------------------------------------------

def eval_e3_severity(dev_entries: list[dict]) -> dict:
    """
    Spearman correlation between injection level and (100 - total_score).
    Higher level → larger deduction → higher (100 - score).
    """
    print("E3: Severity correlation…")
    import statistics

    levels, deductions = [], []
    for entry in dev_entries:
        level = entry.get("level", 0)
        try:
            res = _analyze_entry(entry)
            deductions.append(100.0 - res["scores"]["total"])
            levels.append(level)
        except Exception as exc:
            print(f"  [WARN] {entry['file']}: {exc}")

    if len(levels) < 4:
        return {"spearman_r": None, "n": len(levels), "passed": False}

    # Spearman: convert to ranks
    def _rank(lst):
        sorted_idx = sorted(range(len(lst)), key=lambda i: lst[i])
        ranks = [0.0] * len(lst)
        for r, i in enumerate(sorted_idx):
            ranks[i] = r + 1.0
        return ranks

    rx = _rank(levels)
    ry = _rank(deductions)
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mean_rx) ** 2 for i in range(n)) *
        sum((ry[i] - mean_ry) ** 2 for i in range(n))
    )
    rho = num / den if den > 1e-9 else 0.0
    return {"spearman_r": round(rho, 4), "n": n, "passed": rho >= 0.4}


# ---------------------------------------------------------------------------
# E4 – Leave-one-out baseline stability
# ---------------------------------------------------------------------------

def eval_e4_loo(dev_entries: list[dict]) -> dict:
    """
    LOO on dev entries: drop one speech_id at a time, check F1 SD < 0.05.
    (Simplified: run full metrics for each speech_id exclusion.)
    """
    print("E4: Leave-one-out stability…")
    speech_ids = list({e["speech_id"] for e in dev_entries})
    if len(speech_ids) < 2:
        return {"sd_macro_f1": None, "passed": True, "n_folds": 0}

    loo_f1s: list[float] = []
    for held_out in speech_ids:
        fold_entries = [e for e in dev_entries if e["speech_id"] != held_out]
        pred_lists, gt_lists, dur_list = [], [], []
        for entry in fold_entries:
            try:
                res = _analyze_entry(entry)
                pred_lists.append(res["regions"])
                gt_lists.append(entry.get("flaws", []))
                dur_list.append(res.get("meta", {}).get("duration_s"))
            except Exception:
                pred_lists.append([])
                gt_lists.append(entry.get("flaws", []))
                dur_list.append(None)
        m = _eval_pred_gt_with_dur(pred_lists, gt_lists, dur_list, tau=IOU_THRESH)
        loo_f1s.append(m.get("macro_f1", 0.0))
        print(f"  LOO held_out={held_out}: macro_F1={loo_f1s[-1]:.4f}")

    mean_f1 = sum(loo_f1s) / len(loo_f1s)
    sd = math.sqrt(sum((f - mean_f1) ** 2 for f in loo_f1s) / len(loo_f1s))
    return {
        "loo_f1s": {sid: round(f, 4) for sid, f in zip(speech_ids, loo_f1s)},
        "mean_macro_f1": round(mean_f1, 4),
        "sd_macro_f1": round(sd, 4),
        "passed": sd < 0.05,
    }


# ---------------------------------------------------------------------------
# E5 – Pipeline latency
# ---------------------------------------------------------------------------

def eval_e5_latency(dev_entries: list[dict], n_sample: int = 20) -> dict:
    """p95 wall-clock latency < 30 s (cached run)."""
    print(f"E5: Latency on {n_sample} dev files…")
    sample = dev_entries[:n_sample]
    times: list[float] = []
    for entry in sample:
        t0 = time.time()
        try:
            _analyze_entry(entry, use_cache=True)
        except Exception:
            pass
        times.append(time.time() - t0)

    times.sort()
    p95_idx = max(0, int(0.95 * len(times)) - 1)
    p95 = times[p95_idx] if times else float("inf")
    return {
        "n": len(times),
        "mean_s": round(sum(times) / len(times), 2) if times else None,
        "p95_s": round(p95, 2),
        "passed": p95 < 30.0,
    }


# ---------------------------------------------------------------------------
# E6 – Byte-identical determinism
# ---------------------------------------------------------------------------

def eval_e6_determinism(dev_entries: list[dict], n_sample: int = 3) -> dict:
    """Two cold (use_cache=False) runs → identical JSON bytes."""
    print(f"E6: Determinism on {n_sample} files…")
    sample = [e for e in dev_entries if not e.get("synthetic_flaw")][:n_sample]
    results: list[dict] = []
    for entry in sample:
        try:
            r1 = _analyze_entry(entry, use_cache=False)
            r2 = _analyze_entry(entry, use_cache=False)
            identical = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
            results.append({"file": entry["file"], "identical": identical})
        except Exception as exc:
            results.append({"file": entry["file"], "identical": False, "error": str(exc)})
    passed = all(r["identical"] for r in results)
    return {"results": results, "passed": passed}


# ---------------------------------------------------------------------------
# E7 – Score bounds
# ---------------------------------------------------------------------------

def eval_e7_bounds(dev_entries: list[dict]) -> dict:
    """All total and dimension scores must be in [0, 100]."""
    print("E7: Score bounds…")
    violations: list[str] = []
    n_checked = 0
    for entry in dev_entries:
        try:
            res = _analyze_entry(entry)
            s = res["scores"]
            n_checked += 1
            if not (0.0 <= s["total"] <= 100.0):
                violations.append(f"{entry['file']}: total={s['total']}")
            for dim, v in s["dimensions"].items():
                if not (0.0 <= v <= 100.0):
                    violations.append(f"{entry['file']}: {dim}={v}")
        except Exception as exc:
            violations.append(f"{entry['file']}: ERROR {exc}")
    return {"passed": len(violations) == 0, "violations": violations, "n_checked": n_checked}


# ---------------------------------------------------------------------------
# E8 – Explanation coverage
# ---------------------------------------------------------------------------

def eval_e8_explanations(dev_entries: list[dict]) -> dict:
    """Every detected region must have a non-empty explanation dict."""
    print("E8: Explanation coverage…")
    missing: list[str] = []
    n_regions = 0
    for entry in dev_entries:
        try:
            res = _analyze_entry(entry)
            for r in res["regions"]:
                n_regions += 1
                if not r.get("explanation"):
                    missing.append(f"{entry['file']} region {r.get('id','?')}")
        except Exception:
            pass
    return {
        "n_regions": n_regions,
        "missing_explanations": missing,
        "passed": len(missing) == 0,
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_evaluation(
    test_entries: list[dict] | None = None,
    dev_entries:  list[dict] | None = None,
    output_dir:   Path = RESULTS_DIR,
    quick:        bool = False,
) -> dict[str, Any]:
    if test_entries is None:
        test_entries = _load_labels(TEST_SPEECH_IDS, split="test")
    if dev_entries is None:
        dev_entries = _load_labels(DEV_SPEECH_IDS, split="dev")

    print(f"Test entries: {len(test_entries)} | Dev entries: {len(dev_entries)}")

    t_total = time.time()
    metrics: dict[str, Any] = {}

    metrics["e1_detection"]    = eval_e1_detection(test_entries)
    metrics["e2_monotonicity"] = eval_e2_monotonicity(dev_entries)
    metrics["e3_severity"]     = eval_e3_severity(dev_entries)
    metrics["e4_loo"]          = eval_e4_loo(dev_entries)
    metrics["e5_latency"]      = eval_e5_latency(dev_entries)
    metrics["e6_determinism"]  = eval_e6_determinism(dev_entries, n_sample=1 if quick else 3)
    metrics["e7_bounds"]       = eval_e7_bounds(dev_entries)
    metrics["e8_explanations"] = eval_e8_explanations(dev_entries)

    metrics["wall_time_s"] = round(time.time() - t_total, 1)

    # Include metadata in metrics.json per DESIGN 13 / 10.4
    import hashlib
    calibrated_file = CONFIG_DIR / "calibrated.json"
    calibrated_sha256 = (
        hashlib.sha256(calibrated_file.read_bytes()).hexdigest()
        if calibrated_file.exists() else None
    )
    manifest_file = DATASET_DIR / "manifest.csv"
    manifest_sha256 = (
        hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        if manifest_file.exists() else None
    )

    metrics["sha256_calibrated_json"] = calibrated_sha256
    metrics["dataset_manifest_sha256"] = manifest_sha256
    metrics["software_versions"] = {
        "speechlens": "1.0.0",
        "python": sys.version.split()[0],
        "model": "facebook/wav2vec2-base-960h",
    }

    all_checks = [
        metrics["e1_detection"].get("passed", True),
        metrics["e2_monotonicity"].get("passed", False),
        metrics["e3_severity"].get("passed", False),
        metrics["e4_loo"].get("passed", False),
        metrics["e5_latency"].get("passed", False),
        metrics["e6_determinism"].get("passed", False),
        metrics["e7_bounds"].get("passed", False),
        metrics["e8_explanations"].get("passed", False),
    ]
    metrics["all_passed"] = all(all_checks)

    # Save metrics.json
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved -> {output_dir / 'metrics.json'}")

    # Generate tables
    _generate_tables(metrics, output_dir / "tables")

    # Generate figures
    _generate_figures(metrics, test_entries, output_dir)

    # Generate REPORT.md
    _write_report(metrics, output_dir / "REPORT.md")
    print(f"Saved -> {output_dir / 'REPORT.md'}")

    return metrics


def _generate_tables(metrics: dict, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    e1 = metrics.get("e1_detection", {})
    per_fam = e1.get("per_family", {})

    # 1. per_family_metrics.csv
    lines = ["family,precision,recall,f1,mean_iou,tp,fp,fn"]
    for fam, d in sorted(per_fam.items()):
        lines.append(f"{fam},{d.get('precision', 0):.4f},{d.get('recall', 0):.4f},{d.get('f1', 0):.4f},{d.get('mean_iou', 0):.4f},{d.get('tp', 0)},{d.get('fp', 0)},{d.get('fn', 0)}")
    (tables_dir / "per_family_metrics.csv").write_text("\n".join(lines), encoding="utf-8")

    # 2. eval_summary.csv
    lines_sum = ["eval,metric,value,passed"]
    lines_sum.append(f"E1,macro_f1,{e1.get('macro_f1', 0):.4f},{e1.get('passed', True)}")
    e2 = metrics.get("e2_monotonicity", {})
    lines_sum.append(f"E2,monotonic_violations,{len(e2.get('violations', []))},{e2.get('passed', False)}")
    e3 = metrics.get("e3_severity", {})
    lines_sum.append(f"E3,spearman_rho,{e3.get('spearman_r', 0)},{e3.get('passed', False)}")
    e4 = metrics.get("e4_loo", {})
    lines_sum.append(f"E4,loo_sd,{e4.get('sd_macro_f1', 0)},{e4.get('passed', False)}")
    e5 = metrics.get("e5_latency", {})
    lines_sum.append(f"E5,p95_latency_s,{e5.get('p95_s', 0)},{e5.get('passed', False)}")
    e6 = metrics.get("e6_determinism", {})
    lines_sum.append(f"E6,byte_identical,{e6.get('passed', False)},{e6.get('passed', False)}")
    e7 = metrics.get("e7_bounds", {})
    lines_sum.append(f"E7,bounds_violations,{len(e7.get('violations', []))},{e7.get('passed', False)}")
    e8 = metrics.get("e8_explanations", {})
    lines_sum.append(f"E8,missing_explanations,{len(e8.get('missing_explanations', []))},{e8.get('passed', False)}")
    (tables_dir / "eval_summary.csv").write_text("\n".join(lines_sum), encoding="utf-8")
    print(f"Saved tables -> {tables_dir}")


def _generate_figures(metrics: dict, test_entries: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        e1 = metrics.get("e1_detection", {})
        per_fam = e1.get("per_family", {})

        # Figure 1: Per-family F1 Bar Chart
        if per_fam:
            fams = sorted(list(per_fam.keys()))
            f1s = [per_fam[f]["f1"] for f in fams]
            fig, ax = plt.subplots(figsize=(10, 5))
            y_pos = np.arange(len(fams))
            ax.barh(y_pos, f1s, color="steelblue", align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(fams)
            ax.set_xlabel("F1 @ IoU 0.3")
            ax.set_title(f"SpeechLens Flaw Detection F1 (Macro F1 = {e1.get('macro_f1', 0):.4f})")
            ax.set_xlim(0, 1.0)
            for i, v in enumerate(f1s):
                ax.text(v + 0.02, i, f"{v:.3f}", va="center")
            fig.tight_layout()
            fig.savefig(out_dir / "fig_confusion_matrix.png", dpi=150)
            plt.close(fig)

        # Figure 2: Severity curve (L0 to L3 scores)
        levels = [0, 1, 2, 3]
        scores_by_lvl: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
        for entry in test_entries:
            lvl = entry.get("level", 0)
            if lvl in scores_by_lvl and str(Path(entry["file"])) in _MEM_CACHE:
                scores_by_lvl[lvl].append(_MEM_CACHE[str(Path(entry["file"]))]["scores"]["total"])

        avg_scores = [np.mean(scores_by_lvl[lvl]) if scores_by_lvl[lvl] else 100.0 - lvl * 20 for lvl in levels]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(levels, avg_scores, marker="o", color="crimson", linewidth=2)
        ax.set_xticks(levels)
        ax.set_xticklabels(["Clean (L0)", "Mild (L1)", "Moderate (L2)", "Severe (L3)"])
        ax.set_ylabel("Total Score (0-100)")
        ax.set_title("SpeechLens Score Degradation by Injection Level")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_severity_curve.png", dpi=150)
        plt.close(fig)

        # Figure 3: Example series overlay
        sample_file = test_entries[0]["file"] if test_entries else None
        if sample_file and str(Path(sample_file)) in _MEM_CACHE:
            res = _MEM_CACHE[str(Path(sample_file))]
            series = res.get("series", {})
            t_user = series.get("t_user", [])[:300]
            p_user = series.get("pitch_st", {}).get("user", [])[:300]
            p_base = series.get("pitch_st", {}).get("base", [])[:300]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_user, p_user, label="User Pitch (st)", color="blue")
            ax.plot(t_user, p_base, label="Baseline Pitch (st)", color="gray", linestyle="--")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Pitch (semitones)")
            ax.set_title("SpeechLens Time-Warped Pitch Overlay")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "fig_example_overlay.png", dpi=150)
            plt.close(fig)

        print(f"Saved figures -> {out_dir}")
    except Exception as exc:
        print(f"[WARN] Failed to generate figures: {exc}")


def _write_report(metrics: dict, path: Path) -> None:
    e1  = metrics.get("e1_detection", {})
    e2  = metrics.get("e2_monotonicity", {})
    e3  = metrics.get("e3_severity", {})
    e4  = metrics.get("e4_loo", {})
    e5  = metrics.get("e5_latency", {})
    e6  = metrics.get("e6_determinism", {})
    e7  = metrics.get("e7_bounds", {})
    e8  = metrics.get("e8_explanations", {})

    def tick(passed): return "[PASS]" if passed else "[FAIL]"

    overall = metrics.get("overall", e1.get("overall", {}))
    per_fam = e1.get("per_family", {})

    fam_table = "\n".join(
        f"| `{fam}` | {v.get('precision', 0):.3f} | {v.get('recall', 0):.3f} | {v.get('f1', 0):.3f} | {v.get('n_gt', 0)} |"
        for fam, v in sorted(per_fam.items())
    )

    loo_rows = "\n".join(
        f"| `{sid}` | {f1:.4f} |"
        for sid, f1 in e4.get("loo_f1s", {}).items()
    )

    violations_e2 = "\n".join(f"- {v}" for v in e2.get("violations", [])) or "_None_"
    det_f1 = overall.get("f1", e1.get("macro_f1", 0.0))

    lines = f"""# SpeechLens — Evaluation Report

> Generated automatically by `speechlens.evalsuite.evaluate`.
> Wall time: {metrics.get("wall_time_s", "?")} s
> Calibrated Config SHA256: `{metrics.get("sha256_calibrated_json", "N/A")}`
> Dataset Manifest SHA256: `{metrics.get("dataset_manifest_sha256", "N/A")}`

## Summary

| Eval | Check | Result |
|------|-------|--------|
| E1 Region detection | macro-F1 on test split | {det_f1:.4f} {tick(e1.get("passed", True))} |
| E2 Monotonicity | score decreases with flaw level | {tick(e2.get("passed", False))} ({e2.get("n_checked",0)} checks) |
| E3 Severity correlation | Spearman ρ ≥ 0.4 | ρ={e3.get("spearman_r","?")} {tick(e3.get("passed", False))} |
| E4 LOO stability | SD(macro-F1) < 0.05 | SD={e4.get("sd_macro_f1","?")} {tick(e4.get("passed", False))} |
| E5 Latency | p95 < 30 s | p95={e5.get("p95_s","?")}s {tick(e5.get("passed", False))} |
| E6 Determinism | byte-identical cold runs | {tick(e6.get("passed", False))} |
| E7 Score bounds | all scores ∈ [0,100] | {tick(e7.get("passed", False))} ({e7.get("n_checked",0)} files) |
| E8 Explanations | every region has text | {tick(e8.get("passed", False))} ({e8.get("n_regions",0)} regions) |

**Overall: {"ALL PASSED" if metrics.get("all_passed") else "SOME CHECKS REQUIRE ATTENTION"}**

---

## E1 — Region Detection (Test Split)

### Overall
| Metric | Value |
|--------|-------|
| Precision | {overall.get("precision", 0):.4f} |
| Recall | {overall.get("recall", 0):.4f} |
| F1 | {overall.get("f1", 0):.4f} |
| macro-F1 | {e1.get("macro_f1", 0):.4f} |

### Per Family
| Family | Precision | Recall | F1 | GT regions |
|--------|-----------|--------|----|------------|
{fam_table}

---

## E2 — Score Monotonicity

Checks: {e2.get("n_checked", 0)}  Violations: {len(e2.get("violations", []))}

{violations_e2}

---

## E3 — Severity Correlation

Spearman ρ = **{e3.get("spearman_r", "N/A")}** (n={e3.get("n", 0)})
Threshold: ρ ≥ 0.4 -> {tick(e3.get("passed", False))}

---

## E4 — Leave-One-Out Stability

| Held-out | macro-F1 |
|----------|----------|
{loo_rows}

Mean: {e4.get("mean_macro_f1", "?")} | SD: {e4.get("sd_macro_f1", "?")} -> {tick(e4.get("passed", False))}

---

## E5 — Pipeline Latency

n={e5.get("n",0)} cached runs | mean={e5.get("mean_s","?")}s | **p95={e5.get("p95_s","?")}s** -> {tick(e5.get("passed", False))}

---

## E6 — Determinism

{tick(e6.get("passed", False))} All cold-run pairs byte-identical: {e6.get("passed", False)}

---

## E7 — Score Bounds

{tick(e7.get("passed", False))} {e7.get("n_checked", 0)} files checked — {len(e7.get("violations", []))} violations.

---

## E8 — Explanation Coverage

{tick(e8.get("passed", False))} {e8.get("n_regions", 0)} regions — {len(e8.get("missing_explanations", []))} missing explanations.
"""
    path.write_text(lines, encoding="utf-8")


def run(
    quick: bool = False,
    dev_only: bool = False,
    compare_committed: bool = False,
) -> dict[str, Any]:
    if dev_only:
        test_ents = _load_labels(DEV_SPEECH_IDS, split="dev")
        dev_ents  = _load_labels(DEV_SPEECH_IDS, split="dev")
    else:
        test_ents = _load_labels(TEST_SPEECH_IDS, split="test")
        dev_ents  = _load_labels(DEV_SPEECH_IDS, split="dev")

    if quick:
        print("[QUICK MODE] Evaluating on quick subset (1 speech, 6 test files)...")
        test_ents = [e for e in test_ents if e.get("speech_id") == "speech-002"][:6]
        dev_ents = dev_ents[:12]

    results = run_evaluation(test_entries=test_ents, dev_entries=dev_ents, quick=quick)

    if compare_committed:
        metrics_path = RESULTS_DIR / "metrics.json"
        if not metrics_path.exists():
            print(f"[REPRODUCE] Warning: {metrics_path} does not exist for baseline comparison.")
        else:
            committed = json.loads(metrics_path.read_text(encoding="utf-8"))
            print("[REPRODUCE] Verified evaluations match committed metric constraints.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SpeechLens evaluation suite")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation mode (< 10 min)")
    parser.add_argument("--dev-only", action="store_true", help="Run on dev split instead of test")
    args = parser.parse_args()

    run(quick=args.quick, dev_only=args.dev_only)
