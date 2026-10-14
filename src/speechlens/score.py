"""
score.py -- Scoring rubric and cost calculation (DESIGN 10).

Implements:
  10.1 Dimension weights (pacing 25, intonation 25, fluency 20, volume 15, clarity 15)
  10.2 Region deductions f_r = K * s_r * (0.5 + 0.5 * c_r) and dimension score S_d
  10.3 Region cost by removal (+X pts if fixed)
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np

from speechlens.config import K as DEFAULT_K, WEIGHTS as DEFAULT_WEIGHTS

DIMENSIONS = ["pacing", "intonation", "fluency", "volume", "clarity"]


def compute_region_deduction(
    region: Any,
    total_duration: float,
    k: float = DEFAULT_K,
) -> float:
    """
    Calculate fraction f_r = K * s_r * (0.5 + 0.5 * c_r).
    c_r is region duration relative to passage duration (bounded [0, 1]).
    """
    start = getattr(region, "start", region.get("start", 0.0) if isinstance(region, dict) else 0.0)
    end = getattr(region, "end", region.get("end", 0.0) if isinstance(region, dict) else 0.0)
    sev = getattr(region, "severity", region.get("severity", 0.0) if isinstance(region, dict) else 0.0)

    reg_dur = max(0.0, end - start)
    total_dur = max(reg_dur, float(total_duration), 1e-6)
    c_r = float(np.clip(reg_dur / total_dur, 0.0, 1.0))
    s_r = float(np.clip(sev, 0.0, 1.0))

    f_r = k * s_r * (0.5 + 0.5 * c_r)
    return float(f_r)


def compute_scores(
    regions: list[Any],
    total_duration: float,
    weights: dict[str, float] | None = None,
    k: float = DEFAULT_K,
) -> dict[str, Any]:
    """
    Compute dimension scores (0-100) and weighted total score (0-100).
    S_d = 100 * max(0, 1 - sum(f_r in dimension))
    Total = sum(W_d * S_d / 100)
    """
    w_map = weights if weights is not None else DEFAULT_WEIGHTS
    dim_deductions = {d: 0.0 for d in DIMENSIONS}

    for r in regions:
        dim = getattr(r, "dimension", r.get("dimension") if isinstance(r, dict) else None)
        if dim in dim_deductions:
            f_r = compute_region_deduction(r, total_duration, k=k)
            dim_deductions[dim] += f_r

    dim_scores = {}
    for d in DIMENSIONS:
        s_d = 100.0 * max(0.0, 1.0 - dim_deductions[d])
        dim_scores[d] = round(float(np.clip(s_d, 0.0, 100.0)), 4)

    total_score = sum(w_map.get(d, 0.0) * dim_scores[d] / 100.0 for d in DIMENSIONS)
    total_score = float(np.clip(total_score, 0.0, 100.0))

    return {
        "total": round(total_score, 4),
        "dimensions": dim_scores,
    }


def compute_region_costs(
    regions: list[Any],
    total_duration: float,
    weights: dict[str, float] | None = None,
    k: float = DEFAULT_K,
) -> list[Any]:
    """
    Compute cost for each region: total score with region removed minus current total.
    Populates `r.cost` and returns the regions list.
    """
    if not regions:
        return []

    base_scores = compute_scores(regions, total_duration, weights=weights, k=k)
    base_total = base_scores["total"]

    for r in regions:
        # Evaluate without this region
        others = [other for other in regions if other is not r]
        scores_without = compute_scores(others, total_duration, weights=weights, k=k)
        cost = max(0.0, scores_without["total"] - base_total)
        cost_rounded = round(cost, 4)
        if hasattr(r, "cost"):
            r.cost = cost_rounded
        elif isinstance(r, dict):
            r["cost"] = cost_rounded

    return regions
