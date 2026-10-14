/**
 * Pure helper utilities for the visualization components.
 * These are fully testable without DOM/canvas.
 */

import { FlawRegion } from "@/types/analysis";

// ─── Colour scale ────────────────────────────────────────────────────────────

/**
 * Colour-blind-safe palette for flaw families.
 * Keys must match the `family` field in regions.
 */
export const FAMILY_COLORS: Record<string, string> = {
  pacing: "#3b82f6",       // blue
  intonation: "#8b5cf6",   // violet
  fluency: "#10b981",      // emerald
  volume: "#f59e0b",       // amber
  clarity: "#ec4899",      // pink
  pause: "#06b6d4",        // cyan
  default: "#94a3b8",      // slate
};

/**
 * Map a z-score to a CSS colour string.
 * |z| < 1  → neutral slate
 * 1-2      → amber
 * >2       → rose
 */
export function zToColor(z: number): string {
  const abs = Math.abs(z);
  if (abs < 1) return "#94a3b8";
  if (abs < 2) return "#f59e0b";
  return "#ef4444";
}

/**
 * Return a CSS colour for a word based on its maximum absolute z score across all series.
 */
export function wordZColor(maxAbsZ: number): string {
  return zToColor(maxAbsZ);
}

/**
 * Get a consistent colour for a flaw family.
 */
export function familyColor(family: string): string {
  return FAMILY_COLORS[family] ?? FAMILY_COLORS.default;
}

// ─── Time-warp mapping ───────────────────────────────────────────────────────

/**
 * Build a piecewise-linear warp map from parallel time arrays.
 * Returns a function that maps participant time → baseline time.
 */
export function buildUserToBaseMap(tUser: number[], tBase: number[]): (t: number) => number {
  if (tUser.length === 0 || tUser.length !== tBase.length) return (t) => t;
  return (t: number): number => {
    if (t <= tUser[0]) return tBase[0] + (t - tUser[0]);
    if (t >= tUser[tUser.length - 1]) {
      return tBase[tBase.length - 1] + (t - tUser[tUser.length - 1]);
    }
    // Binary search for the interval
    let lo = 0;
    let hi = tUser.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (tUser[mid] <= t) lo = mid;
      else hi = mid;
    }
    const alpha = (t - tUser[lo]) / (tUser[hi] - tUser[lo]);
    return tBase[lo] + alpha * (tBase[hi] - tBase[lo]);
  };
}

/**
 * Build a piecewise-linear warp map that maps baseline time → participant time.
 */
export function buildBaseToUserMap(tUser: number[], tBase: number[]): (t: number) => number {
  // Swap arguments — base is x, user is y
  return buildUserToBaseMap(tBase, tUser);
}

// ─── Region-to-Plotly-shape conversion ──────────────────────────────────────

export interface PlotlyShape {
  type: "rect";
  xref: string;
  yref: string;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  fillcolor: string;
  opacity: number;
  line: { width: number };
  layer: "below";
}

/**
 * Convert a flaw region to a Plotly layout.shape spanning all y.
 * @param region  The flaw region from the analysis result (start/end are baseline time)
 * @param xAxis   The Plotly x-axis ref (e.g. "x", "x2")
 * @param yAxis   The Plotly y-axis ref (e.g. "y", "y2", "y3")
 * @param useWarp When true, use region.start/end (baseline-time coords).
 *                When false, use participant time via the provided mapper.
 * @param toUserTime Optional mapper from baseline to user time (used when useWarp=false)
 */
export function regionToShape(
  region: FlawRegion,
  xAxis: string,
  yAxis: string,
  useWarp: boolean,
  toUserTime?: (t: number) => number
): PlotlyShape {
  let x0: number;
  let x1: number;
  if (useWarp) {
    x0 = region.start;
    x1 = region.end;
  } else {
    const mapper = toUserTime ?? ((t) => t);
    x0 = mapper(region.start);
    x1 = mapper(region.end);
  }
  return {
    type: "rect",
    xref: xAxis,
    yref: yAxis,
    x0,
    x1,
    y0: 0,
    y1: 1,
    fillcolor: familyColor(region.family),
    opacity: 0.18,
    line: { width: 0 },
    layer: "below",
  };
}

/**
 * Map a baseline-time click on the Plotly chart to the participant's real-time position.
 * Returns the participant seek time in seconds.
 */
export function chartClickToSeekTime(
  baseTime: number,
  tUser: number[],
  tBase: number[]
): number {
  return buildBaseToUserMap(tUser, tBase)(baseTime);
}

// ─── Region click helpers ────────────────────────────────────────────────────

/**
 * Find all region IDs that contain a given time point (in baseline coords).
 */
export function regionsAtTime(regions: FlawRegion[], baseTime: number): string[] {
  return regions
    .filter((r) => r.start <= baseTime && baseTime <= r.end)
    .map((r) => r.id);
}
