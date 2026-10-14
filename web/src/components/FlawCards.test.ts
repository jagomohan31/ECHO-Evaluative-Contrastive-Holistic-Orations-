import { describe, it, expect } from "vitest";
import { FlawRegion } from "@/types/analysis";

// ─── Card ordering helpers (pure) ─────────────────────────────────────────────

function sortByCostDesc(regions: FlawRegion[]): FlawRegion[] {
  return [...regions].sort((a, b) => b.cost - a.cost);
}

// ─── Play-range maths ─────────────────────────────────────────────────────────

/**
 * Given a FlawRegion and the word list, compute the baseline seek time.
 * Uses the base_start of the first word in the region.
 */
function baselineSeekTime(
  region: FlawRegion,
  words: { i: number; start: number; end: number; base_start: number | null; base_end: number | null }[]
): number | null {
  const firstWord = words[region.word_idx[0]];
  return firstWord?.base_start ?? null;
}

const makeRegion = (id: string, cost: number, start: number, end: number): FlawRegion => ({
  id,
  family: "pacing",
  dimension: "pacing",
  start,
  end,
  word_idx: [0, 2],
  severity: 2,
  z_peak: 2.0,
  cost,
  evidence: { z: 2.1 },
  explanation: {
    summary: "Test summary",
    context: "",
    effect: "",
    fix: "",
    math: "z = (x - mu) / sigma",
  },
});

describe("sortByCostDesc", () => {
  it("sorts regions by cost descending", () => {
    const regions = [
      makeRegion("r1", 3.0, 1, 2),
      makeRegion("r2", 7.5, 3, 4),
      makeRegion("r3", 1.2, 5, 6),
    ];
    const sorted = sortByCostDesc(regions);
    expect(sorted[0].id).toBe("r2");
    expect(sorted[1].id).toBe("r1");
    expect(sorted[2].id).toBe("r3");
  });

  it("returns stable order for equal costs", () => {
    const regions = [
      makeRegion("r1", 5.0, 1, 2),
      makeRegion("r2", 5.0, 3, 4),
    ];
    const sorted = sortByCostDesc(regions);
    expect(sorted).toHaveLength(2);
    expect(sorted[0].cost).toBe(5.0);
  });

  it("does not mutate the original array", () => {
    const original = [makeRegion("r1", 3.0, 1, 2), makeRegion("r2", 7.5, 3, 4)];
    const sorted = sortByCostDesc(original);
    expect(original[0].id).toBe("r1"); // untouched
    expect(sorted[0].id).toBe("r2");
  });
});

describe("baselineSeekTime", () => {
  const words = [
    { i: 0, start: 1.0, end: 1.5, base_start: 0.8, base_end: 1.3 },
    { i: 1, start: 1.6, end: 2.0, base_start: 1.4, base_end: 1.8 },
    { i: 2, start: 2.1, end: 2.6, base_start: 1.9, base_end: 2.4 },
  ];

  it("returns base_start of the first word in the region", () => {
    const region = makeRegion("r1", 5.0, 1.0, 2.6);
    region.word_idx = [0, 2];
    expect(baselineSeekTime(region, words)).toBeCloseTo(0.8, 5);
  });

  it("returns null when base_start is null", () => {
    const wordsNoBase = words.map((w) => ({ ...w, base_start: null }));
    const region = makeRegion("r1", 5.0, 1.0, 2.6);
    expect(baselineSeekTime(region, wordsNoBase)).toBeNull();
  });
});
