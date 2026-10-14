import { describe, it, expect } from "vitest";
import {
  zToColor,
  familyColor,
  buildUserToBaseMap,
  buildBaseToUserMap,
  regionToShape,
  chartClickToSeekTime,
  regionsAtTime,
} from "./vizHelpers";
import { FlawRegion } from "@/types/analysis";

// ─── Colour scale ─────────────────────────────────────────────────────────

describe("zToColor", () => {
  it("returns slate for |z| < 1", () => {
    expect(zToColor(0)).toBe("#94a3b8");
    expect(zToColor(0.9)).toBe("#94a3b8");
    expect(zToColor(-0.9)).toBe("#94a3b8");
  });

  it("returns amber for 1 <= |z| < 2", () => {
    expect(zToColor(1.0)).toBe("#f59e0b");
    expect(zToColor(-1.5)).toBe("#f59e0b");
  });

  it("returns rose for |z| >= 2", () => {
    expect(zToColor(2.0)).toBe("#ef4444");
    expect(zToColor(-3.5)).toBe("#ef4444");
  });
});

describe("familyColor", () => {
  it("returns blue for pacing", () => {
    expect(familyColor("pacing")).toBe("#3b82f6");
  });
  it("falls back to default for unknown family", () => {
    expect(familyColor("unknown_xyz")).toBe("#94a3b8");
  });
});

// ─── Time-warp mapping ──────────────────────────────────────────────────────

describe("buildUserToBaseMap", () => {
  it("returns identity-like for empty arrays", () => {
    const map = buildUserToBaseMap([], []);
    expect(map(3.0)).toBe(3.0);
  });

  it("maps interior point via linear interpolation", () => {
    // tUser=[0,2,4], tBase=[0,3,6] → ratio 1.5
    const map = buildUserToBaseMap([0, 2, 4], [0, 3, 6]);
    expect(map(1)).toBeCloseTo(1.5, 5);
    expect(map(2)).toBeCloseTo(3.0, 5);
    expect(map(3)).toBeCloseTo(4.5, 5);
  });

  it("extrapolates linearly beyond right boundary", () => {
    const map = buildUserToBaseMap([0, 1], [0, 2]);
    // past end: last base point + delta * 1
    expect(map(2)).toBeCloseTo(3.0, 5);
  });
});

describe("buildBaseToUserMap", () => {
  it("inverts the user-to-base mapping for interior points", () => {
    const tUser = [0, 2, 4];
    const tBase = [0, 3, 6];
    const fwd = buildUserToBaseMap(tUser, tBase);
    const inv = buildBaseToUserMap(tUser, tBase);
    // Round-trip: user->base->user should recover original
    expect(inv(fwd(1))).toBeCloseTo(1, 4);
    expect(inv(fwd(3))).toBeCloseTo(3, 4);
  });
});

// ─── Region-to-shape ────────────────────────────────────────────────────────

const mockRegion: FlawRegion = {
  id: "r-001",
  family: "pacing",
  dimension: "pacing",
  start: 2.0,
  end: 4.0,
  word_idx: [5, 8],
  severity: 2,
  z_peak: 2.5,
  cost: 5.0,
  evidence: {},
  explanation: {
    summary: "Too fast",
    context: "",
    effect: "",
    fix: "",
    math: "",
  },
};

describe("regionToShape", () => {
  it("uses baseline coords when useWarp=true", () => {
    const shape = regionToShape(mockRegion, "x", "y", true);
    expect(shape.x0).toBe(2.0);
    expect(shape.x1).toBe(4.0);
    expect(shape.fillcolor).toBe("#3b82f6"); // pacing
    expect(shape.type).toBe("rect");
    expect(shape.layer).toBe("below");
  });

  it("applies user-time mapper when useWarp=false", () => {
    // baseline [2,4] → user with mapper t/2
    const shape = regionToShape(mockRegion, "x", "y", false, (t) => t / 2);
    expect(shape.x0).toBeCloseTo(1.0, 5);
    expect(shape.x1).toBeCloseTo(2.0, 5);
  });

  it("uses identity mapper when useWarp=false and no mapper provided", () => {
    const shape = regionToShape(mockRegion, "x", "y", false);
    expect(shape.x0).toBe(2.0);
    expect(shape.x1).toBe(4.0);
  });
});

// ─── chartClickToSeekTime ───────────────────────────────────────────────────

describe("chartClickToSeekTime", () => {
  it("maps base time back to user time", () => {
    const tUser = [0, 2, 4];
    const tBase = [0, 3, 6];
    // baseTime=1.5 → alpha=0.5 → userTime=1.0
    expect(chartClickToSeekTime(1.5, tUser, tBase)).toBeCloseTo(1.0, 4);
  });
});

// ─── regionsAtTime ──────────────────────────────────────────────────────────

describe("regionsAtTime", () => {
  const regions: FlawRegion[] = [
    { ...mockRegion, id: "r-001", start: 1.0, end: 3.0 },
    { ...mockRegion, id: "r-002", start: 2.5, end: 5.0 },
    { ...mockRegion, id: "r-003", start: 6.0, end: 8.0 },
  ];

  it("returns ids for all regions containing the time", () => {
    const ids = regionsAtTime(regions, 2.8);
    expect(ids).toContain("r-001");
    expect(ids).toContain("r-002");
    expect(ids).not.toContain("r-003");
  });

  it("returns empty when no region matches", () => {
    expect(regionsAtTime(regions, 0.5)).toHaveLength(0);
  });
});
