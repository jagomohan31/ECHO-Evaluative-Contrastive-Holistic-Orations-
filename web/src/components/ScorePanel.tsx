"use client";

import React from "react";
import { useSpeechLensStore } from "@/lib/store";
import { DimensionScores } from "@/types/analysis";

const DIMENSIONS: { key: keyof DimensionScores; label: string; weight: number; color: string }[] = [
  { key: "pacing", label: "Pacing", weight: 25, color: "#3b82f6" },
  { key: "intonation", label: "Intonation", weight: 25, color: "#8b5cf6" },
  { key: "fluency", label: "Fluency", weight: 20, color: "#10b981" },
  { key: "volume", label: "Volume", weight: 15, color: "#f59e0b" },
  { key: "clarity", label: "Clarity", weight: 15, color: "#ec4899" },
];

function getRating(score: number): { label: string; colorClass: string } {
  if (score >= 90) return { label: "Excellent", colorClass: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20" };
  if (score >= 75) return { label: "Proficient", colorClass: "bg-blue-500/10 text-blue-500 border-blue-500/20" };
  if (score >= 60) return { label: "Competent", colorClass: "bg-amber-500/10 text-amber-500 border-amber-500/20" };
  return { label: "Needs Practice", colorClass: "bg-rose-500/10 text-rose-500 border-rose-500/20" };
}

export function ScorePanel() {
  const { analysisResult } = useSpeechLensStore();

  if (!analysisResult) return null;

  const { scores, global: g, warnings } = analysisResult;
  const rating = getRating(scores.total);

  // Radar chart SVG calculation
  const size = 200;
  const center = size / 2;
  const radius = center - 24;
  const angleStep = (2 * Math.PI) / DIMENSIONS.length;

  const points = DIMENSIONS.map((dim, i) => {
    const val = scores.dimensions[dim.key] ?? 0;
    const r = (val / 100) * radius;
    const angle = i * angleStep - Math.PI / 2;
    const x = center + r * Math.cos(angle);
    const y = center + r * Math.sin(angle);
    return `${x},${y}`;
  }).join(" ");

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100 shadow-xl space-y-6">
      {/* Header & Overall Score */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Speech Performance</h2>
          <p className="text-sm text-slate-400">
            {analysisResult.meta.mode === "reference" ? "Contrastive Baseline Analysis" : "Population Norm Mode"}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <span className="text-3xl font-extrabold tracking-tight text-white">
              {scores.total.toFixed(1)}
            </span>
            <span className="text-slate-400 text-sm"> / 100</span>
          </div>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${rating.colorClass}`}>
            {rating.label}
          </span>
        </div>
      </div>

      {/* Warnings if any */}
      {warnings && warnings.length > 0 && (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 text-xs text-amber-300 space-y-1">
          {warnings.map((w, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400" />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Grid: Dimension Bars + Radar Chart */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
        {/* Dimension Bars */}
        <div className="space-y-4">
          {DIMENSIONS.map((dim) => {
            const val = scores.dimensions[dim.key] ?? 0;
            return (
              <div key={dim.key} className="space-y-1.5">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-300">
                    {dim.label} <span className="text-slate-400 text-[10px]">({dim.weight}%)</span>
                  </span>
                  <span className="text-slate-200 font-semibold">{val.toFixed(1)}</span>
                </div>
                <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min(100, Math.max(0, val))}%`,
                      backgroundColor: dim.color,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Radar Chart */}
        <div className="flex flex-col items-center justify-center p-2">
          <div className="relative w-[200px] h-[200px]">
            <svg width={size} height={size} className="overflow-visible">
              {/* Background rings */}
              {[0.25, 0.5, 0.75, 1.0].map((level) => {
                const ringPoints = DIMENSIONS.map((_, i) => {
                  const r = level * radius;
                  const angle = i * angleStep - Math.PI / 2;
                  return `${center + r * Math.cos(angle)},${center + r * Math.sin(angle)}`;
                }).join(" ");
                return (
                  <polygon
                    key={level}
                    points={ringPoints}
                    fill="none"
                    stroke="#334155"
                    strokeWidth="1"
                    strokeDasharray={level < 1.0 ? "2,2" : undefined}
                  />
                );
              })}

              {/* Axis lines */}
              {DIMENSIONS.map((_, i) => {
                const angle = i * angleStep - Math.PI / 2;
                const x2 = center + radius * Math.cos(angle);
                const y2 = center + radius * Math.sin(angle);
                return (
                  <line
                    key={i}
                    x1={center}
                    y1={center}
                    x2={x2}
                    y2={y2}
                    stroke="#334155"
                    strokeWidth="1"
                  />
                );
              })}

              {/* Radar Area */}
              <polygon
                points={points}
                fill="rgba(59, 130, 246, 0.25)"
                stroke="#3b82f6"
                strokeWidth="2"
              />

              {/* Data points */}
              {DIMENSIONS.map((dim, i) => {
                const val = scores.dimensions[dim.key] ?? 0;
                const r = (val / 100) * radius;
                const angle = i * angleStep - Math.PI / 2;
                const x = center + r * Math.cos(angle);
                const y = center + r * Math.sin(angle);
                return (
                  <circle
                    key={dim.key}
                    cx={x}
                    cy={y}
                    r="3.5"
                    fill={dim.color}
                    stroke="#0f172a"
                    strokeWidth="1.5"
                  />
                );
              })}
            </svg>
          </div>
          <span className="text-[11px] text-slate-400 mt-2">Rubric Competency Balance</span>
        </div>
      </div>

      {/* Global Speaking Statistics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-slate-800">
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/40">
          <div className="text-[11px] text-slate-400 uppercase tracking-wider">Speaking Rate</div>
          <div className="text-base font-semibold text-white mt-0.5">
            {g.articulation_rate_syl_s.toFixed(2)} <span className="text-xs font-normal text-slate-400">syl/s</span>
          </div>
          {g.baseline_articulation_rate_syl_s && (
            <div className="text-[10px] text-slate-400">Base: {g.baseline_articulation_rate_syl_s.toFixed(2)}</div>
          )}
        </div>

        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/40">
          <div className="text-[11px] text-slate-400 uppercase tracking-wider">Tempo</div>
          <div className="text-base font-semibold text-white mt-0.5">
            {Math.round(g.wpm)} <span className="text-xs font-normal text-slate-400">WPM</span>
          </div>
          <div className="text-[10px] text-slate-400">{analysisResult.words.length} words</div>
        </div>

        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/40">
          <div className="text-[11px] text-slate-400 uppercase tracking-wider">Pause Ratio</div>
          <div className="text-base font-semibold text-white mt-0.5">
            {(g.pause_ratio * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-slate-400">{analysisResult.pauses.length} pauses detected</div>
        </div>

        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/40">
          <div className="text-[11px] text-slate-400 uppercase tracking-wider">Pitch Variation</div>
          <div className="text-base font-semibold text-white mt-0.5">
            {g.f0_sd_st.toFixed(2)} <span className="text-xs font-normal text-slate-400">st (SD)</span>
          </div>
          <div className="text-[10px] text-slate-400">Intonation range</div>
        </div>
      </div>
    </div>
  );
}
