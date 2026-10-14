"use client";

import React, { useState, useCallback } from "react";
import { useSpeechLensStore } from "@/lib/store";
import { familyColor } from "@/lib/vizHelpers";
import { FlawRegion } from "@/types/analysis";

interface FlawCardsProps {
  onSeekUser?: (timeS: number) => void;
  onSeekBase?: (timeS: number) => void;
}

const SEVERITY_LABEL = ["", "Mild", "Moderate", "Severe", "Critical"];

function EvidenceTable({ evidence }: { evidence: FlawRegion["evidence"] }) {
  const rows = Object.entries(evidence).filter(([, v]) => v != null);
  if (rows.length === 0) return null;
  return (
    <table className="w-full text-[11px] mt-1.5 border-collapse">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k} className="border-t border-slate-700/50">
            <td className="py-0.5 pr-3 text-slate-400 font-medium capitalize">
              {k.replace(/_/g, " ")}
            </td>
            <td className="py-0.5 text-slate-200 text-right font-mono">
              {typeof v === "number" ? v.toFixed(3) : String(v)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function FlawCards({ onSeekUser, onSeekBase }: FlawCardsProps) {
  const { analysisResult, selectedRegionId, selectRegion } = useSpeechLensStore();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const handleSelectRegion = useCallback(
    (region: FlawRegion) => {
      selectRegion(region);
    },
    [selectRegion]
  );

  const handleLocate = useCallback(
    (region: FlawRegion) => {
      selectRegion(region);
      onSeekUser?.(region.start);
    },
    [selectRegion, onSeekUser]
  );

  if (!analysisResult) return null;

  // Sort by cost descending
  const sorted = [...analysisResult.regions].sort((a, b) => b.cost - a.cost);

  if (sorted.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-400 text-sm text-center">
        No flaw regions detected. Great delivery!
      </div>
    );
  }

  return (
    <div className="space-y-3" aria-label="Flaw cards">
      {sorted.map((region) => {
        const color = familyColor(region.family);
        const isSelected = selectedRegionId === region.id;
        const isExpanded = expandedId === region.id;

        return (
          <article
            key={region.id}
            className={[
              "bg-slate-900 border rounded-xl p-4 transition-all cursor-pointer",
              isSelected
                ? "border-blue-500/60 ring-1 ring-blue-500/30"
                : "border-slate-800 hover:border-slate-700",
            ].join(" ")}
            onClick={() => handleSelectRegion(region)}
            aria-current={isSelected ? "true" : undefined}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                handleSelectRegion(region);
              }
            }}
          >
            {/* Header row */}
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-2 flex-wrap">
                {/* Time chip */}
                <span className="text-[11px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                  {region.start.toFixed(2)}s – {region.end.toFixed(2)}s
                </span>
                {/* Family badge */}
                <span
                  className="text-[11px] font-semibold px-2 py-0.5 rounded-full border"
                  style={{
                    backgroundColor: color + "22",
                    color,
                    borderColor: color + "55",
                  }}
                >
                  {region.family}
                </span>
                {/* Severity */}
                <span className="text-[11px] text-slate-400">
                  {SEVERITY_LABEL[region.severity] ?? `L${region.severity}`}
                </span>
              </div>

              {/* Cost badge */}
              <span className="shrink-0 text-sm font-bold text-rose-400" title="Rubric cost">
                −{region.cost.toFixed(1)} pts
              </span>
            </div>

            {/* Summary */}
            <p className="mt-2 text-sm text-slate-200">
              {region.explanation.summary}
            </p>

            {/* Fix hint */}
            <p className="mt-1 text-[12px] text-emerald-400 font-medium">
              +{region.cost.toFixed(1)} pts if fixed
            </p>

            {/* Action buttons */}
            <div className="flex flex-wrap gap-2 mt-3">
              <button
                className="text-[11px] px-3 py-1 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 text-blue-400 rounded-full transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  onSeekUser?.(region.start);
                  selectRegion(region);
                }}
                aria-label={`Play your version at ${region.start.toFixed(2)}s`}
              >
                ▶ Play yours
              </button>

              {region.word_idx && (
                <button
                  className="text-[11px] px-3 py-1 bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/30 text-violet-400 rounded-full transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    // Seek baseline audio using base_start of first word in region
                    const firstWord = analysisResult.words[region.word_idx[0]];
                    if (firstWord?.base_start != null) {
                      onSeekBase?.(firstWord.base_start);
                    }
                    selectRegion(region);
                  }}
                  aria-label={`Play baseline version for region ${region.id}`}
                >
                  ▶ Play baseline
                </button>
              )}

              <button
                className="text-[11px] px-3 py-1 bg-slate-700/50 hover:bg-slate-700 border border-slate-600/50 text-slate-300 rounded-full transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  setExpandedId(isExpanded ? null : region.id);
                }}
                aria-expanded={isExpanded}
                aria-label={`Show math for region ${region.id}`}
              >
                {isExpanded ? "Hide math" : "Show math"}
              </button>

              <button
                className="text-[11px] px-3 py-1 bg-slate-700/50 hover:bg-slate-700 border border-slate-600/50 text-slate-300 rounded-full transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  handleLocate(region);
                }}
                aria-label={`Locate region ${region.id} on chart`}
              >
                Locate on chart
              </button>
            </div>

            {/* Expanded: evidence table + math + context */}
            {isExpanded && (
              <div
                className="mt-4 pt-4 border-t border-slate-800 space-y-3"
                onClick={(e) => e.stopPropagation()}
              >
                {/* Evidence table */}
                {Object.keys(region.evidence).length > 0 && (
                  <div>
                    <h4 className="text-[11px] text-slate-400 uppercase tracking-wider mb-1">
                      Evidence
                    </h4>
                    <EvidenceTable evidence={region.evidence} />
                  </div>
                )}

                {/* Math explanation */}
                {region.explanation.math && (
                  <div>
                    <h4 className="text-[11px] text-slate-400 uppercase tracking-wider mb-1">
                      Formula
                    </h4>
                    <code className="text-[11px] text-slate-300 block bg-slate-800/80 rounded p-2 font-mono whitespace-pre-wrap">
                      {region.explanation.math}
                    </code>
                  </div>
                )}

                {/* Effect and fix */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-[12px]">
                  {region.explanation.effect && (
                    <div>
                      <span className="text-slate-400 font-medium">Effect: </span>
                      <span className="text-slate-300">{region.explanation.effect}</span>
                    </div>
                  )}
                  {region.explanation.fix && (
                    <div>
                      <span className="text-emerald-400 font-medium">Fix: </span>
                      <span className="text-slate-300">{region.explanation.fix}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </article>
        );
      })}
    </div>
  );
}
