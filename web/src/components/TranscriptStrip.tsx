"use client";

import React, { useCallback } from "react";
import { useSpeechLensStore } from "@/lib/store";
import { zToColor } from "@/lib/vizHelpers";
import { WordItem } from "@/types/analysis";

interface TranscriptStripProps {
  onSeek?: (timeS: number) => void;
}

function maxAbsZ(word: WordItem): number {
  const vals = [word.z.tempo, word.z.pitch, word.z.energy, word.z.pause]
    .filter((v): v is number => v != null);
  if (vals.length === 0) return 0;
  return Math.max(...vals.map(Math.abs));
}

export function TranscriptStrip({ onSeek }: TranscriptStripProps) {
  const { analysisResult, selectedWordIdx, selectWord } = useSpeechLensStore();

  const handleWordClick = useCallback(
    (word: WordItem) => {
      selectWord(word.i);
      onSeek?.(word.start);
    },
    [selectWord, onSeek]
  );

  if (!analysisResult) return null;

  const { words } = analysisResult;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Transcript
        </h3>
        <span className="text-[11px] text-slate-500">
          colour = z-deviation · click a word to seek
        </span>
      </div>

      <div
        className="flex flex-wrap gap-x-1 gap-y-2 leading-loose"
        aria-label="Colour-coded transcript strip"
      >
        {words.map((word) => {
          const z = maxAbsZ(word);
          const color = zToColor(z);
          const isSelected = selectedWordIdx === word.i;
          const isProminent = (word.prominence.base ?? 0) > 0.5;
          const isUserProminent = (word.prominence.user ?? 0) > 0.5;
          const hasFlaw = word.flags.length > 0;

          return (
            <button
              key={word.i}
              onClick={() => handleWordClick(word)}
              title={`z≈${z.toFixed(2)} · t=${word.start.toFixed(2)}-${word.end.toFixed(2)}s${hasFlaw ? " · " + word.flags.join(", ") : ""}`}
              className={[
                "relative px-1 py-0.5 rounded text-sm font-medium transition-all cursor-pointer",
                "focus:outline-none focus:ring-2 focus:ring-blue-500",
                isSelected
                  ? "ring-2 ring-blue-400 bg-slate-700"
                  : "hover:bg-slate-800",
              ].join(" ")}
              style={{ color }}
              aria-pressed={isSelected}
              aria-label={`${word.text}, deviation ${z.toFixed(1)} sigma`}
            >
              {word.text}
              {/* Stress markers */}
              {isProminent && (
                <span
                  className="absolute -top-1 left-1/2 -translate-x-1/2 text-[8px] text-violet-400 leading-none"
                  aria-hidden="true"
                >
                  ●
                </span>
              )}
              {isUserProminent && !isProminent && (
                <span
                  className="absolute -top-1 left-1/2 -translate-x-1/2 text-[8px] text-blue-400 leading-none"
                  aria-hidden="true"
                >
                  ○
                </span>
              )}
              {/* Flaw underline */}
              {hasFlaw && (
                <span
                  className="absolute bottom-0 left-0 right-0 h-0.5 rounded"
                  style={{ backgroundColor: color }}
                  aria-hidden="true"
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex gap-4 mt-3 text-[11px] text-slate-500">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-[#94a3b8]" />
          Normal
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-[#f59e0b]" />
          Moderate deviation
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-[#ef4444]" />
          High deviation
        </span>
        <span className="flex items-center gap-1">
          <span className="text-violet-400">●</span>
          Baseline stress
        </span>
        <span className="flex items-center gap-1">
          <span className="text-blue-400">○</span>
          Your stress
        </span>
      </div>
    </div>
  );
}
