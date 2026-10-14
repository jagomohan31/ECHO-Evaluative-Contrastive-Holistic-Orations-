"use client";

import React, { useCallback, useRef } from "react";
import { useSpeechLensStore } from "@/lib/store";
import { ControlPanel } from "@/components/ControlPanel";
import { ScorePanel } from "@/components/ScorePanel";
import { WaveformViewer } from "@/components/WaveformViewer";
import { OverlayChart } from "@/components/OverlayChart";
import { TranscriptStrip } from "@/components/TranscriptStrip";
import { FlawCards } from "@/components/FlawCards";
import { LoadingSkeleton } from "@/components/LoadingSkeleton";
import { ErrorAlert } from "@/components/ErrorAlert";

export default function DashboardPage() {
  const { isAnalyzing, analysisResult, error, setError, audioUrl, warpMode, setWarpMode } =
    useSpeechLensStore();

  // Waveform ref for seek control from other components
  const waveformSeekRef = useRef<((t: number) => void) | null>(null);

  const handleSeek = useCallback((t: number) => {
    waveformSeekRef.current?.(t);
  }, []);

  return (
    <div className="space-y-8">
      {/* Intro hero banner */}
      <div className="space-y-2">
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
          Contrastive Oratory Analytics Dashboard
        </h1>
        <p className="text-sm text-slate-400 max-w-3xl">
          Upload or record oratorical speech to detect delivery flaws, inspect rubric deductions,
          and explore time-warped pitch, energy, and pacing overlays against historical reference readers.
        </p>
      </div>

      {/* Control Panel (Speech selector, upload, recording, transcript, analyze) */}
      <ControlPanel />

      {/* Error alert if any */}
      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {/* Dynamic results or loading skeleton */}
      {isAnalyzing && <LoadingSkeleton />}

      {!isAnalyzing && analysisResult && (
        <div className="space-y-6">
          {/* Score Panel */}
          <ScorePanel />

          {/* Waveform (user audio) */}
          {audioUrl && (
            <WaveformViewer audioUrl={audioUrl} />
          )}

          {/* Warp mode toggle */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400 font-medium">x-axis:</span>
            <button
              onClick={() => setWarpMode("base")}
              className={[
                "px-3 py-1 rounded-full text-xs font-semibold border transition-colors",
                warpMode === "base"
                  ? "bg-blue-500/20 text-blue-400 border-blue-500/40"
                  : "bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200",
              ].join(" ")}
              aria-pressed={warpMode === "base"}
            >
              Baseline time (warped)
            </button>
            <button
              onClick={() => setWarpMode("user")}
              className={[
                "px-3 py-1 rounded-full text-xs font-semibold border transition-colors",
                warpMode === "user"
                  ? "bg-blue-500/20 text-blue-400 border-blue-500/40"
                  : "bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200",
              ].join(" ")}
              aria-pressed={warpMode === "user"}
            >
              Your real time
            </button>
          </div>

          {/* Overlay Chart (Plotly 3-panel) */}
          <OverlayChart onSeek={handleSeek} />

          {/* Transcript Strip */}
          <TranscriptStrip onSeek={handleSeek} />

          {/* Flaw Cards sorted by cost */}
          <div>
            <h2 className="text-base font-semibold text-slate-200 mb-3">
              Flaw Analysis
            </h2>
            <FlawCards onSeekUser={handleSeek} />
          </div>
        </div>
      )}
    </div>
  );
}
