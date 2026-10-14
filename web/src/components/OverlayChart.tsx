"use client";

import React, { useEffect, useRef, useCallback } from "react";
import { useSpeechLensStore } from "@/lib/store";
import { regionToShape, buildBaseToUserMap, chartClickToSeekTime } from "@/lib/vizHelpers";
import { FlawRegion, SeriesData, PauseItem } from "@/types/analysis";

interface OverlayChartProps {
  onSeek?: (timeS: number) => void;
}

export function OverlayChart({ onSeek }: OverlayChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<any>(null);

  const { analysisResult, warpMode, selectedRegionId, selectRegion } =
    useSpeechLensStore();

  const buildPlot = useCallback(async () => {
    if (!containerRef.current || !analysisResult) return;

    const Plotly = await import("plotly.js-dist-min" as any);
    const { series, regions, pauses } = analysisResult;
    const { t_user, t_base, pitch_st, energy_db, tempo } = series;

    const xUser = t_user;
    const xBase = t_base;

    // x-axis depends on warp toggle: baseline time or participant real time
    const xPitch = warpMode === "base" ? xBase : xUser;
    const xEnergy = warpMode === "base" ? xBase : xUser;
    const xTempo = warpMode === "base" ? xBase : xUser;

    const toUserTime = buildBaseToUserMap(t_user, t_base);

    const regionShapes = regions.flatMap((r: FlawRegion) => [
      regionToShape(r, "x", "y domain", warpMode === "base", toUserTime),
      regionToShape(r, "x", "y2 domain", warpMode === "base", toUserTime),
      regionToShape(r, "x", "y3 domain", warpMode === "base", toUserTime),
    ]);

    // Highlight selected region with more opacity
    const highlightShapes = selectedRegionId
      ? regions
          .filter((r: FlawRegion) => r.id === selectedRegionId)
          .flatMap((r: FlawRegion) => [
            { ...regionToShape(r, "x", "y domain", warpMode === "base", toUserTime), opacity: 0.35 },
            { ...regionToShape(r, "x", "y2 domain", warpMode === "base", toUserTime), opacity: 0.35 },
            { ...regionToShape(r, "x", "y3 domain", warpMode === "base", toUserTime), opacity: 0.35 },
          ])
      : [];

    // Pause bars in tempo panel
    const pauseShapes = pauses
      .filter((p: PauseItem) => !p.expected)
      .map((p: PauseItem) => ({
        type: "rect" as const,
        xref: "x",
        yref: "y3 domain",
        x0: warpMode === "base" ? p.start : toUserTime(p.start),
        x1: warpMode === "base" ? p.end : toUserTime(p.end),
        y0: 0,
        y1: 1,
        fillcolor: "#06b6d4",
        opacity: 0.15,
        line: { width: 0 },
        layer: "below" as const,
      }));

    const pitchUser = pitch_st.user.map((v) => v ?? null);
    const pitchBase = pitch_st.base.map((v) => v ?? null);
    const pitchLo = pitch_st.lo.map((v) => v ?? null);
    const pitchHi = pitch_st.hi.map((v) => v ?? null);

    const data: any[] = [
      // ── Pitch panel (row 1) ──────────────────────────────────────────
      {
        x: xBase, y: pitchHi, name: "Pitch hi", xaxis: "x", yaxis: "y",
        type: "scatter", mode: "lines", line: { color: "#334155", width: 0 },
        fill: "none", showlegend: false, hoverinfo: "skip",
      },
      {
        x: xBase, y: pitchLo, name: "Pitch lo", xaxis: "x", yaxis: "y",
        type: "scatter", mode: "lines", line: { color: "#334155", width: 0 },
        fill: "tonexty", fillcolor: "rgba(51,65,85,0.35)", showlegend: false, hoverinfo: "skip",
      },
      {
        x: xBase, y: pitchBase, name: "Baseline pitch", xaxis: "x", yaxis: "y",
        type: "scatter", mode: "lines", line: { color: "#8b5cf6", width: 1.5, dash: "dash" },
        showlegend: true,
      },
      {
        x: xUser, y: pitchUser, name: "Your pitch", xaxis: "x", yaxis: "y",
        type: "scatter", mode: "lines", line: { color: "#a78bfa", width: 2 },
        showlegend: true,
        hovertemplate: "t=%{x:.2f}s<br>Pitch=%{y:.1f} st<extra></extra>",
      },

      // ── Energy panel (row 2) ─────────────────────────────────────────
      {
        x: xBase, y: energy_db.hi, name: "Energy hi", xaxis: "x", yaxis: "y2",
        type: "scatter", mode: "lines", line: { color: "#334155", width: 0 },
        fill: "none", showlegend: false, hoverinfo: "skip",
      },
      {
        x: xBase, y: energy_db.lo, name: "Energy lo", xaxis: "x", yaxis: "y2",
        type: "scatter", mode: "lines", line: { color: "#334155", width: 0 },
        fill: "tonexty", fillcolor: "rgba(51,65,85,0.35)", showlegend: false, hoverinfo: "skip",
      },
      {
        x: xBase, y: energy_db.base, name: "Baseline energy", xaxis: "x", yaxis: "y2",
        type: "scatter", mode: "lines", line: { color: "#f59e0b", width: 1.5, dash: "dash" },
        showlegend: true,
      },
      {
        x: xUser, y: energy_db.user, name: "Your energy", xaxis: "x", yaxis: "y2",
        type: "scatter", mode: "lines", line: { color: "#fbbf24", width: 2 },
        showlegend: true,
        hovertemplate: "t=%{x:.2f}s<br>Energy=%{y:.1f} dB<extra></extra>",
      },

      // ── Tempo panel (row 3) ──────────────────────────────────────────
      {
        x: xBase, y: tempo.base, name: "Baseline tempo", xaxis: "x", yaxis: "y3",
        type: "scatter", mode: "lines", line: { color: "#10b981", width: 1.5, dash: "dash" },
        showlegend: true,
      },
      {
        x: xUser, y: tempo.user, name: "Your tempo", xaxis: "x", yaxis: "y3",
        type: "scatter", mode: "lines+markers", line: { color: "#34d399", width: 2 },
        marker: { size: 4, color: "#34d399" },
        showlegend: true,
        hovertemplate: "t=%{x:.2f}s<br>Tempo=%{y:.2f} syl/s<extra></extra>",
      },
    ];

    const layout: any = {
      paper_bgcolor: "#0f172a",
      plot_bgcolor: "#0f172a",
      font: { color: "#94a3b8", size: 11 },
      margin: { t: 10, b: 40, l: 55, r: 10 },
      height: 420,
      grid: { rows: 3, columns: 1, pattern: "independent", roworder: "top to bottom" },
      xaxis: { color: "#475569", gridcolor: "#1e293b", zeroline: false, title: "" },
      yaxis: { color: "#475569", gridcolor: "#1e293b", zeroline: false, title: "Pitch (st)", domain: [0.7, 1.0] },
      yaxis2: { color: "#475569", gridcolor: "#1e293b", zeroline: false, title: "Energy (dB)", domain: [0.37, 0.66] },
      yaxis3: { color: "#475569", gridcolor: "#1e293b", zeroline: false, title: "Tempo (syl/s)", domain: [0.0, 0.33] },
      legend: { bgcolor: "rgba(15,23,42,0.8)", bordercolor: "#1e293b", borderwidth: 1, x: 1.01, y: 1 },
      shapes: [...pauseShapes, ...regionShapes, ...highlightShapes],
      clickmode: "event",
    };

    if (!plotRef.current) {
      plotRef.current = await Plotly.newPlot(containerRef.current, data, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
        displaylogo: false,
      });

      // Click handler: translate chart click → seek time
      (containerRef.current as any).on("plotly_click", (eventData: any) => {
        if (!eventData?.points?.length) return;
        const baseTime = eventData.points[0].x;
        const seekTime = warpMode === "base"
          ? chartClickToSeekTime(baseTime, t_user, t_base)
          : baseTime;

        // Find the region at this time
        const hit = regions.find(
          (r: FlawRegion) => r.start <= baseTime && baseTime <= r.end
        );
        if (hit) selectRegion(hit);
        onSeek?.(seekTime);
      });
    } else {
      await Plotly.react(containerRef.current, data, layout);
    }
  }, [analysisResult, warpMode, selectedRegionId, onSeek, selectRegion]);

  useEffect(() => {
    buildPlot();
  }, [buildPlot]);

  if (!analysisResult) return null;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Pitch · Energy · Tempo
        </h3>
        <span className="text-[11px] text-slate-500">
          {warpMode === "base" ? "x = baseline time (time-warped)" : "x = your real time"}
        </span>
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  );
}
