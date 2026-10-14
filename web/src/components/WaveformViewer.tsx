"use client";

import React, { useEffect, useRef, useCallback, useMemo } from "react";
import { useSpeechLensStore } from "@/lib/store";
import { familyColor } from "@/lib/vizHelpers";

interface WaveformViewerProps {
  audioUrl: string;
}

export function WaveformViewer({ audioUrl }: WaveformViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<any>(null);
  const regionsPluginRef = useRef<any>(null);

  const { analysisResult, selectedRegionId, selectRegion, setCurrentTime } =
    useSpeechLensStore();

  const regions = useMemo(() => analysisResult?.regions ?? [], [analysisResult]);

  const buildRegions = useCallback(
    (ws: any, regionsPlugin: any) => {
      regionsPlugin.clearRegions();
      regions.forEach((r) => {
        regionsPlugin.addRegion({
          id: r.id,
          start: r.start,
          end: r.end,
          color: familyColor(r.family) + "33", // 20% alpha
          drag: false,
          resize: false,
          content: r.family,
        });
      });
    },
    [regions]
  );

  // Initialise WaveSurfer lazily (dynamic import to avoid SSR issues)
  useEffect(() => {
    if (!containerRef.current || !audioUrl) return;

    let cancelled = false;

    (async () => {
      const WaveSurfer = (await import("wavesurfer.js")).default;
      const RegionsPlugin = (await import("wavesurfer.js/dist/plugins/regions.js")).default;

      if (cancelled || !containerRef.current) return;

      // Destroy previous instance
      if (wsRef.current) {
        wsRef.current.destroy();
        wsRef.current = null;
      }

      const regionsPlugin = RegionsPlugin.create();
      regionsPluginRef.current = regionsPlugin;

      const ws = WaveSurfer.create({
        container: containerRef.current,
        waveColor: "#475569",
        progressColor: "#3b82f6",
        cursorColor: "#93c5fd",
        height: 80,
        normalize: true,
        interact: true,
        plugins: [regionsPlugin],
      });

      wsRef.current = ws;

      ws.on("ready", () => {
        if (cancelled) return;
        buildRegions(ws, regionsPlugin);
      });

      ws.on("timeupdate", (t: number) => {
        if (!cancelled) setCurrentTime(t);
      });

      // Region click → shared selection
      regionsPlugin.on("region-clicked", (region: any, e: Event) => {
        e.stopPropagation();
        const match = regions.find((r) => r.id === region.id) ?? null;
        selectRegion(match);
        region.play();
      });

      await ws.load(audioUrl);
    })();

    return () => {
      cancelled = true;
      if (wsRef.current) {
        wsRef.current.destroy();
        wsRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  // Rebuild regions when analysis result changes
  useEffect(() => {
    if (wsRef.current && regionsPluginRef.current) {
      buildRegions(wsRef.current, regionsPluginRef.current);
    }
  }, [buildRegions]);

  // Highlight selected region
  useEffect(() => {
    if (!regionsPluginRef.current) return;
    const rPlugin = regionsPluginRef.current;
    const all = rPlugin.getRegions ? rPlugin.getRegions() : {};
    Object.values(all).forEach((r: any) => {
      r.setOptions({
        color:
          r.id === selectedRegionId
            ? familyColor(
                regions.find((reg) => reg.id === r.id)?.family ?? "default"
              ) + "66"
            : familyColor(
                regions.find((reg) => reg.id === r.id)?.family ?? "default"
              ) + "33",
      });
    });
  }, [selectedRegionId, regions]);

  // Keyboard: Space = play/pause
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.code === "Space" &&
        wsRef.current &&
        !(e.target instanceof HTMLInputElement) &&
        !(e.target instanceof HTMLTextAreaElement)
      ) {
        e.preventDefault();
        wsRef.current.playPause();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Waveform
        </h3>
        <span className="text-[11px] text-slate-500">
          Space = play/pause · click region to loop
        </span>
      </div>
      <div ref={containerRef} className="w-full" />
      {/* Region legend */}
      {regions.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-3">
          {Array.from(new Set(regions.map((r) => r.family))).map((fam) => (
            <span
              key={fam}
              className="text-[11px] px-2 py-0.5 rounded-full font-medium border"
              style={{
                backgroundColor: familyColor(fam) + "22",
                color: familyColor(fam),
                borderColor: familyColor(fam) + "55",
              }}
            >
              {fam}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
