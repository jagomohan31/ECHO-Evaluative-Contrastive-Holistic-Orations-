import React from "react";
import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { ScorePanel } from "./ScorePanel";
import { useSpeechLensStore } from "@/lib/store";
import { AnalysisResult } from "@/types/analysis";

describe("ScorePanel Component", () => {
  beforeEach(() => {
    useSpeechLensStore.getState().reset();
  });

  it("renders null when no analysisResult exists", () => {
    const { container } = render(<ScorePanel />);
    expect(container.firstChild).toBeNull();
  });

  it("renders total score, rating, dimension labels, and global stats when analysisResult is present", () => {
    const mockResult: AnalysisResult = {
      schema_version: "1.0",
      meta: {
        speech_id: "speech-001",
        mode: "reference",
        duration_s: 75.0,
        n_words: 272,
        baseline_readers: ["r1_amb"],
        software: { speechlens: "1.0.0", model: "wav2vec2" },
        input_sha256: "abc",
      },
      scores: {
        total: 78.0,
        dimensions: {
          pacing: 85.0,
          intonation: 80.0,
          fluency: 90.0,
          volume: 75.0,
          clarity: 60.0,
        },
      },
      global: {
        articulation_rate_syl_s: 4.25,
        baseline_articulation_rate_syl_s: 4.10,
        wpm: 145,
        pause_ratio: 0.18,
        f0_sd_st: 3.2,
      },
      words: [],
      series: {
        hop_s: 0.02,
        t_user: [],
        t_base: [],
        pitch_st: { user: [], base: [], lo: [], hi: [] },
        energy_db: { user: [], base: [], lo: [], hi: [] },
        tempo: { user: [], base: [] },
      },
      pauses: [],
      regions: [],
      warnings: [],
    };

    useSpeechLensStore.getState().setAnalysisResult(mockResult);

    render(<ScorePanel />);

    // Total score (78.0 is unique across all rendered text)
    expect(screen.getByText("78.0")).toBeDefined();
    // Rating badge
    expect(screen.getByText("Proficient")).toBeDefined();

    // Dimension labels — rendered in a nested span; check at least one per label text
    // The component renders dimension labels like "Pacing (25%)" split across spans
    // so we look for the dimension label container text inclusively
    expect(screen.getAllByText(/Pacing/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Intonation/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Fluency/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Volume/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Clarity/).length).toBeGreaterThan(0);

    // Global stats
    expect(screen.getByText("4.25")).toBeDefined();
    // WPM = Math.round(145) = "145"
    expect(screen.getByText("145")).toBeDefined();
    // Speech Performance heading
    expect(screen.getByText("Speech Performance")).toBeDefined();
  });
});
