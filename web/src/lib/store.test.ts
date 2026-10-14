import { describe, it, expect, beforeEach } from "vitest";
import { useSpeechLensStore } from "./store";
import { FlawRegion } from "@/types/analysis";

describe("SpeechLens Zustand Store", () => {
  beforeEach(() => {
    useSpeechLensStore.getState().reset();
  });

  it("initializes with default state", () => {
    const state = useSpeechLensStore.getState();
    expect(state.selectedSpeechId).toBe("speech-001");
    expect(state.isNormMode).toBe(false);
    expect(state.isAnalyzing).toBe(false);
    expect(state.analysisResult).toBeNull();
    expect(state.selectedRegionId).toBeNull();
  });

  it("updates speech selection and transcript", () => {
    const store = useSpeechLensStore.getState();
    store.setSpeeches([
      {
        id: "speech-002",
        title: "Speech 2",
        speaker: "Speaker 2",
        duration_s: 40.0,
        n_words: 100,
        transcript: "Testing speech two transcript",
        baseline_audio_url: "/api/media/speech-002/baseline.wav",
        n_readers: 2,
      },
    ]);

    store.setSelectedSpeechId("speech-002");
    expect(useSpeechLensStore.getState().selectedSpeechId).toBe("speech-002");
    expect(useSpeechLensStore.getState().customTranscript).toBe("Testing speech two transcript");
  });

  it("selectRegion updates selectedRegionId, selectedWordIdx, and currentTime", () => {
    const store = useSpeechLensStore.getState();
    const mockRegion: FlawRegion = {
      id: "reg_01",
      family: "pace_fast",
      dimension: "pacing",
      start: 12.5,
      end: 15.0,
      word_idx: [20, 25],
      severity: 0.8,
      z_peak: 2.5,
      cost: 5.0,
      evidence: {},
      explanation: { summary: "Pace too fast", context: "", effect: "", fix: "", math: "" },
    };

    store.selectRegion(mockRegion);
    const updated = useSpeechLensStore.getState();
    expect(updated.selectedRegionId).toBe("reg_01");
    expect(updated.selectedWordIdx).toBe(20);
    expect(updated.currentTime).toBe(12.5);

    // Unselect
    store.selectRegion(null);
    expect(useSpeechLensStore.getState().selectedRegionId).toBeNull();
  });

  it("selectWord updates selectedWordIdx and finds overlapping region", () => {
    const store = useSpeechLensStore.getState();
    const mockRegion: FlawRegion = {
      id: "reg_02",
      family: "volume_low",
      dimension: "volume",
      start: 5.0,
      end: 7.0,
      word_idx: [10, 15],
      severity: 0.6,
      z_peak: -2.0,
      cost: 3.0,
      evidence: {},
      explanation: { summary: "Low volume", context: "", effect: "", fix: "", math: "" },
    };

    store.setAnalysisResult({
      schema_version: "1.0",
      meta: {
        speech_id: "speech-001",
        mode: "reference",
        duration_s: 30,
        n_words: 50,
        baseline_readers: [],
        software: { speechlens: "1.0.0", model: "w2v" },
        input_sha256: "hash",
      },
      scores: { total: 80, dimensions: { pacing: 80, intonation: 80, fluency: 80, volume: 80, clarity: 80 } },
      global: { articulation_rate_syl_s: 4, baseline_articulation_rate_syl_s: null, wpm: 120, pause_ratio: 0.2, f0_sd_st: 3 },
      words: [{ i: 12, text: "word", start: 5.5, end: 6.0, base_start: 5.5, base_end: 6.0, z: {}, prominence: {}, align_score: 0.9, flags: [] }],
      series: { hop_s: 0.02, t_user: [], t_base: [], pitch_st: { user: [], base: [], lo: [], hi: [] }, energy_db: { user: [], base: [], lo: [], hi: [] }, tempo: { user: [], base: [] } },
      pauses: [],
      regions: [mockRegion],
      warnings: [],
    });

    store.selectWord(12);
    const updated = useSpeechLensStore.getState();
    expect(updated.selectedWordIdx).toBe(12);
    expect(updated.selectedRegionId).toBe("reg_02");
    expect(updated.currentTime).toBe(5.5);
  });
});
