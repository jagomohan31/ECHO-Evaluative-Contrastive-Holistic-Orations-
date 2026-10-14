import { describe, it, expect, vi, beforeEach } from "vitest";
import { getHealth, getSpeeches, getBaselineAudioUrl, analyzeAudio, ApiError } from "./api";

describe("SpeechLens API Client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("getBaselineAudioUrl constructs correct URL", () => {
    const url = getBaselineAudioUrl("speech-001");
    expect(url).toContain("/api/media/speech-001/baseline.wav");
  });

  it("getHealth returns server health status", async () => {
    const mockData = { status: "ok", version: "1.0.0", model_loaded: true };
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => mockData,
    } as Response);

    const res = await getHealth();
    expect(res).toEqual(mockData);
  });

  it("getHealth throws ApiError on failed response", async () => {
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: false,
      status: 500,
    } as Response);

    await expect(getHealth()).rejects.toThrow(ApiError);
  });

  it("getSpeeches returns list of speeches", async () => {
    const mockSpeeches = [
      {
        id: "speech-001",
        title: "Gettysburg Address",
        speaker: "Abraham Lincoln",
        duration_s: 75.0,
        n_words: 272,
        transcript: "Four score and seven years ago...",
        baseline_audio_url: "/api/media/speech-001/baseline.wav",
        n_readers: 4,
      },
    ];
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => mockSpeeches,
    } as Response);

    const res = await getSpeeches();
    expect(res).toHaveLength(1);
    expect(res[0].id).toBe("speech-001");
  });

  it("analyzeAudio sends multipart form and returns AnalysisResult", async () => {
    const mockResult = {
      schema_version: "1.0",
      meta: {
        speech_id: "speech-001",
        mode: "reference",
        duration_s: 75.0,
        n_words: 272,
        baseline_readers: ["r1_amb"],
        software: { speechlens: "1.0.0", model: "wav2vec2" },
        input_sha256: "dummy_sha",
      },
      scores: {
        total: 88.5,
        dimensions: { pacing: 90, intonation: 85, fluency: 90, volume: 85, clarity: 90 },
      },
      global: {
        articulation_rate_syl_s: 4.2,
        baseline_articulation_rate_syl_s: 4.1,
        wpm: 140,
        pause_ratio: 0.2,
        f0_sd_st: 3.5,
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

    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    } as Response);

    const dummyBlob = new Blob(["dummy audio"], { type: "audio/wav" });
    const res = await analyzeAudio(dummyBlob, "speech-001");
    expect(res.meta.speech_id).toBe("speech-001");
    expect(res.scores.total).toBe(88.5);
  });
});
