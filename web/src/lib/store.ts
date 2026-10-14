/**
 * Zustand Shared Selection Store (DESIGN §15 item 7).
 * Synchronizes controls, score panel, waveform, overlay chart, transcript strip, and flaw cards.
 */

import { create } from "zustand";
import { AnalysisResult, SpeechMetadata, FlawRegion } from "@/types/analysis";

export interface SpeechLensStore {
  // Speech & Control state
  speeches: SpeechMetadata[];
  selectedSpeechId: string;
  isNormMode: boolean;
  customTranscript: string;
  audioBlob: Blob | File | null;
  audioUrl: string | null;

  // Analysis status & result
  isAnalyzing: boolean;
  analysisResult: AnalysisResult | null;
  error: string | null;

  // Shared selection & playback state (DESIGN §15 item 7)
  selectedRegionId: string | null;
  hoveredRegionId: string | null;
  selectedWordIdx: number | null;
  currentTime: number;
  isPlaying: boolean;
  warpMode: "base" | "user";

  // Actions
  setSpeeches: (speeches: SpeechMetadata[]) => void;
  setSelectedSpeechId: (id: string) => void;
  setIsNormMode: (isNorm: boolean) => void;
  setCustomTranscript: (text: string) => void;
  setAudioBlob: (blob: Blob | File | null) => void;
  setAudioUrl: (url: string | null) => void;
  setIsAnalyzing: (loading: boolean) => void;
  setAnalysisResult: (result: AnalysisResult | null) => void;
  setError: (error: string | null) => void;

  setSelectedRegionId: (id: string | null) => void;
  setHoveredRegionId: (id: string | null) => void;
  setSelectedWordIdx: (idx: number | null) => void;
  setCurrentTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setWarpMode: (mode: "base" | "user") => void;

  // Coordinated Selection Actions
  selectRegion: (region: FlawRegion | null) => void;
  selectWord: (wordIdx: number | null) => void;
  reset: () => void;
}

export const useSpeechLensStore = create<SpeechLensStore>((set, get) => ({
  speeches: [],
  selectedSpeechId: "speech-001",
  isNormMode: false,
  customTranscript: "",
  audioBlob: null,
  audioUrl: null,

  isAnalyzing: false,
  analysisResult: null,
  error: null,

  selectedRegionId: null,
  hoveredRegionId: null,
  selectedWordIdx: null,
  currentTime: 0,
  isPlaying: false,
  warpMode: "base",

  setSpeeches: (speeches) => {
    set({ speeches });
    if (speeches.length > 0 && !get().selectedSpeechId) {
      set({ selectedSpeechId: speeches[0].id });
    }
  },

  setSelectedSpeechId: (selectedSpeechId) => {
    set({
      selectedSpeechId,
      selectedRegionId: null,
      selectedWordIdx: null,
    });
  },

  setIsNormMode: (isNormMode) => set({ isNormMode }),
  setCustomTranscript: (customTranscript) => set({ customTranscript }),

  setAudioBlob: (blob) => {
    const prevUrl = get().audioUrl;
    if (prevUrl) {
      URL.revokeObjectURL(prevUrl);
    }
    const audioUrl = blob ? URL.createObjectURL(blob) : null;
    set({ audioBlob: blob, audioUrl, error: null });
  },

  setAudioUrl: (audioUrl) => set({ audioUrl, error: null }),

  setIsAnalyzing: (isAnalyzing) => set({ isAnalyzing }),
  setAnalysisResult: (analysisResult) =>
    set({
      analysisResult,
      isAnalyzing: false,
      error: null,
      selectedRegionId: null,
      selectedWordIdx: null,
    }),
  setError: (error) => set({ error, isAnalyzing: false }),

  setSelectedRegionId: (selectedRegionId) => set({ selectedRegionId }),
  setHoveredRegionId: (hoveredRegionId) => set({ hoveredRegionId }),
  setSelectedWordIdx: (selectedWordIdx) => set({ selectedWordIdx }),
  setCurrentTime: (currentTime) => set({ currentTime }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setWarpMode: (warpMode) => set({ warpMode }),

  selectRegion: (region) => {
    if (!region) {
      set({ selectedRegionId: null, selectedWordIdx: null });
      return;
    }
    set({
      selectedRegionId: region.id,
      selectedWordIdx: region.word_idx[0],
      currentTime: region.start,
    });
  },

  selectWord: (wordIdx) => {
    if (wordIdx === null) {
      set({ selectedWordIdx: null });
      return;
    }
    const res = get().analysisResult;
    let matchingRegionId = null;
    if (res) {
      const match = res.regions.find(
        (r) => wordIdx >= r.word_idx[0] && wordIdx <= r.word_idx[1]
      );
      if (match) {
        matchingRegionId = match.id;
      }
      const w = res.words.find((word) => word.i === wordIdx);
      if (w) {
        set({ currentTime: w.start });
      }
    }
    set({
      selectedWordIdx: wordIdx,
      selectedRegionId: matchingRegionId,
    });
  },

  reset: () => {
    const prevUrl = get().audioUrl;
    if (prevUrl) {
      URL.revokeObjectURL(prevUrl);
    }
    set({
      audioBlob: null,
      audioUrl: null,
      analysisResult: null,
      error: null,
      selectedRegionId: null,
      hoveredRegionId: null,
      selectedWordIdx: null,
      currentTime: 0,
      isPlaying: false,
    });
  },
}));
