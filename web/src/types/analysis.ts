/**
 * Analysis result TypeScript interfaces matching schema/analysis_result.schema.json (DESIGN §12).
 */

export interface AnalysisMeta {
  speech_id: string | null;
  mode: "reference" | "norm";
  duration_s: number;
  n_words: number;
  baseline_readers: string[];
  software: {
    speechlens: string;
    model: string;
  };
  input_sha256: string;
}

export interface DimensionScores {
  pacing: number;
  intonation: number;
  fluency: number;
  volume: number;
  clarity: number;
}

export interface Scores {
  total: number;
  dimensions: DimensionScores;
}

export interface GlobalFeatures {
  articulation_rate_syl_s: number;
  baseline_articulation_rate_syl_s: number | null;
  wpm: number;
  pause_ratio: number;
  f0_sd_st: number;
}

export interface WordItem {
  i: number;
  text: string;
  start: number;
  end: number;
  base_start: number | null;
  base_end: number | null;
  z: {
    tempo?: number | null;
    pitch?: number | null;
    energy?: number | null;
    pause?: number | null;
  };
  prominence: {
    user?: number | null;
    base?: number | null;
  };
  align_score: number;
  flags: string[];
}

export interface SeriesPitch {
  user: (number | null)[];
  base: (number | null)[];
  lo: (number | null)[];
  hi: (number | null)[];
}

export interface SeriesEnergy {
  user: number[];
  base: number[];
  lo: number[];
  hi: number[];
}

export interface SeriesTempo {
  user: number[];
  base: number[];
}

export interface SeriesData {
  hop_s: number;
  t_user: number[];
  t_base: number[];
  pitch_st: SeriesPitch;
  energy_db: SeriesEnergy;
  tempo: SeriesTempo;
}

export interface PauseItem {
  start: number;
  end: number;
  kind: "silent" | "filled" | "other";
  expected: boolean;
}

export interface RegionEvidence {
  metric?: string;
  unit?: string;
  observed?: number;
  expected?: number;
  sigma?: number;
  z?: number;
  formula?: string;
  [key: string]: any;
}

export interface RegionExplanation {
  summary: string;
  context: string;
  effect: string;
  fix: string;
  math: string;
}

export interface FlawRegion {
  id: string;
  family: string;
  dimension: "pacing" | "intonation" | "fluency" | "volume" | "clarity";
  start: number;
  end: number;
  word_idx: [number, number];
  severity: number;
  z_peak: number;
  cost: number;
  evidence: RegionEvidence;
  explanation: RegionExplanation;
}

export interface AnalysisResult {
  schema_version: "1.0";
  meta: AnalysisMeta;
  scores: Scores;
  global: GlobalFeatures;
  words: WordItem[];
  series: SeriesData;
  pauses: PauseItem[];
  regions: FlawRegion[];
  warnings: string[];
}

export interface SpeechMetadata {
  id: string;
  title: string;
  speaker: string;
  duration_s: number;
  n_words: number;
  transcript: string;
  baseline_audio_url: string;
  n_readers: number;
}

export interface DemoClip {
  id: string;
  label: string;
  level: number;
  family: string;
  audio_url: string;
  result_url: string;
}
