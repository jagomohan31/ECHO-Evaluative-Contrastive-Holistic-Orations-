/**
 * Typed API Client for SpeechLens Backend (DESIGN §14).
 */

import { AnalysisResult, SpeechMetadata, DemoClip } from "@/types/analysis";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

export class ApiError extends Error {
  constructor(
    public code: string,
    public detail: string,
    public status: number,
  ) {
    super(`[${code}] ${detail}`);
    this.name = "ApiError";
  }
}

export async function getHealth(): Promise<{ status: string; version: string; model_loaded: boolean }> {
  const resp = await fetch(`${API_BASE}/api/health`);
  if (!resp.ok) {
    throw new ApiError("health_failed", "Health check failed", resp.status);
  }
  return resp.json();
}

export async function getSpeeches(): Promise<SpeechMetadata[]> {
  const resp = await fetch(`${API_BASE}/api/speeches`);
  if (!resp.ok) {
    throw new ApiError("speeches_failed", "Failed to fetch speeches list", resp.status);
  }
  return resp.json();
}

export function getBaselineAudioUrl(speechId: string): string {
  return `${API_BASE}/api/media/${speechId}/baseline.wav`;
}

export async function getDemoClips(): Promise<DemoClip[]> {
  try {
    const resp = await fetch(`${API_BASE}/api/demo`);
    if (resp.ok) {
      return await resp.json();
    }
  } catch {}
  // Offline fallback to static demo index
  const staticResp = await fetch("/demo/index.json");
  if (!staticResp.ok) {
    throw new ApiError("demo_failed", "Failed to fetch demo clips", staticResp.status);
  }
  return staticResp.json();
}

export async function analyzeAudio(
  audioBlob: Blob | File,
  speechId?: string | null,
  transcript?: string | null,
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");
  if (speechId) {
    formData.append("speech_id", speechId);
  }
  if (transcript) {
    formData.append("transcript", transcript);
  }

  const resp = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!resp.ok) {
    let errBody: { error?: string; detail?: string } = {};
    try {
      errBody = await resp.json();
    } catch {
      errBody = { error: "http_error", detail: resp.statusText };
    }
    throw new ApiError(
      errBody.error || "unknown_error",
      errBody.detail || "Analysis request failed",
      resp.status,
    );
  }

  return resp.json();
}

export async function loadDemoResult(resultUrl: string): Promise<AnalysisResult> {
  const url = resultUrl.startsWith("http") ? resultUrl : `${API_BASE}${resultUrl}`;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new ApiError("demo_load_failed", `Failed to load demo result from ${url}`, resp.status);
  }
  return resp.json();
}
