"use client";

import React, { useState, useRef, useEffect } from "react";
import { useSpeechLensStore } from "@/lib/store";
import { getBaselineAudioUrl, analyzeAudio, getSpeeches, getDemoClips, loadDemoResult } from "@/lib/api";
import { Mic, Square, Upload, Play, Pause, Sparkles, RefreshCw, Volume2 } from "lucide-react";

export function ControlPanel() {
  const {
    speeches,
    setSpeeches,
    selectedSpeechId,
    setSelectedSpeechId,
    isNormMode,
    setIsNormMode,
    customTranscript,
    setCustomTranscript,
    audioBlob,
    setAudioBlob,
    setAudioUrl,
    isAnalyzing,
    setIsAnalyzing,
    setAnalysisResult,
    error,
    setError,
    reset,
  } = useSpeechLensStore();

  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [isPlayingBaseline, setIsPlayingBaseline] = useState(false);
  const [demoList, setDemoList] = useState<any[]>([]);
  const [isDemoActive, setIsDemoActive] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordTimerRef = useRef<NodeJS.Timeout | null>(null);
  const baselineAudioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load speeches on mount
  useEffect(() => {
    const FALLBACK_SPEECHES = [
      { id: "speech-001", title: "The Gettysburg Address", speaker: "Abraham Lincoln", duration_s: 75.0, n_words: 91, transcript: "", baseline_audio_url: "/api/media/speech-001/baseline.wav", n_readers: 3 },
      { id: "speech-002", title: "Lincoln Second Inaugural Address", speaker: "Abraham Lincoln", duration_s: 49.2, n_words: 74, transcript: "", baseline_audio_url: "/api/media/speech-002/baseline.wav", n_readers: 1 },
      { id: "speech-003", title: "Give Me Liberty or Give Me Death", speaker: "Patrick Henry", duration_s: 70.0, n_words: 106, transcript: "", baseline_audio_url: "/api/media/speech-003/baseline.wav", n_readers: 1 },
      { id: "speech-004", title: "Emancipation Proclamation (excerpt)", speaker: "Abraham Lincoln", duration_s: 70.0, n_words: 99, transcript: "", baseline_audio_url: "/api/media/speech-004/baseline.wav", n_readers: 1 },
    ];
    getSpeeches()
      .then((data) => {
        const speeches = data.length > 0 ? data : FALLBACK_SPEECHES;
        setSpeeches(speeches);
        if (speeches.length > 0 && !selectedSpeechId) {
          setSelectedSpeechId(speeches[0].id);
        }
      })
      .catch(() => {
        setSpeeches(FALLBACK_SPEECHES);
        if (!selectedSpeechId) setSelectedSpeechId(FALLBACK_SPEECHES[0].id);
      });

    getDemoClips()
      .then((data) => setDemoList(data))
      .catch(() => {});
  }, [setSpeeches, setSelectedSpeechId, selectedSpeechId]);

  const activeSpeech = speeches.find((s) => s.id === selectedSpeechId);

  // File Upload Handlers
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setIsDemoActive(false);
      setAudioBlob(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setIsDemoActive(false);
      setAudioBlob(file);
    }
  };

  // Microphone Recording Handlers
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        setAudioBlob(audioBlob);
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start(100);
      setIsDemoActive(false);
      setIsRecording(true);
      setRecordDuration(0);
      recordTimerRef.current = setInterval(() => {
        setRecordDuration((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      setError(`Microphone access error: ${err.message}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (recordTimerRef.current) {
        clearInterval(recordTimerRef.current);
      }
    }
  };

  // Baseline Audio Playback
  const toggleBaseline = () => {
    if (!baselineAudioRef.current) return;
    if (isPlayingBaseline) {
      baselineAudioRef.current.pause();
      setIsPlayingBaseline(false);
    } else {
      baselineAudioRef.current.play();
      setIsPlayingBaseline(true);
    }
  };

  // Analyze Action
  const handleAnalyze = async () => {
    if (!audioBlob) {
      setError("Please upload an audio file or record speech before analyzing.");
      return;
    }
    const transcriptToUse = isNormMode ? customTranscript : activeSpeech?.transcript || customTranscript;
    if (isNormMode && (!transcriptToUse || !transcriptToUse.trim())) {
      setError("Please provide a transcript for Norm mode analysis.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const result = await analyzeAudio(
        audioBlob,
        isNormMode ? null : selectedSpeechId,
        transcriptToUse
      );
      setAnalysisResult(result);
    } catch (err: any) {
      setError(err.detail || err.message || "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Load Demo Item
  const handleLoadDemo = async (demo: any) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const result = await loadDemoResult(demo.result_url);
      setAnalysisResult(result);
      setIsDemoActive(true);
      if (demo.audio_url) {
        setAudioUrl(demo.audio_url);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load demo");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100 shadow-xl space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Speech Input & Controls</h2>
          <p className="text-sm text-slate-400">Select reference passage or provide your own speech</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer bg-slate-800/80 hover:bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700/60 transition-colors">
            <input
              type="checkbox"
              checked={isNormMode}
              onChange={(e) => setIsNormMode(e.target.checked)}
              className="rounded bg-slate-700 border-slate-600 text-blue-500 focus:ring-blue-500"
            />
            <span>Custom Speech (Norm Mode)</span>
          </label>
        </div>
      </div>

      {/* Speech Selector & Baseline Player (Reference Mode) */}
      {!isNormMode && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-slate-800/40 p-4 rounded-lg border border-slate-700/50">
          <div className="md:col-span-2 space-y-1.5">
            <label className="text-xs font-semibold text-slate-300">Reference Passage</label>
            <select
              aria-label="Select reference passage"
              value={selectedSpeechId}
              onChange={(e) => setSelectedSpeechId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
            >
              {speeches.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.title} — {s.speaker} ({s.duration_s.toFixed(0)}s, {s.n_words} words)
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1.5 flex flex-col justify-end">
            <span className="text-xs font-semibold text-slate-300">Reference Reader</span>
            {activeSpeech && (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={toggleBaseline}
                  className="flex-1 flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-2 rounded-lg text-xs font-medium border border-slate-700 transition-colors"
                >
                  {isPlayingBaseline ? <Pause className="w-4 h-4 text-amber-400" /> : <Play className="w-4 h-4 text-emerald-400" />}
                  <span>{isPlayingBaseline ? "Pause Baseline" : "Play Baseline"}</span>
                </button>
                <audio
                  ref={baselineAudioRef}
                  src={getBaselineAudioUrl(selectedSpeechId)}
                  onEnded={() => setIsPlayingBaseline(false)}
                  className="hidden"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Audio Input: shown only in Norm Mode or when a file is already loaded */}
      {(isNormMode || audioBlob) && !isDemoActive && (
      <div className="space-y-3">
        <label className="text-xs font-semibold text-slate-300">Your Audio Recording</label>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Drag & Drop Upload */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-slate-700 hover:border-blue-500/50 bg-slate-950/40 hover:bg-slate-950/70 rounded-xl cursor-pointer transition-colors text-center group"
          >
            <Upload className="w-8 h-8 text-slate-400 group-hover:text-blue-400 transition-colors mb-2" />
            <span className="text-xs font-medium text-slate-300">Click to upload or drag audio file</span>
            <span className="text-[11px] text-slate-400 mt-1">FLAC, WAV, MP3, OGG up to 20 MB</span>
            <input
              ref={fileInputRef}
              data-testid="audio-file-input"
              type="file"
              accept="audio/*,.flac,.wav,.mp3,.ogg"
              onChange={handleFileChange}
              className="hidden"
            />
          </div>

          {/* Live Mic Recording */}
          <div className="flex flex-col items-center justify-center p-6 border border-slate-800 bg-slate-950/40 rounded-xl text-center space-y-3">
            <div className="flex items-center gap-2">
              <span className={`w-3 h-3 rounded-full ${isRecording ? "bg-rose-500 animate-pulse" : "bg-slate-600"}`} />
              <span className="text-xs font-semibold text-slate-300">
                {isRecording ? `Recording... (${recordDuration}s)` : "Live Microphone"}
              </span>
            </div>
            {isRecording ? (
              <button
                type="button"
                onClick={stopRecording}
                className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 text-white px-5 py-2 rounded-lg text-xs font-semibold shadow-lg shadow-rose-900/40 transition-colors"
              >
                <Square className="w-4 h-4 fill-white" />
                <span>Stop Recording</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={startRecording}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-5 py-2 rounded-lg text-xs font-semibold shadow-lg shadow-blue-900/40 transition-colors"
              >
                <Mic className="w-4 h-4" />
                <span>Start Recording</span>
              </button>
            )}
          </div>
        </div>

        {audioBlob && (
          <div className="flex items-center justify-between text-xs bg-slate-800/60 px-3 py-2 rounded-lg border border-slate-700/60">
            <span className="text-slate-300">
              Selected: <span className="text-blue-400 font-medium">{(audioBlob as File).name || "recorded_audio.webm"}</span> ({(audioBlob.size / 1024 / 1024).toFixed(2)} MB)
            </span>
            <button
              type="button"
              onClick={() => setAudioBlob(null)}
              className="text-slate-400 hover:text-rose-400 transition-colors text-[11px]"
            >
              Remove
            </button>
          </div>
        )}
      </div>
      )}

      {/* Transcript Textarea (Norm Mode or custom) */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center text-xs">
          <label className="font-semibold text-slate-300">
            {isNormMode ? "Speech Transcript (Required)" : "Reference Transcript"}
          </label>
          <span className="text-slate-400">
            {customTranscript ? `${customTranscript.split(/\s+/).filter(Boolean).length} words` : "0 words"}
          </span>
        </div>
        <textarea
          rows={3}
          value={isNormMode ? customTranscript : activeSpeech?.transcript || ""}
          onChange={(e) => setCustomTranscript(e.target.value)}
          readOnly={!isNormMode}
          placeholder="Enter the speech transcript..."
          className={`w-full rounded-lg px-3 py-2 text-xs font-mono focus:outline-none transition-colors ${
            isNormMode
              ? "bg-slate-950 border border-slate-700 text-slate-200 focus:border-blue-500"
              : "bg-slate-950/50 border border-slate-800 text-slate-400 cursor-not-allowed"
          }`}
        />
      </div>

      {/* Action Buttons: Analyze & Reset */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={isAnalyzing || !audioBlob}
            className="flex-1 sm:flex-none flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold px-6 py-2.5 rounded-lg text-sm shadow-lg shadow-blue-900/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isAnalyzing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Analyzing Speech...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Analyze Performance</span>
              </>
            )}
          </button>
          <button
            type="button"
            onClick={reset}
            className="px-4 py-2.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            Clear
          </button>
        </div>

        {/* Demo scenarios buttons */}
        {demoList.length > 0 && (
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[11px] text-slate-400 mr-1">Demo Clips:</span>
            {demoList.map((d) => (
              <button
                key={d.id}
                type="button"
                onClick={() => handleLoadDemo(d)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white px-2.5 py-1 rounded text-[11px] font-medium border border-slate-700/60 transition-colors"
              >
                {d.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg p-3 text-xs text-rose-300">
          {error}
        </div>
      )}
    </div>
  );
}
