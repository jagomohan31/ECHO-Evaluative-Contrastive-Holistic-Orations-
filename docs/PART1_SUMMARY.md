# SpeechLens — Part 1 Summary

> ECHO Hackathon 2026 | Submission before 14 Oct EOD

## Checklist

### T01 – T05: Project Scaffold & Data Pipeline
- [x] **T01** Repository layout: `src/speechlens/`, `tests/`, `data/`, `config/`, `results/`, `scripts/`
- [x] **T02** `pyproject.toml` with all dependencies; `.venv` created; `speechlens` importable
- [x] **T03** `config.py` — all DESIGN §4 constants (`SR`, `Z_TOL`, `Z_MAX`, `SIGMA_FLOOR_MULT`, `K`, `ROUND_DECIMALS`, flaw families, dimension weights)
- [x] **T04** `audio.py` — `load()` with mono-resample to 16 kHz float32
- [x] **T05** Dataset build — `data/library/speech-001…004/`, `baseline.json`, `norms.json`, `meta.json`, `transcript.txt`; `data/dataset/speech-001…004/labels.json`

### T06 – T08: Alignment & Features
- [x] **T06** `align/ctc.py` — `align(samples, transcript, use_cache=True)` using `facebook/wav2vec2-base-960h`; CTC trellis + backtrack; SHA-256 cache under `.cache/align/`; returns `AlignResult` with `.words` (`AlignWord`) and `.pauses` (`PauseInfo`)
- [x] **T07** `features/frame.py` — `compute_frame_features(samples, SR)` → `FrameFeatures` (F0 via RAPT/Praat, energy_db, voiced_mask, times); `features/word.py` — `compute_word_features`, `classify_pauses`, `compute_global_features`, `time_warp`
- [x] **T08** `features/word.py` — `WordFeatures` dataclass with all 20+ fields including `punct_after`, `share`, `art_rate`, `f0_mean_st`, `f0_delta_st`, `en_mean_rel`, `prominence`, `align_score`, `pause_before`, etc.

### T09 – T10: Baseline & Norms
- [x] **T09** `baseline.py` — `build_baseline(speech_id)` aggregates reader alignments into per-word `mu_*`/`sigma_*` stat dicts and `artic_rate`; saves `baseline.json`
- [x] **T10** `norms.py` — `build_norms()` computes population-level `sigma_pool_*`, `sigma_D`, `artic_rate_mean/sd`; saves `norms.json`

### T11: Flaw Detection
- [x] **T11** `detect.py` — `detect_all(words, pauses, baseline, norms, ff)` and `detect_norm(words, pauses, norms, ff)` implementing all 13 detection rules:
  - `pace_fast`, `pace_slow` (articulation rate z-score, DESIGN §9.1–9.2)
  - `pause_excess`, `pause_missing` (duration vs. baseline mu, DESIGN §9.3–9.4)
  - `monotone` (F0 SD z-score, DESIGN §9.5)
  - `terminal_contour` (clause-final F0 delta, DESIGN §9.6)
  - `stress_missed` (prominence z-score per word, DESIGN §9.7)
  - `hesitation_filled` (filled-pause detector via CTC alignment + keyword match, DESIGN §9.8)
  - `volume_low` (energy z-score, DESIGN §9.9)
  - `trailing_off` (final-word energy drop, DESIGN §9.10)
  - `clipping` (peak amplitude saturation, DESIGN §9.11)
  - `muffled` (spectral centroid / HF energy ratio, DESIGN §9.12)
  - `word_swallowed` (per-word CTC alignment score < threshold, DESIGN §9.13)
  - `Region` dataclass: `family, dimension, start, end, word_start, word_end, severity, z_peak, cost, id, evidence, explanation`; `.to_dict()` method
- [x] **T11 tests** — `tests/test_t11_detect.py`: 18 tests, all 13 rules covered + norm mode + integration

### T12: Scoring & Explanations
- [x] **T12** `score.py` — `compute_scores(regions, total_dur, weights, k)` → `{"total": float, "dimensions": {...}}`; `compute_region_costs(regions, total_dur)` (mutates `r.cost`)
- [x] **T12** `explain.py` — `explain_all(regions)` with 14 family explanation templates; evidence-only numbers; sets `r.explanation`
- [x] **T12 tests** — `tests/test_t12_score_explain.py`: 8 tests (weights, bounds, monotonicity, cost=difference, all templates)

### T13: Pipeline
- [x] **T13** `pipeline.py` — `analyze(audio_path, speech_id, …, use_cache)` end-to-end:
  - Audio load → transcript resolve → mode detection (reference vs. norm)
  - CTC alignment (SHA-256 cache) → frame + word features
  - Flaw detection → scoring + costs → explanations
  - Time-warp series (t_user → t_base via mirror reader's `align.json`)
  - Pitch / energy / tempo grids at 50 Hz
  - Schema-compliant output dict; validated by `AnalysisResult.model_validate`
  - SHA-256 analysis cache under `.cache/analysis/`
- [x] **T13 tests** — `tests/test_t13_pipeline.py`: 10 fast tests (schema, bounds, regions fields, series lengths, cached-run <30 s); 2 slow (determinism, wrong-passage warning)
- [x] **Timing**: 20.7 s for 74.9 s clip; cache hit 0.55 s; byte-identical verified

### T14: Calibration
- [x] **T14** `evalsuite/calibrate.py` — grid search over `Z_TOL ∈ {1.25,1.5,1.75,2.0}` × `SIGMA_FLOOR_MULT ∈ {0.75,1.0,1.25}` × `K ∈ {0.5,0.65,0.8,0.95}` (48 configs)
  - Phase 1: analytic `remap_words_and_pauses` for 73 synthetic files + CTC (cached) for 4 clean readers; Praat (~7s/file); 77 files in ~565 s
  - Phase 2: reload `detect_all` with patched config; ~0.3 s per config
  - Output: `config/calibrated.json` with best `(z_tol, sigma_floor_mult, k)` maximising macro-F1 on dev split
- [x] `dataset/remap.py` — analytic time-remap for all 5 synthetic flaw families (pace, pause, hesitation, trailing_off, muffled)
- [x] `evalsuite/grounding.py` — `temporal_iou`, `match_file_regions`, `compute_metrics`; greedy IoU≥0.3 matching per family

### T15: Evaluation Suite
- [x] **T15** `evalsuite/evaluate.py` — E1–E8 evaluation suite:
  - **E1** Region detection (P/R/F1, macro-F1) on test split
  - **E2** Score monotonicity: `score(L0) > score(L1) > score(L2) > score(L3)`
  - **E3** Severity correlation: Spearman ρ ≥ 0.4 between injection level and deduction
  - **E4** Leave-one-out baseline stability: SD(macro-F1) < 0.05
  - **E5** Pipeline latency: p95 < 30 s (cached)
  - **E6** Byte-identical determinism (cold runs)
  - **E7** Score bounds: all scores ∈ [0, 100]
  - **E8** Explanation coverage: every region has non-empty explanation
  - Outputs: `results/metrics.json` + `results/REPORT.md`

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| T11 detect | 18 | ✅ All pass |
| T12 score+explain | 8 | ✅ All pass |
| T13 pipeline | 10 fast + 2 slow | ✅ Fast pass |
| T14 calibration | 4 | ✅ All pass |
| T15 evaluation | 4 fast + 1 slow | ✅ Fast pass |
| T01–T10 baseline | 108 | ✅ All pass |
| **Total fast** | **152** | ✅ **All pass** |

## Calibration Results (T14)

- Winning hyperparameters: `Z_TOL = 1.75`, `SIGMA_FLOOR_MULT = 1.25`, `K = 0.5`
- Optimization objective: dev split macro-F1 @ IoU 0.3
- Saved artifacts:
  - `config/calibrated.json` (overrides default constants automatically via `config.py`)
  - `results/calibration.json` (full 48-entry grid search data)
  - `results/fig_calibration.png` (grid search heatmap)

## Key Design Decisions

### Alignment caching
CTC inference takes ~12s first call; SHA-256 cache on `(audio_bytes + transcript + model_id)` makes subsequent calls <0.1s.

### Calibration strategy
Synthetic dataset files have known analytic mappings — `remap_words_and_pauses` gives exact word timings without re-running CTC (~0ms vs ~80s). Only clean library readers need CTC (cached). This reduced calibration Phase 1 from ~100 min to ~9 min.

### Time-warp reference
`time_warp(t_user, align_words_user, align_words_mirror)` uses the **mirror reader's `.align.json`** word timestamps (not baseline stat dicts which only have `mu_*`/`sigma_*`). The series output maps user time → baseline time for dashboard overlay.

### Score formula (DESIGN §10)
\[
\text{score}_\text{dim} = 100 \times \max\!\left(0,\, 1 - K \cdot \frac{\sum_r \text{cost}_r}{D_\text{dim}}\right)
\]
where $D_\text{dim}$ = total duration × dimension weight, and $\text{cost}_r = (\text{severity}_r)^{1.5} \times \text{dur}_r$.

## Files Delivered

```
src/speechlens/
  config.py          pipeline.py        models.py
  audio.py           score.py           explain.py
  detect.py          cli.py
  align/
    ctc.py
  features/
    frame.py  word.py
  evalsuite/
    grounding.py  calibrate.py  evaluate.py
  dataset/
    remap.py
tests/
  test_t11_detect.py          test_t12_score_explain.py   test_t13_pipeline.py
  test_t14_calibration.py     test_t15_evaluation.py
scripts/
  eval_dev.py                 cache_dataset_alignments.py
config/
  calibrated.json
results/
  calibration.json            dev_pre_calibration.json   timing.json
  metrics.json                REPORT.md
  fig_calibration.png         fig_confusion_matrix.png
  fig_example_overlay.png     fig_severity_curve.png
  tables/
    eval_summary.csv          per_family_metrics.csv
docs/
  PART1_SUMMARY.md            KNOWN_ISSUES.md
```

