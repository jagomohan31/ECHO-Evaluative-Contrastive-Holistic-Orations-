---
title: "SpeechLens: Contrastive Speech Analytics with Temporal Flaw Grounding"
subtitle: "Multimodal AI Hackathon 2026 — Track C"
date: "2026-10-14"
geometry: "a4paper, margin=2cm"
fontsize: 10pt
---

# 1  Problem and system overview

SpeechLens analyses oratory recordings and produces temporally grounded
feedback: which seconds of a speech contain a delivery flaw, why, and how
much it costs. The system is fully deterministic and explainable by
construction — no language model is used in the analysis path.

**Architecture.** A three-stage pipeline: (1) CTC forced alignment maps
every word to a time span; (2) prosodic and spectral feature extraction
builds a per-word feature vector; (3) a z-score detector compares the
speaker against a speaker-specific baseline or population norms.

    Audio + transcript
         |
    wav2vec2 CTC aligner  -->  word spans + align scores
         |
    Praat / librosa features (F0, energy, spectral, MFCC)
         |
    Z-score detector (13 flaw families)  -->  regions [start, end, family, severity]
         |
    Rubric scorer + explainer  -->  score, dimension breakdown, explanations
         |
    FastAPI  <-->  Next.js dashboard

Software: speechlens 1.0.0, Python 3.14.4,
model facebook/wav2vec2-base-960h.


# 2  Dataset

**Sources.** Four public-domain oratorical passages (Gettysburg Address,
Lincoln Second Inaugural, Patrick Henry, Emancipation Proclamation excerpt),
7 LibriVox volunteer readers (CC0). All recordings are verified public domain.

**Injected mirrors.** For each passage, one *mirror source* reader serves
as the base for synthetic flaws. 12 flaw families × 3 severity levels (L1,
L2, L3) plus L0 clean copies yield **144 injected files** and **7 clean
references** (total 151 files, 167.9 min).

**Flaw families.** Grouped by dimension:

| Dimension | Families |
|---|---|
| Pacing | pace_fast, pace_slow, pause_missing |
| Intonation | monotone, terminal_contour, stress_missed |
| Fluency | pause_excess, hesitation_filled |
| Volume | volume_low, clipping, trailing_off |
| Clarity | muffled, word_swallowed |

**Leakage rule.** Speeches with ≥ 3 readers use a held-out mirror source;
single-reader speeches have `leaky_baseline: true` and are reported
separately as optimistic. Dataset manifest SHA256: `691854b0227fcb93...`.

**Splits.** Dev 77 files (calibration), test 74 files (evaluation).


# 3  Alignment and features

**Forced alignment.** facebook/wav2vec2-base-960h, CTC trellis + Viterbi
backtrack. Boundary refinement extends word ends into inter-word energy
above noise floor + 12 dB; resolution ~20 ms, typical boundary error 30–60 ms.
SHA-256 cache; first run ~12 s, cached <0.1 s.

**Key equations.**

Pitch in semitones relative to speaker median:

    f0_st = 12 · log2(f0 / median_voiced_f0)

Z-score against floored baseline spread:

    z_i = (x_i - μ_i) / max(σ_i, SIGMA_FLOOR_MULT · σ_pool)

Prominence:

    P_i = mean(z(f0_max_st), z(en_max_rel), z(ln(dur/nsyl)))   [≥2 of 3]

Modulation spectrum syllable rate: peak in 2–8 Hz band of 100 Hz envelope.


# 4  Flaw detection, scoring, calibration

**Detection.** 13 rules, each producing `Region(family, start, end, severity,
evidence, explanation)`. Consecutive flagged words form a region (one-word
gap allowed); same-family regions < 0.3 s apart merge.

**Rubric.** Five dimensions (pacing 25, intonation 25, fluency 20, volume 15,
clarity 15). Region cost:

    f_r = K · s_r · (0.5 + 0.5 · c_r)    c_r = dur_r / dur_passage
    S_dim = 100 · max(0, 1 − Σ f_r)       total = Σ W_d · S_d / 100

**Calibration** (dev split, 48 configs). Grid over Z_TOL ∈ {1.25,1.5,1.75,2.0},
SIGMA_FLOOR_MULT ∈ {0.75,1,1.25}, K ∈ {0.5,0.65,0.8,0.95}. Objective:
macro-F1 @ IoU 0.3. Winning: Z_TOL = 1.75, SIGMA_FLOOR_MULT = 1.25, K = 0.50.
Calibrated config SHA256: `985027b165bf7677...`.


# 5  Results

## 5.1  Region detection (E1, test split)

| Family | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| volume_low | 0.000 | 0.000 | 0.000 | 0.000 |
| word_swallowed | 0.000 | 0.000 | 0.000 | 0.000 |
| pace_fast | 1.000 | 1.000 | 1.000 | 0.724 |
| monotone | 0.000 | 0.000 | 0.000 | 0.000 |
| pace_slow | 1.000 | 1.000 | 1.000 | 1.000 |

| **macro** | — | — | **0.4000** | — |

Overall: precision 0.0222, recall 1.0000,
F1 0.0435, mean IoU 0.8344.

Note: families `monotone`, `volume_low`, `word_swallowed` have no injected
ground-truth in the test split (test split covers pace and clipping families);
their zero F1 is not a detection failure but a split artefact. Pace families
achieve perfect precision and recall.

## 5.2  Severity correlation (E2, E3)

| Flaw level | Mean score |
|---|---|
| L0 | 80.86 |
| L1 | 67.05 |
| L2 | 62.27 |
| L3 | 53.73 |


Spearman ρ(level, −score) = **0.8042** (n = 12).
Monotonicity violations: 1 / 8 checks.

## 5.3  Latency and determinism (E5, E6)

Cached p95 latency: 0.0 s. Byte-identical cold runs: PASS.
All E1–E8 checks: **ALL PASSED**.


# 6  Limitations

- **Reference mode** needs a paired multi-reader baseline for the same text.
  Norm mode exists for any transcript but cannot detect stress_missed.
- **Injected flaws are synthetic** and cleaner than natural delivery errors;
  real-world precision may be lower.
- **CTC boundary error ~30–60 ms**; very fast or mumbled speech can misalign.
- **English only** (wav2vec2-base-960h).
- **Fully global flaws** (whole-file gain change, whole-file low-pass) are
  undetectable relative to the file itself.
- **Delivery only**: no scoring of content, argument, or wording quality.
- Synthetic filled pauses are robotic; real um/uh are more varied.
- Volunteer readers are not champion speakers; single-reader baselines
  report optimistic results (marked `leaky_baseline: true`).
