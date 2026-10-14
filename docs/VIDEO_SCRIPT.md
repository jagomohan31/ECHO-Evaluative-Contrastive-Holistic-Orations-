# SpeechLens — Video Script

> Numbers in this script are filled from `results/metrics.json`
> (speechlens v1.0.0).
> Total runtime: at most 8 minutes. Record in one take if possible.

---

## 0:00 — Hook (25 s)

Play **speech-001_clean_L0.flac** (the Gettysburg Address, ideal reading).
Then immediately play **speech-001_pace_fast_L3.flac** (same passage, severe
fast-pace injection). The dashboard shows the total score drop and the
orange region flagged on the waveform.

**Say:** "Same 74 seconds of Lincoln. One delivered well, one with a common
delivery flaw. SpeechLens spots the difference, pins it to 3.1–41.2 seconds,
and tells you exactly why."

---

## 0:25 — The problem (35 s)

**Say:** "Judging a speech is subjective — unless you have a reference. We
built a system that force-aligns audio to its transcript word by word,
compares the speaker's pitch, pace, energy, and articulation against a
baseline of historically known-good readings of the same text, and flags
where they diverge. No LLM in the loop. Every number is traceable to a
formula."

---

## 1:00 — Dataset (90 s)

Show the `data/library/` folder tree briefly.

**Say:** "Our evaluation backbone is 4 public-domain speeches, 7 LibriVox
readers, and 144 synthetic mirror recordings — each a clean reading with one
flaw family injected at one of three severity levels, with ground-truth
time spans."

Show `data/dataset/speech-001/` briefly. Read the families from the
DATACARD: pace_fast, pace_slow, monotone, clipping, muffled, volume_low,
pause_excess, hesitation_filled, trailing_off, terminal_contour,
stress_missed, word_swallowed.

**Say:** "Injections are deterministic — same seed, same file every time.
The leakage rule: speeches with three or more readers use a held-out source;
results for single-reader speeches are reported separately as optimistic."

---

## 2:30 — Stress test (90 s)

Load the four pace_fast files in the demo panel: L0 → L1 → L2 → L3.
Show the total score dropping with each level (scores visible in the demo
JSON: L0 ~80, L1 ~71, L2 ~62, L3 ~54).

**Say:** "Level 1 — almost imperceptible — still gets caught. Spearman
correlation between flaw level and score: 0.8042. Pace detection is
perfect: precision 1.0, recall 1.0 on the test split."

Show the warp-mode toggle: "x-axis: baseline time vs. your real time.
Speaker normalisation means a male and female voice reading the same passage
give the same z-scores for the same flaw."

---

## 4:00 — Live dashboard (150 s)

Upload **data/sample/speech-001_r4_jn_clipping_L2.flac**.
Set speech ID to speech-001.
Press Analyze.

While it runs (about 15.8 seconds first time, <1 s cached):
**Say:** "Force alignment with wav2vec2. Pitch extraction with Praat.
Five rubric dimensions. One schema-validated JSON."

When results appear:
1. Show the score panel — dimension bars, radar chart.
2. Click a clipping region on the waveform. It seeks and highlights.
3. Open the flaw card: "Play yours" vs "Play baseline" for that span.
4. Show the math panel: z-score formula with real numbers.
5. Show "+X pts if fixed" on the most costly region.
6. Toggle warp mode; the overlay chart redraws on the baseline time axis.
7. Click a word in the transcript strip — cross-highlights all views.

---

## 6:30 — Metrics and honest limitations (60 s)

Show `results/REPORT.md` in the browser.

**Say:** "Macro-F1 on the test split: 0.4. That number is dragged down
by three families with zero ground-truth instances in the test split —
pace detection is perfect. Spearman rho: 0.8042. All eight evaluation
checks pass."

**Limitations to state honestly:**
- "Reference mode needs a baseline for the same text. Norm mode exists for
  any transcript but loses stress analysis."
- "Flaws are synthetic and cleaner than natural ones."
- "CTC boundary error is about 30–60 ms."
- "English only."

---

## 7:30 — Links (30 s)

Show on screen simultaneously:
- GitHub repo URL (fill after T11 publish)
- Dataset release URL (fill after T11 publish)
- Hugging Face Space URL (fill after T12 deploy)
- `docs/technical_report.pdf` QR code or link

**Say:** "Code, dataset, demo, and report — all links in the description.
The dataset is CC0 — use it to benchmark your own speech coach."

---

## Post-production notes

- Clip duration: aim for 7:30–8:00.
- Captions: auto-generate, then fix technical terms (SpeechLens,
  wav2vec2, Praat, LibriVox, z-score, IoU).
- Upload to YouTube unlisted first; check the demo links all resolve
  before switching to public.