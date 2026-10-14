# SpeechLens â€” Known Issues & Technical Limitations

## 1. Aligner Confidence and `word_swallowed` Sensitivity
- **Issue**: Facebook `wav2vec2-base-960h` alignment scores for unstressed short syllables (e.g., "a", "of", "the", "in") naturally cluster between 0.20 and 0.45 even in high-quality reference recordings.
- **Impact**: Setting a fixed `SWALLOW_SCORE_MAX = 0.30` without duration or energy gating produces false positive swallowed word flags on unstressed function words.
- **Mitigation / Next Step**: Calibration tuned `Z_TOL = 1.75` and `SIGMA_FLOOR_MULT = 1.25`. For Part 2, we gate `word_swallowed` flags by both duration (< 60 ms) and relative energy drop rather than aligner confidence alone.

## 2. Global Pace Detection on Short Clips
- **Issue**: Global speech tempo estimation relies on total utterance duration and syllable count.
- **Impact**: For recordings under 5 seconds (< 10 words), articulation rate estimation has higher variance (Â±0.8 syl/s) compared to longer oratorical passages.
- **Status**: Pipeline warns when speech duration is below minimum recommended length.

## 3. CPU Latency for Uncached Audio
- **Issue**: Initial analysis of a 75-second audio clip takes ~20 seconds on CPU (CTC Viterbi decoding + Praat pitch/formant extraction).
- **Status**: Cached analysis returns in < 0.6 seconds. Analytic remapping provides instant alignments for all synthetic dataset files.

## 4. Multi-reader Baseline Availability
- **Issue**: Speeches with fewer than 2 baseline readers (`speech-002`, `speech-003`, `speech-004`) use conservative population norms from `norms.json` rather than multi-speaker empirical distributions.
- **Status**: Explicitly declared in metadata `meta.json` and handled gracefully by pipeline `norm` vs `reference` modes.


## 5. Docker Runtime Not Verified on Build Machine
- **Issue**: Docker Desktop may not be installed on the development machine.
- **Impact**: `docker build` and `docker run` were not exercised. The Dockerfile is syntactically correct and matches the structure of the Hugging Face Space config.
- **Mitigation**: The same Dockerfile is used for HF Spaces (T12). Verify by running `docker build -t speechlens .` on any machine with Docker. Expected image size: ~3-5 GB (PyTorch CPU wheels dominate).
