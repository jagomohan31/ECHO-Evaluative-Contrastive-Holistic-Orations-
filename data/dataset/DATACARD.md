# SpeechLens Dataset v1 -- DATACARD

## Summary
Contrastive speech dataset with injected prosodic flaws at three severity levels,
grounded to time spans for evaluation of spoken-language analysis systems.

## Licence
All source recordings are LibriVox public-domain (CC0). Synthetic modifications
are licensed CC0 for research use.

## Ethics
Flawed files are synthetic modifications of real people's recordings, made only
for evaluation. They are flagged `synthetic_flaw: true`, keep source attribution,
and **must not be redistributed as authentic speech** (DESIGN 5.8).

## Speeches
- **speech-001**: The Gettysburg Address
- **speech-002**: Lincoln Second Inaugural Address
- **speech-003**: Give Me Liberty or Give Me Death
- **speech-004**: Emancipation Proclamation (excerpt)


## Flaw families
- pace_fast (pacing)
- pace_slow (pacing)
- pause_missing (pacing)
- monotone (intonation)
- terminal_contour (intonation)
- stress_missed (intonation)
- pause_excess (fluency)
- hesitation_filled (fluency)
- volume_low (volume)
- clipping (volume)
- trailing_off (volume)
- muffled (clarity)

## Statistics
- Injected files: 144
- L0 ideal recordings: 7
- Total duration: 167.9 min

## Splits
- Dev: 77 files
- Test: 74 files

## Columns in manifest.csv
file, speech_id, reader_id, role, level, families, n_flaws, duration_s, split, sha256, leaky_baseline

## Leakage policy
For speeches with fewer than 3 readers, mirrors are built from the same recording
as the baseline (leaky_baseline=true). These results are reported separately in
the evaluation as optimistic upper bounds.
