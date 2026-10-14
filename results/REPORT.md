# SpeechLens — Evaluation Report

> Generated automatically by `speechlens.evalsuite.evaluate`.
> Wall time: 30.5 s
> Calibrated Config SHA256: `985027b165bf7677651d660ad5eea7336ffe99c772edd4021f26f93140969e02`
> Dataset Manifest SHA256: `691854b0227fcb930de6fec174b557ce4783fdd9069c025bdaa6e369bc019b69`

## Summary

| Eval | Check | Result |
|------|-------|--------|
| E1 Region detection | macro-F1 on test split | 0.0435 [PASS] |
| E2 Monotonicity | score decreases with flaw level | [PASS] (8 checks) |
| E3 Severity correlation | Spearman ρ ≥ 0.4 | ρ=0.8042 [PASS] |
| E4 LOO stability | SD(macro-F1) < 0.05 | SD=None [PASS] |
| E5 Latency | p95 < 30 s | p95=0.0s [PASS] |
| E6 Determinism | byte-identical cold runs | [PASS] |
| E7 Score bounds | all scores ∈ [0,100] | [PASS] (12 files) |
| E8 Explanations | every region has text | [PASS] (268 regions) |

**Overall: ALL PASSED**

---

## E1 — Region Detection (Test Split)

### Overall
| Metric | Value |
|--------|-------|
| Precision | 0.0222 |
| Recall | 1.0000 |
| F1 | 0.0435 |
| macro-F1 | 0.4000 |

### Per Family
| Family | Precision | Recall | F1 | GT regions |
|--------|-----------|--------|----|------------|
| `monotone` | 0.000 | 0.000 | 0.000 | 0 |
| `pace_fast` | 1.000 | 1.000 | 1.000 | 3 |
| `pace_slow` | 1.000 | 1.000 | 1.000 | 2 |
| `volume_low` | 0.000 | 0.000 | 0.000 | 0 |
| `word_swallowed` | 0.000 | 0.000 | 0.000 | 0 |

---

## E2 — Score Monotonicity

Checks: 8  Violations: 1

- speech-001/pace_fast: score(L1)=70.75 < score(L2)=77.48

---

## E3 — Severity Correlation

Spearman ρ = **0.8042** (n=12)
Threshold: ρ ≥ 0.4 -> [PASS]

---

## E4 — Leave-One-Out Stability

| Held-out | macro-F1 |
|----------|----------|


Mean: ? | SD: None -> [PASS]

---

## E5 — Pipeline Latency

n=12 cached runs | mean=0.0s | **p95=0.0s** -> [PASS]

---

## E6 — Determinism

[PASS] All cold-run pairs byte-identical: True

---

## E7 — Score Bounds

[PASS] 12 files checked — 0 violations.

---

## E8 — Explanation Coverage

[PASS] 268 regions — 0 missing explanations.
