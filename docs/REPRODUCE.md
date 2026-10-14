# Reproducing SpeechLens Results

This document gives the exact commands to reproduce `results/metrics.json`
and all figures from scratch on any supported platform.

## Prerequisites

- Python 3.11+ with `speechlens` installed (`pip install -e ".[dev]"`)
- Node 22 with `npm ci` run inside `web/`
- Internet access for the first run (downloads `facebook/wav2vec2-base-960h`)

## Windows PowerShell

```powershell
# 1. Set up environment (skip if already done)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m speechlens warmup      # download + cache model weights

# 2. Run the quick reproducibility check (used by CI)
python -m speechlens reproduce

# 3. Full evaluation on the complete test split
python -m speechlens evaluate

# 4. Check the output
Get-Content results\metrics.json | python -m json.tool | Select-Object -First 20
```

## macOS / Linux

```sh
# 1. Set up environment (skip if already done)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m speechlens warmup

# 2. Quick check (same as CI)
python -m speechlens reproduce
# or via make
make reproduce

# 3. Full evaluation
python -m speechlens evaluate
```

## What `reproduce` does

`python -m speechlens reproduce` runs `evaluate --quick` (one dev speech,
six files) and then asserts that the numbers in the freshly produced
`results/metrics.json` match the committed constraints:

- Counts (`total_tp`, `total_fp`, `total_fn`) must be **exact**.
- Rates (`f1`, `spearman_r`, etc.) must be within **0.01** of the committed
  value.
- All eight evals (E1–E8) must still report `all_passed: true`.

Exit code 0 = reproduced. Exit code 1 = mismatch; the diff is printed.

## Outputs

After a full `evaluate` run, these files are updated:

| Path | Contents |
|---|---|
| `results/metrics.json` | All E1–E8 metrics, software versions, sha256 hashes |
| `results/REPORT.md` | Human-readable evaluation report |
| `results/tables/eval_summary.csv` | One row per eval check |
| `results/tables/per_family_metrics.csv` | E1 per-flaw-family P/R/F1 |
| `results/fig_calibration.png` | Grid-search heatmap (T14) |
| `results/fig_severity_curve.png` | Score vs. flaw level curves |
| `results/fig_confusion_matrix.png` | Predicted vs. injected families |
| `results/fig_example_overlay.png` | Example pitch/energy/tempo overlay |

## Notes

- CTC alignment is SHA-256 cached in `.cache/align/`; the first run per
  unique audio+transcript takes ~12 s, cached runs are <0.1 s.
- Model weights are cached by HuggingFace in `$HF_HOME` (default
  `~/.cache/huggingface`); `warmup` pre-populates this.
- `--quick` processes six files from one dev speech. The full evaluation
  processes all 74 test files and takes ~30 min on CPU the first time (all
  cached on re-runs, ~2 min).