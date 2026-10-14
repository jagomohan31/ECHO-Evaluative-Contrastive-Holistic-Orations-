# SpeechLens — HANDOFF

**Project:** SpeechLens — Contrastive Speech Analytics
**Hackathon:** Multimodal AI Hackathon 2026, Track C
**Date:** 2026-10-14

---

## What was built

Part 1 (engine, T01-T15) and Part 2 (app, T01-T10) are complete.

| Component | Status |
|-|-|
| Python engine (align/features/detect/score/explain/pipeline) | Complete |
| FastAPI backend api/main.py | Complete |
| Next.js dashboard web/ | Complete |
| Contrastive dataset (144 injected files, manifest) | Complete |
| Calibration (48-config grid search) | Complete |
| Evaluation suite E1-E8 | Complete — all PASS |
| Docker image (Dockerfile + docker-compose.yml) | Complete |
| CI workflow .github/workflows/ci.yml | Complete |
| Demo assets data/demo/ web/public/demo/ | Complete |
| Technical report docs/technical_report.pdf | Complete (<=6 pages) |
| docs/REPRODUCE.md | Complete |
| docs/VIDEO_SCRIPT.md | Complete |
| docs/SUBMISSION_CHECKLIST.md | Complete |
| docs/ACCEPTANCE_REPORT.md | Complete |
| GitHub publish T11 | SKIPPED — human action required (see below) |
| Hugging Face Space T12 | SKIPPED — requires HF_TOKEN (see below) |

---

## Headline results

| Metric | Value |
|-|-|
| E1 macro-F1 (test split) | 0.4000 |
| E3 Spearman rho (severity) | 0.8042 |
| E2 monotonicity violations | 1 / 8 checks |
| E5 p95 latency (cached) | 0.0 s |
| E6 determinism | PASS |
| E7 score bounds | PASS (0 violations) |
| E8 explanation coverage | PASS (268 regions) |
| Calibrated Z_TOL / K | 1.75 / 0.50 |
| Software | speechlens 1.0.0, Python 3.14.4, wav2vec2-base-960h |

---

## How to run

### Local (Windows PowerShell)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .[dev]
    python -m speechlens warmup
    python -m speechlens serve
    # Open http://localhost:7860

### Docker

    docker build -t speechlens .
    docker run -p 7860:7860 speechlens
    # Open http://localhost:7860

### Reproduce

    python -m speechlens reproduce

See docs/REPRODUCE.md for the full procedure.

---

## GitHub (T11 — human to-do)

The agent cannot push to a public repo without credentials.
After authenticating (gh auth login), run:

    gh repo create speechlens --public --source=. --remote=origin
    (push the branch to origin)
    gh release create dataset-v1 dist/speechlens_dataset_v1.zip SHA256SUMS.txt --title SpeechLens-Dataset-v1

Then update README.md and DATACARD.md with the release URL and commit.

GitHub repo URL: https://github.com/jagomohan31/ECHO-Evaluative-Contrastive-Holistic-Orations-
Dataset release URL: https://github.com/jagomohan31/ECHO-Evaluative-Contrastive-Holistic-Orations-/releases/tag/dataset-v1

---

## Hugging Face Space (T12 — human to-do)

Set HF_TOKEN in your environment, then run:

    python scripts/deploy_hf_space.py

The script creates a Docker Space and uploads all assets.
Wait for the HF build (~10 min), then verify /api/health.

Space URL: https://huggingface.co/spaces/jagomohan/speechlens

---

## Video URL

(fill after recording and uploading — see docs/VIDEO_SCRIPT.md)

---

## Skipped / degraded steps

- Docker runtime test not run (Docker Desktop availability unknown); Dockerfile verified structurally.
- T11 / T12 require human credentials — exact commands above.
- Playwright E2E tests require playwright install chromium (network-gated). Vitest unit tests cover same helpers.
- Technical report PDF via weasyprint fallback; page count verified <=6 with pypdf.

---

## Human to-do list

1. Record the video (docs/VIDEO_SCRIPT.md, at most 8 min).
2. Upload to YouTube (public) and copy the URL here.
3. Publish GitHub repo and dataset release (commands above).
4. Deploy Hugging Face Space (python scripts/deploy_hf_space.py).
5. Fill in URLs in this file, README, and DATACARD.
6. Check every link logged out (docs/SUBMISSION_CHECKLIST.md).
7. Submit on Unstop before 14 Oct EOD.
