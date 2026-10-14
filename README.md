# SpeechLens

**Contrastive Speech Analytics & Temporal Flaw Grounding**
Multimodal AI Hackathon 2026, Track C

SpeechLens takes a recording plus its transcript, force-aligns the text to the audio,
extracts speaker-normalised prosodic features, compares them word-by-word with an expected
band built from good recordings of the same text, finds flaw regions (start and end in
seconds), explains each region in numbers and plain words, scores the delivery on a
transparent rubric, and shows everything in a web dashboard.

## Quick Start

### Docker (recommended)
```sh
docker build -t speechlens .
docker run -p 7860:7860 speechlens
# Open http://localhost:7860
```

### Local — Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m speechlens warmup        # downloads model weights once
python -m speechlens serve         # starts API at http://localhost:7860
```

### Local — macOS / Linux
```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m speechlens warmup
python -m speechlens serve
```

## Reproduce results
```sh
python -m speechlens reproduce
```

## Repo map
```
src/speechlens/   Python library (align, features, detect, score, explain, pipeline)
api/              FastAPI backend
web/              Next.js dashboard
data/library/     Source recordings + alignments (committed)
data/dataset/     Built by make dataset (not committed, rebuilt from library)
results/          metrics.json, figures, REPORT.md (committed)
config/           calibrated.json (committed after calibration)
docs/             Technical report, video script, handoff
```

## Licences
Code: MIT. Dataset sources: public domain / CC0 / CC-BY / CC-BY-SA / US-government — see DATACARD.md.