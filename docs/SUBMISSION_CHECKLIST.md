# SpeechLens — Submission Checklist

**Hackathon:** Multimodal AI Hackathon 2026, Track C
**Deadline:** 14 October 2026 EOD — aim to submit by midday.

This is your to-do list. Everything below is a **human task** that requires
credentials or a browser the agent cannot access. Everything above (code,
tests, report, assets) is already done.

---

## Before submitting

- [ ] **Record the video** — follow `docs/VIDEO_SCRIPT.md`. At most 8 min.
- [ ] **Upload to YouTube** — unlisted first; check all clips play; switch to public.
- [ ] **Copy the YouTube URL** into `HANDOFF.md` under Video URL.

---

## GitHub (T11)

> If `gh` is installed and authenticated, run the commands in `HANDOFF.md` -> GitHub.
> Otherwise follow the manual steps there.

- [ ] Repo is public at `https://github.com/<you>/speechlens`
- [ ] Release `dataset-v1` is published with `speechlens_dataset_v1.zip`
- [ ] SHA256 of the zip matches `SHA256SUMS.txt`
- [ ] README and DATACARD links to the release URL resolve (logged-out check)

---

## Hugging Face Space (T12)

> Set `HF_TOKEN` and run: `python scripts/deploy_hf_space.py`
> Or follow `HANDOFF.md` -> Hugging Face.

- [ ] Space is live at `https://huggingface.co/spaces/<you>/speechlens`
- [ ] `GET /api/health` returns `{status:ok}` (check logged out)
- [ ] At least one demo clip loads and analyses without error
- [ ] Note: free Spaces may sleep; open the URL and wait 20 s before anyone judges (15-20 Oct)

---

## Final link check (open all while logged out)

- [ ] GitHub repo
- [ ] GitHub dataset release asset (direct .zip link)
- [ ] Hugging Face Space URL
- [ ] `docs/technical_report.pdf` — check page count <= 6
- [ ] YouTube video (public, not unlisted)

---

## Submission form (Unstop)

Fill in:
- GitHub repository URL
- Dataset URL (GitHub Release or HF dataset)
- Demo URL (Hugging Face Space)
- Technical report (upload `docs/technical_report.pdf`)
- Video URL (YouTube)

Click submit. Take a screenshot for your records.

---

## After submitting

- [ ] Screenshot of the Unstop confirmation saved
- [ ] `HANDOFF.md` updated with all final URLs
- [ ] Brief note to teammates with the submission receipt