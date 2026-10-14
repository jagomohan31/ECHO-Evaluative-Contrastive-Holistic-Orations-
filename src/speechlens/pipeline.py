"""
pipeline.py -- End-to-end speech analysis pipeline (DESIGN 12, 13, 14).

Orchestrates:
  audio load -> forced align -> frame & word features -> baseline/norms comparison
  -> flaw detection -> scoring & cost -> explanations -> time-warped series -> schema JSON
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

from speechlens.config import (
    SR, ROUND_DECIMALS, Z_TOL, Z_MAX, SIGMA_FLOOR_MULT,
)
from speechlens import audio
from speechlens.align.ctc import align
from speechlens.features.frame import compute_frame_features
from speechlens.features.word import (
    compute_word_features, compute_global_features, classify_pauses, time_warp,
)
from speechlens.detect import detect_all, detect_norm
from speechlens.score import compute_scores, compute_region_costs
from speechlens.explain import explain_all
from speechlens.models import AnalysisResult

CACHE_DIR = Path(".cache/analysis")


def _r(val: float | None, decimals: int = ROUND_DECIMALS) -> float | None:
    if val is None or math.isnan(val) or math.isinf(val):
        return None
    return round(float(val), decimals)


def analyze(
    audio_path: str | Path,
    speech_id: str | None = None,
    transcript_path: str | Path | None = None,
    transcript_text: str | None = None,
    library_dir: Path = Path("data/library"),
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Run end-to-end analysis on an audio file.
    Returns schema-compliant dictionary (DESIGN §12).
    """
    audio_file = Path(audio_path)
    samples = audio.load(audio_file)
    input_sha256 = hashlib.sha256(samples.tobytes()).hexdigest()
    total_dur = len(samples) / SR
    warnings: list[str] = []

    # 1. Resolve transcript
    transcript = ""
    if transcript_text is not None and transcript_text.strip():
        transcript = transcript_text.strip()
    elif transcript_path is not None and Path(transcript_path).exists():
        transcript = Path(transcript_path).read_text(encoding="utf-8").strip()
    elif speech_id is not None and (library_dir / speech_id / "transcript.txt").exists():
        transcript = (library_dir / speech_id / "transcript.txt").read_text(encoding="utf-8").strip()
    else:
        # Check faster-whisper fallback
        try:
            from faster_whisper import WhisperModel
            wm = WhisperModel("base.en", device="cpu", compute_type="int8")
            segments, _ = wm.transcribe(samples)
            transcript = " ".join(seg.text for seg in segments).strip()
            warnings.append("Auto-transcribed with faster-whisper")
        except Exception:
            raise ValueError("Transcript missing in norm mode; no transcript provided.")

    # Cache check
    config_tag = f"{Z_TOL}_{Z_MAX}_{SIGMA_FLOOR_MULT}"
    cache_key = hashlib.sha256(
        (input_sha256 + (speech_id or "") + transcript + config_tag).encode()
    ).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 2. Mode determination (reference vs norm)
    mode = "reference"
    baseline: dict[str, Any] = {}
    base_words: list[dict] = []
    meta_info: dict[str, Any] = {}
    baseline_readers: list[str] = []

    if speech_id is not None and (library_dir / speech_id / "baseline.json").exists():
        baseline_path = library_dir / speech_id / "baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        base_words = baseline.get("words", [])
        meta_path = library_dir / speech_id / "meta.json"
        if meta_path.exists():
            meta_info = json.loads(meta_path.read_text(encoding="utf-8"))
            baseline_readers = [r["id"] for r in meta_info.get("readers", [])]

        lib_trans_path = library_dir / speech_id / "transcript.txt"
        if lib_trans_path.exists():
            lib_trans = lib_trans_path.read_text(encoding="utf-8").strip()
            u_words = transcript.lower().split()
            l_words = lib_trans.lower().split()
            if l_words and abs(len(u_words) - len(l_words)) / len(l_words) > 0.10:
                mode = "norm"
                warnings.append("Norm mode: transcript differs by >10% of words from reference baseline.")
    else:
        mode = "norm"
        if speech_id is not None:
            warnings.append(f"Norm mode: baseline not found for speech '{speech_id}'.")
        else:
            warnings.append("Norm mode: no speech ID provided. Using population norms.")

    norms_path = library_dir / "norms.json"
    norms = json.loads(norms_path.read_text(encoding="utf-8")) if norms_path.exists() else {}

    # 3. Forced alignment
    align_res = align(samples, transcript, use_cache=True)
    align_words = [
        {"i": w.i, "text": w.text, "text_norm": w.text_norm, "punct_after": w.punct_after,
         "start": w.start, "end": w.end, "score": w.score, "nsyl": w.nsyl}
        for w in align_res.words
    ]
    align_pauses = [
        {"start": p.start, "end": p.end, "before_word": p.before_word, "after_word": p.after_word}
        for p in align_res.pauses
    ]

    scores = [w.score for w in align_res.words]
    mean_score = float(np.mean(scores)) if scores else 0.0
    if mean_score < 0.35:
        warnings.append("Wrong passage or noisy audio: mean alignment score < 0.35")

    # 4. Feature extraction
    ff = compute_frame_features(samples, SR)
    word_feats = compute_word_features(align_words, align_pauses, ff, total_dur)
    base_pause_dict = {
        i: bw.get("mu_pause_before", 0.0)
        for i, bw in enumerate(base_words)
    } if mode == "reference" and base_words else None
    classified_pauses = classify_pauses(align_pauses, ff, baseline_pause_mu=base_pause_dict)
    global_feats = compute_global_features(word_feats, classified_pauses, ff, total_dur)

    # 5. Flaw detection
    if mode == "reference":
        regions = detect_all(word_feats, classified_pauses, baseline, norms, ff=ff)
    else:
        regions = detect_norm(word_feats, classified_pauses, norms, ff=ff)

    # 6. Scoring & costs
    scores_dict = compute_scores(regions, total_dur)
    compute_region_costs(regions, total_dur)

    # 7. Explanations
    explain_all(regions)

    # 8. Series & Time warp (downsampled to 50 Hz, hop_s = 0.02)
    hop_s = 0.02
    t_user = np.arange(0.0, total_dur, hop_s, dtype=float)
    # base_words (stat dicts) already populated above; load mirror reader's aligned timestamps for time_warp
    ref_align_words: list[dict] = []
    if mode == "reference" and speech_id is not None:
        mirror_id = meta_info.get("mirror_source")
        if not mirror_id:
            readers = meta_info.get("readers", [])
            mirror_id = readers[0]["id"] if readers else None
        if mirror_id:
            ref_align_path = library_dir / speech_id / "readers" / f"{mirror_id}.align.json"
            if ref_align_path.exists():
                ref_align_data = json.loads(ref_align_path.read_text(encoding="utf-8"))
                ref_align_words = ref_align_data.get("words", [])

    if mode == "reference" and ref_align_words:
        t_base = time_warp(t_user, align_words, ref_align_words)
    else:
        t_base = t_user.copy()

    # Pitch interpolation onto t_user grid
    f0_times = ff.times
    user_pitch_grid: list[float | None] = []
    if ff.f0_st is not None and len(ff.f0_st) > 0:
        # Interpolate where voiced
        voiced_mask = ~np.isnan(ff.f0_st)
        if np.any(voiced_mask):
            f0_interp = np.interp(t_user, f0_times[voiced_mask], ff.f0_st[voiced_mask])
            # Zero out where unvoiced in nearest frame
            f0_indices = np.clip((t_user / (f0_times[1] - f0_times[0])).astype(int), 0, len(ff.f0_st) - 1)
            for idx, ti in enumerate(t_user):
                if voiced_mask[f0_indices[idx]]:
                    user_pitch_grid.append(_r(f0_interp[idx]))
                else:
                    user_pitch_grid.append(None)
        else:
            user_pitch_grid = [None] * len(t_user)
    else:
        user_pitch_grid = [None] * len(t_user)

    # Baseline pitch & energy bands
    base_pitch_line: list[float | None] = [None] * len(t_user)
    base_pitch_lo: list[float | None] = [None] * len(t_user)
    base_pitch_hi: list[float | None] = [None] * len(t_user)
    base_en_line: list[float | None] = [None] * len(t_user)
    base_en_lo: list[float | None] = [None] * len(t_user)
    base_en_hi: list[float | None] = [None] * len(t_user)

    if mode == "reference" and base_words and ref_align_words:
        # Use mirror reader's aligned timestamps for the time axis; mu/sigma from baseline stat dicts
        n_bw = min(len(base_words), len(ref_align_words))
        b_mid_times = [0.5 * (ref_align_words[k].get("start", 0.0) + ref_align_words[k].get("end", 0.0)) for k in range(n_bw)]
        b_pitch_mu = [base_words[k].get("mu_f0_mean_st", 0.0) for k in range(n_bw)]
        b_pitch_sig = [max(base_words[k].get("sigma_f0_mean_st", 0.0), float(norms.get("sigma_pool_f0_mean_st", 1.5))) for k in range(n_bw)]
        b_en_mu = [base_words[k].get("mu_en_mean_rel", -10.0) for k in range(n_bw)]
        b_en_sig = [max(base_words[k].get("sigma_en_mean_rel", 0.0), float(norms.get("sigma_pool_en_mean_rel", 3.0))) for k in range(n_bw)]

        if len(b_mid_times) >= 2:
            p_mu_interp = np.interp(t_base, b_mid_times, b_pitch_mu)
            p_sig_interp = np.interp(t_base, b_mid_times, b_pitch_sig)
            en_mu_interp = np.interp(t_base, b_mid_times, b_en_mu)
            en_sig_interp = np.interp(t_base, b_mid_times, b_en_sig)

            for i in range(len(t_user)):
                base_pitch_line[i] = _r(p_mu_interp[i])
                base_pitch_lo[i] = _r(p_mu_interp[i] - p_sig_interp[i])
                base_pitch_hi[i] = _r(p_mu_interp[i] + p_sig_interp[i])
                base_en_line[i] = _r(en_mu_interp[i])
                base_en_lo[i] = _r(en_mu_interp[i] - en_sig_interp[i])
                base_en_hi[i] = _r(en_mu_interp[i] + en_sig_interp[i])

    # User energy grid
    en_interp = np.interp(t_user, ff.times, ff.energy_db)
    user_en_grid = [_r(val) for val in en_interp]

    # Tempo series
    u_mid_times = [0.5 * (w.start + w.end) for w in word_feats] if word_feats else [0.0]
    u_art_rates = [w.art_rate for w in word_feats] if word_feats else [0.0]
    u_tempo_interp = np.interp(t_user, u_mid_times, u_art_rates)
    user_tempo_grid = [_r(val) for val in u_tempo_interp]

    b_tempo_grid: list[float | None] = []
    base_art = baseline.get("artic_rate", float(norms.get("artic_rate_mean", 4.2)))
    b_tempo_grid = [_r(base_art) for _ in t_user]

    # 9. Words output list
    flagged_families_per_word: dict[int, list[str]] = {w.i: [] for w in word_feats}
    for r in regions:
        for wi in range(r.word_start, r.word_end + 1):
            if wi in flagged_families_per_word:
                flagged_families_per_word[wi].append(r.family)

    words_out: list[dict[str, Any]] = []
    for i in range(len(word_feats)):
        wf = word_feats[i]
        bw = base_words[i] if (mode == "reference" and i < len(base_words)) else {}
        ref_bw = ref_align_words[i] if (mode == "reference" and i < len(ref_align_words)) else {}
        b_start = _r(ref_bw.get("start")) if ref_bw else None
        b_end = _r(ref_bw.get("end")) if ref_bw else None

        # z-scores
        z_tempo = None
        z_pitch = None
        z_energy = None
        z_pause = None
        if bw:
            sp_tempo = float(norms.get("sigma_D", 0.12))
            mu_share = bw.get("mu_share", 0.0)
            if mu_share > 0 and wf.share > 0:
                z_tempo = _r(math.log(wf.share / mu_share) / sp_tempo)
            sp_pitch = max(bw.get("sigma_f0_mean_st", 0.0), float(norms.get("sigma_pool_f0_mean_st", 1.5)))
            if not math.isnan(wf.f0_mean_st):
                z_pitch = _r((wf.f0_mean_st - bw.get("mu_f0_mean_st", 0.0)) / max(sp_pitch, 1e-6))
            sp_en = max(bw.get("sigma_en_mean_rel", 0.0), float(norms.get("sigma_pool_en_mean_rel", 3.0)))
            if not math.isnan(wf.en_mean_rel):
                z_energy = _r((wf.en_mean_rel - bw.get("mu_en_mean_rel", 0.0)) / max(sp_en, 1e-6))
            if wf.pause_before > 0:
                sp_pause = max(bw.get("sigma_pause_before", 0.0), float(norms.get("sigma_pool_pause_before", 0.2)))
                z_pause = _r((wf.pause_before - bw.get("mu_pause_before", 0.0)) / max(sp_pause, 1e-6))

        words_out.append({
            "i": wf.i,
            "text": wf.text,
            "start": _r(wf.start),
            "end": _r(wf.end),
            "base_start": b_start,
            "base_end": b_end,
            "z": {
                "tempo": z_tempo,
                "pitch": z_pitch,
                "energy": z_energy,
                "pause": z_pause,
            },
            "prominence": {
                "user": _r(wf.prominence),
                "base": _r(bw.get("mu_prominence")) if bw else None,
            },
            "align_score": _r(wf.align_score),
            "flags": flagged_families_per_word.get(wf.i, []),
        })

    # 10. Pauses output
    pauses_out = [
        {
            "start": _r(p.start),
            "end": _r(p.end),
            "kind": p.kind,
            "expected": bool(p.is_expected),
        }
        for p in classified_pauses
    ]

    # 11. Regions output
    regions_out = [
        {
            "id": r.id,
            "family": r.family,
            "dimension": r.dimension,
            "start": _r(r.start),
            "end": _r(r.end),
            "word_idx": [r.word_start, r.word_end],
            "severity": _r(r.severity),
            "z_peak": _r(r.z_peak),
            "cost": _r(r.cost),
            "evidence": r.evidence,
            "explanation": r.explanation,
        }
        for r in regions
    ]

    # 12. Assemble result dict
    result = {
        "schema_version": "1.0",
        "meta": {
            "speech_id": speech_id if mode == "reference" else None,
            "mode": mode,
            "duration_s": _r(total_dur),
            "n_words": len(words_out),
            "baseline_readers": baseline_readers,
            "software": {
                "speechlens": "1.0.0",
                "model": "facebook/wav2vec2-base-960h",
            },
            "input_sha256": input_sha256,
        },
        "scores": scores_dict,
        "global": {
            "articulation_rate_syl_s": _r(global_feats.artic_rate),
            "baseline_articulation_rate_syl_s": _r(baseline.get("artic_rate")) if mode == "reference" else None,
            "wpm": _r(global_feats.words_per_minute),
            "pause_ratio": _r(global_feats.pause_ratio),
            "f0_sd_st": _r(global_feats.f0_sd_st),
        },
        "words": words_out,
        "series": {
            "hop_s": hop_s,
            "t_user": [_r(t) for t in t_user],
            "t_base": [_r(t) for t in t_base],
            "pitch_st": {
                "user": user_pitch_grid,
                "base": base_pitch_line,
                "lo": base_pitch_lo,
                "hi": base_pitch_hi,
            },
            "energy_db": {
                "user": user_en_grid,
                "base": base_en_line,
                "lo": base_en_lo,
                "hi": base_en_hi,
            },
            "tempo": {
                "user": user_tempo_grid,
                "base": b_tempo_grid,
            },
        },
        "pauses": pauses_out,
        "regions": regions_out,
        "warnings": warnings,
    }

    # Validate against Pydantic model
    AnalysisResult.model_validate(result)

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result
