"""detect.py -- Flaw detection engine (DESIGN 9)."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from speechlens.config import (
    Z_TOL, Z_MAX, SIGMA_FLOOR_MULT, CLIP_FRAC_MIN,
    SWALLOW_SCORE_MAX, SWALLOW_DUR_MAX_S,
)
from speechlens.features.word import WordFeatures, PauseInfo

FAMILY_DIM = {
    "pace_fast": "pacing", "pace_slow": "pacing", "pace_global": "pacing",
    "pause_missing": "pacing", "monotone": "intonation", "terminal_contour": "intonation",
    "stress_missed": "intonation", "pause_excess": "fluency", "hesitation_filled": "fluency",
    "volume_low": "volume", "clipping": "volume", "trailing_off": "volume",
    "muffled": "clarity", "word_swallowed": "clarity",
}

MIN_REGION = {
    "pace_fast": 4, "pace_slow": 4, "pause_missing": 1, "pause_excess": 1,
    "hesitation_filled": 1, "monotone": 6, "terminal_contour": 1, "stress_missed": 1,
    "volume_low": 2, "clipping": 2, "trailing_off": 3, "muffled": 3,
    "word_swallowed": 1, "pace_global": 0,
}

MERGE_GAP_S = 0.3
_SENTENCE_END_RE = re.compile(r"[.!?]")


@dataclass
class Region:
    family: str
    dimension: str
    start: float
    end: float
    word_start: int
    word_end: int
    severity: float
    z_peak: float
    cost: float = 0.0
    id: str = ""
    evidence: dict = field(default_factory=dict)
    explanation: dict = field(default_factory=dict)
    boundary_refined: bool = False

    @property
    def word_idx(self) -> list[int]:
        return [self.word_start, self.word_end]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "dimension": self.dimension,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "word_idx": [self.word_start, self.word_end],
            "severity": round(self.severity, 4),
            "z_peak": round(self.z_peak, 4),
            "cost": round(self.cost, 4),
            "evidence": self.evidence,
            "explanation": self.explanation,
        }


def _safe(v: float) -> float:
    return 0.0 if math.isnan(v) or math.isinf(v) else float(v)


def _z(x: float, mu: float, sp: float) -> float:
    return 0.0 if sp < 1e-9 else float((x - mu) / sp)


def _sev(z: float) -> float:
    return float(np.clip((abs(z) - Z_TOL) / (Z_MAX - Z_TOL), 0.0, 1.0))


def _fs(si: float, sp: float) -> float:
    return float(max(si, SIGMA_FLOOR_MULT * sp))


def _build_regions(
    flagged: list[int],
    words: list[Any],
    z_vals: dict[int, float],
    family: str,
    min_size: int,
    evidence_fn: Callable[[list[int], float], dict] | None = None,
) -> list[Region]:
    if not flagged:
        return []
    runs: list[list[int]] = []
    cur = [flagged[0]]
    for a, b in zip(flagged, flagged[1:]):
        if b - a <= 2:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)

    valid = [r for r in runs if len(r) >= min_size]
    if not valid:
        return []

    dim = FAMILY_DIM.get(family, "clarity")
    regions: list[Region] = []
    for run in valid:
        wi_s, wi_e = run[0], run[-1]
        zr = [z_vals.get(i, 0.0) for i in run]
        zp = max(zr, key=abs)
        ev = evidence_fn(run, zp) if evidence_fn else {}
        regions.append(Region(
            family=family,
            dimension=dim,
            start=words[wi_s].start,
            end=words[wi_e].end,
            word_start=wi_s,
            word_end=wi_e,
            severity=_sev(zp),
            z_peak=zp,
            evidence=ev,
        ))

    merged = [regions[0]]
    for r in regions[1:]:
        prev = merged[-1]
        if r.start - prev.end < MERGE_GAP_S:
            zp = max(prev.z_peak, r.z_peak, key=abs)
            wi_s = prev.word_start
            wi_e = r.word_end
            merged_run = list(range(wi_s, wi_e + 1))
            ev = evidence_fn(merged_run, zp) if evidence_fn else prev.evidence
            merged[-1] = Region(
                family=family,
                dimension=dim,
                start=prev.start,
                end=r.end,
                word_start=wi_s,
                word_end=wi_e,
                severity=_sev(zp),
                z_peak=zp,
                evidence=ev,
            )
        else:
            merged.append(r)
    return merged


# ---------------------------------------------------------------------------
# Detectors per family (DESIGN 9)
# ---------------------------------------------------------------------------

def detect_pace(
    words: list[Any],
    baseline_mu_share: list[float],
    sigma_D: float,
    base_art_rate: float | None = None,
) -> list[Region]:
    regions: list[Region] = []
    if not words or not baseline_mu_share:
        return regions
    n = min(len(words), len(baseline_mu_share))
    sigma_D = max(float(sigma_D), 0.10)

    rho = []
    for i in range(n):
        mu = baseline_mu_share[i]
        sh = getattr(words[i], "share", 0.0)
        rho.append(math.log(sh / mu) if mu > 0 and sh > 0 else 0.0)
    med = float(np.median(rho))
    d = [r - med for r in rho]

    win = 5
    ff: list[int] = []
    fs2: list[int] = []
    zf: dict[int, float] = {}
    zs: dict[int, float] = {}
    d_win_vals: dict[int, float] = {}

    for i in range(n):
        lo, hi = max(0, i - win // 2), min(n, i + win // 2 + 1)
        D = float(np.mean(d[lo:hi]))
        d_win_vals[i] = D
        z = D / sigma_D
        if z <= -Z_TOL:
            ff.append(i)
            zf[i] = z
        elif z >= Z_TOL:
            fs2.append(i)
            zs[i] = z

    def _ev_fast(run: list[int], zp: float) -> dict:
        t0, t1 = words[run[0]].start, words[run[-1]].end
        txt = " ".join(words[i].text for i in range(run[0], run[-1] + 1))
        # Find peak D
        peak_d = min(d_win_vals.get(i, 0.0) for i in run)
        ratio = float(math.exp(peak_d))
        pct = round(abs(1.0 - ratio) * 100, 1)
        return {
            "metric": "tempo_ratio",
            "unit": "ratio",
            "observed": round(ratio, 3),
            "expected": 1.0,
            "sigma": round(sigma_D, 3),
            "z": round(zp, 2),
            "formula": f"z = D / sigma_D = ({peak_d:.3f}) / {sigma_D:.3f} = {zp:.2f}",
            "t0": round(t0, 2),
            "t1": round(t1, 2),
            "text": txt,
            "pct": pct,
        }

    def _ev_slow(run: list[int], zp: float) -> dict:
        t0, t1 = words[run[0]].start, words[run[-1]].end
        txt = " ".join(words[i].text for i in range(run[0], run[-1] + 1))
        peak_d = max(d_win_vals.get(i, 0.0) for i in run)
        ratio = float(math.exp(peak_d))
        pct = round(abs(ratio - 1.0) * 100, 1)
        return {
            "metric": "tempo_ratio",
            "unit": "ratio",
            "observed": round(ratio, 3),
            "expected": 1.0,
            "sigma": round(sigma_D, 3),
            "z": round(zp, 2),
            "formula": f"z = D / sigma_D = ({peak_d:.3f}) / {sigma_D:.3f} = {zp:.2f}",
            "t0": round(t0, 2),
            "t1": round(t1, 2),
            "text": txt,
            "pct": pct,
        }

    regions += _build_regions(ff, words, zf, "pace_fast", MIN_REGION["pace_fast"], _ev_fast)
    regions += _build_regions(fs2, words, zs, "pace_slow", MIN_REGION["pace_slow"], _ev_slow)

    # pace_global
    user_art = sum(w.nsyl for w in words[:n]) / max(sum(w.dur for w in words[:n]), 1e-6)
    if base_art_rate is not None and base_art_rate > 0:
        base_art = float(base_art_rate)
    else:
        base_dur = sum(w.dur / max(w.share, 1e-9) * mu for w, mu in zip(words[:n], baseline_mu_share[:n]))
        base_art = sum(w.nsyl for w in words[:n]) / base_dur if base_dur > 0 else user_art
    if base_art > 0:
        g = math.log(user_art / base_art)
        if abs(g) > math.log(1.12):
            s = float(np.clip((abs(g) - math.log(1.12)) / (math.log(1.8) - math.log(1.12)), 0.0, 1.0))
            pct = round((user_art / base_art - 1.0) * 100, 1)
            regions.append(Region(
                family="pace_global",
                dimension="pacing",
                start=words[0].start,
                end=words[-1].end,
                word_start=0,
                word_end=n - 1,
                severity=s,
                z_peak=g,
                evidence={
                    "metric": "articulation_rate",
                    "unit": "syl/s",
                    "observed": round(user_art, 2),
                    "expected": round(base_art, 2),
                    "sigma": round(sigma_D, 2),
                    "z": round(g, 2),
                    "formula": f"g = ln(obs / exp) = ln({user_art:.2f} / {base_art:.2f}) = {g:.2f}",
                    "obs": round(user_art, 2),
                    "exp": round(base_art, 2),
                    "pct": pct,
                },
            ))
    return regions


def detect_pause_missing(
    words: list[Any],
    mu_pause: list[float],
    sigma_pause: list[float],
    sp_pause: float,
) -> list[Region]:
    regions: list[Region] = []
    n = min(len(words), len(mu_pause))
    for i in range(n):
        mu = mu_pause[i]
        if mu < 0.25:
            continue
        sp = _fs(sigma_pause[i] if i < len(sigma_pause) else 0.0, sp_pause)
        pb = getattr(words[i], "pause_before", 0.0)
        if pb < max(0.4 * mu, 0.10):
            z = _z(pb, mu, sp)
            prev_txt = words[i - 1].text if i > 0 else ""
            punct = getattr(words[i - 1], "punct_after", "") if i > 0 else ""
            obs_ms = round(pb * 1000)
            exp_ms = round(mu * 1000)
            sig_ms = round(sp * 1000)
            regions.append(Region(
                family="pause_missing",
                dimension="pacing",
                start=words[i].start,
                end=words[i].end,
                word_start=i,
                word_end=i,
                severity=_sev(z),
                z_peak=z,
                evidence={
                    "metric": "pause_before",
                    "unit": "ms",
                    "observed": obs_ms,
                    "expected": exp_ms,
                    "sigma": sig_ms,
                    "z": round(z, 2),
                    "formula": f"z = (obs - mu) / sigma' = ({obs_ms} - {exp_ms}) / {sig_ms} = {z:.2f}",
                    "word": words[i].text,
                    "t": round(words[i].start, 3),
                    "obs_ms": obs_ms,
                    "exp_ms": exp_ms,
                    "sig_ms": sig_ms,
                    "prev": prev_txt,
                    "punct": punct,
                },
            ))
    return regions


def detect_pause_excess(
    words: list[Any],
    pauses: list[Any],
    mu_pause: list[float],
    sigma_pause: list[float],
    sp_pause: float,
) -> list[Region]:
    regions: list[Region] = []
    for p in pauses:
        kind = getattr(p, "kind", p.get("kind") if isinstance(p, dict) else "other")
        if kind != "silent":
            continue
        p_start = getattr(p, "start", p.get("start") if isinstance(p, dict) else 0.0)
        p_end = getattr(p, "end", p.get("end") if isinstance(p, dict) else 0.0)
        dur = p_end - p_start
        if dur < 0.45:
            continue
        wi = getattr(p, "after_word", p.get("after_word") if isinstance(p, dict) else None)
        if wi is None or wi >= len(words):
            continue
        mu = mu_pause[wi] if wi < len(mu_pause) else 0.0
        sp = _fs(sigma_pause[wi] if wi < len(sigma_pause) else 0.0, sp_pause)
        if dur >= mu + max(2 * sp, 0.35):
            z = _z(dur, mu, sp)
            obs_ms = round(dur * 1000)
            exp_ms = round(mu * 1000)
            sig_ms = round(sp * 1000)
            regions.append(Region(
                family="pause_excess",
                dimension="fluency",
                start=p_start,
                end=p_end,
                word_start=wi,
                word_end=wi,
                severity=_sev(z),
                z_peak=z,
                evidence={
                    "metric": "pause_duration",
                    "unit": "ms",
                    "observed": obs_ms,
                    "expected": exp_ms,
                    "sigma": sig_ms,
                    "z": round(z, 2),
                    "formula": f"z = (obs - mu) / sigma' = ({obs_ms} - {exp_ms}) / {sig_ms} = {z:.2f}",
                    "word": words[wi].text,
                    "t": round(p_start, 3),
                    "obs_ms": obs_ms,
                    "exp_ms": exp_ms,
                    "sig_ms": sig_ms,
                },
            ))
    return regions


def detect_hesitation_filled(
    pauses: list[Any],
    words: list[Any] | None = None,
) -> list[Region]:
    regions: list[Region] = []
    for p in pauses:
        kind = getattr(p, "kind", p.get("kind") if isinstance(p, dict) else "other")
        if kind != "filled":
            continue
        p_start = getattr(p, "start", p.get("start") if isinstance(p, dict) else 0.0)
        p_end = getattr(p, "end", p.get("end") if isinstance(p, dict) else 0.0)
        dur = p_end - p_start
        if dur < 0.25:
            continue
        s = float(np.clip((dur - 0.25) / 0.75, 0.3, 1.0))
        after_w = getattr(p, "after_word", p.get("after_word") if isinstance(p, dict) else None)
        before_w = getattr(p, "before_word", p.get("before_word") if isinstance(p, dict) else 0)
        wi = after_w if after_w is not None else (before_w or 0)
        w_text = words[wi].text if (words and wi < len(words)) else ""
        ms_val = round(dur * 1000)
        regions.append(Region(
            family="hesitation_filled",
            dimension="fluency",
            start=p_start,
            end=p_end,
            word_start=wi,
            word_end=wi,
            severity=s,
            z_peak=dur,
            evidence={
                "metric": "filled_pause_dur",
                "unit": "ms",
                "observed": ms_val,
                "expected": 0,
                "sigma": 0,
                "z": round(dur, 2),
                "formula": f"dur = {dur:.2f} s",
                "ms": ms_val,
                "t": round(p_start, 3),
                "word": w_text,
            },
        ))
    return regions


def detect_monotone(
    words: list[Any],
    mu_f0sd: list[float],
    sigma_f0sd: list[float],
    sp_f0sd: float,
) -> list[Region]:
    flagged: list[int] = []
    z_vals: dict[int, float] = {}
    n = min(len(words), len(mu_f0sd))

    for i in range(n):
        vsw = getattr(words[i], "voiced_share_win", 0.0)
        f0_sd = getattr(words[i], "f0_sd_st", float("nan"))
        if vsw < 0.4 or math.isnan(f0_sd):
            continue
        sp = _fs(sigma_f0sd[i] if i < len(sigma_f0sd) else 0.0, sp_f0sd)
        z = _z(f0_sd, mu_f0sd[i], sp)
        if z <= -Z_TOL:
            flagged.append(i)
            z_vals[i] = z

    def _ev(run: list[int], zp: float) -> dict:
        t0, t1 = words[run[0]].start, words[run[-1]].end
        # Find peak word index
        p_idx = min(run, key=lambda idx: z_vals.get(idx, 0.0))
        obs = round(getattr(words[p_idx], "f0_sd_st", 0.0), 2)
        exp = round(mu_f0sd[p_idx], 2)
        sp = _fs(sigma_f0sd[p_idx] if p_idx < len(sigma_f0sd) else 0.0, sp_f0sd)
        sig = round(sp, 2)
        return {
            "metric": "f0_sd_st",
            "unit": "st",
            "observed": obs,
            "expected": exp,
            "sigma": sig,
            "z": round(zp, 2),
            "formula": f"z = (obs - mu) / sigma' = ({obs:.2f} - {exp:.2f}) / {sig:.2f} = {zp:.2f}",
            "t0": round(t0, 2),
            "t1": round(t1, 2),
            "obs": obs,
            "exp": exp,
            "sig": sig,
        }

    return _build_regions(flagged, words, z_vals, "monotone", MIN_REGION["monotone"], _ev)


def detect_terminal_contour(
    words: list[Any],
    mu_slope: list[float],
    sigma_slope: list[float],
    sp_slope: float,
) -> list[Region]:
    regions: list[Region] = []
    n = min(len(words), len(mu_slope))
    has_any_sentence_punct = any(_SENTENCE_END_RE.search(getattr(w, "punct_after", "")) for w in words[:n])

    for i in range(n):
        if mu_slope[i] > 0 or math.isnan(getattr(words[i], "f0_slope", float("nan"))):
            continue
        # Check sentence final condition
        if has_any_sentence_punct:
            is_sentence_final = bool(_SENTENCE_END_RE.search(getattr(words[i], "punct_after", ""))) or (i == n - 1)
            if not is_sentence_final:
                continue

        slope = getattr(words[i], "f0_slope", 0.0)
        if slope - mu_slope[i] < 2.0:
            continue
        sp = _fs(sigma_slope[i] if i < len(sigma_slope) else 0.0, sp_slope)
        z = _z(slope, mu_slope[i], sp)
        if z >= Z_TOL:
            obs = round(slope, 2)
            exp = round(mu_slope[i], 2)
            sig = round(sp, 2)
            regions.append(Region(
                family="terminal_contour",
                dimension="intonation",
                start=words[i].start,
                end=words[i].end,
                word_start=i,
                word_end=i,
                severity=_sev(z),
                z_peak=z,
                evidence={
                    "metric": "f0_slope",
                    "unit": "st/s",
                    "observed": obs,
                    "expected": exp,
                    "sigma": sig,
                    "z": round(z, 2),
                    "formula": f"z = (obs - mu) / sigma' = ({obs:.2f} - {exp:.2f}) / {sig:.2f} = {z:.2f}",
                    "word": words[i].text,
                    "t": round(words[i].start, 3),
                    "obs": obs,
                    "exp": exp,
                },
            ))
    return regions


def detect_stress_missed(
    words: list[Any],
    mu_prom: list[float],
    sigma_prom: list[float],
    sp_prom: float,
) -> list[Region]:
    regions: list[Region] = []
    n = min(len(words), len(mu_prom))
    for i in range(n):
        if mu_prom[i] < 1.0:
            continue
        prom = getattr(words[i], "prominence", float("nan"))
        if math.isnan(prom) or prom > 0.3:
            continue
        sp = _fs(sigma_prom[i] if i < len(sigma_prom) else 0.0, sp_prom)
        z = _z(prom, mu_prom[i], sp)
        if z <= -Z_TOL:
            obs = round(_safe(prom), 2)
            exp = round(mu_prom[i], 2)
            sig = round(sp, 2)
            regions.append(Region(
                family="stress_missed",
                dimension="intonation",
                start=words[i].start,
                end=words[i].end,
                word_start=i,
                word_end=i,
                severity=_sev(z),
                z_peak=z,
                evidence={
                    "metric": "prominence",
                    "unit": "z",
                    "observed": obs,
                    "expected": exp,
                    "sigma": sig,
                    "z": round(z, 2),
                    "formula": f"z = (obs - mu) / sigma' = ({obs:.2f} - {exp:.2f}) / {sig:.2f} = {z:.2f}",
                    "word": words[i].text,
                    "obs": obs,
                    "exp": exp,
                },
            ))
    return regions


def detect_volume_low(
    words: list[Any],
    mu_en: list[float],
    sigma_en: list[float],
    sp_en: float,
) -> list[Region]:
    flagged: list[int] = []
    z_vals: dict[int, float] = {}
    n = min(len(words), len(mu_en))

    for i in range(n):
        en = getattr(words[i], "en_mean_rel", float("nan"))
        if math.isnan(en):
            continue
        sp = _fs(sigma_en[i] if i < len(sigma_en) else 0.0, sp_en)
        z = _z(en, mu_en[i], sp)
        if z <= -Z_TOL:
            flagged.append(i)
            z_vals[i] = z

    def _ev(run: list[int], zp: float) -> dict:
        t0, t1 = words[run[0]].start, words[run[-1]].end
        p_idx = min(run, key=lambda idx: z_vals.get(idx, 0.0))
        obs = round(getattr(words[p_idx], "en_mean_rel", 0.0), 1)
        exp = round(mu_en[p_idx], 1)
        sp = _fs(sigma_en[p_idx] if p_idx < len(sigma_en) else 0.0, sp_en)
        sig = round(sp, 1)
        return {
            "metric": "en_mean_rel",
            "unit": "dB",
            "observed": obs,
            "expected": exp,
            "sigma": sig,
            "z": round(zp, 2),
            "formula": f"z = (obs - mu) / sigma' = ({obs:.1f} - {exp:.1f}) / {sig:.1f} = {zp:.2f}",
            "t0": round(t0, 2),
            "t1": round(t1, 2),
            "obs": obs,
            "exp": exp,
            "sig": sig,
        }

    return _build_regions(flagged, words, z_vals, "volume_low", MIN_REGION["volume_low"], _ev)


def detect_clipping(words: list[Any]) -> list[Region]:
    flagged: list[int] = []
    z_vals: dict[int, float] = {}
    for i, w in enumerate(words):
        cf = getattr(w, "clip_frac", 0.0)
        if cf >= CLIP_FRAC_MIN:
            flagged.append(i)
            z_vals[i] = float(np.clip(cf / 0.01, 0.2, 1.0))

    def _ev(run: list[int], zp: float) -> dict:
        t0, t1 = words[run[0]].start, words[run[-1]].end
        max_clip = max(getattr(words[i], "clip_frac", 0.0) for i in run)
        pct = round(max_clip * 100, 1)
        return {
            "metric": "clip_frac",
            "unit": "%",
            "observed": pct,
            "expected": 0.0,
            "sigma": 0.0,
            "z": round(zp, 2),
            "formula": f"clip_fraction = {pct:.1f}%",
            "t0": round(t0, 2),
            "t1": round(t1, 2),
            "pct": pct,
        }

    return _build_regions(flagged, words, z_vals, "clipping", MIN_REGION["clipping"], _ev)


def detect_trailing_off(
    words: list[Any],
    mu_slope: list[float],
    sigma_slope: list[float],
    sp_slope: float,
) -> list[Region]:
    flagged: list[int] = []
    z_vals: dict[int, float] = {}
    n = min(len(words), len(mu_slope))
    has_any_sentence_punct = any(_SENTENCE_END_RE.search(getattr(w, "punct_after", "")) for w in words[:n])

    # Find candidate indices: within last 3 words before .?! or the end
    eligible = set()
    if has_any_sentence_punct:
        sentence_ends = [i for i in range(n) if _SENTENCE_END_RE.search(getattr(words[i], "punct_after", "")) or i == n - 1]
        for se in sentence_ends:
            for offset in range(3):
                if se - offset >= 0:
                    eligible.add(se - offset)
    else:
        eligible = set(range(n))

    for i in range(n):
        if i not in eligible:
            continue
        slope = getattr(words[i], "en_slope", float("nan"))
        if math.isnan(slope):
            continue
        sp = _fs(sigma_slope[i] if i < len(sigma_slope) else 0.0, sp_slope)
        z = _z(slope, mu_slope[i], sp)
        if z <= -Z_TOL:
            flagged.append(i)
            z_vals[i] = z

    def _ev(run: list[int], zp: float) -> dict:
        wi_e = run[-1]
        p_idx = min(run, key=lambda idx: z_vals.get(idx, 0.0))
        obs = round(getattr(words[p_idx], "en_slope", 0.0), 1)
        exp = round(mu_slope[p_idx], 1)
        sp = _fs(sigma_slope[p_idx] if p_idx < len(sigma_slope) else 0.0, sp_slope)
        sig = round(sp, 1)
        return {
            "metric": "en_slope",
            "unit": "dB/s",
            "observed": obs,
            "expected": exp,
            "sigma": sig,
            "z": round(zp, 2),
            "formula": f"z = (obs - mu) / sigma' = ({obs:.1f} - {exp:.1f}) / {sig:.1f} = {zp:.2f}",
            "word": words[wi_e].text,
            "obs": obs,
            "exp": exp,
        }

    return _build_regions(flagged, words, z_vals, "trailing_off", MIN_REGION["trailing_off"], _ev)


def detect_muffled(
    words: list[Any],
    mu_tilt: list[float],
    mu_hnr: list[float],
    sigma_tilt: list[float],
    sigma_hnr: list[float],
    sp_tilt: float,
    sp_hnr: float,
) -> list[Region]:
    flagged: list[int] = []
    z_vals: dict[int, float] = {}
    n = min(len(words), len(mu_tilt), len(mu_hnr))

    for i in range(n):
        tilt = getattr(words[i], "tilt_rel", float("nan"))
        hnr = getattr(words[i], "hnr_rel", float("nan"))
        if math.isnan(tilt) or math.isnan(hnr):
            continue
        spt = _fs(sigma_tilt[i] if i < len(sigma_tilt) else 0.0, sp_tilt)
        sph = _fs(sigma_hnr[i] if i < len(sigma_hnr) else 0.0, sp_hnr)
        zt = _z(tilt, mu_tilt[i], spt)
        zh = _z(hnr, mu_hnr[i], sph)
        if zt <= -Z_TOL and zh <= -Z_TOL:
            zp = min(zt, zh)
            flagged.append(i)
            z_vals[i] = zp

    def _ev(run: list[int], zp: float) -> dict:
        t0, t1 = words[run[0]].start, words[run[-1]].end
        p_idx = min(run, key=lambda idx: z_vals.get(idx, 0.0))
        spt = _fs(sigma_tilt[p_idx] if p_idx < len(sigma_tilt) else 0.0, sp_tilt)
        sph = _fs(sigma_hnr[p_idx] if p_idx < len(sigma_hnr) else 0.0, sp_hnr)
        obs_hnr = round(getattr(words[p_idx], "hnr_rel", 0.0), 1)
        exp_hnr = round(mu_hnr[p_idx], 1)
        obs_tilt = round(getattr(words[p_idx], "tilt_rel", 0.0), 1)
        exp_tilt = round(mu_tilt[p_idx], 1)
        return {
            "metric": "muffled_hnr_tilt",
            "unit": "dB",
            "observed": obs_hnr,
            "expected": exp_hnr,
            "sigma": round(sph, 1),
            "z": round(zp, 2),
            "formula": f"z_hnr = {zh:.2f}, z_tilt = {zt:.2f}",
            "t0": round(t0, 2),
            "t1": round(t1, 2),
            "obs": obs_hnr,
            "exp": exp_hnr,
            "sig": round(sph, 1),
            "obs2": obs_tilt,
            "exp2": exp_tilt,
        }

    return _build_regions(flagged, words, z_vals, "muffled", MIN_REGION["muffled"], _ev)


def detect_word_swallowed(words: list[Any]) -> list[Region]:
    regions: list[Region] = []
    for i, w in enumerate(words):
        score = getattr(w, "align_score", 1.0)
        dur = getattr(w, "dur", 0.0)
        nsyl = getattr(w, "nsyl", 1)
        if score < SWALLOW_SCORE_MAX:
            s = float(np.clip((SWALLOW_SCORE_MAX - score) / SWALLOW_SCORE_MAX, 0.1, 1.0))
        elif dur < SWALLOW_DUR_MAX_S and nsyl >= 2:
            s = float(np.clip((SWALLOW_DUR_MAX_S - dur) / SWALLOW_DUR_MAX_S, 0.1, 1.0))
        else:
            continue
        regions.append(Region(
            family="word_swallowed",
            dimension="clarity",
            start=w.start,
            end=w.end,
            word_start=i,
            word_end=i,
            severity=s,
            z_peak=-1.0,
                evidence={
                    "metric": "align_score",
                    "unit": "score",
                    "observed": round(score, 3),
                    "expected": 1.0,
                    "sigma": 0.0,
                    "z": -1.0,
                    "formula": f"score = {score:.2f}, dur = {dur*1000:.0f} ms",
                    "word": w.text,
                    "t": round(w.start, 3),
                    "score": round(score, 2),
                    "ms": round(dur * 1000),
                },
            ))
    return regions


# ---------------------------------------------------------------------------
# Boundary refinement (DESIGN 9)
# ---------------------------------------------------------------------------

def refine_boundaries(
    regions: list[Region],
    ff: Any,
    words: list[Any],
) -> list[Region]:
    """Refine start/end for volume_low, monotone, muffled, clipping using frame features."""
    if ff is None or getattr(ff, "times", None) is None:
        return regions

    times = ff.times
    for r in regions:
        if r.family not in {"volume_low", "monotone", "muffled", "clipping"}:
            continue
        w_start = words[r.word_start]
        w_end = words[r.word_end]
        mask = (times >= w_start.start) & (times <= w_end.end)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue

        if r.family == "clipping" and getattr(ff, "clip_frac", None) is not None:
            active = ff.clip_frac[idx] >= CLIP_FRAC_MIN
            if np.any(active):
                act_idx = idx[active]
                r.start = float(times[act_idx[0]])
                r.end = float(times[act_idx[-1]])
                r.boundary_refined = True
        elif r.family == "volume_low" and getattr(ff, "en_rel", None) is not None:
            active = ff.en_rel[idx] <= -10.0
            if np.any(active):
                act_idx = idx[active]
                r.start = float(times[act_idx[0]])
                r.end = float(times[act_idx[-1]])
                r.boundary_refined = True
    return regions


# ---------------------------------------------------------------------------
# All detectors (Reference mode: DESIGN 9.1)
# ---------------------------------------------------------------------------

def detect_all(
    words: list[Any],
    pauses: list[Any],
    baseline: dict[str, Any],
    norms: dict[str, Any],
    ff: Any = None,
) -> list[Region]:
    if not words:
        return []
    bw = baseline.get("words", [])
    n = min(len(words), len(bw))

    def _mu(feat: str) -> list[float]:
        return [bw[i].get(f"mu_{feat}", 0.0) for i in range(n)]

    def _sig(feat: str) -> list[float]:
        return [bw[i].get(f"sigma_{feat}", 0.0) for i in range(n)]

    FEAT_NORM_MAP = {
        "share": "ln_share_sd",
        "pause_before": "pause_sd",
        "f0_mean_st": "f0_mean_sd",
        "f0_slope": "f0_slope_sd",
        "en_mean_rel": "en_mean_sd",
        "en_slope": "en_slope_sd",
        "prominence": "prominence_sd",
        "hnr_rel": "hnr_sd",
        "tilt_rel": "tilt_sd",
        "f0_sd_st": "sigma_win",
    }

    def _sp(feat: str) -> float:
        k = f"sigma_pool_{feat}"
        if k in norms:
            return float(norms[k])
        norm_key = FEAT_NORM_MAP.get(feat, f"{feat}_sd")
        if norm_key in norms:
            return float(norms[norm_key])
        return float(norms.get(f"{feat}_sd", 1.0))

    sigma_D = float(norms.get("sigma_D", 0.12))

    all_r: list[Region] = []
    base_art_rate = baseline.get("artic_rate")
    all_r += detect_pace(words[:n], _mu("share"), sigma_D, base_art_rate=base_art_rate)
    all_r += detect_pause_missing(words[:n], _mu("pause_before"), _sig("pause_before"), _sp("pause_before"))
    all_r += detect_pause_excess(words[:n], pauses, _mu("pause_before"), _sig("pause_before"), _sp("pause_before"))
    all_r += detect_hesitation_filled(pauses, words[:n])
    all_r += detect_monotone(words[:n], _mu("f0_sd_st"), _sig("f0_sd_st"), _sp("f0_sd_st"))
    all_r += detect_terminal_contour(words[:n], _mu("f0_slope"), _sig("f0_slope"), _sp("f0_slope"))
    all_r += detect_stress_missed(words[:n], _mu("prominence"), _sig("prominence"), _sp("prominence"))
    all_r += detect_volume_low(words[:n], _mu("en_mean_rel"), _sig("en_mean_rel"), _sp("en_mean_rel"))
    all_r += detect_clipping(words[:n])
    all_r += detect_trailing_off(words[:n], _mu("en_slope"), _sig("en_slope"), _sp("en_slope"))
    all_r += detect_muffled(
        words[:n],
        _mu("tilt_rel"),
        _mu("hnr_rel"),
        _sig("tilt_rel"),
        _sig("hnr_rel"),
        _sp("tilt_rel"),
        _sp("hnr_rel"),
    )
    all_r += detect_word_swallowed(words[:n])

    if ff is not None:
        all_r = refine_boundaries(all_r, ff, words[:n])

    all_r.sort(key=lambda r: (r.start, r.end))
    for idx, r in enumerate(all_r, start=1):
        r.id = f"reg_{idx}"
    return all_r


# ---------------------------------------------------------------------------
# Norm mode detection (DESIGN 9.2)
# ---------------------------------------------------------------------------

def detect_norm(
    words: list[Any],
    pauses: list[Any],
    norms: dict[str, Any],
    ff: Any = None,
) -> list[Region]:
    """Detect flaws without a reference speech using population norms (DESIGN 9.2)."""
    if not words:
        return []
    n = len(words)
    all_r: list[Region] = []

    # 1. pace_global
    art_mean = float(norms.get("artic_rate_mean", 4.2))
    art_sd = float(norms.get("artic_rate_sd", 0.6))
    user_art = sum(w.nsyl for w in words) / max(sum(w.dur for w in words), 1e-6)
    if user_art < art_mean - 2.0 * art_sd or user_art > art_mean + 2.0 * art_sd:
        g = math.log(user_art / max(art_mean, 1e-6))
        s = float(np.clip(abs(user_art - art_mean) / (3.0 * art_sd), 0.2, 1.0))
        all_r.append(Region(
            family="pace_global", dimension="pacing",
            start=words[0].start, end=words[-1].end,
            word_start=0, word_end=n - 1, severity=s, z_peak=g,
            evidence={
                "metric": "articulation_rate", "unit": "syl/s",
                "observed": round(user_art, 2), "expected": round(art_mean, 2),
                "sigma": round(art_sd, 2), "z": round((user_art - art_mean) / art_sd, 2),
                "obs": round(user_art, 2), "exp": round(art_mean, 2),
                "pct": round((user_art / art_mean - 1.0) * 100, 1),
            },
        ))

    # 2. pace_fast / pace_slow relative to own median window rate
    win = 5
    durs = [w.dur for w in words]
    syls = [w.nsyl for w in words]
    win_rates = []
    for i in range(n):
        lo, hi = max(0, i - win // 2), min(n, i + win // 2 + 1)
        sub_d = sum(durs[lo:hi])
        sub_s = sum(syls[lo:hi])
        win_rates.append(sub_s / sub_d if sub_d > 0 else art_mean)
    med_rate = float(np.median(win_rates))
    sigma_rate = float(norms.get("artic_rate_sd", 0.6))

    ff_idx, fs_idx = [], []
    zf, zs = {}, {}
    for i in range(n):
        z = (win_rates[i] - med_rate) / sigma_rate
        if z >= Z_TOL:
            ff_idx.append(i)
            zf[i] = -z  # negative z represents fast tempo in DESIGN
        elif z <= -Z_TOL:
            fs_idx.append(i)
            zs[i] = -z

    all_r += _build_regions(ff_idx, words, zf, "pace_fast", MIN_REGION["pace_fast"])
    all_r += _build_regions(fs_idx, words, zs, "pace_slow", MIN_REGION["pace_slow"])

    # 3. pause_missing: sentence-final punctuation followed by a pause under 0.12 s (expected >= 0.30 s)
    for i in range(n - 1):
        punct = getattr(words[i], "punct_after", "")
        if _SENTENCE_END_RE.search(punct):
            p_after = getattr(words[i], "pause_after", 0.0)
            if p_after < 0.12:
                all_r.append(Region(
                    family="pause_missing", dimension="pacing",
                    start=words[i + 1].start, end=words[i + 1].end,
                    word_start=i + 1, word_end=i + 1, severity=0.6, z_peak=-2.0,
                    evidence={
                        "word": words[i + 1].text, "t": round(words[i + 1].start, 3),
                        "obs_ms": round(p_after * 1000), "exp_ms": 300, "sig_ms": 100,
                    },
                ))

    # 4. pause_excess & hesitation_filled
    all_r += detect_pause_excess(words, pauses, [0.0] * n, [0.1] * n, float(norms.get("pause_sd", 0.2)))
    all_r += detect_hesitation_filled(pauses, words)

    # 5. monotone: phrase F0 SD below norms P10
    f0_p10 = float(norms.get("phrase_f0_sd_p10", 1.2))
    flagged_mono, z_mono = [], {}
    for i in range(n):
        f0_sd = getattr(words[i], "f0_sd_st", float("nan"))
        if not math.isnan(f0_sd) and f0_sd < f0_p10:
            flagged_mono.append(i)
            z_mono[i] = -Z_TOL
    all_r += _build_regions(flagged_mono, words, z_mono, "monotone", MIN_REGION["monotone"])

    # 6. terminal_contour: sentence-final slope above norms P90
    slope_p90 = float(norms.get("terminal_contour_p90", 2.0))
    for i in range(n):
        if _SENTENCE_END_RE.search(getattr(words[i], "punct_after", "")):
            slope = getattr(words[i], "f0_slope", float("nan"))
            if not math.isnan(slope) and slope > slope_p90:
                all_r.append(Region(
                    family="terminal_contour", dimension="intonation",
                    start=words[i].start, end=words[i].end,
                    word_start=i, word_end=i, severity=0.7, z_peak=Z_TOL,
                    evidence={"word": words[i].text, "t": round(words[i].start, 3), "obs": round(slope, 2), "exp": 0.0},
                ))

    # 7. volume_low, clipping, trailing_off, muffled, word_swallowed
    sp_en = float(norms.get("en_mean_sd", 3.0))
    all_r += detect_volume_low(words, [0.0] * n, [0.0] * n, sp_en)
    all_r += detect_clipping(words)
    all_r += detect_trailing_off(words, [0.0] * n, [0.0] * n, float(norms.get("en_slope_sd", 8.0)))
    sp_tilt = float(norms.get("tilt_sd", 4.0))
    sp_hnr = float(norms.get("hnr_sd", 3.0))
    all_r += detect_muffled(words, [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n, sp_tilt, sp_hnr)
    all_r += detect_word_swallowed(words)

    if ff is not None:
        all_r = refine_boundaries(all_r, ff, words)

    all_r.sort(key=lambda r: (r.start, r.end))
    for idx, r in enumerate(all_r, start=1):
        r.id = f"reg_{idx}"
    return all_r
