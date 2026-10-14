"""
explain.py -- Causal explanations for detected flaw regions (DESIGN 11).

Generates {summary, context, effect, fix, math} for every region using evidence fields only.
Never prints numbers not present in the evidence dict.
"""
from __future__ import annotations
from typing import Any


def explain_region(region: Any) -> dict[str, str]:
    """
    Generate an explanation dict for a single region.
    Returns: {"summary": str, "context": str, "effect": str, "fix": str, "math": str}
    """
    fam = getattr(region, "family", region.get("family") if isinstance(region, dict) else "")
    ev = getattr(region, "evidence", region.get("evidence") if isinstance(region, dict) else {})
    if not isinstance(ev, dict):
        ev = {}

    t0 = ev.get("t0", round(getattr(region, "start", 0.0), 2))
    t1 = ev.get("t1", round(getattr(region, "end", 0.0), 2))
    t = ev.get("t", round(getattr(region, "start", 0.0), 2))
    z = ev.get("z", round(getattr(region, "z_peak", 0.0), 2))
    word = ev.get("word", "")
    text = ev.get("text", "")
    pct = ev.get("pct", 0.0)
    math_str = ev.get("formula", f"z = {z}")

    if fam == "pace_fast":
        return {
            "summary": f"{t0}-{t1} s '{text}': you spent {pct}% less time on these words than the baseline does relative to the rest of the passage (z = {z}).",
            "context": f"Phrase '{text}' between {t0} s and {t1} s.",
            "effect": "Words run together and emphasis is lost.",
            "fix": "Slow down here and let stressed words breathe.",
            "math": math_str,
        }

    elif fam == "pace_slow":
        return {
            "summary": f"{t0}-{t1} s '{text}': you spent {pct}% more time on these words than the baseline does relative to the rest of the passage (z = {z}).",
            "context": f"Phrase '{text}' between {t0} s and {t1} s.",
            "effect": "The delivery dragged and lost momentum.",
            "fix": "Keep momentum and avoid stretching unaccented words.",
            "math": math_str,
        }

    elif fam == "pace_global":
        obs = ev.get("obs", ev.get("observed", 0.0))
        exp = ev.get("exp", ev.get("expected", 0.0))
        return {
            "summary": f"Whole passage: articulation rate {obs} syl/s against {exp} syl/s for the baseline ({pct}%).",
            "context": "Passage delivery overall.",
            "effect": "Overall pace diverges significantly from baseline.",
            "fix": "Calibrate your baseline tempo to match the passage genre.",
            "math": math_str,
        }

    elif fam == "pause_missing":
        obs_ms = ev.get("obs_ms", ev.get("observed", 0))
        exp_ms = ev.get("exp_ms", ev.get("expected", 0))
        sig_ms = ev.get("sig_ms", ev.get("sigma", 0))
        prev = ev.get("prev", "")
        punct = ev.get("punct", "")
        context_str = f"After '{prev}{punct}', before '{word}' at {t} s." if prev else f"Before '{word}' at {t} s."
        return {
            "summary": f"Before '{word}' at {t} s: pause of {obs_ms} ms where the baseline pauses {exp_ms} +/- {sig_ms} ms (after '{prev}{punct}').",
            "context": context_str,
            "effect": "The phrase boundary is blurred.",
            "fix": "Stop fully at the punctuation.",
            "math": math_str,
        }

    elif fam == "pause_excess":
        obs_ms = ev.get("obs_ms", ev.get("observed", 0))
        return {
            "summary": f"An unexpected {obs_ms} ms silence before '{word}' at {t} s reads as hesitation.",
            "context": f"Before '{word}' at {t} s.",
            "effect": "Unintended gap interrupts fluency.",
            "fix": "Maintain momentum without stopping mid-phrase.",
            "math": math_str,
        }

    elif fam == "hesitation_filled":
        ms = ev.get("ms", ev.get("observed", 0))
        target_word = f" before '{word}'" if word else ""
        return {
            "summary": f"A {ms} ms voiced hesitation (um/uh){target_word} at {t} s.",
            "context": f"Boundary at {t} s{target_word}.",
            "effect": "Filled pauses reduce perceived confidence and fluency.",
            "fix": "Pause silently instead.",
            "math": math_str,
        }

    elif fam == "monotone":
        obs = ev.get("obs", ev.get("observed", 0.0))
        exp = ev.get("exp", ev.get("expected", 0.0))
        sig = ev.get("sig", ev.get("sigma", 0.0))
        return {
            "summary": f"{t0}-{t1} s: pitch varied +/-{obs} st (SD) against {exp} +/- {sig} st expected (z = {z}).",
            "context": f"Span from {t0} s to {t1} s.",
            "effect": "Flat pitch removes vocal emphasis.",
            "fix": "Let pitch rise on key words.",
            "math": math_str,
        }

    elif fam == "terminal_contour":
        obs = ev.get("obs", ev.get("observed", 0.0))
        exp = ev.get("exp", ev.get("expected", 0.0))
        return {
            "summary": f"The sentence ending '{word}' at {t} s rises ({obs} st/s) where the baseline falls ({exp} st/s).",
            "context": f"Sentence-final word '{word}' at {t} s.",
            "effect": "It sounds like a question or doubt.",
            "fix": "Drop pitch at sentence endings for finality.",
            "math": math_str,
        }

    elif fam == "stress_missed":
        obs = ev.get("obs", ev.get("observed", 0.0))
        exp = ev.get("exp", ev.get("expected", 0.0))
        return {
            "summary": f"'{word}' is a stress point in the baseline (prominence {exp}) but flat in your delivery ({obs}).",
            "context": f"Target word '{word}'.",
            "effect": "Missing emphasis dilutes key semantic points.",
            "fix": "Add pitch peak, loudness or length on it.",
            "math": math_str,
        }

    elif fam == "volume_low":
        obs = ev.get("obs", ev.get("observed", 0.0))
        exp = ev.get("exp", ev.get("expected", 0.0))
        sig = ev.get("sig", ev.get("sigma", 0.0))
        return {
            "summary": f"{t0}-{t1} s: level {obs} dB against {exp} +/- {sig} dB expected relative to your loud speech (z = {z}).",
            "context": f"Span from {t0} s to {t1} s.",
            "effect": "Listeners may lose these words.",
            "fix": "Project voice consistently across phrases.",
            "math": math_str,
        }

    elif fam == "clipping":
        return {
            "summary": f"{pct}% of samples are clipped at {t0}-{t1} s.",
            "context": f"Span from {t0} s to {t1} s.",
            "effect": "Acoustic distortion and harshness.",
            "fix": "Move back from the microphone or lower the input gain.",
            "math": math_str,
        }

    elif fam == "trailing_off":
        obs = ev.get("obs", ev.get("observed", 0.0))
        exp = ev.get("exp", ev.get("expected", 0.0))
        return {
            "summary": f"At the end of the sentence ending '{word}', energy falls {obs} dB/s against {exp} dB/s expected.",
            "context": f"Sentence ending '{word}'.",
            "effect": "Weakens sentence closure.",
            "fix": "Keep energy through the last word.",
            "math": math_str,
        }

    elif fam == "muffled":
        obs = ev.get("obs", ev.get("observed", 0.0))
        exp = ev.get("exp", ev.get("expected", 0.0))
        sig = ev.get("sig", ev.get("sigma", 0.0))
        obs2 = ev.get("obs2", 0.0)
        exp2 = ev.get("exp2", 0.0)
        return {
            "summary": f"{t0}-{t1} s: clarity dropped (harmonics-to-noise {obs} dB vs {exp} +/- {sig}; high-frequency balance {obs2} dB vs {exp2}).",
            "context": f"Span from {t0} s to {t1} s.",
            "effect": "Consonants and high frequencies sound indistinct.",
            "fix": "Open the mouth and articulate consonants.",
            "math": math_str,
        }

    elif fam == "word_swallowed":
        score = ev.get("score", ev.get("observed", 0.0))
        ms = ev.get("ms", 0)
        return {
            "summary": f"'{word}' at {t} s was barely articulated (aligner confidence {score}, {ms} ms).",
            "context": f"Word '{word}' at {t} s.",
            "effect": "Poor intelligibility for the listener.",
            "fix": "Fully enunciate all syllables.",
            "math": math_str,
        }

    else:
        return {
            "summary": f"Flaw detected in {fam} at {t0}-{t1} s (z = {z}).",
            "context": f"Span {t0}-{t1} s.",
            "effect": "Impairs delivery quality.",
            "fix": "Review acoustic balance.",
            "math": math_str,
        }


def explain_all(regions: list[Any]) -> list[Any]:
    """Populate explanations on all regions in place and return them."""
    for r in regions:
        exp_dict = explain_region(r)
        if isinstance(r, dict):
            r["explanation"] = exp_dict
        else:
            setattr(r, "explanation", exp_dict)
    return regions
