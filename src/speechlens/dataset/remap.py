"""
dataset/remap.py -- Analytic time remap for synthetic flaws (DESIGN 5.7).

Maps original reader word and pause alignments to the time-domain coordinates
of synthetically flawed recordings with zero CTC alignment error.
"""
from __future__ import annotations
import copy
from typing import Any
from speechlens.features.word import PauseInfo


def remap_words_and_pauses(
    base_words: list[dict[str, Any]],
    base_pauses: list[dict[str, Any]],
    flaws: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[PauseInfo]]:
    """
    Analytically remap words and pauses based on synthetic flaw parameters.
    """
    words = [copy.deepcopy(w) for w in base_words]
    # Convert base_pauses to list of PauseInfo-like objects
    pauses: list[PauseInfo] = []
    for p in base_pauses:
        pauses.append(PauseInfo(
            start=p["start"],
            end=p["end"],
            before_word=p.get("before_word", 0),
            after_word=p.get("after_word", 1),
            kind=p.get("kind", "silent"),
        ))

    if not flaws:
        return words, pauses

    # Sort flaws by affected start time so shifts apply in temporal order
    for flaw in flaws:
        fam = flaw["family"]
        params = flaw.get("params", {})
        wi_s, wi_e = flaw["word_idx"]

        if fam in ("pace_fast", "pace_slow"):
            factor = params.get("factor", 1.0)
            orig_ts = base_words[wi_s]["start"]
            orig_te = base_words[wi_e]["end"]
            orig_span_dur = orig_te - orig_ts
            new_span_dur = orig_span_dur / factor
            shift = new_span_dur - orig_span_dur

            for w in words:
                idx = w["i"]
                if idx < wi_s:
                    continue
                elif wi_s <= idx <= wi_e:
                    # Scale inside span
                    w_orig_s = base_words[idx]["start"]
                    w_orig_e = base_words[idx]["end"]
                    rel_s = (w_orig_s - orig_ts) / max(orig_span_dur, 1e-6)
                    rel_e = (w_orig_e - orig_ts) / max(orig_span_dur, 1e-6)
                    w["start"] = orig_ts + rel_s * new_span_dur
                    w["end"] = orig_ts + rel_e * new_span_dur
                else:
                    # Shift subsequent words
                    w["start"] += shift
                    w["end"] += shift

            for p in pauses:
                if p.start >= orig_te:
                    p.start += shift
                    p.end += shift

        elif fam == "pause_missing":
            ratio = params.get("ratio")
            target_s = params.get("target_s")
            # The shortened pause occurred right before wi_s or after wi_s
            # Find the pause that was shortened
            for p in pauses:
                if p.after_word == wi_s or (wi_s <= p.after_word <= wi_e):
                    orig_dur = p.end - p.start
                    tgt = orig_dur * ratio if ratio is not None else target_s
                    tgt = max(0.060, tgt if tgt is not None else 0.060)
                    shift = tgt - orig_dur
                    p.end = p.start + tgt
                    # Shift subsequent words and pauses
                    for w in words:
                        if w["i"] >= p.after_word:
                            w["start"] += shift
                            w["end"] += shift
                    for p2 in pauses:
                        if p2.start > p.start:
                            p2.start += shift
                            p2.end += shift
                    break

        elif fam == "pause_excess":
            dur_s = params.get("dur_s", 0.45)
            # Find boundary word
            split_idx = wi_s
            if split_idx < len(words):
                insert_time = words[split_idx]["end"]
                # Insert pause
                new_p = PauseInfo(
                    start=insert_time,
                    end=insert_time + dur_s,
                    before_word=split_idx,
                    after_word=split_idx + 1,
                    kind="silent",
                )
                pauses.append(new_p)
                for w in words:
                    if w["i"] > split_idx:
                        w["start"] += dur_s
                        w["end"] += dur_s
                for p in pauses:
                    if p is not new_p and p.start >= insert_time:
                        p.start += dur_s
                        p.end += dur_s

        elif fam == "hesitation_filled":
            dur_s = params.get("dur_s", 0.35)
            split_idx = wi_s
            if split_idx < len(words):
                insert_time = words[split_idx]["end"]
                new_p = PauseInfo(
                    start=insert_time,
                    end=insert_time + dur_s,
                    before_word=split_idx,
                    after_word=split_idx + 1,
                    kind="filled",
                )
                pauses.append(new_p)
                for w in words:
                    if w["i"] > split_idx:
                        w["start"] += dur_s
                        w["end"] += dur_s
                for p in pauses:
                    if p is not new_p and p.start >= insert_time:
                        p.start += dur_s
                        p.end += dur_s

    pauses.sort(key=lambda p: p.start)
    return words, pauses
