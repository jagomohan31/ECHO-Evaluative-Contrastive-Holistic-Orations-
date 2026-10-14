"""tests/test_t04_alignments.py - T04: every reader has a valid align.json."""
import json
from pathlib import Path
import pytest

LIB = Path("data/library")

def _all_align_files():
    return list(LIB.rglob("*.align.json"))

def test_every_reader_has_align():
    for sp in LIB.iterdir():
        if not sp.is_dir(): continue
        for flac in (sp/"readers").glob("*.flac"):
            assert flac.with_suffix(".align.json").exists(), f"{flac} missing .align.json"

def test_align_json_valid():
    for af in _all_align_files():
        d = json.loads(af.read_text())
        assert "words" in d
        assert "pauses" in d
        assert d["duration_s"] > 0
        assert len(d["words"]) > 0

def test_words_in_order():
    for af in _all_align_files():
        words = json.loads(af.read_text())["words"]
        for i in range(len(words)-1):
            assert words[i]["end"] <= words[i+1]["start"] + 0.05, f"{af}: word order violated at {i}"

def test_no_zero_duration_words():
    for af in _all_align_files():
        for w in json.loads(af.read_text())["words"]:
            assert w["end"] > w["start"], f"{af}: zero duration word {w['text']}"

def test_words_within_audio():
    for af in _all_align_files():
        d = json.loads(af.read_text())
        dur = d["duration_s"]
        for w in d["words"]:
            assert w["end"] <= dur + 0.1, f"{af}: word end {w['end']} > duration {dur}"
