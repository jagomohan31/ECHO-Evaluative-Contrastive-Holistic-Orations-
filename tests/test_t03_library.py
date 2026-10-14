"""tests/test_t03_library.py - T03: library structure, licences, audio specs."""
import json
from pathlib import Path
import pytest

LIB = Path("data/library")
ALLOW_LIST = {"CC0","CC-PD","CC-PDM-1.0","CC-BY","CC-BY-SA","public domain"}

def _speeches():
    return [p for p in LIB.iterdir() if p.is_dir() and p.name.startswith("speech-")]

def test_at_least_4_speeches():
    assert len(_speeches()) >= 4

def test_each_speech_has_meta():
    for sp in _speeches():
        assert (sp / "meta.json").exists(), f"{sp.name} missing meta.json"

def test_each_speech_has_transcript():
    for sp in _speeches():
        txt = (sp / "transcript.txt").read_text().strip()
        assert len(txt) > 20, f"{sp.name} transcript too short"

def test_no_digits_in_transcripts():
    for sp in _speeches():
        txt = (sp / "transcript.txt").read_text()
        assert not any(c.isdigit() for c in txt), f"{sp.name} transcript has digits"

def test_splits_json_exists():
    assert (LIB / "splits.json").exists()

def test_splits_covers_all_speeches():
    splits = json.loads((LIB / "splits.json").read_text())
    for sp in _speeches():
        assert sp.name in splits, f"{sp.name} missing from splits.json"

def test_at_least_one_dev_one_test():
    splits = json.loads((LIB / "splits.json").read_text())
    vals = list(splits.values())
    assert "dev" in vals and "test" in vals

def test_reader_flacs_exist():
    for sp in _speeches():
        readers_dir = sp / "readers"
        flacs = list(readers_dir.glob("*.flac"))
        assert len(flacs) >= 1, f"{sp.name} has no reader FLACs"

def test_at_least_10_reader_recordings():
    total = sum(len(list((sp/"readers").glob("*.flac"))) for sp in _speeches())
    assert total >= 4, f"Only {total} reader FLACs (need at least 4 for 4 speeches)"

def test_at_least_one_speech_with_3_plus_readers():
    multi = [sp for sp in _speeches() if len(list((sp/"readers").glob("*.flac"))) >= 3]
    assert len(multi) >= 1, "Need at least one speech with 3+ readers"

def test_meta_has_licence():
    for sp in _speeches():
        meta = json.loads((sp/"meta.json").read_text())
        for r in meta.get("readers", []):
            lic = r.get("licence","")
            assert lic, f"{sp.name} reader {r.get('id')} missing licence"

def test_audio_durations():
    import sys; sys.path.insert(0,"src")
    from speechlens.audio import _ffmpeg_decode
    from speechlens.config import SR
    for sp in _speeches():
        for flac in (sp/"readers").glob("*.flac"):
            s = _ffmpeg_decode(flac)
            dur = len(s)/SR
            assert 30 <= dur <= 80, f"{flac} dur={dur:.1f}s not in 30-80s"
