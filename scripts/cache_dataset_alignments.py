"""
cache_dataset_alignments.py
Populate .cache/align for all synthetic files using analytic remap.
This enables instant (<5s) pipeline analysis during evaluation.
"""
import json
from pathlib import Path
from speechlens import audio
from speechlens.config import SR
from speechlens.align.ctc import _cache_key, _save_cache, _load_cache, tokenise
from speechlens.dataset.remap import remap_words_and_pauses

def main():
    speeches = ["speech-001", "speech-002", "speech-003", "speech-004"]
    total = 0
    cached = 0

    for sid in speeches:
        labels_file = Path(f"data/dataset/{sid}/labels.json")
        if not labels_file.exists():
            continue
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        
        meta = json.loads(Path(f"data/library/{sid}/meta.json").read_text(encoding="utf-8"))
        ref_id = meta.get("mirror_source") or (meta.get("readers")[0]["id"] if meta.get("readers") else "r1")
        ref_align_p = Path(f"data/library/{sid}/readers/{ref_id}.align.json")
        if not ref_align_p.exists():
            continue
        ref_align = json.loads(ref_align_p.read_text(encoding="utf-8"))
        
        trans = Path(f"data/library/{sid}/transcript.txt").read_text(encoding="utf-8").strip()
        toks = tokenise(trans)
        norm_trans = " ".join(t.text_norm for t in toks if t.text_norm)
        
        for entry in labels:
            flac_path = Path(entry["file"])
            if not flac_path.exists():
                continue
            total += 1
            samples = audio.load(flac_path)
            k = _cache_key(samples, norm_trans)
            
            if _load_cache(k) is not None:
                cached += 1
                continue
                
            flaws = entry.get("flaws", [])
            words, pauses = remap_words_and_pauses(ref_align.get("words", []), ref_align.get("pauses", []), flaws)
            pause_dicts = [
                {"start": p.start, "end": p.end, "before_word": p.before_word, "after_word": p.after_word}
                for p in pauses
            ]
            _save_cache(k, {
                "model": "facebook/wav2vec2-base-960h",
                "stride_s": 0.02,
                "duration_s": len(samples) / SR,
                "words": words,
                "pauses": pause_dicts,
            })
            cached += 1
            if cached % 20 == 0:
                print(f"Cached {cached}/{total} alignments...")

    print(f"Done! All {cached}/{total} dataset alignments cached in .cache/align.")

if __name__ == "__main__":
    main()
