"""
speechlens/cli.py
CLI entry-point.  Every command is a thin wrapper over the library.

Usage:
    python -m speechlens <command> [options]

Commands:
    fetch           Download and convert source recordings.
    align           Force-align a recording against its transcript.
    dataset         Build, validate or release the contrastive dataset.
    baseline        Build baseline models from the library.
    calibrate       Grid-search thresholds on the dev split.
    analyze         Analyze a recording and print/save JSON.
    evaluate        Run the full evaluation suite.
    warmup          Pre-download and cache model weights.
    serve           Start the FastAPI server.
    reproduce       Run evaluate --quick and compare with committed metrics.
"""
from __future__ import annotations

import argparse
import sys


def _cmd_fetch(args: argparse.Namespace) -> int:
    from speechlens.dataset import build as _build
    print("fetch: reading data/sources.yaml and downloading recordings...")
    try:
        from speechlens import fetcher
        fetcher.run()
    except ImportError:
        print("fetcher not yet implemented.")
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    from speechlens.align.ctc import align_file
    result = align_file(
        audio_path=args.audio,
        transcript=args.transcript,
        speech_id=getattr(args, "speech", None),
    )
    import json, sys
    print(json.dumps(result, indent=2))
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    sub = args.subcommand
    if sub == "build":
        from speechlens.dataset.build import main as _main
        _main()
    elif sub == "validate":
        from speechlens.dataset.validate import main as _main
        _main()
    elif sub == "release":
        from speechlens.dataset.release import main as _main
        _main()
    else:
        print(f"Unknown dataset subcommand: {sub}", file=sys.stderr)
        return 1
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    from speechlens.baseline import build as _build
    _build()
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from speechlens.evalsuite.calibrate import run as _run
    _run()
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    import json
    from speechlens.pipeline import analyze
    result = analyze(
        audio_path=args.audio,
        speech_id=getattr(args, "speech", None),
        transcript_path=getattr(args, "transcript", None),
    )
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        import pathlib
        pathlib.Path(args.out).write_text(output, encoding="utf-8")
        print(f"Saved to {args.out}")
    else:
        print(output)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from speechlens.evalsuite.evaluate import run as _run
    _run(quick=args.quick)
    return 0


def _cmd_warmup(args: argparse.Namespace) -> int:
    print("Warming up: downloading model weights...")
    from speechlens.align.ctc import load_model
    load_model()
    print("Model ready.")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from api.main import app
    import uvicorn
    print(f"Starting SpeechLens API server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    from speechlens.evalsuite.evaluate import run as _run
    _run(quick=True, compare_committed=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speechlens",
        description="SpeechLens — contrastive speech analytics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch
    sub.add_parser("fetch", help="Download source recordings.")

    # align
    p_align = sub.add_parser("align", help="Force-align a recording.")
    p_align.add_argument("audio", help="Path to audio file.")
    p_align.add_argument("--transcript", help="Path to transcript .txt.")
    p_align.add_argument("--speech", help="Speech ID for library lookup.")

    # dataset
    p_ds = sub.add_parser("dataset", help="Build/validate/release the dataset.")
    p_ds.add_argument("subcommand", choices=["build", "validate", "release"])

    # baseline
    sub.add_parser("baseline", help="Build baseline models.")

    # calibrate
    sub.add_parser("calibrate", help="Grid-search thresholds (dev split only).")

    # analyze
    p_an = sub.add_parser("analyze", help="Analyze a recording.")
    p_an.add_argument("audio", help="Path to audio file.")
    p_an.add_argument("--speech", help="Speech ID (reference mode).")
    p_an.add_argument("--transcript", help="Path to transcript .txt (norm mode).")
    p_an.add_argument("--out", help="Write JSON to this path.")

    # evaluate
    p_ev = sub.add_parser("evaluate", help="Run the evaluation suite.")
    p_ev.add_argument("--quick", action="store_true", help="Quick subset (CI).")

    # warmup
    sub.add_parser("warmup", help="Pre-cache model weights.")

    # serve
    p_sv = sub.add_parser("serve", help="Start the API server.")
    p_sv.add_argument("--host", default="0.0.0.0")
    p_sv.add_argument("--port", type=int, default=7860)

    # reproduce
    sub.add_parser("reproduce", help="Reproduce committed metrics (quick).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _dispatch = {
        "fetch":      _cmd_fetch,
        "align":      _cmd_align,
        "dataset":    _cmd_dataset,
        "baseline":   _cmd_baseline,
        "calibrate":  _cmd_calibrate,
        "analyze":    _cmd_analyze,
        "evaluate":   _cmd_evaluate,
        "warmup":     _cmd_warmup,
        "serve":      _cmd_serve,
        "reproduce":  _cmd_reproduce,
    }
    fn = _dispatch.get(args.command)
    if fn is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)
    sys.exit(fn(args) or 0)


if __name__ == "__main__":
    main()