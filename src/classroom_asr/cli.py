"""Small CLI entry point (`classroom-asr`).

Subcommands:
  demo   run the stub pipeline on the synthetic lesson and print the transcript
         + headline metrics (final WER, candidate-oracle WER gate).
"""

from __future__ import annotations

import argparse
import json
import sys

from .evaluate import evaluate_run
from .pipeline.orchestrator import Orchestrator, assemble_transcript
from .synthetic import example_lesson


def _cmd_demo(args: argparse.Namespace) -> int:
    lesson = example_lesson()
    result = Orchestrator().run(lesson)
    transcript = assemble_transcript(result.package)
    ev = evaluate_run(result.package)

    if args.json:
        print(json.dumps(
            {
                "transcript": transcript,
                "final_wer": ev.final.wer,
                "baseline_wer": ev.oracle.baseline.wer,
                "oracle_wer": ev.oracle.oracle.wer,
                "oracle_headroom": ev.oracle.headroom,
                "spans_expanded": result.n_expanded,
                "spans_frozen": result.n_frozen,
            },
            ensure_ascii=False, indent=2,
        ))
        return 0

    print("Verbatim transcript (timeline order):")
    for row in transcript:
        print(f"  [{row['start']:6.2f}-{row['end']:6.2f}] {row['speaker']:7s} {row['text']}")
    print()
    print(f"Spans: {ev.n_spans}  expanded(selector)={result.n_expanded}  "
          f"frozen(1-best)={result.n_frozen}")
    print(f"Session lexicon terms: {len(result.session_lexicon)}")
    print()
    print(f"Final verbatim WER : {ev.final.wer:6.3f}  "
          f"(sub={ev.final.substitutions} del={ev.final.deletions} ins={ev.final.insertions})")
    print(f"Baseline 1-best WER: {ev.oracle.baseline.wer:6.3f}")
    print(f"Candidate-oracle WER: {ev.oracle.oracle.wer:6.3f}  "
          f"(headroom {ev.oracle.headroom:+.3f})  [§18.2 gate]")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Transcripts carry IPA and Cyrillic; force UTF-8 so a legacy Windows code
    # page (cp1251) doesn't crash on output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="classroom-asr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the stub pipeline on synthetic data")
    demo.add_argument("--json", action="store_true", help="emit JSON instead of text")
    demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
