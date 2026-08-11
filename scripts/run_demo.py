"""End-to-end demo: run the stub pipeline on the synthetic lesson.

    python scripts/run_demo.py

Runs with zero ML dependencies. Shows the verbatim transcript and the two
headline numbers from the design doc (§18.2 / §28): the final WER and the
candidate-oracle WER gate that says whether the correct answer is already in the
candidate pool.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classroom_asr.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["demo"]))
