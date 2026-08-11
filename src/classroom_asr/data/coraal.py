"""CORAAL transcript parsing + verbatim-preserving reference cleaning.

CORAAL (Corpus of Regional African American Language) ships tab-separated
transcripts, one utterance per line::

    Line   Spkr        StTime    Content                         EnTime
    1      INT_se0_ag2 0.0000    Okay so um (pause 0.34) yeah     3.4500

The transcription is careful and verbatim: fillers ("um", "uh"), repetitions,
and false starts are written out — exactly the property this project needs
(§1.2, §16.4). Our job when *scoring* is to strip the annotation markup (pauses,
non-linguistic events, redactions) while KEEPING the spoken words, then hand the
result to the standard scoring normalizer (§29).

CORAAL markup we remove / keep:
* ``(pause 0.34)`` / ``(pause)``            -> remove (non-speech)
* ``<laugh>``, ``<cough>``, ``<ts>`` ...    -> remove (non-linguistic / notes)
* ``/RD-NAME-2/``, ``/unintelligible/`` ... -> remove (redaction / can't score)
* ``[word]``                                -> keep ``word`` (uncertain best-guess)

This module is stdlib-only and unit-tested; audio slicing (which needs an audio
lib) stays in the notebook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_ANGLE = re.compile(r"<[^>]*>")          # <laugh>, <ts ...>
_PAREN = re.compile(r"\([^)]*\)")        # (pause 0.34), (unintelligible)
_SLASH = re.compile(r"/[^/]*/")          # /RD-NAME-2/, /inaudible/
_BRACK = re.compile(r"[\[\]]")           # keep inner text, drop the brackets
_WS = re.compile(r"\s+")


def clean_content(raw: str) -> str:
    """Strip CORAAL markup, preserving the verbatim spoken words."""
    t = raw
    t = _ANGLE.sub(" ", t)
    t = _PAREN.sub(" ", t)
    t = _SLASH.sub(" ", t)
    t = _BRACK.sub(" ", t)
    return _WS.sub(" ", t).strip()


@dataclass(frozen=True)
class CoraalSegment:
    line: int
    speaker: str
    start: float
    end: float
    raw: str
    text: str            # cleaned verbatim reference

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_interviewer(self) -> bool:
        # Interviewer speaker codes start with "int" in CORAAL.
        return self.speaker.lower().startswith("int")


def parse_transcript(path: str | Path) -> list[CoraalSegment]:
    """Parse a CORAAL ``*.txt`` transcript into cleaned segments."""
    rows: list[CoraalSegment] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        cols = {name.strip().lower(): i for i, name in enumerate(header)}
        # tolerate either exact header or positional fallback
        li = cols.get("line", 0)
        si = cols.get("spkr", 1)
        sti = cols.get("sttime", 2)
        ci = cols.get("content", 3)
        eni = cols.get("entime", 4)
        for lineno, line in enumerate(fh, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(ci, eni):
                continue
            try:
                start = float(parts[sti])
                end = float(parts[eni])
            except ValueError:
                continue
            raw = parts[ci]
            text = clean_content(raw)
            try:
                lnum = int(parts[li])
            except ValueError:
                lnum = lineno
            rows.append(CoraalSegment(lnum, parts[si].strip(), start, end, raw, text))
    return rows


def select_segments(
    segments: list[CoraalSegment],
    *,
    max_seconds: float = 3600.0,
    min_dur: float = 0.4,
    max_dur: float = 28.0,
    min_words: int = 1,
    include_interviewer: bool = True,
) -> list[CoraalSegment]:
    """Filter to scoreable speech segments up to a total-duration budget.

    Drops pure-annotation lines (empty after cleaning), too-short/too-long
    segments (Whisper handles <=30 s per call), and — optionally — interviewer
    turns. Accumulates in timeline order until ``max_seconds`` of audio.
    """
    out: list[CoraalSegment] = []
    total = 0.0
    for s in sorted(segments, key=lambda x: x.start):
        if not s.text or len(s.text.split()) < min_words:
            continue
        if not (min_dur <= s.duration <= max_dur):
            continue
        if s.is_interviewer and not include_interviewer:
            continue
        if total + s.duration > max_seconds:
            break
        out.append(s)
        total += s.duration
    return out


def iter_transcripts(text_dir: str | Path) -> Iterator[Path]:
    """Yield CORAAL transcript ``.txt`` files (skips metadata sidecars)."""
    for p in sorted(Path(text_dir).rglob("*.txt")):
        name = p.name.lower()
        if "metadata" in name or name.endswith("_readme.txt"):
            continue
        yield p
