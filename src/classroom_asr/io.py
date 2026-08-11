"""Lesson-package on-disk layout and (de)serialization (§22.1).

The canonical layout from §22.1::

    lesson_YYYYMMDD_id/
      audio/
        teacher_raw.flac
        student_raw.flac | student_zoom.flac
      timeline.parquet
      candidates.parquet
      phones.parquet
      session_lexicon.json
      transcript_gold_or_final.jsonl
      corrections.jsonl
      metadata.json

Parquet is optional (needs pyarrow/pandas). To keep the core dependency-free we
persist the same records as JSONL by default and use parquet only when the
``data`` extra is installed. Immutable evidence and revisable interpretation are
written to different files so a re-selection pass never rewrites provenance
(§22.3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .datamodel import Correction, LessonPackage, Span
from .lexicon import Lexicon


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class LessonStore:
    """Reads/writes a lesson package directory (§22.1)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    def save(self, pkg: LessonPackage) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        # Candidate graph (immutable evidence) — one JSON object per span.
        write_jsonl(self.root / "candidates.jsonl", (s.to_dict() for s in pkg.sorted_spans()))
        # Compact timeline index for quick range queries.
        write_jsonl(
            self.root / "timeline.jsonl",
            (
                {
                    "span_id": s.span_id,
                    "speaker": s.speaker.value,
                    "audio_source": s.audio_source.value,
                    "start": s.interval.start,
                    "end": s.interval.end,
                    "overlap_span_ids": s.overlap_span_ids,
                    "flags": sorted(f.value for f in s.flags),
                }
                for s in pkg.sorted_spans()
            ),
        )
        # Final/gold transcript (revisable interpretation), verbatim per §1.2.
        write_jsonl(
            self.root / "transcript_gold_or_final.jsonl",
            (
                {
                    "span_id": s.span_id,
                    "speaker": s.speaker.value,
                    "start": s.interval.start,
                    "end": s.interval.end,
                    "text": s.resolved_text(),
                    "lexical_target": s.lexical_target,
                    "realized_phones": s.realized_phones,
                }
                for s in pkg.sorted_spans()
                if s.resolved_text() is not None
            ),
        )
        write_jsonl(self.root / "corrections.jsonl", (c.to_dict() for c in pkg.corrections))
        (self.root / "metadata.json").write_text(
            json.dumps({"lesson_id": pkg.lesson_id, **pkg.metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_lexicon(self, lex: Lexicon) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        lex.save(self.root / "session_lexicon.json")

    def load_lexicon(self) -> Lexicon | None:
        p = self.root / "session_lexicon.json"
        return Lexicon.load(p) if p.exists() else None

    def load(self) -> LessonPackage:
        meta_path = self.root / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        lesson_id = meta.pop("lesson_id", self.root.name)
        spans = [Span.from_dict(d) for d in read_jsonl(self.root / "candidates.jsonl")]
        corrections = []
        cpath = self.root / "corrections.jsonl"
        if cpath.exists():
            corrections = [Correction.from_dict(d) for d in read_jsonl(cpath)]
        return LessonPackage(lesson_id=lesson_id, spans=spans,
                             corrections=corrections, metadata=meta)
