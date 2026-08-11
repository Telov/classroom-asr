"""Immutable evidence data model and the per-span candidate graph (§1.2, §22).

Design invariant (§22.3): raw audio, phone observations, model hypotheses, and
user corrections are **immutable provenance**. A later whole-lesson pass may
change the *selected orthographic candidate* but must never erase the evidence.
We encode this by making every evidence object a frozen dataclass, while the
revisable interpretation lives in the mutable :class:`Span.selection` and the
derived canonical fields.

The canonical transcript is strictly **verbatim** (§1.2): ``canonical_text``
records what was spoken. ``lexical_target`` (intended word) and
``realized_phones`` (actual pronunciation) are separate, non-overwriting fields
(Appendix A.2: do not replace realized /friː/ with /θriː/).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .timeline import Interval
from .types import AudioSource, CandidateSource, Language, Role, SpanFlag


# --------------------------------------------------------------------------- #
# Immutable evidence                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TextCandidate:
    """An orthographic hypothesis from an ASR branch (§12.1–12.2).

    ``score`` is the branch's own (uncalibrated) score; calibration onto a
    common scale happens later (§13). ``no_lexical_bias`` marks the
    acoustic-faithfulness diagnostic candidate (§12.1).
    """

    id: str
    text: str
    source: CandidateSource
    score: float = 0.0
    beam_rank: int | None = None
    language: Language | None = None
    no_lexical_bias: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {"id": self.id, "text": self.text, "source": self.source.value, "score": self.score}
        if self.beam_rank is not None:
            d["beam_rank"] = self.beam_rank
        if self.language is not None:
            d["language"] = self.language.value
        if self.no_lexical_bias:
            d["no_lexical_bias"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextCandidate":
        return cls(
            id=d["id"],
            text=d["text"],
            source=CandidateSource(d["source"]),
            score=float(d.get("score", 0.0)),
            beam_rank=d.get("beam_rank"),
            language=Language(d["language"]) if d.get("language") else None,
            no_lexical_bias=bool(d.get("no_lexical_bias", False)),
        )


@dataclass(frozen=True)
class PhonePath:
    """A phone hypothesis with posterior mass (§10.2).

    We keep the top-K paths and posteriors — never collapse to one IPA string,
    because that loses exactly the uncertainty robust P2G needs (§10.2).
    ``hidden_ref`` optionally points to cached encoder states on disk (§21.2).
    """

    id: str
    ipa: str
    prob: float
    hidden_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"id": self.id, "ipa": self.ipa, "p": self.prob}
        if self.hidden_ref:
            d["hidden_ref"] = self.hidden_ref
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhonePath":
        return cls(id=d["id"], ipa=d["ipa"], prob=float(d.get("p", 0.0)),
                   hidden_ref=d.get("hidden_ref"))


@dataclass(frozen=True)
class P2GCandidate:
    """A phone->grapheme spelling with its probability (§10.3).

    ``script`` follows the project's transcription policy (Latin/Cyrillic).
    For a genuinely novel pronunciation the spelling is not uniquely recoverable
    (§10.3/§10.5), so ``spelling_confidence`` is reported separately from ``prob``.
    """

    id: str
    text: str
    prob: float
    script: str | None = None          # "latin" | "cyrillic"
    ipa: str | None = None
    spelling_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "text": self.text, "p": self.prob}
        for k in ("script", "ipa"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.spelling_confidence is not None:
            d["spelling_confidence"] = self.spelling_confidence
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "P2GCandidate":
        return cls(id=d["id"], text=d["text"], prob=float(d.get("p", 0.0)),
                   script=d.get("script"), ipa=d.get("ipa"),
                   spelling_confidence=d.get("spelling_confidence"))


@dataclass(frozen=True)
class RagMatch:
    """A phonetic-RAG hit against the session/persistent lexicon (§10.4)."""

    term: str
    similarity: float
    canonical_spelling: str | None = None
    persistent: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"term": self.term, "similarity": self.similarity}
        if self.canonical_spelling and self.canonical_spelling != self.term:
            d["canonical_spelling"] = self.canonical_spelling
        if self.persistent:
            d["persistent"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RagMatch":
        return cls(term=d["term"], similarity=float(d["similarity"]),
                   canonical_spelling=d.get("canonical_spelling"),
                   persistent=bool(d.get("persistent", False)))


@dataclass(frozen=True)
class Confidence:
    """Calibrated uncertainty for a span (§13.1).

    Sources are not directly comparable raw (§13); these fields are expected to
    be on a common calibrated scale after §13 estimators run.
    """

    word: float | None = None
    span: float | None = None
    phone_path_probability: float | None = None
    phonetic_rag_similarity: float | None = None
    p2g_candidate_probability: float | None = None
    source_reliability: str | None = None       # "raw" | "zoom"
    model_disagreement_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Confidence":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Revisable interpretation                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Selection:
    """The selector's constrained choice over a span's candidates (§14.5).

    Mirrors the structured selector output: it picks candidate *IDs*, it does
    not compose text. ``needs_novel_candidate``/``novel_text`` are gated
    (§14.5): free-form ``NEW`` output is disabled initially and only allowed
    when every orthographic candidate is inadequate but phone/P2G evidence
    strongly supports a novel form.
    """

    orthographic_candidate_id: str | None = None
    phone_candidate_id: str | None = None
    needs_novel_candidate: bool = False
    novel_text: str | None = None
    confidence: float | None = None
    selector: str | None = None                # model/impl that made the choice

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v is not False}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Selection":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Span = the candidate graph record (§22.2)                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Span:
    """One timeline span with all evidence and its (revisable) interpretation.

    Evidence collections hold frozen objects (immutable provenance). Mutating a
    span means adding candidates or setting :attr:`selection`, never editing an
    existing candidate in place.
    """

    span_id: int
    speaker: Role
    audio_source: AudioSource
    interval: Interval

    # --- immutable evidence ------------------------------------------------ #
    qwen: list[TextCandidate] = field(default_factory=list)
    gigaam: list[TextCandidate] = field(default_factory=list)
    extra: list[TextCandidate] = field(default_factory=list)   # generic extra branches
    phones: list[PhonePath] = field(default_factory=list)
    p2g: list[P2GCandidate] = field(default_factory=list)
    rag: list[RagMatch] = field(default_factory=list)
    mbr: list[TextCandidate] = field(default_factory=list)
    confidence: Confidence = field(default_factory=Confidence)
    flags: set[SpanFlag] = field(default_factory=set)
    overlap_span_ids: list[int] = field(default_factory=list)   # §5.5

    # --- revisable interpretation ----------------------------------------- #
    selection: Selection | None = None
    canonical_text: str | None = None       # verbatim, what was spoken (§1.2)
    lexical_target: str | None = None       # intended word, optional (§1.2)
    realized_phones: str | None = None      # actual pronunciation, immutable once observed
    language: Language | None = None

    # -- convenience -------------------------------------------------------- #
    def text_candidates(self) -> list[TextCandidate]:
        """Primary orthographic candidates (ASR branches + consensus), in order."""
        return [*self.qwen, *self.gigaam, *self.extra, *self.mbr]

    def selectable_candidates(self) -> list[TextCandidate]:
        """Every candidate the selector may choose by ID (§14.4–14.5).

        Beyond the ASR branches, the P2G spellings and RAG canonical spellings
        are exposed as selectable orthographic options so the judge can pick a
        reconstructed nonce spelling by ID (Appendix A.3) rather than having to
        author it — free-form ``NEW`` output stays gated (§14.5). P2G keeps its
        own ``x*`` ids; RAG matches get synthetic ``r*`` ids.
        """
        out = self.text_candidates()
        out += [
            TextCandidate(p.id, p.text, CandidateSource.P2G, score=p.prob)
            for p in self.p2g
        ]
        out += [
            TextCandidate(f"r{i}", r.canonical_spelling or r.term,
                          CandidateSource.RAG, score=r.similarity)
            for i, r in enumerate(self.rag)
        ]
        return out

    def candidate_by_id(self, cid: str) -> TextCandidate | None:
        for c in self.selectable_candidates():
            if c.id == cid:
                return c
        return None

    def add_flag(self, flag: SpanFlag) -> None:
        self.flags.add(flag)

    def resolved_text(self) -> str | None:
        """The verbatim text implied by the current selection, if any.

        Priority: an explicit gated novel form > the chosen orthographic
        candidate > an already-assembled ``canonical_text``. Returns ``None``
        when the span is still unresolved.
        """
        if self.selection is not None:
            sel = self.selection
            if sel.needs_novel_candidate and sel.novel_text is not None:
                return sel.novel_text
            if sel.orthographic_candidate_id is not None:
                cand = self.candidate_by_id(sel.orthographic_candidate_id)
                if cand is not None:
                    return cand.text
        return self.canonical_text

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "span_id": self.span_id,
            "speaker": self.speaker.value,
            "audio_source": self.audio_source.value,
            "start": self.interval.start,
            "end": self.interval.end,
        }
        if self.qwen:
            d["qwen"] = [c.to_dict() for c in self.qwen]
        if self.gigaam:
            d["gigaam"] = [c.to_dict() for c in self.gigaam]
        if self.extra:
            d["extra"] = [c.to_dict() for c in self.extra]
        if self.phones:
            d["phones"] = [p.to_dict() for p in self.phones]
        if self.p2g:
            d["p2g"] = [p.to_dict() for p in self.p2g]
        if self.rag:
            d["rag"] = [r.to_dict() for r in self.rag]
        if self.mbr:
            d["mbr"] = [c.to_dict() for c in self.mbr]
        conf = self.confidence.to_dict()
        if conf:
            d["confidence"] = conf
        if self.flags:
            d["flags"] = sorted(f.value for f in self.flags)
        if self.overlap_span_ids:
            d["overlap_span_ids"] = list(self.overlap_span_ids)
        if self.selection is not None:
            d["selection"] = self.selection.to_dict()
        for k in ("canonical_text", "lexical_target", "realized_phones"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.language is not None:
            d["language"] = self.language.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Span":
        return cls(
            span_id=int(d["span_id"]),
            speaker=Role(d["speaker"]),
            audio_source=AudioSource(d["audio_source"]),
            interval=Interval(float(d["start"]), float(d["end"])),
            qwen=[TextCandidate.from_dict(x) for x in d.get("qwen", [])],
            gigaam=[TextCandidate.from_dict(x) for x in d.get("gigaam", [])],
            extra=[TextCandidate.from_dict(x) for x in d.get("extra", [])],
            phones=[PhonePath.from_dict(x) for x in d.get("phones", [])],
            p2g=[P2GCandidate.from_dict(x) for x in d.get("p2g", [])],
            rag=[RagMatch.from_dict(x) for x in d.get("rag", [])],
            mbr=[TextCandidate.from_dict(x) for x in d.get("mbr", [])],
            confidence=Confidence.from_dict(d.get("confidence", {})),
            flags={SpanFlag(f) for f in d.get("flags", [])},
            overlap_span_ids=list(d.get("overlap_span_ids", [])),
            selection=Selection.from_dict(d["selection"]) if d.get("selection") else None,
            canonical_text=d.get("canonical_text"),
            lexical_target=d.get("lexical_target"),
            realized_phones=d.get("realized_phones"),
            language=Language(d["language"]) if d.get("language") else None,
        )


@dataclass
class Correction:
    """A human/lesson-review correction kept as immutable provenance (§16.2)."""

    span_id: int
    old_text: str | None
    new_text: str
    source: str = "manual"       # manual | lesson_review | teacher_correction
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Correction":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class LessonPackage:
    """A whole lesson: spans on a common timeline plus session state (§22.1)."""

    lesson_id: str
    spans: list[Span] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def spans_for(self, role: Role) -> list[Span]:
        return [s for s in self.spans if s.speaker is role]

    def sorted_spans(self) -> list[Span]:
        """Spans in timeline order (ties broken by span_id for determinism)."""
        return sorted(self.spans, key=lambda s: (s.interval.start, s.span_id))

    def with_selection(self, span_id: int, selection: Selection) -> Span:
        """Attach a selection to a span (returns the updated span)."""
        for s in self.spans:
            if s.span_id == span_id:
                s.selection = selection
                return s
        raise KeyError(f"no span with id {span_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "spans": [s.to_dict() for s in self.spans],
            "corrections": [c.to_dict() for c in self.corrections],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LessonPackage":
        return cls(
            lesson_id=d["lesson_id"],
            spans=[Span.from_dict(x) for x in d.get("spans", [])],
            corrections=[Correction.from_dict(x) for x in d.get("corrections", [])],
            metadata=dict(d.get("metadata", {})),
        )


def clone_span(span: Span, **changes: Any) -> Span:
    """Shallow-copy a span with field overrides (evidence objects are shared,
    which is safe because they are frozen)."""
    return replace(span, **changes)
