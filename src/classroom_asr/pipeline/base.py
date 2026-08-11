"""Stage interfaces for the acoustic + selection pipeline (§6–§14).

These abstract bases are the seams the design doc's modularity depends on
("the model names are replaceable" — §30). A real integration subclasses each
one against a concrete checkpoint; the orchestrator and everything above it are
written only against these interfaces.

Audio abstraction
-----------------
A :class:`SpeechSegment` carries the timeline interval, known role/source, and
either real ``waveform`` samples (production) or a scripted :class:`SpokenUnit`
of ground truth (synthetic/eval data). Real backends read ``waveform`` and
ignore ``truth``; the dependency-free stubs read ``truth``. This lets the entire
pipeline run and be tested with no audio stack, while keeping the interface
honest for real models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..datamodel import P2GCandidate, PhonePath, Selection, Span, TextCandidate
from ..timeline import Interval
from ..types import AudioSource, Language, Role


@dataclass
class SpokenUnit:
    """Ground-truth content of a segment, used by synthetic data and the stubs.

    Mirrors what an annotator would label (Appendix C): verbatim text, realized
    IPA, language, and the acoustic difficulty markers that drive deletion-slice
    metrics (§18.1).
    """

    text: str                       # verbatim spoken form (§1.2)
    ipa: str = ""                   # realized pronunciation
    language: Language | None = None
    energy: float = 1.0             # 0..1 local RMS proxy; <~0.3 == "quiet" (§6.2)
    is_nonce: bool = False          # invented/OOV word (§1.1)
    canonical_spelling: str | None = None   # if a known/established spelling exists
    spelled_aloud: bool = False     # speaker spelled it out (§10.5 exact recovery)
    # This utterance *establishes* the exact spelling of ``canonical_spelling``
    # (pronunciation ``ipa``) for the whole lesson, e.g. the teacher spelling a
    # nonce out (§10.5, §15.1.3). Distinct from ``is_nonce``: the utterance
    # itself may be an ordinary sentence whose verbatim text must be preserved.
    spelling_event: bool = False


@dataclass
class SpeechSegment:
    """One VAD segment on the lesson timeline."""

    interval: Interval
    role: Role
    audio_source: AudioSource
    waveform: Any | None = None      # e.g. np.ndarray at 16 kHz (§5.1); None for stubs
    truth: SpokenUnit | None = None  # present only for synthetic/eval data


@dataclass
class LessonInput:
    """Two logical streams for one lesson (§5). Overlap is preserved (§5.5)."""

    lesson_id: str
    teacher: list[SpeechSegment] = field(default_factory=list)
    student: list[SpeechSegment] = field(default_factory=list)

    def all_segments(self) -> list[SpeechSegment]:
        return [*self.teacher, *self.student]


# --------------------------------------------------------------------------- #
# Stage interfaces                                                             #
# --------------------------------------------------------------------------- #
class VAD(ABC):
    """Non-destructive segmentation (§6.1). Must not discard low-energy edges."""

    @abstractmethod
    def segment(self, segments: Sequence[SpeechSegment]) -> list[SpeechSegment]:
        """Refine/pad segments; may split or pad but never drop audio (§6.1)."""


class AcousticModel(ABC):
    """An ASR branch producing N-best text candidates (§7.1, §12.1)."""

    source_name: str = "asr"

    @abstractmethod
    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        ...


class PhoneEncoder(ABC):
    """A phone-recognition branch producing a top-K phone lattice (§7.3, §10.2)."""

    @abstractmethod
    def recognize(self, segment: SpeechSegment, *, top_k: int) -> list[PhonePath]:
        ...


class P2G(ABC):
    """Phone->grapheme reconstruction over an uncertain phone lattice (§10.2–10.3).

    Must consume the *lattice* (top-K + posteriors), not a single IPA string
    (§10.2), and emit character-level candidates so arbitrary strings are not
    tokenizer OOVs (§10.3).
    """

    @abstractmethod
    def convert(self, phones: Sequence[PhonePath], *, n_best: int) -> list[P2GCandidate]:
        ...


@dataclass
class SelectorContext:
    """Retrieved whole-lesson context for one uncertain span (§14.3–14.4).

    A *compact* package, not the whole lesson (§14.3): recent/relevant earlier
    turns, relevant later turns (future context is an asset — §15.2), and lesson
    vocabulary.
    """

    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    vocabulary: list[str] = field(default_factory=list)


class Selector(ABC):
    """The constrained conversation-LLM judge (§14).

    Selects evidence-supported candidate *IDs*; it does not compose transcripts
    (§14.1/§14.5). ``allow_novel`` gates the free-form ``NEW`` escape hatch.
    """

    name: str = "selector"

    @abstractmethod
    def select(self, span: Span, context: SelectorContext, *, allow_novel: bool) -> Selection:
        ...
