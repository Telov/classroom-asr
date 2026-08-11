"""Enumerations shared across the pipeline.

These encode the *known* metadata the design doc says we get for free from the
application: speaker role from channel topology (§11.1) and audio provenance
from the capture path (§5.3). They are deliberately closed enums so that a
mislabeled source is a loud failure, not a silent domain leak (§5.3 asks that
raw vs. zoom be treated as distinct domains during evaluation/adaptation).
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """Which logical channel produced a span. Known, never inferred (§11.1)."""

    TEACHER = "teacher"
    STUDENT = "student"


class AudioSource(str, Enum):
    """Capture provenance for a span (§5.3). Drives domain conditioning."""

    TEACHER_RAW = "teacher_raw"
    STUDENT_RAW = "student_raw"      # endpoint capture, preferred (§5.2)
    STUDENT_ZOOM = "student_zoom"    # conferencing fallback (§5.3)

    @property
    def role(self) -> Role:
        return Role.TEACHER if self is AudioSource.TEACHER_RAW else Role.STUDENT

    @property
    def is_conferenced(self) -> bool:
        """True if the signal passed through conferencing DSP (lossy domain)."""
        return self is AudioSource.STUDENT_ZOOM


class Language(str, Enum):
    """Language tag for a lexical span (§16.4 / Appendix C ``language_span``)."""

    RU = "ru"
    EN = "en"
    OTHER = "other"
    UNCERTAIN = "uncertain"


class SpanFlag(str, Enum):
    """Uncertainty / event flags attached to a span (§13.1, §15.1.5, Appendix C).

    Flags drive the candidate-expansion policy (§12.5) and active learning
    (§16.6): the orchestrator only expands and re-opens *flagged* spans.
    """

    NONCE_CANDIDATE = "nonce_candidate"
    NONCE_KNOWN = "nonce_known"
    NONCE_NOVEL = "nonce_novel"
    MODEL_DISAGREEMENT = "model_disagreement"
    LOW_CONFIDENCE = "low_confidence"
    SWITCH_BOUNDARY = "switch_boundary"
    PHONE_TEXT_MISMATCH = "phone_text_mismatch"
    QUIET_WORD = "quiet_word"
    SHORT_WORD = "short_word"
    OVERLAP = "overlap"
    OOV = "oov"
    TEACHER_LEAK = "teacher_leak"
    SELF_CORRECTION = "self_correction"
    FALSE_START = "false_start"
    PRONUNCIATION_DRILL = "pronunciation_drill"
    TEACHER_CORRECTION = "teacher_correction"


class CandidateSource(str, Enum):
    """Which branch produced a text/phone candidate (§12)."""

    QWEN = "qwen"                 # multilingual ASR backbone (§7.1)
    GIGAAM = "gigaam"             # Russian specialist text hyp (§12.2)
    PHONE = "phone"              # phone lattice (§12.3)
    P2G = "p2g"                  # phone->grapheme (§10)
    RAG = "rag"                  # session/persistent lexicon match (§10.4)
    MBR = "mbr"                  # consensus / "do nothing clever" (§12.4)
    NOVEL = "novel"              # selector-authored novel form (§14.5, gated)
