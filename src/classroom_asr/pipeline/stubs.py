"""Dependency-free reference backends (§30 "model names are replaceable").

These are NOT models. They are deterministic stand-ins driven by the scripted
:class:`SpokenUnit` ground truth so the whole pipeline — timeline, candidate
graph, lexicon/RAG, selector, assembly, metrics — runs and is testable without
torch or any checkpoint. They deliberately reproduce the design doc's failure
modes so the demo shows real behavior:

* the acoustic 1-best is *wrong* on hard spans (quiet / nonce / switch) but the
  correct form appears lower in the N-best, so candidate-oracle WER beats the
  1-best baseline (§18.2 — the whole point of the gate);
* nonce words surface as "cleaned" dictionary words in the 1-best (Appendix
  A.3 "aboba" -> "above a"), which only phone/P2G/RAG can rescue;
* quiet function words are dropped by the 1-best (Appendix A.5).

Swap any one of these for a real subclass of the matching `base` interface
without touching the orchestrator.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from ..datamodel import P2GCandidate, PhonePath, Selection, Span, TextCandidate
from ..timeline import Interval, merge_intervals
from ..types import CandidateSource, SpanFlag
from .base import (
    VAD,
    AcousticModel,
    P2G,
    PhoneEncoder,
    Selector,
    SelectorContext,
    SpeechSegment,
)


def _rng(*parts: object) -> float:
    """Deterministic pseudo-random float in [0,1) from the arguments."""
    h = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


QUIET_ENERGY = 0.35     # below this a word is "quiet" (§6.2 / §18.1)


def is_hard(unit) -> bool:
    return unit.energy < QUIET_ENERGY or unit.is_nonce


# --------------------------------------------------------------------------- #
# VAD                                                                          #
# --------------------------------------------------------------------------- #
class StubVAD(VAD):
    """Pads and merges segments; never drops audio (§6.1)."""

    def __init__(self, pre_pad_s: float = 0.75, post_pad_s: float = 0.75,
                 merge_gap_s: float = 0.2) -> None:
        self.pre, self.post, self.gap = pre_pad_s, post_pad_s, merge_gap_s

    def segment(self, segments: Sequence[SpeechSegment]) -> list[SpeechSegment]:
        # Here segments are already utterance-level; we only apply padding and
        # keep everything. Real VAD would derive segments from the waveform.
        out: list[SpeechSegment] = []
        for s in segments:
            out.append(
                SpeechSegment(
                    interval=s.interval.padded(self.pre, self.post),
                    role=s.role,
                    audio_source=s.audio_source,
                    waveform=s.waveform,
                    truth=s.truth,
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Acoustic branches                                                            #
# --------------------------------------------------------------------------- #
def _degrade(unit, seed: object) -> str:
    """Produce a plausible *wrong* transcription of a hard unit."""
    if unit.energy < QUIET_ENERGY and len(unit.text.split()) == 1:
        return ""  # quiet single function word gets deleted (Appendix A.5)
    if unit.is_nonce and unit.canonical_spelling:
        # nonce forced into a dictionary reading (Appendix A.3)
        return _dictionary_reading(unit.canonical_spelling)
    words = unit.text.split()
    if len(words) > 1 and _rng(seed, "drop") < 0.5:
        drop = int(_rng(seed, "which") * len(words))
        return " ".join(w for i, w in enumerate(words) if i != drop)
    return unit.text  # fall back to correct if we can't sensibly degrade


_READINGS = {"aboba": "above a", "amogus": "among us", "chel": "chill"}


def _dictionary_reading(spelling: str) -> str:
    return _READINGS.get(spelling.lower(), spelling)


class StubQwenASR(AcousticModel):
    """Multilingual backbone stand-in (§7.1). Correct on easy spans, wrong-but-
    recoverable on hard ones."""

    source_name = "qwen"

    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        u = segment.truth
        if u is None:
            return []
        cands: list[TextCandidate] = []
        if is_hard(u):
            wrong = _degrade(u, ("qwen", u.text))
            cands.append(TextCandidate("q1", wrong, CandidateSource.QWEN, score=0.60, beam_rank=0))
            cands.append(TextCandidate("q2", u.text, CandidateSource.QWEN, score=0.30, beam_rank=1))
        else:
            # Easy span: the 1-best is correct. No spurious alternative that
            # would fake cross-branch disagreement.
            cands.append(TextCandidate("q1", u.text, CandidateSource.QWEN, score=0.90, beam_rank=0))
        return cands[:n_best]


class StubGigaAM(AcousticModel):
    """Russian-specialist text hypotheses for candidate diversity (§12.2)."""

    source_name = "gigaam"

    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        u = segment.truth
        if u is None:
            return []
        # GigaAM contributes an independent, Russian-biased hypothesis. For a
        # nonce it tends to keep an acoustic (non-dictionary) rendering, which is
        # exactly the diversity §12.2 wants.
        text = u.text
        return [TextCandidate("g1", text, CandidateSource.GIGAAM, score=0.5)][:n_best]


# --------------------------------------------------------------------------- #
# Phone + P2G                                                                  #
# --------------------------------------------------------------------------- #
_NEAR_SWAPS = {"θ": "s", "ð": "z", "w": "v", "ŋ": "n"}


class StubPhoneEncoder(PhoneEncoder):
    """Emits a small phone lattice around the realized IPA (§10.2)."""

    def recognize(self, segment: SpeechSegment, *, top_k: int) -> list[PhonePath]:
        u = segment.truth
        if u is None or not u.ipa:
            return []
        paths = [PhonePath("p1", u.ipa, 0.6)]
        # one near-miss path from an accent substitution, preserving uncertainty
        swapped = u.ipa
        for a, b in _NEAR_SWAPS.items():
            if a in swapped:
                swapped = swapped.replace(a, b)
                break
        if swapped != u.ipa:
            paths.append(PhonePath("p2", swapped, 0.3))
        return paths[:top_k]


_IPA2LAT = {
    "ɐ": "a", "ə": "a", "ʌ": "u", "ɪ": "i", "iː": "ee", "i": "i", "uː": "oo",
    "ʊ": "u", "ɔː": "o", "oʊ": "o", "o": "o", "b": "b", "g": "g", "m": "m",
    "s": "s", "θ": "th", "f": "f", "t": "t", "r": "r", "l": "l", "n": "n",
    "k": "k", "d": "d", "p": "p", "e": "e", "a": "a", "u": "u",
}


class StubP2G(P2G):
    """Naive phone->grapheme over the lattice (§10.3).

    Character-level so arbitrary strings never become OOVs. Emits one spelling
    per phone path, weighted by the path posterior.
    """

    def convert(self, phones: Sequence[PhonePath], *, n_best: int) -> list[P2GCandidate]:
        out: list[P2GCandidate] = []
        for i, path in enumerate(phones):
            spelling = self._graphemize(path.ipa)
            out.append(
                P2GCandidate(
                    id=f"x{i+1}",
                    text=spelling,
                    prob=path.prob,
                    script="latin",
                    ipa=path.ipa,
                    spelling_confidence=path.prob,   # novel spelling not unique (§10.5)
                )
            )
        return out[:n_best]

    @staticmethod
    def _graphemize(ipa: str) -> str:
        from ..phonetics import strip_ipa

        return "".join(_IPA2LAT.get(sym, sym) for sym in strip_ipa(ipa))


# --------------------------------------------------------------------------- #
# Selector                                                                     #
# --------------------------------------------------------------------------- #
class RuleBasedSelector(Selector):
    """A deterministic stand-in for the 9B/12B judge implementing §13.2 logic.

    It never writes free text (novel stays gated). It picks a candidate ID by:

    1. If a RAG match is strong (established/known spelling), pick it — this is
       the Appendix A.3 "aboba" recovery via session memory.
    2. Else prefer the highest-scoring *acoustically supported* candidate; when
       acoustic confidence is high it trusts acoustics over a "nicer" reading
       (§13.2, Appendix A.1 — do not clean "didn't went").
    3. Else fall back to the MBR/consensus candidate (§12.4).

    Crucially it does NOT prefer the more grammatical candidate — the adversarial
    lesson of §11.3 / Appendix A.1.
    """

    name = "rule_based_stub"

    def select(self, span: Span, context: SelectorContext, *, allow_novel: bool) -> Selection:
        phone_id = span.phones[0].id if span.phones else None

        # A span is only treated as a nonce recovery when it was flagged as one
        # (short span, pronunciation matches a known nonce term — §10.4).
        if SpanFlag.NONCE_CANDIDATE in span.flags:
            # 1. established/known spelling via session memory (Appendix A.3).
            if span.rag:
                best_idx = max(range(len(span.rag)), key=lambda i: span.rag[i].similarity)
                if span.rag[best_idx].similarity >= 0.85:
                    return Selection(
                        orthographic_candidate_id=f"r{best_idx}",
                        phone_candidate_id=phone_id,
                        confidence=span.rag[best_idx].similarity,
                        selector=self.name,
                    )
            # 2. no dictionary support: trust the phone->grapheme reconstruction
            #    over the "cleaned" ASR reading (§10.1).
            if span.p2g:
                best_p2g = max(span.p2g, key=lambda p: p.prob)
                return Selection(
                    orthographic_candidate_id=best_p2g.id,
                    phone_candidate_id=phone_id,
                    confidence=best_p2g.prob,
                    selector=self.name,
                )

        # Ordinary spans select among the ASR orthographic candidates only;
        # phonetic (P2G/RAG) candidates are reserved for the nonce branch above.
        candidates = span.text_candidates()
        # If there is acoustic evidence of speech (a phone path), do not select a
        # deletion: weak evidence is still evidence (§6.2, Appendix A.5). Only an
        # evidence-free span may resolve to empty.
        if span.phones:
            candidates = [c for c in candidates if c.text.strip()] or candidates
        if not candidates:
            return Selection(selector=self.name)

        # 3. highest-scoring acoustically supported candidate; it must NOT prefer
        #    the more grammatical reading (§11.3, Appendix A.1). Ties -> earliest.
        best = max(range(len(candidates)), key=lambda i: (candidates[i].score, -i))
        chosen = candidates[best]
        return Selection(
            orthographic_candidate_id=chosen.id,
            phone_candidate_id=phone_id,
            confidence=chosen.score,
            selector=self.name,
        )
