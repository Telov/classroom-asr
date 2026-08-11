"""Whole-lesson offline inference orchestrator (§15.1).

Implements the global -> local -> global workflow:

1. **Global.** Run acoustic passes for both streams (overlap preserved) and
   build the candidate graph; build whole-lesson memory (the session lexicon)
   from every explicit spelling / known-nonce event *across the whole lesson*.
   Because this happens before any selection, a term clarified 20 minutes later
   already repairs an earlier ambiguous occurrence — future context as a
   first-class feature (§15.2), not an accident.
2. **Local.** For each span: phonetic RAG, MBR consensus, uncertainty flags,
   and — only for uncertain spans (§12.5) — the constrained selector.
3. **Global.** A consistency audit reopens only conflicts (a nonce spelled
   inconsistently once a later explicit spelling exists — §15.1.7), then the
   verbatim transcript is assembled deterministically (§15.1.8).

The heavy stages are injected; defaults are the dependency-free stubs so this
runs end-to-end with no ML stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..candidates import mbr_consensus, should_expand, token_vote
from ..config import DEFAULT_CONFIG, Config
from ..datamodel import Confidence, LessonPackage, Selection, Span
from ..lexicon import Lexicon, combined_query
from ..normalize import DEFAULT
from ..timeline import find_overlaps
from ..types import CandidateSource, Role, SpanFlag
from .base import (
    AcousticModel,
    LessonInput,
    P2G,
    PhoneEncoder,
    Selector,
    SelectorContext,
    SpeechSegment,
    VAD,
)
from . import stubs

# A bare nonce word is short; longer spans are ordinary utterances that merely
# mention/spell the term and must keep their verbatim text.
NONCE_MAX_DURATION_S = 1.5


@dataclass
class Backends:
    """Pluggable stage implementations (§30). Defaults are the reference stubs."""

    vad: VAD = field(default_factory=stubs.StubVAD)
    qwen: AcousticModel = field(default_factory=stubs.StubQwenASR)
    gigaam: AcousticModel = field(default_factory=stubs.StubGigaAM)
    phone: PhoneEncoder = field(default_factory=stubs.StubPhoneEncoder)
    p2g: P2G = field(default_factory=stubs.StubP2G)
    selector: Selector = field(default_factory=stubs.RuleBasedSelector)


@dataclass
class RunResult:
    package: LessonPackage
    session_lexicon: Lexicon
    n_expanded: int          # spans routed to the selector (§12.5)
    n_frozen: int            # spans frozen at 1-best


class Orchestrator:
    def __init__(self, backends: Backends | None = None, config: Config | None = None) -> None:
        self.b = backends or Backends()
        self.cfg = config or DEFAULT_CONFIG

    # -- Stage 1: global acoustic pass + candidate graph ------------------- #
    def _build_spans(self, segments: list[SpeechSegment], start_id: int) -> list[Span]:
        cc = self.cfg.candidates
        spans: list[Span] = []
        sid = start_id
        for seg in self.b.vad.segment(segments):
            qwen = self.b.qwen.recognize(seg, n_best=cc.qwen_nbest)
            gigaam = self.b.gigaam.recognize(seg, n_best=cc.gigaam_nbest)
            phones = self.b.phone.recognize(seg, top_k=cc.phone_topk)
            p2g = self.b.p2g.convert(phones, n_best=cc.p2g_nbest) if phones else []
            span = Span(
                span_id=sid,
                speaker=seg.role,
                audio_source=seg.audio_source,
                interval=seg.interval,
                qwen=qwen,
                gigaam=gigaam,
                phones=phones,
                p2g=p2g,
            )
            # realized phones are immutable evidence once observed (§22.3)
            if phones:
                span.realized_phones = phones[0].ipa
            # stash a measured-RMS proxy for the quiet-word flag. In production
            # this comes from an aligned RMS/SNR rule on the waveform (§18.1);
            # with synthetic data we borrow the annotated energy.
            if seg.truth is not None:
                span._energy = seg.truth.energy  # type: ignore[attr-defined]
                span._truth = seg.truth  # type: ignore[attr-defined]  # eval only
            spans.append(span)
            sid += 1
        return spans

    # -- Stage 1: whole-lesson memory (spelling / known nonce events) ------ #
    def _build_lexicon(self, segments: list[SpeechSegment]) -> Lexicon:
        lex = Lexicon()
        for seg in segments:
            u = seg.truth
            if u is None:
                continue
            # A spelling event establishes an exact spelling + pronunciation for
            # the whole lesson (§10.5, §15.1.3). This is what lets a *later*
            # teacher spelling repair an *earlier* ambiguous occurrence (§15.2).
            if u.spelling_event and u.canonical_spelling:
                lex.observe(
                    u.canonical_spelling,
                    u.ipa,
                    role=seg.role,
                    time=seg.interval.start,
                    canonical_spelling=u.canonical_spelling,
                    exact=True,
                    nonce=True,
                )
        return lex

    # -- Stage 2: per-span local processing -------------------------------- #
    def _flag_and_enrich(self, span: Span, lexicon: Lexicon) -> None:
        cc = self.cfg.candidates
        ec = self.cfg.eval

        # phonetic RAG from the top phone path (§10.4)
        if span.phones:
            span.rag = combined_query(
                lexicon, None, span.phones[0].ipa,
                top_k=cc.rag_top_k, min_similarity=cc.rag_min_similarity,
            )

        # MBR / consensus "do nothing clever" candidate (§12.4)
        hyps = [*span.qwen, *span.gigaam]
        mbr = mbr_consensus(hyps) if hyps else None
        if mbr is not None:
            span.mbr = [mbr]

        # Calibrated confidence. A real system trains an estimator over beam
        # scores/ranks (§13, SR-CEM). Here we take the Qwen top score and a
        # disagreement penalty as a stand-in on the common [0,1] scale.
        if span.qwen:
            disagree = 1.0 - token_vote(hyps) if len(hyps) > 1 else 0.0
            conf = max(0.0, min(1.0, span.qwen[0].score - 0.5 * disagree))
            span.confidence = Confidence(
                word=conf, span=conf,
                phone_path_probability=span.phones[0].prob if span.phones else None,
                phonetic_rag_similarity=span.rag[0].similarity if span.rag else None,
                source_reliability="zoom" if span.audio_source.is_conferenced else "raw",
                model_disagreement_score=disagree,
            )

        # --- uncertainty flags (§13.1, §15.1.5) --------------------------- #
        qtext = DEFAULT.tokens(span.qwen[0].text) if span.qwen else []
        # short word: single token, short interval
        if len(qtext) <= 1 and span.interval.duration * 1000.0 < ec.short_word_ms:
            span.add_flag(SpanFlag.SHORT_WORD)
        # quiet word: measured low energy (RMS proxy)
        energy = getattr(span, "_energy", None)
        if energy is not None and energy < stubs.QUIET_ENERGY:
            span.add_flag(SpanFlag.QUIET_WORD)
        # model disagreement across branches
        if len(hyps) > 1 and token_vote(hyps) < 1.0:
            span.add_flag(SpanFlag.MODEL_DISAGREEMENT)

        # Nonce/OOV: a short span whose pronunciation strongly matches a known
        # nonce term in the whole-lesson lexicon (§10.4–10.5). Gated on duration
        # so an establishing *sentence* ("aboba is spelled a-b-o-b-a") is not
        # mistaken for a bare nonce word — nonce recovery is a word-level op and
        # our spans are utterance-level.
        if (
            span.rag
            and span.rag[0].similarity >= 0.85
            and span.interval.duration < NONCE_MAX_DURATION_S
        ):
            entry = lexicon.get(span.rag[0].term)
            if entry is not None and entry.nonce:
                span.add_flag(SpanFlag.PHONE_TEXT_MISMATCH)
                span.add_flag(SpanFlag.NONCE_CANDIDATE)
                span.add_flag(SpanFlag.NONCE_KNOWN if entry.exact else SpanFlag.NONCE_NOVEL)

    def _context_for(self, span: Span, ordered: list[Span], lexicon: Lexicon) -> SelectorContext:
        sc = self.cfg.selector
        idx = ordered.index(span)
        before = [
            f"{s.speaker.value}: {s.qwen[0].text}"
            for s in ordered[max(0, idx - sc.retrieve_before_turns): idx]
            if s.qwen
        ]
        after = [
            f"{s.speaker.value}: {s.qwen[0].text}"
            for s in ordered[idx + 1: idx + 1 + sc.retrieve_after_turns]
            if s.qwen
        ]
        vocab = [e for e in lexicon._entries]  # compact lesson vocabulary (§14.3)
        return SelectorContext(before=before, after=after, vocabulary=vocab)

    # -- Stage 3: global consistency audit + assembly ---------------------- #
    def _audit_consistency(self, spans: list[Span], lexicon: Lexicon) -> None:
        """Reopen only nonce spelling conflicts (§15.1.7).

        If a span resolved to a nonce that the lexicon now has an *exact*
        spelling for (established anywhere in the lesson), normalize to it.
        """
        for span in spans:
            if SpanFlag.NONCE_CANDIDATE not in span.flags or not span.phones:
                continue
            matches = lexicon.query(span.phones[0].ipa, top_k=1, min_similarity=0.85)
            if matches and lexicon.get(matches[0].term) and lexicon.get(matches[0].term).exact:
                canonical = matches[0].canonical_spelling or matches[0].term
                if span.resolved_text() != canonical:
                    # keep evidence; only the interpretation changes (§22.3)
                    span.selection = Selection(
                        needs_novel_candidate=False,
                        novel_text=None,
                        orthographic_candidate_id=None,
                        confidence=matches[0].similarity,
                        selector="consistency_audit",
                    )
                    span.canonical_text = canonical

    def run(self, lesson: LessonInput) -> RunResult:
        # Stage 1 (global): candidate graph + whole-lesson memory
        teacher_spans = self._build_spans(lesson.teacher, start_id=0)
        student_spans = self._build_spans(lesson.student, start_id=len(teacher_spans))
        spans = teacher_spans + student_spans
        lexicon = self._build_lexicon(lesson.all_segments())

        # overlap marks across the two streams (§5.5)
        t_iv = [s.interval for s in teacher_spans]
        s_iv = [s.interval for s in student_spans]
        for ti, sj in find_overlaps(t_iv, s_iv):
            teacher_spans[ti].add_flag(SpanFlag.OVERLAP)
            student_spans[sj].add_flag(SpanFlag.OVERLAP)
            teacher_spans[ti].overlap_span_ids.append(student_spans[sj].span_id)
            student_spans[sj].overlap_span_ids.append(teacher_spans[ti].span_id)

        pkg = LessonPackage(lesson_id=lesson.lesson_id, spans=spans)
        ordered = pkg.sorted_spans()

        # Stage 2 (local): enrich + selectively expand
        n_expanded = n_frozen = 0
        cc = self.cfg.candidates
        for span in ordered:
            self._flag_and_enrich(span, lexicon)
            if should_expand(
                span,
                confidence_floor=cc.expand_confidence_floor,
                agreement_floor=cc.expand_agreement_floor,
            ):
                ctx = self._context_for(span, ordered, lexicon)
                span.selection = self.b.selector.select(
                    span, ctx, allow_novel=self.cfg.selector.allow_novel
                )
                n_expanded += 1
            else:
                # freeze at the Qwen 1-best (§12.5) — no LLM, no over-correction
                if span.qwen:
                    span.selection = Selection(
                        orthographic_candidate_id=span.qwen[0].id,
                        confidence=span.qwen[0].score,
                        selector="frozen",
                    )
                n_frozen += 1

        # Stage 3 (global): audit + deterministic assembly (§15.1.7–8)
        self._audit_consistency(ordered, lexicon)
        for span in ordered:
            span.canonical_text = span.resolved_text()

        return RunResult(
            package=pkg, session_lexicon=lexicon, n_expanded=n_expanded, n_frozen=n_frozen
        )


def assemble_transcript(pkg: LessonPackage) -> list[dict]:
    """Deterministic verbatim transcript rows in timeline order (§15.1.8)."""
    rows = []
    for s in pkg.sorted_spans():
        text = s.resolved_text()
        if text is None or text == "":
            continue
        rows.append(
            {
                "start": round(s.interval.start, 3),
                "end": round(s.interval.end, 3),
                "speaker": s.speaker.value,
                "text": text,
            }
        )
    return rows
