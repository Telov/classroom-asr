"""Candidate-graph construction helpers (§12).

The final selector must not be asked to infer a transcript from three one-best
strings; it receives a candidate graph rich enough that the correct answer is
already present whenever the acoustics heard it (§12). This module supplies the
non-model logic around that graph:

* :func:`mbr_consensus` — the conservative "do nothing clever" candidate (§12.4).
* :func:`should_expand` — the candidate-expansion policy (§12.5): only expand
  uncertain regions; freeze confident, agreeing spans so the LLM gets fewer
  chances to over-correct already-correct text.
* :func:`span_candidate_texts` — flatten a span's evidence into the candidate
  string set used by the oracle metric (§18.2).
"""

from __future__ import annotations

import math
from collections import Counter

from .datamodel import Span, TextCandidate
from .metrics import score
from .normalize import DEFAULT, Normalizer
from .types import CandidateSource, SpanFlag


def _softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps] if total else [1.0 / len(scores)] * len(scores)


def mbr_consensus(
    hyps: list[TextCandidate],
    *,
    norm: Normalizer = DEFAULT,
    temperature: float = 1.0,
    candidate_id: str = "mbr1",
) -> TextCandidate | None:
    """Minimum-Bayes-risk consensus over text hypotheses (§12.4).

    Treats branch scores as a posterior over hypotheses and returns the
    hypothesis minimizing expected WER against the rest — a ROVER/MBR-style
    "safe" candidate. Returns ``None`` for an empty input.
    """
    if not hyps:
        return None
    if len(hyps) == 1:
        return TextCandidate(candidate_id, hyps[0].text, CandidateSource.MBR, score=1.0)

    probs = _softmax([h.score for h in hyps], temperature)
    best_idx, best_risk = 0, math.inf
    for i, hi in enumerate(hyps):
        risk = 0.0
        for j, hj in enumerate(hyps):
            if i == j:
                continue
            risk += probs[j] * score(hj.text, hi.text, norm=norm).wer
        if risk < best_risk:
            best_idx, best_risk = i, risk
    chosen = hyps[best_idx]
    # Consensus confidence: how much posterior mass agrees exactly with choice.
    agree = sum(
        probs[j] for j, h in enumerate(hyps)
        if norm.tokens(h.text) == norm.tokens(chosen.text)
    )
    return TextCandidate(candidate_id, chosen.text, CandidateSource.MBR, score=agree)


def token_vote(hyps: list[TextCandidate], *, norm: Normalizer = DEFAULT) -> float:
    """Fraction of branches whose full token sequence equals the plurality one.

    A cheap cross-branch agreement signal for :func:`should_expand` and for the
    ``model_disagreement`` flag (§13.1).
    """
    if not hyps:
        return 1.0
    keys = [tuple(norm.tokens(h.text)) for h in hyps]
    top, n = Counter(keys).most_common(1)[0]
    return n / len(hyps)


# Flags that always warrant candidate expansion regardless of confidence (§12.5,
# §15.1.5): these are the known failure modes the graph exists to rescue.
_EXPAND_FLAGS = frozenset({
    SpanFlag.OOV,
    SpanFlag.NONCE_CANDIDATE,
    SpanFlag.NONCE_NOVEL,
    SpanFlag.PHONE_TEXT_MISMATCH,
    SpanFlag.SWITCH_BOUNDARY,
    SpanFlag.QUIET_WORD,
    SpanFlag.SHORT_WORD,
    SpanFlag.MODEL_DISAGREEMENT,
    SpanFlag.OVERLAP,
})


def should_expand(
    span: Span,
    *,
    confidence_floor: float = 0.85,
    agreement_floor: float = 0.99,
    norm: Normalizer = DEFAULT,
) -> bool:
    """Candidate-expansion policy (§12.5).

    Expand (generate more candidates / invoke the selector) when a span is
    uncertain. Freeze an obvious span when calibrated Qwen confidence is high,
    the branches agree, and no failure-mode flag is set — this both saves cost
    and, more importantly, removes opportunities for the LLM to over-correct
    correct text (§12.5).
    """
    if span.flags & _EXPAND_FLAGS:
        return True
    conf = span.confidence.span if span.confidence.span is not None else span.confidence.word
    if conf is not None and conf < confidence_floor:
        return True
    hyps = span.text_candidates()
    if len(hyps) > 1 and token_vote(hyps, norm=norm) < agreement_floor:
        return True
    return False


def span_candidate_texts(span: Span, *, include_p2g: bool = True,
                         include_rag: bool = True) -> list[str]:
    """All candidate spellings for a span, de-duplicated in branch-priority order.

    Ordering is Qwen → GigaAM → extra → MBR → P2G → RAG canonical spellings,
    matching the branch precedence used elsewhere. This is the candidate set fed to
    :func:`classroom_asr.metrics.candidate_oracle_wer`.
    """
    texts: list[str] = []
    seen: set[str] = set()

    def _add(t: str | None) -> None:
        if t is None:
            return
        key = t.strip()
        if key and key.casefold() not in seen:
            seen.add(key.casefold())
            texts.append(key)

    for c in span.qwen:
        _add(c.text)
    for c in span.gigaam:
        _add(c.text)
    for c in span.extra:
        _add(c.text)
    for c in span.mbr:
        _add(c.text)
    if include_p2g:
        for p in span.p2g:
            _add(p.text)
    if include_rag:
        for r in span.rag:
            _add(r.canonical_spelling or r.term)
    return texts
