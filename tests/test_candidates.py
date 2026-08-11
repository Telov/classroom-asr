from classroom_asr.candidates import (
    mbr_consensus,
    should_expand,
    span_candidate_texts,
    token_vote,
)
from classroom_asr.datamodel import Confidence, P2GCandidate, Span, TextCandidate
from classroom_asr.timeline import Interval
from classroom_asr.types import AudioSource, CandidateSource, Role, SpanFlag


def _cand(cid, text, score=0.5, source=CandidateSource.QWEN):
    return TextCandidate(cid, text, source, score=score)


def test_mbr_picks_majority():
    hyps = [_cand("a", "i thought", 1.0), _cand("b", "i thought", 0.9),
            _cand("c", "i sought", 0.2)]
    mbr = mbr_consensus(hyps)
    assert mbr.text == "i thought"
    assert mbr.source is CandidateSource.MBR


def test_token_vote():
    hyps = [_cand("a", "x"), _cand("b", "x"), _cand("c", "y")]
    assert abs(token_vote(hyps) - 2 / 3) < 1e-9


def _span(**kw):
    base = dict(span_id=1, speaker=Role.STUDENT, audio_source=AudioSource.STUDENT_RAW,
                interval=Interval(0.0, 1.0))
    base.update(kw)
    return Span(**base)


def test_should_expand_on_flag():
    s = _span(qwen=[_cand("q1", "hi", 0.99)], flags={SpanFlag.NONCE_CANDIDATE})
    assert should_expand(s)


def test_should_freeze_confident_agreeing():
    s = _span(
        qwen=[_cand("q1", "hello", 0.99)],
        gigaam=[_cand("g1", "hello", 0.9, CandidateSource.GIGAAM)],
        confidence=Confidence(span=0.95),
    )
    assert not should_expand(s)


def test_should_expand_on_low_confidence():
    s = _span(qwen=[_cand("q1", "hello", 0.6)], confidence=Confidence(span=0.6))
    assert should_expand(s)


def test_span_candidate_texts_dedup_and_order():
    s = _span(
        qwen=[_cand("q1", "above a", 0.6)],
        gigaam=[_cand("g1", "above a", 0.5, CandidateSource.GIGAAM)],  # dup
        p2g=[P2GCandidate("x1", "aboba", 0.5)],
    )
    texts = span_candidate_texts(s)
    assert texts == ["above a", "aboba"]     # deduped, branch-priority order
