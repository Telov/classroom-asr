from classroom_asr.datamodel import (
    LessonPackage,
    P2GCandidate,
    PhonePath,
    RagMatch,
    Selection,
    Span,
    TextCandidate,
)
from classroom_asr.io import LessonStore
from classroom_asr.timeline import Interval
from classroom_asr.types import AudioSource, CandidateSource, Role, SpanFlag


def _rich_span():
    return Span(
        span_id=163,
        speaker=Role.STUDENT,
        audio_source=AudioSource.STUDENT_RAW,
        interval=Interval(522.12, 522.61),
        qwen=[TextCandidate("q1", "above a", CandidateSource.QWEN, score=0.6)],
        extra=[TextCandidate("v1", "aboba", CandidateSource.QWEN, score=0.55)],
        phones=[PhonePath("p1", "ɐbobə", 0.59)],
        p2g=[P2GCandidate("x1", "aboba", 0.51)],
        rag=[RagMatch("aboba", 0.94, canonical_spelling="aboba")],
        flags={SpanFlag.NONCE_CANDIDATE, SpanFlag.MODEL_DISAGREEMENT},
    )


def test_span_roundtrip():
    s = _rich_span()
    back = Span.from_dict(s.to_dict())
    assert back.span_id == 163
    assert back.qwen[0].text == "above a"
    assert back.phones[0].ipa == "ɐbobə"
    assert back.p2g[0].text == "aboba"
    assert back.rag[0].similarity == 0.94
    assert SpanFlag.NONCE_CANDIDATE in back.flags
    assert back.extra[0].id == "v1"
    assert any(c.id == "v1" for c in back.text_candidates())


def test_selectable_candidate_ids():
    s = _rich_span()
    ids = {c.id for c in s.selectable_candidates()}
    assert "q1" in ids and "x1" in ids and "r0" in ids


def test_resolved_text_priority():
    s = _rich_span()
    # picking the P2G reconstruction by id resolves to its text
    s.selection = Selection(orthographic_candidate_id="x1")
    assert s.resolved_text() == "aboba"
    # a gated novel form wins when present
    s.selection = Selection(needs_novel_candidate=True, novel_text="abobb")
    assert s.resolved_text() == "abobb"


def test_lesson_store_roundtrip(tmp_path):
    s = _rich_span()
    s.selection = Selection(orthographic_candidate_id="x1", confidence=0.9)
    s.canonical_text = "aboba"
    pkg = LessonPackage(lesson_id="L1", spans=[s], metadata={"note": "demo"})
    store = LessonStore(tmp_path / "L1")
    store.save(pkg)
    back = store.load()
    assert back.lesson_id == "L1"
    assert back.metadata["note"] == "demo"
    assert back.spans[0].resolved_text() == "aboba"
