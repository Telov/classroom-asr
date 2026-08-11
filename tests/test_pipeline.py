"""End-to-end pipeline behavior on the synthetic lesson.

These assert the *design requirements*, not implementation details: verbatim
preservation (A.1), pronunciation-based nonce recovery via future context
(A.3), quiet-word retention (A.5), overlap preservation (A.4), and a positive
candidate-oracle gate (§18.2).
"""

from classroom_asr.evaluate import evaluate_run
from classroom_asr.pipeline.orchestrator import Orchestrator, assemble_transcript
from classroom_asr.synthetic import example_lesson
from classroom_asr.types import SpanFlag


def run():
    return Orchestrator().run(example_lesson())


def test_final_wer_zero_and_oracle_headroom():
    result = run()
    ev = evaluate_run(result.package)
    assert ev.final.wer == 0.0                      # transcript matches truth
    assert ev.oracle.baseline.wer > 0.0             # naive 1-best makes errors
    assert ev.oracle.headroom > 0.0                 # answer already in the pool


def test_grammar_not_cleaned():
    # A.1 — "I didn't went there" must survive verbatim.
    rows = assemble_transcript(run().package)
    assert any(r["text"] == "I didn't went there" for r in rows)


def test_nonce_recovered_from_pronunciation():
    # A.3 — the early ambiguous "aboba" (1-best "above a") is recovered.
    rows = assemble_transcript(run().package)
    texts = [r["text"] for r in rows]
    assert "aboba" in texts
    assert "above a" not in texts


def test_quiet_word_retained():
    # A.5 — the quiet "to" is not deleted.
    rows = assemble_transcript(run().package)
    assert any(r["text"] == "to" for r in rows)


def test_overlap_marked_both_sides():
    # A.4 — overlapping teacher/student spans reference each other.
    pkg = run().package
    overlapped = [s for s in pkg.spans if SpanFlag.OVERLAP in s.flags]
    assert len(overlapped) >= 2
    assert all(s.overlap_span_ids for s in overlapped)


def test_future_context_builds_lexicon():
    result = run()
    assert "aboba" in result.session_lexicon
    assert result.session_lexicon.get("aboba").exact


def test_some_spans_frozen_some_expanded():
    # §12.5 — obvious spans are frozen; only uncertain ones reach the selector.
    result = run()
    assert result.n_frozen >= 1
    assert result.n_expanded >= 1
