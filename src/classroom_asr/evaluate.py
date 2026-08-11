"""Turn a completed run into the headline numbers of the evaluation spec (§18).

Works on spans that still carry scripted ``_truth`` (synthetic/eval data). The
two numbers that matter first (§18.2, §28): final verbatim WER and the
candidate-oracle WER gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidates import span_candidate_texts
from .datamodel import LessonPackage
from .metrics import OracleResult, WERResult, candidate_oracle_wer, corpus_wer
from .normalize import DEFAULT, Normalizer


@dataclass
class LessonEval:
    final: WERResult
    oracle: OracleResult
    n_spans: int


def evaluate_run(pkg: LessonPackage, *, norm: Normalizer = DEFAULT) -> LessonEval:
    spans = [s for s in pkg.sorted_spans() if getattr(s, "_truth", None) is not None]
    refs = [s._truth.text for s in spans]                       # type: ignore[attr-defined]
    hyps = [s.resolved_text() or "" for s in spans]
    cand_sets = [span_candidate_texts(s) for s in spans]
    baseline = [s.qwen[0].text if s.qwen else "" for s in spans]

    final = corpus_wer(zip(refs, hyps), norm=norm)
    oracle = candidate_oracle_wer(refs, cand_sets, baseline_choice=baseline, norm=norm)
    return LessonEval(final=final, oracle=oracle, n_spans=len(spans))
