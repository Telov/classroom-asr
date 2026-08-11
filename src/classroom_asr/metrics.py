"""Evaluation metrics (§18).

Everything is alignment-based so the same edit path drives WER *and* the
deletion-slice metrics the product actually cares about (§18.1): deletions of
short, low-energy, switch-boundary, and nonce words are the headline failure
mode. The star of the file is :func:`candidate_oracle_wer` — the §18.2
development gate that separates "candidate generation" problems from
"selection/reasoning" problems.

Pure stdlib; no numpy required. For large corpora these are O(n*m) per pair,
which is fine at lesson scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .normalize import DEFAULT, Normalizer


class Op(str, Enum):
    MATCH = "match"
    SUB = "sub"
    DEL = "del"      # reference token with no hyp counterpart (deletion)
    INS = "ins"      # hyp token with no ref counterpart (insertion)


@dataclass(frozen=True)
class AlignOp:
    op: Op
    ref_index: int | None
    hyp_index: int | None


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[AlignOp]:
    """Levenshtein alignment (unit costs) with a deterministic backtrace.

    Tie-break order at equal cost: substitution/match, then deletion, then
    insertion. Stable ordering matters so slice metrics are reproducible (§29).
    """
    n, m = len(ref), len(hyp)
    # cost[i][j] = edit distance between ref[:i] and hyp[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            sub = cost[i - 1][j - 1] + (0 if ri == hyp[j - 1] else 1)
            dele = cost[i - 1][j] + 1
            ins = cost[i][j - 1] + 1
            cost[i][j] = min(sub, dele, ins)

    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if cost[i][j] == cost[i - 1][j - 1] + (0 if same else 1):
                ops.append(AlignOp(Op.MATCH if same else Op.SUB, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(AlignOp(Op.DEL, i - 1, None))
            i -= 1
            continue
        ops.append(AlignOp(Op.INS, None, j - 1))
        j -= 1
    ops.reverse()
    return ops


@dataclass(frozen=True)
class WERResult:
    ref_len: int
    substitutions: int
    deletions: int
    insertions: int
    ops: tuple[AlignOp, ...] = field(default=(), repr=False)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.ref_len if self.ref_len else (0.0 if self.errors == 0 else 1.0)

    @property
    def deletion_rate(self) -> float:
        return self.deletions / self.ref_len if self.ref_len else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "ref_len": self.ref_len,
            "sub": self.substitutions,
            "del": self.deletions,
            "ins": self.insertions,
            "errors": self.errors,
            "wer": self.wer,
            "deletion_rate": self.deletion_rate,
        }


def score(ref: str, hyp: str, *, norm: Normalizer = DEFAULT) -> WERResult:
    """WER for a single reference/hypothesis string pair."""
    r, h = norm.tokens(ref), norm.tokens(hyp)
    return _score_tokens(r, h)


def _score_tokens(r: Sequence[str], h: Sequence[str]) -> WERResult:
    ops = align(r, h)
    subs = sum(1 for o in ops if o.op is Op.SUB)
    dels = sum(1 for o in ops if o.op is Op.DEL)
    ins = sum(1 for o in ops if o.op is Op.INS)
    return WERResult(len(r), subs, dels, ins, tuple(ops))


def corpus_wer(pairs: Iterable[tuple[str, str]], *, norm: Normalizer = DEFAULT) -> WERResult:
    """Aggregate WER over many (ref, hyp) pairs (micro-average over tokens)."""
    R = S = D = I = 0
    for ref, hyp in pairs:
        res = score(ref, hyp, norm=norm)
        R += res.ref_len
        S += res.substitutions
        D += res.deletions
        I += res.insertions
    return WERResult(R, S, D, I)


def cer(ref: str, hyp: str, *, norm: Normalizer = DEFAULT) -> float:
    """Character error rate (§18.1 nonce CER, IPA-CER)."""
    r, h = norm.chars(ref), norm.chars(hyp)
    return _score_tokens(r, h).wer


# --------------------------------------------------------------------------- #
# Deletion-slice metrics (§18.1)                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SliceDeletion:
    """Deletion rate restricted to reference tokens carrying a given property.

    ``marked`` is the set of reference-token indices in the *slice* (e.g. tokens
    shorter than 300 ms, or locally low-energy tokens — §18.1). We report how
    many of those specific tokens were deleted.
    """

    slice_name: str
    marked: int
    deleted: int

    @property
    def rate(self) -> float:
        return self.deleted / self.marked if self.marked else 0.0


def slice_deletion_rate(
    result: WERResult, marked_ref_indices: set[int], *, slice_name: str = "slice"
) -> SliceDeletion:
    """Given an aligned :class:`WERResult`, count deletions among ``marked``."""
    deleted = sum(
        1 for o in result.ops if o.op is Op.DEL and o.ref_index in marked_ref_indices
    )
    return SliceDeletion(slice_name, len(marked_ref_indices), deleted)


# --------------------------------------------------------------------------- #
# Candidate-oracle WER — the §18.2 development gate                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OracleResult:
    oracle: WERResult
    baseline: WERResult
    chosen: tuple[str, ...]          # per-span oracle-chosen candidate text
    headroom: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "headroom", self.baseline.wer - self.oracle.wer)


def candidate_oracle_wer(
    span_refs: Sequence[str],
    span_candidates: Sequence[Sequence[str]],
    *,
    baseline_choice: Sequence[str] | None = None,
    norm: Normalizer = DEFAULT,
) -> OracleResult:
    """Best transcript attainable if an oracle picked the right candidate per span.

    For each span the oracle picks the candidate with the fewest word errors
    against that span's reference; ties break toward the earliest candidate
    (branch priority order). The chosen candidates are concatenated and scored
    against the concatenated references. This is exactly the §18.2 gate:

    * large ``headroom`` (oracle << baseline)  -> the answer is already in the
      candidate pool; invest in the selector.
    * small ``headroom``                       -> improve upstream acoustics /
      candidate generation, *not* the LLM judge.

    ``baseline_choice`` defaults to each span's first candidate (typically the
    1-best of the primary branch) so the comparison mirrors §18.2 Example A/B.
    """
    if len(span_refs) != len(span_candidates):
        raise ValueError("span_refs and span_candidates length mismatch")

    chosen: list[str] = []
    baseline: list[str] = []
    for idx, (ref, cands) in enumerate(zip(span_refs, span_candidates)):
        cand_list = list(cands)
        rtok = norm.tokens(ref)
        if not cand_list:
            chosen.append("")
        else:
            chosen.append(min(cand_list, key=lambda c: _score_tokens(rtok, norm.tokens(c)).errors))
        if baseline_choice is not None:
            baseline.append(baseline_choice[idx])
        else:
            baseline.append(cand_list[0] if cand_list else "")

    # Aggregate per utterance (micro-average), consistent with corpus_wer, so
    # the baseline here matches a corpus_wer over the same 1-best list. A single
    # concatenated alignment would let matches cross utterance boundaries and
    # disagree with the per-utterance baseline print — the source of the earlier
    # confusing "headroom +0.000".
    oracle = _aggregate(span_refs, chosen, norm)
    base = _aggregate(span_refs, baseline, norm)
    return OracleResult(oracle=oracle, baseline=base, chosen=tuple(chosen))


def _aggregate(refs: Sequence[str], hyps: Sequence[str], norm: Normalizer) -> WERResult:
    R = S = D = I = 0
    for ref, hyp in zip(refs, hyps):
        res = _score_tokens(norm.tokens(ref), norm.tokens(hyp))
        R += res.ref_len
        S += res.substitutions
        D += res.deletions
        I += res.insertions
    return WERResult(R, S, D, I)


# --------------------------------------------------------------------------- #
# Calibration (§18.1 ECE)                                                      #
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, bins: int = 10
) -> float:
    """Expected calibration error over confidence/correctness pairs (§13, §18.1).

    Answers "can confidence be trusted for routing and active learning?" (§18.1).
    """
    if len(confidences) != len(correct):
        raise ValueError("length mismatch")
    if not confidences:
        return 0.0
    n = len(confidences)
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        # last bin is closed on the right so conf == 1.0 lands somewhere
        members = [
            i for i, c in enumerate(confidences)
            if (lo <= c < hi) or (b == bins - 1 and c == hi)
        ]
        if not members:
            continue
        acc = sum(1 for i in members if correct[i]) / len(members)
        conf = sum(confidences[i] for i in members) / len(members)
        ece += (len(members) / n) * abs(acc - conf)
    return ece
