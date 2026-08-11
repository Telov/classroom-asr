"""Turn (ref, hyp) pairs into a readable breakdown of *what went wrong* (§18).

Scores alone don't tell you whether the system is dropping fillers, confusing a
handful of word pairs, or hallucinating. This module classifies every edit from
the alignment and surfaces the most frequent deletions, substitutions, and
insertions, plus the worst individual utterances — the raw material for the
deletion-slice metrics (§18.1) and active learning (§16.6).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .metrics import Op, align, score
from .normalize import DEFAULT, Normalizer


@dataclass
class ErrorReport:
    ref_words: int
    substitutions: int
    deletions: int
    insertions: int
    top_deletions: list[tuple[str, int]] = field(default_factory=list)
    top_insertions: list[tuple[str, int]] = field(default_factory=list)
    top_substitutions: list[tuple[str, int]] = field(default_factory=list)  # "ref->hyp"

    @property
    def wer(self) -> float:
        e = self.substitutions + self.deletions + self.insertions
        return e / self.ref_words if self.ref_words else 0.0

    def format(self, top: int = 12) -> str:
        lines = [
            f"WER {self.wer:.3f}  |  S={self.substitutions}  D={self.deletions}  I={self.insertions}"
            f"  (ref words={self.ref_words})",
            f"  deletions are {self._pct(self.deletions)}% of errors — "
            f"substitutions {self._pct(self.substitutions)}%, insertions {self._pct(self.insertions)}%",
            "",
            "Most-deleted words (ref words the hyp dropped):",
            *[f"    {w!r:20} x{n}" for w, n in self.top_deletions[:top]],
            "",
            "Most-common substitutions (ref -> hyp):",
            *[f"    {pair:30} x{n}" for pair, n in self.top_substitutions[:top]],
            "",
            "Most-inserted words (hyp words with no ref):",
            *[f"    {w!r:20} x{n}" for w, n in self.top_insertions[:top]],
        ]
        return "\n".join(lines)

    def _pct(self, x: int) -> int:
        e = self.substitutions + self.deletions + self.insertions
        return round(100 * x / e) if e else 0


def error_report(pairs, *, norm: Normalizer = DEFAULT) -> ErrorReport:
    """Aggregate an :class:`ErrorReport` over (ref, hyp) string pairs."""
    R = S = D = I = 0
    dels: Counter[str] = Counter()
    ins: Counter[str] = Counter()
    subs: Counter[str] = Counter()
    for ref, hyp in pairs:
        r, h = norm.tokens(ref), norm.tokens(hyp)
        R += len(r)
        for op in align(r, h):
            if op.op is Op.SUB:
                S += 1
                subs[f"{r[op.ref_index]} -> {h[op.hyp_index]}"] += 1
            elif op.op is Op.DEL:
                D += 1
                dels[r[op.ref_index]] += 1
            elif op.op is Op.INS:
                I += 1
                ins[h[op.hyp_index]] += 1
    return ErrorReport(
        ref_words=R, substitutions=S, deletions=D, insertions=I,
        top_deletions=dels.most_common(50),
        top_insertions=ins.most_common(50),
        top_substitutions=subs.most_common(50),
    )


def worst_utterances(pairs, *, n: int = 15, min_ref_words: int = 3,
                     norm: Normalizer = DEFAULT):
    """Return the ``n`` highest-WER (ref, hyp, wer) triples for eyeballing."""
    scored = []
    for ref, hyp in pairs:
        if len(norm.tokens(ref)) < min_ref_words:
            continue
        scored.append((ref, hyp, score(ref, hyp, norm=norm).wer))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:n]
