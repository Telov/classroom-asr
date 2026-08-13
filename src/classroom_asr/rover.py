"""Reference-free candidate graph + ROVER fusion over whole-recording branch transcripts.

The candidate-oracle (§18.2) shows how much is *recoverable* by picking the right branch per
word; a real system has no reference, so it must build the per-span candidate set by aligning
the branches to **each other** and then choosing. This module is that substrate:

* :func:`build_graph` aligns every branch transcript to a pivot (the most complete one) and
  produces one :class:`Slot` per pivot position — the candidate set + per-candidate vote count
  the design's LLM selector consumes (§12, §14.4). ``NULL`` (drop the word) is a candidate too.
* :func:`fuse` is the deterministic default selector — ROVER-style majority vote — which both
  gives a strong no-LLM baseline (it typically beats the best single branch) and settles the
  confident, agreeing spans the design says to freeze rather than hand to the LLM (§12.5). The
  LLM judge is a drop-in that overrides the choice on the *uncertain* slots.

Substitution + deletion voting only (no insertions): the fused transcript stays within the
pivot's word slots, so the fusion never invents words no branch placed there — the "select,
don't compose" guarantee (§14.1). Pure stdlib + the package's own aligner.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .metrics import Op, align
from .normalize import DEFAULT, Normalizer

NULL = None  # a slot candidate meaning "drop this word" (a deletion vote)


@dataclass
class Slot:
    """One pivot word position and every branch's aligned candidate for it."""

    pivot: str
    votes: Counter = field(default_factory=Counter)   # candidate token (or NULL) -> vote count

    @property
    def agreed(self) -> bool:
        """True when every branch offered the same token (nothing for the LLM to decide)."""
        return len(self.votes) == 1

    def winner(self) -> str | None:
        """Majority vote; ties break toward keeping the pivot's own word (conservative)."""
        return max(self.votes.items(), key=lambda kv: (kv[1], kv[0] == self.pivot))[0]


def _aligned_to_pivot(pivot: list[str], hyp: list[str]) -> list[str | None]:
    """For each pivot index, the hyp token aligned there, or NULL if hyp dropped it.

    Insertions (hyp words with no pivot slot) are intentionally ignored — fusion never adds
    words outside the pivot's positions."""
    out: list[str | None] = [NULL] * len(pivot)
    for op in align(pivot, hyp):
        if op.op in (Op.MATCH, Op.SUB):
            out[op.ref_index] = hyp[op.hyp_index]
    return out


def build_graph(token_lists: list[list[str]]) -> list[Slot]:
    """Align every (non-empty) branch to the most complete one → one :class:`Slot` per pivot word."""
    lists = [t for t in token_lists if t]
    if not lists:
        return []
    pivot = max(lists, key=len)                      # most complete branch anchors the graph
    slots = [Slot(pivot=w) for w in pivot]
    for h in lists:
        aligned = list(pivot) if h is pivot else _aligned_to_pivot(pivot, h)
        for i, tok in enumerate(aligned):
            slots[i].votes[tok] += 1
    return slots


def select(slots: list[Slot], chooser=None) -> list[str]:
    """Assemble a token list from the graph. ``chooser(slot) -> token|NULL`` overrides the
    default majority vote per slot (this is the seam the LLM selector plugs into); a NULL
    choice drops the word."""
    pick = chooser or (lambda s: s.winner())
    return [tok for s in slots if (tok := pick(s)) is not NULL]


def fuse(transcripts, *, norm: Normalizer = DEFAULT, chooser=None) -> str:
    """Fuse whole-recording branch transcripts into one string via the candidate graph.

    ``transcripts`` is one string per branch for the *same* recording. Returns the fused
    transcript (space-joined tokens). ``chooser`` defaults to ROVER majority vote."""
    token_lists = [norm.tokens(t) for t in transcripts if t and t.strip()]
    slots = build_graph(token_lists)
    return " ".join(select(slots, chooser)).strip()
