"""Reference-free candidate alignment over whole-recording branch transcripts.

A real system has no reference, so it builds the per-position candidate set by aligning the
branches to **each other**, then chooses. This module is that substrate:

* :func:`build_graph` aligns every branch to a designated primary pivot (or the longest one for
  generic callers) and produces an
  interleaved **confusion network**: a ``"word"`` slot per pivot position *and* an ``"ins"`` slot
  for every gap between them. A word another branch heard where the pivot didn't lands in the
  adjacent ``"ins"`` slot, so it stays selectable (it is not silently discarded). ``NULL`` (emit
  nothing) is a candidate in every slot.
* :func:`realizable_oracle_tokens` — the **honest** ceiling: for a given reference it builds an
  actual transcript by choosing, per slot, the candidate that best matches the reference
  (including emitting an ``"ins"`` candidate to recover a word the pivot dropped, or ``NULL`` to
  drop a spurious pivot word). It is a real transcript scored by ordinary WER, so — unlike a
  "fraction of reference words some branch matched" recall count — it *includes insertions* and
  never understates the achievable error.

Pure stdlib + the package's own aligner (an ``opcodes`` argument lets the notebook feed a fast
rapidfuzz alignment for whole-recording inputs).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .metrics import Op, align

NULL = None  # a slot candidate meaning "emit nothing here"


@dataclass
class Slot:
    kind: str                                         # "word" (a pivot position) or "ins" (a gap)
    pivot: str | None = None                          # the pivot word for a "word" slot; None for "ins"
    votes: Counter = field(default_factory=Counter)   # candidate token (or NULL) -> vote count

    @property
    def agreed(self) -> bool:
        """True when every branch offered the same candidate (nothing to decide)."""
        return len(self.votes) == 1

    def anchor(self) -> str | None:
        """The primary-backbone value: pivot word, or no insertion for a gap slot."""
        return self.pivot if self.kind == "word" else NULL


def build_graph(token_lists: list[list[str]], *, pivot_index: int | None = None) -> list[Slot]:
    """Interleaved candidate graph.

    ``pivot_index`` anchors the graph to a designated primary backbone (Qwen in the production
    path). When omitted, the legacy longest-hypothesis behavior is retained for generic callers.
    """
    lists = [t for t in token_lists if t]
    if not lists:
        return []
    pivot = (lists[pivot_index] if pivot_index is not None else max(lists, key=len))
    P = len(pivot)
    word_votes = [Counter() for _ in range(P)]
    ins_votes = [Counter() for _ in range(P + 1)]
    for h in lists:
        aligned: list[str | None] = [NULL] * P        # this branch's word per pivot slot (NULL=dropped)
        inserts: list[list[str]] = [[] for _ in range(P + 1)]
        p = 0                                          # pivot tokens consumed so far -> current gap
        if h is pivot:
            aligned = list(pivot)
        else:
            for op in align(pivot, h):
                if op.op in (Op.MATCH, Op.SUB):
                    aligned[op.ref_index] = h[op.hyp_index]; p = op.ref_index + 1
                elif op.op is Op.DEL:
                    p = op.ref_index + 1               # pivot word this branch didn't produce
                elif op.op is Op.INS:
                    inserts[p].append(h[op.hyp_index])  # word with no pivot slot -> this gap
        for i, tok in enumerate(aligned):
            word_votes[i][tok] += 1
        for g in range(P + 1):
            ins_votes[g][" ".join(inserts[g]) if inserts[g] else NULL] += 1
    slots: list[Slot] = []
    for i in range(P):
        slots.append(Slot("ins", None, ins_votes[i]))
        slots.append(Slot("word", pivot[i], word_votes[i]))
    slots.append(Slot("ins", None, ins_votes[P]))
    return slots


# --------------------------------------------------------------------------- #
# Realizable oracle — the honest ceiling for a selector over this network      #
# --------------------------------------------------------------------------- #
def _ref_maps(pivot: list[str], ref: list[str], opcodes=None):
    """Return ``(ref_at, gap_refs)``: the reference word aligned to each pivot slot (or None if
    the pivot word is spurious), and the reference words that fall in each gap (pivot dropped
    them). ``opcodes`` is an optional rapidfuzz-style ``(tag, i0, i1, j0, j1)`` alignment of
    ``pivot`` vs ``ref`` (tags equal/replace/delete/insert); without it we align locally."""
    ref_at: list[str | None] = [None] * len(pivot)
    gap_refs: list[list[str]] = [[] for _ in range(len(pivot) + 1)]
    if opcodes is not None:
        for tag, i0, i1, j0, j1 in opcodes:
            if tag in ("equal", "replace"):
                for pi, rj in zip(range(i0, i1), range(j0, j1)):
                    ref_at[pi] = ref[rj]
            elif tag == "insert":                     # ref words with no pivot slot -> gap at i0
                gap_refs[i0].extend(ref[j0:j1])
            # "delete": pivot words with no ref -> stay None (spurious)
    else:
        p = 0
        for op in align(pivot, ref):
            if op.op in (Op.MATCH, Op.SUB):
                ref_at[op.ref_index] = ref[op.hyp_index]; p = op.ref_index + 1
            elif op.op is Op.DEL:
                p = op.ref_index + 1
            elif op.op is Op.INS:
                gap_refs[p].append(ref[op.hyp_index])
    return ref_at, gap_refs


def realizable_oracle_tokens(graph: list[Slot], ref: list[str], *, opcodes=None) -> list[str]:
    """Best transcript actually assemblable from the network for this reference (the honest
    ceiling). Recovers a reference word only where some slot's candidate provides it, drops a
    spurious pivot word only where ``NULL`` is available — everything else is a real error."""
    pivot = [s.pivot for s in graph if s.kind == "word"]
    ref_at, gap_refs = _ref_maps(pivot, ref, opcodes)
    out: list[str] = []
    wi = gi = 0
    for s in graph:
        cand_words = {w for c in s.votes if c is not NULL for w in str(c).split()}
        if s.kind == "ins":
            for w in gap_refs[gi]:                    # recover pivot-dropped words a branch caught
                if w in cand_words:
                    out.append(w)
            gi += 1
        else:
            target = ref_at[wi]
            if target is None:                        # pivot word spurious vs ref
                if NULL not in s.votes:
                    out.append(s.pivot)               # can't drop it -> forced insertion (an error)
            elif target in cand_words:
                out.append(target)                    # some branch heard the right word here
            else:
                out.append(s.pivot)                   # nobody did -> unavoidable substitution
            wi += 1
    return out
