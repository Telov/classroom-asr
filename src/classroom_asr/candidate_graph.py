"""Reference-free candidate alignment over whole-recording branch transcripts.

A real system has no reference, so it builds the per-position candidate set by aligning the
branches to **each other**, then chooses. This module is that substrate:

* :func:`build_graph` aligns every branch to a designated primary pivot (or the longest one for
  generic callers) and produces an
  interleaved **confusion network**: a ``"word"`` slot per pivot position *and* an ``"ins"`` slot
  for every gap between them. A word another branch heard where the pivot didn't lands in the
  adjacent ``"ins"`` slot, so it stays selectable (it is not silently discarded). ``NULL`` (emit
  nothing) is a candidate in every slot.
* :func:`realizable_oracle_distance` — the **honest** ceiling: exact token edit distance between
  the reference and the best complete path through the graph. It jointly chooses one offered
  candidate per slot, so alignment ties cannot turn a selectable better path into a worse score.
  Unlike a "fraction of reference words some branch matched" recall count, it includes insertions.
* :func:`realizable_oracle_tokens` — a fast alignment-conditioned witness path retained for
  inspection. It is realizable, but local alignment ties mean it must not be used as the exact
  oracle score.

Pure stdlib + the package's own aligner (an ``opcodes`` argument lets the notebook feed a fast
rapidfuzz alignment for whole-recording inputs).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

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


Opcode = tuple[str, int, int, int, int]


def build_graph(
    token_lists: list[list[str]],
    *,
    pivot_index: int | None = None,
    opcodes_fn: Callable[[list[str], list[str]], Iterable[Opcode]] | None = None,
) -> list[Slot]:
    """Interleaved candidate graph.

    ``pivot_index`` anchors the graph to a designated primary backbone (Qwen in the production
    path). When omitted, the legacy longest-hypothesis behavior is retained for generic callers.

    ``opcodes_fn`` optionally supplies a fast sequence aligner such as
    ``rapidfuzz.distance.Levenshtein.opcodes(...).as_list()``. The dependency-free default keeps
    using :func:`classroom_asr.metrics.align`; whole-recording callers can opt into the compiled
    path without making RapidFuzz a core dependency.
    """
    lists = [t for t in token_lists if t]
    if not lists:
        return []
    if pivot_index is not None:
        if not 0 <= pivot_index < len(token_lists):
            raise IndexError(f"pivot_index {pivot_index} outside {len(token_lists)} token lists")
        pivot = token_lists[pivot_index]
        if not pivot:
            raise ValueError("designated pivot transcript is empty")
    else:
        pivot = max(lists, key=len)
    P = len(pivot)
    word_votes = [Counter() for _ in range(P)]
    ins_votes = [Counter() for _ in range(P + 1)]
    for h in lists:
        aligned: list[str | None] = [NULL] * P        # this branch's word per pivot slot (NULL=dropped)
        inserts: list[list[str]] = [[] for _ in range(P + 1)]
        p = 0                                          # pivot tokens consumed so far -> current gap
        if h is pivot:
            aligned = list(pivot)
        elif opcodes_fn is not None:
            for tag, i0, i1, j0, j1 in opcodes_fn(pivot, h):
                if tag in ("equal", "replace"):
                    paired = min(i1 - i0, j1 - j0)
                    for offset in range(paired):
                        aligned[i0 + offset] = h[j0 + offset]
                    # An unequal replacement tail is an insertion after its paired prefix;
                    # unmatched pivot words stay NULL by construction.
                    if j0 + paired < j1:
                        inserts[i0 + paired].extend(h[j0 + paired:j1])
                elif tag == "insert":
                    inserts[i0].extend(h[j0:j1])
                elif tag != "delete":
                    raise ValueError(f"unsupported alignment opcode: {tag!r}")
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
# Realizable paths and exact oracle distance                                    #
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
    """Build a fast, alignment-conditioned transcript from offered graph candidates.

    The result is always realizable, but it is not necessarily the globally minimum-edit path:
    a pivot/reference alignment tie can assign a reference word to the adjacent gap instead of a
    word slot that offers it. Use :func:`realizable_oracle_distance` for the exact oracle score.
    """
    pivot = [s.pivot for s in graph if s.kind == "word"]
    ref_at, gap_refs = _ref_maps(pivot, ref, opcodes)
    out: list[str] = []
    wi = gi = 0
    for s in graph:
        if s.kind == "ins":
            target = gap_refs[gi]
            # An insertion-slot candidate is atomic: if branches offered "a b" and "a c", a
            # selector may choose [], [a,b], or [a,c] -- never the synthetic [a,b,c] union. Pick
            # the whole offered sequence with the smallest local edit distance to this reference
            # gap. Counter preserves branch order, so ties deterministically prefer the anchor
            # (NULL) and then earlier branches.
            choices = [([] if candidate is NULL else str(candidate).split())
                       for candidate in s.votes]
            chosen = min(choices, key=lambda words: _edit_distance(words, target), default=[])
            out.extend(chosen)
            gi += 1
        else:
            target = ref_at[wi]
            cand_words = {c for c in s.votes if c is not NULL}
            if target is None:                        # pivot word spurious vs ref
                if NULL not in s.votes:
                    out.append(s.pivot)               # can't drop it -> forced insertion (an error)
            elif target in cand_words:
                out.append(target)                    # some branch heard the right word here
            else:
                out.append(s.pivot)                   # nobody did -> unavoidable substitution
            wi += 1
    return out


def _slot_sequences(slot: Slot) -> tuple[tuple[str, ...], ...]:
    """Every complete token sequence a selector may emit for one atomic slot."""
    return tuple(
        () if candidate is NULL else tuple(str(candidate).split())
        for candidate in slot.votes
    )


def realizable_oracle_distance(graph: list[Slot], ref: list[str]) -> int:
    """Exact edit distance from ``ref`` to the best complete path through ``graph``.

    This is Levenshtein dynamic programming over an acyclic word lattice. ``previous[j]`` is the
    best cost after all prior slots and the first ``j`` reference tokens. For each new slot we
    advance that row through every *whole* offered sequence, then take the elementwise minimum.
    Therefore alternatives remain mutually exclusive and candidate selection and alignment are
    optimized jointly.

    NumPy is an optional acceleration only; the package remains dependency-free and uses the
    equivalent stdlib implementation when NumPy is absent. Multiple one-token alternatives share
    one row update because scalar edit distance only needs to know whether *any* offered token
    matches each reference position.
    """
    if not graph:
        return len(ref)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - exercised in dependency-minimal installations
        return _realizable_oracle_distance_python(graph, ref)

    token_ids = {token: index for index, token in enumerate(dict.fromkeys(ref))}
    reference = np.asarray([token_ids[token] for token in ref], dtype=np.int32)
    positions = np.arange(len(ref) + 1, dtype=np.int32)
    previous = positions.copy()
    max_output = sum(
        max((len(sequence) for sequence in _slot_sequences(slot)), default=0)
        for slot in graph
    )
    infinity = np.int32(len(ref) + max_output + 1)

    def advance(row, accepted_ids):
        if len(accepted_ids) == 1:
            matches = reference == accepted_ids[0]
        else:
            matches = np.zeros(len(ref), dtype=bool)
            for token_id in accepted_ids:
                if token_id >= 0:
                    matches |= reference == token_id
        base = np.empty(len(ref) + 1, dtype=np.int32)
        base[0] = row[0] + 1
        base[1:] = np.minimum(row[1:] + 1, row[:-1] + np.logical_not(matches))
        # Close over any number of reference deletions: min_k(base[k] + j-k).
        return positions + np.minimum.accumulate(base - positions)

    for slot in graph:
        sequences = _slot_sequences(slot)
        if len(sequences) == 1:
            row = previous
            for token in sequences[0]:
                row = advance(row, (token_ids.get(token, -1),))
            previous = row
            continue

        best = previous.copy() if () in sequences else np.full(
            len(ref) + 1, infinity, dtype=np.int32
        )
        single_ids = tuple(
            dict.fromkeys(token_ids.get(sequence[0], -1)
                          for sequence in sequences if len(sequence) == 1)
        )
        if single_ids:
            np.minimum(best, advance(previous, single_ids), out=best)
        for sequence in sequences:
            if len(sequence) <= 1:
                continue
            row = previous
            for token in sequence:
                row = advance(row, (token_ids.get(token, -1),))
            np.minimum(best, row, out=best)
        previous = best
    return int(previous[-1])


def _realizable_oracle_distance_python(graph: list[Slot], ref: list[str]) -> int:
    """Dependency-free counterpart of :func:`realizable_oracle_distance`."""
    previous = list(range(len(ref) + 1))

    def advance(row: list[int], accepted: set[str]) -> list[int]:
        current = [row[0] + 1]
        for j, target in enumerate(ref, 1):
            current.append(min(
                row[j] + 1,
                current[j - 1] + 1,
                row[j - 1] + (target not in accepted),
            ))
        return current

    for slot in graph:
        sequences = _slot_sequences(slot)
        best = previous.copy() if () in sequences else [10**18] * (len(ref) + 1)
        singles = {sequence[0] for sequence in sequences if len(sequence) == 1}
        if singles:
            best = [min(left, right) for left, right in zip(best, advance(previous, singles))]
        for sequence in sequences:
            if len(sequence) <= 1:
                continue
            row = previous
            for token in sequence:
                row = advance(row, {token})
            best = [min(left, right) for left, right in zip(best, row)]
        previous = best
    return previous[-1]


def _edit_distance(left: list[str], right: list[str]) -> int:
    """Dependency-free token edit distance for the tiny alternatives inside one graph slot."""
    return sum(operation.op is not Op.MATCH for operation in align(left, right))
