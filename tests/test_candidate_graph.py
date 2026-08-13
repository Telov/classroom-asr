from itertools import product

import pytest

from classroom_asr.candidate_graph import (
    NULL,
    build_graph,
    realizable_oracle_distance,
    realizable_oracle_tokens,
)
from classroom_asr.normalize import Normalizer
from classroom_asr.metrics import score

N = Normalizer()  # plain: lowercase + edge punct, no number/spelling folding


def _words(graph):
    return [s for s in graph if s.kind == "word"]


def test_designated_backbone_is_the_graph_pivot():
    graph = build_graph(
        [N.tokens("qwen short"), N.tokens("another much longer transcript")], pivot_index=0)
    assert [slot.pivot for slot in graph if slot.kind == "word"] == ["qwen", "short"]


def test_designated_pivot_index_is_not_shifted_when_another_branch_is_empty():
    graph = build_graph(
        [[], N.tokens("qwen backbone"), N.tokens("other words")], pivot_index=1
    )
    assert [slot.pivot for slot in graph if slot.kind == "word"] == ["qwen", "backbone"]

    with pytest.raises(ValueError, match="designated pivot transcript is empty"):
        build_graph([[], N.tokens("fallback must not silently win")], pivot_index=0)


def test_fast_opcode_path_preserves_replacements_and_insertions():
    pivot = ["we", "went", "home"]
    other = ["we", "uh", "go", "home"]

    def opcodes(ref, hyp):
        assert ref == pivot and hyp == other
        return [
            ("equal", 0, 1, 0, 1),
            ("insert", 1, 1, 1, 2),
            ("replace", 1, 2, 2, 3),
            ("equal", 2, 3, 3, 4),
        ]

    graph = build_graph([pivot, other], pivot_index=0, opcodes_fn=opcodes)
    insertion_slots = [slot for slot in graph if slot.kind == "ins"]
    word_slots = [slot for slot in graph if slot.kind == "word"]

    assert insertion_slots[1].votes["uh"] == 1
    assert word_slots[1].votes["go"] == 1
    assert word_slots[2].votes["home"] == 2


def test_anchor_is_backbone_word_or_no_insertion():
    graph = build_graph(
        [N.tokens("i went home"), N.tokens("i really want home")], pivot_index=0)
    assert [slot.anchor() for slot in graph] == [None, "i", None, "went", None, "home", None]


def test_graph_marks_agreement():
    words = _words(build_graph([N.tokens("a b c"), N.tokens("a x c")]))
    assert words[0].agreed and words[2].agreed         # all branches agree on "a" and "c"
    assert not words[1].agreed                          # b vs x -> the LLM's job
    assert NULL not in words[1].votes


def test_insertion_candidate_is_kept_selectable():
    # review finding #2: pivot "a b c d", other branch "a x b c" -> "x" must survive in the graph
    graph = build_graph([N.tokens("a b c d"), N.tokens("a x b c")])
    all_candidates = {c for s in graph for c in s.votes if c is not NULL}
    assert "x" in all_candidates                        # the inserted word is not discarded


def test_realizable_oracle_counts_insertions():
    # review finding #1: ref "a b", only hypothesis "a x b".
    # a recall count says 0.0 (both a,b matched); the realizable oracle must show the true 0.5.
    graph = build_graph([N.tokens("a x b")])
    oracle = " ".join(realizable_oracle_tokens(graph, N.tokens("a b")))
    assert oracle == "a x b"                             # x is a forced, unavoidable insertion
    assert score("a b", oracle, norm=N).wer == 0.5

    # but if a branch actually heard "a b" (no x), the oracle can drop x -> 0.0
    graph2 = build_graph([N.tokens("a x b"), N.tokens("a b")])
    oracle2 = " ".join(realizable_oracle_tokens(graph2, N.tokens("a b")))
    assert score("a b", oracle2, norm=N).wer == 0.0


def test_oracle_recovers_a_pivot_deletion_from_an_insertion_slot():
    # pivot dropped "really"; another branch has it -> oracle recovers it via the ins slot
    graph = build_graph([N.tokens("i do"), N.tokens("i really do")])
    oracle = " ".join(realizable_oracle_tokens(graph, N.tokens("i really do")))
    assert oracle == "i really do"


def test_oracle_cannot_union_parts_of_mutually_exclusive_insertion_candidates():
    # Every branch is aligned against pivot "x". The graph therefore offers one insertion slot
    # with exactly [], [a,b], or [a,c]. A word-union implementation incorrectly emitted [a,b,c]
    # and claimed a perfect, but unassemblable, oracle transcript.
    graph = build_graph(
        [N.tokens("x"), N.tokens("a b x"), N.tokens("a c x")], pivot_index=0
    )

    oracle = realizable_oracle_tokens(graph, N.tokens("a b c x"))

    assert oracle == N.tokens("a b x")
    offered = {
        tuple([] if candidate is NULL else candidate.split())
        for slot in graph if slot.kind == "ins" for candidate in slot.votes
    }
    assert tuple(oracle[:-1]) in offered
    assert score("a b c x", " ".join(oracle), norm=N).wer == 0.25


def test_adding_a_branch_cannot_worsen_the_realizable_oracle():
    ref = N.tokens("i really want to go")
    pivot = N.tokens("i want go")
    first_alternative = N.tokens("i really want go")
    second_alternative = N.tokens("i want to go")

    smaller = build_graph([pivot, first_alternative], pivot_index=0)
    larger = build_graph([pivot, first_alternative, second_alternative], pivot_index=0)
    assert realizable_oracle_distance(larger, ref) <= realizable_oracle_distance(smaller, ref)


def _offered_paths(graph):
    choices = [
        [() if candidate is NULL else tuple(str(candidate).split()) for candidate in slot.votes]
        for slot in graph
    ]
    for selected in product(*choices):
        yield [token for sequence in selected for token in sequence]


def test_exact_oracle_optimizes_candidate_choice_and_alignment_jointly():
    # Pivot/reference edit alignment has two equal-cost choices here. The old local oracle maps
    # reference "b" into the preceding gap, then emits pivot "a" (2 errors), even though the
    # word slot explicitly offers "b" and that realizable path has only one deletion.
    graph = build_graph([["a"], ["b"]], pivot_index=0)

    assert realizable_oracle_tokens(graph, ["b", "c"]) == ["a"]
    assert realizable_oracle_distance(graph, ["b", "c"]) == 1


def test_exact_oracle_matches_exhaustive_paths_on_small_graphs():
    sequences = [list(words) for length in (1, 2)
                 for words in product(("a", "b"), repeat=length)]
    references = [[]] + sequences

    for pivot in sequences:
        for alternative in sequences:
            graph = build_graph([pivot, alternative], pivot_index=0)
            for reference in references:
                brute = min(
                    score(" ".join(reference), " ".join(path), norm=N).errors
                    for path in _offered_paths(graph)
                )
                assert realizable_oracle_distance(graph, reference) == brute


def test_empty_graph():
    assert build_graph([]) == []
