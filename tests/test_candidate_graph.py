from classroom_asr.candidate_graph import build_graph, realizable_oracle_tokens, NULL
from classroom_asr.normalize import Normalizer
from classroom_asr.metrics import score

N = Normalizer()  # plain: lowercase + edge punct, no number/spelling folding


def _words(graph):
    return [s for s in graph if s.kind == "word"]


def test_designated_backbone_is_the_graph_pivot():
    graph = build_graph(
        [N.tokens("qwen short"), N.tokens("another much longer transcript")], pivot_index=0)
    assert [slot.pivot for slot in graph if slot.kind == "word"] == ["qwen", "short"]


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
    smaller_text = " ".join(realizable_oracle_tokens(smaller, ref))
    larger_text = " ".join(realizable_oracle_tokens(larger, ref))

    assert score(" ".join(ref), larger_text, norm=N).wer <= score(
        " ".join(ref), smaller_text, norm=N
    ).wer


def test_empty_graph():
    assert build_graph([]) == []
