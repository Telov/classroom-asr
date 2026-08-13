from classroom_asr.rover import build_graph, fuse, realizable_oracle_tokens, NULL
from classroom_asr.normalize import Normalizer
from classroom_asr.metrics import score

N = Normalizer()  # plain: lowercase + edge punct, no number/spelling folding


def _words(graph):
    return [s for s in graph if s.kind == "word"]


def test_majority_fixes_a_substitution():
    # two branches say "went", one says "want" -> vote picks "went"
    out = fuse(["i went to the store", "i want to the store", "i went to the store"], norm=N)
    assert out == "i went to the store"


def test_majority_deletion_drops_a_word():
    # pivot (longest) has "really"; the other two dropped it -> NULL wins, word dropped
    out = fuse(["i really went home", "i went home", "i went home"], norm=N)
    assert out == "i went home"


def test_pivot_keeps_word_on_a_tie():
    # 1 keep vs 1 drop is a tie -> conservative: keep the pivot's word
    out = fuse(["i really went home", "i went home"], norm=N)
    assert out == "i really went home"


def test_fusion_beats_the_best_single_branch():
    ref = "the quick brown fox jumps"
    a = "the quick brown box jumps"      # 1 sub (fox->box)
    b = "the quick brown fox jumped"     # 1 sub (jumps->jumped)
    c = "the quick brown fox jumps"      # perfect
    fused = fuse([a, b, c], norm=N)
    assert fused == ref                  # majority recovers both -> 0 WER
    assert score(ref, fused, norm=N).wer == 0.0
    assert score(ref, a, norm=N).wer > 0.0


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


def test_empty_and_single():
    assert fuse([], norm=N) == ""
    assert fuse(["", "  "], norm=N) == ""
    assert fuse(["only one branch here"], norm=N) == "only one branch here"
