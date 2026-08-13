from classroom_asr.rover import build_graph, fuse, NULL
from classroom_asr.normalize import Normalizer
from classroom_asr.metrics import score

N = Normalizer()  # plain: lowercase + edge punct, no number/spelling folding


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
    slots = build_graph([N.tokens("a b c"), N.tokens("a x c")])
    assert slots[0].agreed and slots[2].agreed        # all branches agree on "a" and "c"
    assert not slots[1].agreed                         # b vs x -> the LLM's job
    assert NULL not in slots[1].votes


def test_empty_and_single():
    assert fuse([], norm=N) == ""
    assert fuse(["", "  "], norm=N) == ""
    assert fuse(["only one branch here"], norm=N) == "only one branch here"
