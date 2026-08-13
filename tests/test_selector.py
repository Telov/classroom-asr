from classroom_asr.normalize import Normalizer
from classroom_asr.rover import build_graph, NULL
from classroom_asr.selector import (
    build_decisions, format_batch, parse_batch, assemble, select_transcript,
)

N = Normalizer()


def _graph(*texts):
    return build_graph([N.tokens(t) for t in texts])


def test_only_contested_slots_become_decisions():
    # "went" has a 2/3 majority -> frozen; a 1-1-1 split would be contested.
    g = _graph("i went to the store", "i want to the store", "i went to the store")
    decs = build_decisions(g)
    assert decs == []                                   # majority everywhere -> nothing to ask

    g2 = _graph("their house", "there house", "they're house")   # 3-way split on word 0
    decs2 = build_decisions(g2)
    assert len(decs2) == 1 and decs2[0].slot == 0
    toks = {tok for _, tok in decs2[0].options}
    assert toks == {"their", "there", "they're"}


def test_prompt_and_parse_roundtrip():
    g = _graph("their house", "there house", "they're house")
    decs = build_decisions(g)
    prompt = format_batch(decs)
    assert "1. context:" in prompt and "[?]" in prompt
    # letters are assigned in the option order shown in the prompt
    letter_for = {tok: L for L, tok in decs[0].options}
    resp = f"1:{letter_for['there']}"
    choices = parse_batch(resp, decs)
    assert choices == {0: "there"}


def test_llm_choice_overrides_only_its_slot():
    g = _graph("their house", "there house", "they're house")
    decs = build_decisions(g)
    letter_for = {tok: L for L, tok in decs[0].options}
    choices = parse_batch(f"1:{letter_for['there']}", decs)
    assert assemble(g, choices) == ["there", "house"]


def test_bad_llm_output_falls_back_to_rover():
    # llm returns garbage -> abstain -> ROVER majority ("went" 2/3) stands
    g_texts = ["i went home now", "i want home now", "i went home now"]
    out, n, nc = select_transcript(g_texts, lambda p: "no idea", norm=N)
    assert out == "i went home now"                     # unchanged by the useless judge
    assert nc == 0                                       # nothing parsed -> nothing decided

    # a working judge flips a genuinely contested slot
    def judge(prompt):
        # answer "A" for every item; option A is the top-voted token, so this is a no-op-ish
        import re
        return "\n".join(f"{m}:A" for m in re.findall(r"^(\d+)\.", prompt, re.M))
    out2, n2, nc2 = select_transcript(["a b c", "a x c", "a y c"], judge, norm=N)
    assert out2.split()[0] == "a" and out2.split()[-1] == "c"
    assert nc2 >= 1                                      # the working judge decided the contested slot


def test_drop_candidate_is_offered_when_branches_delete():
    # pivot has "really"; two branches drop it -> NULL is a candidate (but 2/3 is a majority,
    # so it's frozen, not contested). Force contest with an even split:
    g = _graph("i really do", "i do", "i truly do", "i do")   # slot1: really/∅/truly/∅ = 1/2/1
    decs = build_decisions(g)
    # slot for word-after-"i": ∅ has 2 votes of 4 -> not a strict majority -> contested
    slot1 = [d for d in decs if d.slot == 1]
    assert slot1, "the really/truly/drop slot should be contested"
    assert any(tok is NULL for _, tok in slot1[0].options)
