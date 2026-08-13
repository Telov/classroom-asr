from classroom_asr.normalize import Normalizer
from classroom_asr.rover import build_graph, NULL
from classroom_asr.selector import (
    build_decisions, format_batch, parse_batch, generated_token_ids, assemble, select_transcript,
)

N = Normalizer()


def _graph(*texts):
    return build_graph([N.tokens(t) for t in texts])


def test_only_contested_slots_become_decisions():
    # "went" has a 2/3 majority -> frozen; a 1-1-1 split would be contested.
    g = _graph("i went to the store", "i want to the store", "i went to the store")
    assert build_decisions(g) == []                     # majority everywhere -> nothing to ask

    g2 = _graph("their house", "there house", "they're house")   # 3-way split on word 0
    decs2 = build_decisions(g2)
    assert len(decs2) == 1
    toks = {tok for _, tok in decs2[0].options}
    assert toks == {"their", "there", "they're"}


def test_prompt_and_parse_roundtrip():
    g = _graph("their house", "there house", "they're house")
    decs = build_decisions(g)
    prompt = format_batch(decs)
    assert "1. context:" in prompt and "[?]" in prompt
    # letters are assigned in the option order shown in the prompt
    letter_for = {tok: L for L, tok in decs[0].options}
    choices = parse_batch(f"1:{letter_for['there']}", decs)
    assert choices == {decs[0].slot: "there"}


def test_prompt_includes_deduplicated_acoustic_evidence():
    g = _graph("i thought so", "i sought so", "i fought so")
    base = build_decisions(g)
    evidence = {
        base[0].slot: [
            "PhoneticXEUS p1 (p=1.000): /aɪ sɔːt/",
            "wav2vec2-phone p1 (p=0.740): /aɪ s ɔː t/",
        ]
    }
    decs = build_decisions(g, evidence_by_slot=evidence)
    prompt = format_batch([decs[0], decs[0]])

    assert prompt.count("E1:") == 1
    assert prompt.count("acoustic evidence: E1") == 2
    assert "/aɪ sɔːt/" in prompt


def test_parser_accepts_bounded_markdown_and_json_variants():
    g = _graph("their house", "there house", "they're house")
    decs = build_decisions(g)
    letter_for = {tok: letter for letter, tok in decs[0].options}
    wanted = letter_for["there"]

    variants = [
        f"- 1: option {wanted}",
        f"**answers**\n1 -> {wanted}",
        '{"1": "' + wanted + '"}',
        '[{"item": 1, "choice": "' + wanted + '"}]',
    ]
    for response in variants:
        assert parse_batch(response, decs) == {decs[0].slot: "there"}


def test_generated_token_ids_supports_full_sequence_and_completion_only():
    prompt = [[10, 20, 30]]
    assert generated_token_ids([[10, 20, 30, 40, 50]], prompt) == [40, 50]
    assert generated_token_ids([[40, 50]], prompt) == [40, 50]


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
    # pivot has "really"; an even really/truly/drop/drop split is contested (∅ = 2 of 4, no
    # strict majority) -> a decision whose options include NULL (drop) must be offered.
    g = _graph("i really do", "i do", "i truly do", "i do")
    decs = build_decisions(g)
    drop_slots = [d for d in decs if any(tok is NULL for _, tok in d.options)
                  and {"really", "truly"} & {tok for _, tok in d.options}]
    assert drop_slots, "the really/truly/drop slot should be contested and offer ∅"
