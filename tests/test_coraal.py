from classroom_asr.data.coraal import (
    CoraalSegment,
    clean_content,
    parse_transcript,
    select_segments,
)


def test_clean_keeps_disfluencies_strips_markup():
    raw = "Okay so um (pause 0.34) yeah <laugh> I [think] /RD-NAME-2/ went"
    # fillers kept ("um", "yeah", "think"); pause/laugh/redaction removed;
    # brackets dropped but the word kept.
    assert clean_content(raw) == "Okay so um yeah I think went"


def test_clean_pure_annotation_becomes_empty():
    assert clean_content("(pause 1.20)") == ""
    assert clean_content("<laugh>") == ""


def test_parse_transcript(tmp_path):
    p = tmp_path / "DCB_se1.txt"
    p.write_text(
        "Line\tSpkr\tStTime\tContent\tEnTime\n"
        "1\tINT_dcb\t0.0000\tSo tell me (pause 0.5) about it\t3.2000\n"
        "2\tDCB_se1_ag1\t3.2000\tWell um I didn't went there\t6.5000\n"
        "3\tDCB_se1_ag1\t6.5000\t<laugh>\t7.0000\n",
        encoding="utf-8",
    )
    segs = parse_transcript(p)
    assert len(segs) == 3
    assert segs[0].is_interviewer
    assert segs[1].text == "Well um I didn't went there"   # verbatim, not cleaned
    assert segs[2].text == ""                              # pure non-linguistic


def test_select_segments_budget_and_filters():
    segs = [
        CoraalSegment(1, "INT", 0.0, 3.0, "", ""),          # empty -> drop
        CoraalSegment(2, "SUB", 3.0, 5.0, "a b c", "a b c"),
        CoraalSegment(3, "SUB", 5.0, 5.1, "hi", "hi"),      # too short -> drop
        CoraalSegment(4, "SUB", 6.0, 12.0, "x y", "x y"),
    ]
    kept = select_segments(segs, max_seconds=5.0, min_dur=0.4)
    # segment 2 (2 s) fits; segment 4 (6 s) would exceed 5 s budget -> stop
    assert [s.line for s in kept] == [2]
