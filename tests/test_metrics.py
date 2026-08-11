from classroom_asr.metrics import (
    Op,
    align,
    candidate_oracle_wer,
    cer,
    corpus_wer,
    expected_calibration_error,
    score,
    slice_deletion_rate,
)


def test_wer_counts_sub_del_ins():
    # ref: a b c d   hyp: a x c   -> 1 sub (b->x), 1 del (d)
    r = score("a b c d", "a x c")
    assert r.ref_len == 4
    assert r.substitutions == 1
    assert r.deletions == 1
    assert r.insertions == 0
    assert abs(r.wer - 0.5) < 1e-9


def test_insertion():
    r = score("a b", "a b c")
    assert r.insertions == 1
    assert abs(r.wer - 0.5) < 1e-9


def test_perfect_match():
    assert score("hello world", "hello world").wer == 0.0


def test_empty_ref():
    r = score("", "")
    assert r.wer == 0.0
    r2 = score("", "spurious")
    assert r2.insertions == 1


def test_corpus_micro_average():
    pairs = [("a b", "a b"), ("c d e", "c d x")]  # 0 errors + 1 sub over 5 tokens
    assert abs(corpus_wer(pairs).wer - 0.2) < 1e-9


def test_cer():
    assert abs(cer("cat", "car") - 1 / 3) < 1e-9


def test_slice_deletion_rate():
    # ref tokens indices 0..3; token 2 ("c") deleted and it is in the slice
    r = score("a b c d", "a b d")
    marked = {2}  # e.g. a short/quiet word
    sd = slice_deletion_rate(r, marked, slice_name="short")
    assert sd.marked == 1
    assert sd.deleted == 1
    assert sd.rate == 1.0


def test_candidate_oracle_headroom():
    # Two spans; the 1-best (first candidate) is wrong on span 2 but the correct
    # answer is present as a lower candidate -> oracle beats baseline.
    refs = ["i thought", "aboba"]
    cands = [["i thought", "i sought"], ["above a", "aboba"]]
    res = candidate_oracle_wer(refs, cands)
    assert res.oracle.wer == 0.0
    assert res.baseline.wer > 0.0
    assert res.headroom > 0.0
    assert res.chosen == ("i thought", "aboba")


def test_candidate_oracle_explicit_baseline():
    refs = ["to"]
    cands = [["to"]]              # empty 1-best degraded away, only "to" survives
    res = candidate_oracle_wer(refs, cands, baseline_choice=[""])
    assert res.oracle.wer == 0.0
    assert res.baseline.deletions == 1


def test_ece_perfect_calibration():
    conf = [0.05, 0.95]
    correct = [False, True]
    assert expected_calibration_error(conf, correct, bins=10) < 0.1


def test_align_ops_shape():
    ops = align(["a", "b"], ["a", "b"])
    assert all(o.op is Op.MATCH for o in ops)
