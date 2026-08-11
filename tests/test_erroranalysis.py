from classroom_asr.erroranalysis import error_report, worst_utterances
from classroom_asr.metrics import candidate_oracle_wer, corpus_wer


def test_error_report_counts_and_top_deletions():
    pairs = [("i did not go", "i go"), ("we were here", "we were here")]
    rep = error_report(pairs)
    # "did" and "not" deleted from the first utterance
    assert rep.deletions == 2
    deleted = dict(rep.top_deletions)
    assert deleted.get("did") == 1 and deleted.get("not") == 1
    assert rep.substitutions == 0 and rep.insertions == 0


def test_error_report_substitution_pair():
    rep = error_report([("we were talking", "we was talking")])
    assert ("were -> was", 1) in rep.top_substitutions


def test_worst_utterances_order():
    pairs = [("a b c d", "a b c d"), ("e f g h", "x y z w")]
    worst = worst_utterances(pairs, n=1)
    assert worst[0][0] == "e f g h"  # the fully-wrong one ranks first


def test_oracle_baseline_matches_corpus_wer():
    # The oracle's baseline must equal a corpus_wer over the same 1-best list
    # (the fix for the confusing headroom).
    refs = ["so um i did not go there", "we were talking about it"]
    cands = [["so i did not go there", "so um i did not go there"],
             ["we were talking about it"]]
    baseline = [c[0] for c in cands]
    res = candidate_oracle_wer(refs, cands, baseline_choice=baseline)
    cw = corpus_wer(zip(refs, baseline))
    assert abs(res.baseline.wer - cw.wer) < 1e-9
    assert res.oracle.wer <= res.baseline.wer
    assert res.headroom >= 0.0
