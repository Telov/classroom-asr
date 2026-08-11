from classroom_asr.normalize import Normalizer
from classroom_asr.metrics import score

SCORE = Normalizer(fold_numbers=True, fold_spelling=True)


def test_cardinals_fold_to_digits():
    n = SCORE
    assert n.tokens("i have fifteen") == ["i", "have", "15"]
    assert n.tokens("twenty five dollars") == ["25", "dollars"]
    assert n.tokens("one hundred") == ["100"]
    assert n.tokens("two thousand") == ["2000"]
    # digit form already matches the folded word form
    assert n.tokens("15") == ["15"]


def test_ordinals_and_spelling():
    n = SCORE
    assert n.tokens("september tenth") == ["september", "10th"]
    assert n.tokens("cause it's ok") == ["because", "it's", "okay"]
    assert n.tokens("mm-hm") == ["mmhm"]


def test_folding_removes_formatting_only_errors():
    # "fifteen" vs "15" and "September tenth" vs "September 10th" should be 0 WER
    # under the scoring normalizer, but the default normalizer counts them wrong.
    assert score("it was fifteen", "it was 15", norm=SCORE).wer == 0.0
    assert score("September tenth", "September 10th", norm=SCORE).wer == 0.0
    assert score("it was fifteen", "it was 15").wer > 0.0  # default: still an error


def test_fillers_are_preserved():
    # crucial: we do NOT drop fillers — deletion of them is a real metric.
    assert SCORE.tokens("um i like uh it") == ["um", "i", "like", "uh", "it"]
