import numpy as np

from classroom_asr.backends import iter_overlapping_silence_chunks, merge_overlapping_text


def test_overlapping_windows_cover_every_core_sample_once_and_stay_bounded():
    sr = 1_000
    waveform = np.ones(55 * sr, dtype=np.float32)
    windows = list(iter_overlapping_silence_chunks(waveform, sr))

    assert windows[0][2] == 0
    assert windows[-1][3] == len(waveform)
    assert all(a[3] == b[2] for a, b in zip(windows, windows[1:]))
    assert all(len(window[4]) <= 25 * sr for window in windows)
    assert all(a[1] > b[0] for a, b in zip(windows, windows[1:]))


def test_merge_overlapping_text_uses_longest_case_and_punctuation_normalized_match():
    parts = [
        "Это очень важная Фраза, на границе",
        "фраза на границе окна и дальше",
        "окна и дальше без потери",
    ]
    assert merge_overlapping_text(parts) == (
        "Это очень важная Фраза, на границе окна и дальше без потери"
    )


def test_merge_overlapping_text_retains_disagreements():
    assert merge_overlapping_text(["one boundary reading", "different continuation"]) == (
        "one boundary reading different continuation"
    )


def test_merge_overlapping_text_does_not_drop_a_single_coincidental_word():
    assert merge_overlapping_text(["we finished that", "that starts another thought"]) == (
        "we finished that that starts another thought"
    )
