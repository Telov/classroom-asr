from classroom_asr.phonetics import best_phone_subsequence


def test_best_phone_subsequence_localizes_without_returning_the_full_window():
    score, excerpt = best_phone_subsequence("/aɪ sɔːt/", "/ðə tiːtʃə aɪ sɔːt ɪt əgɛn/")

    assert score == 1.0
    assert "aɪsɔt" in excerpt
    assert len(excerpt) < len("ðətiːtʃəaɪsɔːtɪtəgɛn")


def test_best_phone_subsequence_uses_accent_aware_costs():
    score, excerpt = best_phone_subsequence("/θriː/", "/aɪ sɛd sriː taɪmz/")

    assert score > 0.8
    assert "sri" in excerpt
