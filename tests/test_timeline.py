from classroom_asr.timeline import (
    ClockSync,
    Interval,
    find_overlaps,
    merge_intervals,
)


def test_overlap_and_duration():
    a = Interval(0.0, 2.0)
    b = Interval(1.5, 3.0)
    assert a.overlaps(b)
    assert abs(a.overlap_duration(b) - 0.5) < 1e-9
    assert not a.overlaps(Interval(2.0, 3.0))  # half-open, touching is not overlap


def test_min_overlap_threshold():
    a = Interval(0.0, 1.0)
    b = Interval(0.99, 2.0)
    assert a.overlaps(b)                      # 0.01 s overlap
    assert not a.overlaps(b, min_overlap=0.05)


def test_padded_clamps_at_zero():
    iv = Interval(0.2, 1.0).padded(0.5)
    assert iv.start == 0.0
    assert abs(iv.end - 1.5) < 1e-9


def test_invalid_interval():
    try:
        Interval(2.0, 1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for end < start")


def test_merge_intervals_with_gap():
    ivs = [Interval(0, 1), Interval(1.1, 2), Interval(5, 6)]
    merged = merge_intervals(ivs, gap=0.2)
    assert merged == [Interval(0, 2), Interval(5, 6)]


def test_find_overlaps():
    a = [Interval(0, 2), Interval(10, 12)]
    b = [Interval(1, 3), Interval(20, 21)]
    assert find_overlaps(a, b) == [(0, 0)]


def test_clock_sync_linear_map():
    cs = ClockSync(offset=1.0, drift=0.0)
    assert cs.to_lesson_time(5.0) == 6.0
    mapped = cs.map_interval(Interval(0.0, 1.0))
    assert mapped == Interval(1.0, 2.0)
