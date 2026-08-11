"""Lesson timeline and overlap representation (§5.4–5.5).

Both streams are mapped into a common lesson time (§5.4). Overlap is *not*
serialized into an artificial token order (§5.5): it is represented as
intersecting speaker intervals, and the selector receives overlap metadata so a
simultaneously started student utterance is not treated as a response to words
the teacher had not finished saying (§5.5, Appendix A.4).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Interval:
    """A half-open time interval ``[start, end)`` in lesson seconds."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"interval end {self.end} precedes start {self.start}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: "Interval", *, min_overlap: float = 0.0) -> bool:
        """True if the intersection with ``other`` is longer than ``min_overlap``."""
        return self.overlap_duration(other) > min_overlap

    def overlap_duration(self, other: "Interval") -> float:
        lo = max(self.start, other.start)
        hi = min(self.end, other.end)
        return max(0.0, hi - lo)

    def contains_time(self, t: float) -> bool:
        return self.start <= t < self.end

    def padded(self, pre: float, post: float | None = None) -> "Interval":
        """Grow the interval by ``pre``/``post`` seconds (§6.1 VAD padding).

        Clamped at zero on the low end; never mutates in place.
        """
        post = pre if post is None else post
        return Interval(max(0.0, self.start - pre), self.end + post)


def merge_intervals(intervals: list[Interval], *, gap: float = 0.0) -> list[Interval]:
    """Union of intervals, merging any that touch or sit within ``gap`` seconds."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for iv in ordered[1:]:
        last = merged[-1]
        if iv.start <= last.end + gap:
            merged[-1] = Interval(last.start, max(last.end, iv.end))
        else:
            merged.append(iv)
    return merged


def find_overlaps(
    a: list[Interval], b: list[Interval], *, min_overlap: float = 0.0
) -> list[tuple[int, int]]:
    """Indices ``(i, j)`` where ``a[i]`` and ``b[j]`` overlap.

    Used to mark cross-speaker overlap (teacher vs. student) so both intervals
    are preserved and flagged rather than interleaved (§5.5).
    """
    pairs: list[tuple[int, int]] = []
    b_sorted = sorted(range(len(b)), key=lambda j: b[j].start)
    for i, ai in enumerate(a):
        for j in b_sorted:
            bj = b[j]
            if bj.start >= ai.end:
                break
            if ai.overlaps(bj, min_overlap=min_overlap):
                pairs.append((i, j))
    return pairs


@dataclass
class ClockSync:
    """Cross-device clock offset/drift for mapping a track into lesson time (§5.4).

    ``lesson_t = local_t * (1 + drift) + offset``. Sample-perfect sync is not
    required for text ASR (§5.4); this is a coarse linear model estimated from
    periodic sample-counter / monotonic-timestamp checkpoints.
    """

    offset: float = 0.0
    drift: float = 0.0

    def to_lesson_time(self, local_t: float) -> float:
        return local_t * (1.0 + self.drift) + self.offset

    def map_interval(self, iv: Interval) -> Interval:
        return Interval(self.to_lesson_time(iv.start), self.to_lesson_time(iv.end))
