"""Synthetic two-channel lesson mirroring the Appendix A error cases.

This is scripted ground truth (annotator-style, Appendix C) used by the demo and
tests. It exercises exactly the behaviors the design doc says must work:

* A.1 — "I didn't went there" must NOT be cleaned to "I didn't go there".
* A.2 — "three" realized as /friː/: orthography ``three``, realized phones /friː/.
* A.3 — nonce "aboba" first heard early, spelled out by the teacher later; the
        earlier ambiguous occurrence is repaired by that future context (§15.2).
* A.4 — simultaneous teacher/student speech kept as two intervals, not merged.
* A.5 — a very quiet "to" that the 1-best drops but the candidate pool recovers.
"""

from __future__ import annotations

from .pipeline.base import LessonInput, SpeechSegment, SpokenUnit
from .timeline import Interval
from .types import AudioSource, Language, Role


def _seg(role, source, start, end, unit) -> SpeechSegment:
    return SpeechSegment(interval=Interval(start, end), role=role,
                         audio_source=source, truth=unit)


def example_lesson() -> LessonInput:
    T, S = Role.TEACHER, Role.STUDENT
    TR, SR = AudioSource.TEACHER_RAW, AudioSource.STUDENT_RAW
    en, ru = Language.EN, Language.RU

    teacher = [
        _seg(T, TR, 1.0, 2.2, SpokenUnit("Say three.", "seɪ θriː", en, energy=0.9)),
        # A.4 overlap partner (starts before the student's response)
        _seg(T, TR, 30.20, 33.10,
             SpokenUnit("Нет, я имею в виду three", "net ja θriː", ru, energy=0.9)),
        # A.3 the teacher spells the nonce out — establishes exact spelling (later
        # in time). The utterance itself is an ordinary sentence (verbatim text
        # preserved); ``spelling_event`` seeds the whole-lesson lexicon.
        _seg(T, TR, 41.0, 44.0,
             SpokenUnit("aboba пишется a b o b a", "ɐbobə", ru, energy=0.9,
                        canonical_spelling="aboba", spelling_event=True)),
    ]

    student = [
        # A.3 first, ambiguous occurrence of the nonce (before the spelling event)
        _seg(S, SR, 8.0, 8.6,
             SpokenUnit("aboba", "ɐbobə", ru, energy=0.5,
                        is_nonce=True, canonical_spelling="aboba")),
        # A.5 very quiet function word "to"
        _seg(S, SR, 10.0, 10.25, SpokenUnit("to", "tu", en, energy=0.2)),
        # A.1 grammar must be preserved verbatim
        _seg(S, SR, 12.0, 14.0,
             SpokenUnit("I didn't went there", "aɪ dɪdnt went ðer", en, energy=0.9)),
        # A.2 realized /friː/ for target "three"
        _seg(S, SR, 2.5, 3.2, SpokenUnit("three", "friː", en, energy=0.9)),
        # A.4 overlapping response
        _seg(S, SR, 32.05, 33.42, SpokenUnit("А three понял", "a θriː ponʲal", ru, energy=0.9)),
    ]
    return LessonInput(lesson_id="lesson_20260811_demo", teacher=teacher, student=student)
