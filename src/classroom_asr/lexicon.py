"""Session and persistent pronunciation-indexed lexicons (§10.4–10.5).

The session lexicon grows *during* a lesson and the persistent one *across*
lessons (§10.4). Both index entries by observed pronunciation so a later
mention of a nonce word can be matched to an earlier one by sound — the core of
the Appendix A.3 "aboba" example, where an early ``[ɐbobə]`` is reconciled with
a later explicit spelling ``a-b-o-b-a``.

Key distinction (§10.5): if a word was already typed / spelled / introduced, we
can recover the *exact* known spelling; if the convention is genuinely unknown
we can only offer a plausible graphemization. Entries therefore carry an
``exact`` flag so evaluation can separate "known OOV exact recovery" from "novel
P2G plausibility" (§10.5, §18.1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .datamodel import RagMatch
from .phonetics import phonetic_similarity
from .types import Role


@dataclass
class LexiconEntry:
    """One term with its canonical spelling and observed pronunciations."""

    canonical_spelling: str
    pronunciations: list[str] = field(default_factory=list)   # observed IPA strings
    teacher_prons: list[str] = field(default_factory=list)
    student_prons: list[str] = field(default_factory=list)
    exact: bool = False        # spelling was explicitly established (§10.5)
    nonce: bool = False        # a weird/OOV term, not an ordinary dictionary word
    count: int = 0
    first_seen_time: float | None = None

    def observe(self, ipa: str, role: Role | None, time: float | None) -> None:
        if ipa and ipa not in self.pronunciations:
            self.pronunciations.append(ipa)
        if role is Role.TEACHER and ipa and ipa not in self.teacher_prons:
            self.teacher_prons.append(ipa)
        elif role is Role.STUDENT and ipa and ipa not in self.student_prons:
            self.student_prons.append(ipa)
        self.count += 1
        if self.first_seen_time is None and time is not None:
            self.first_seen_time = time

    def best_similarity(self, ipa: str) -> float:
        if not self.pronunciations:
            return 0.0
        return max(phonetic_similarity(ipa, p) for p in self.pronunciations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_spelling": self.canonical_spelling,
            "pronunciations": self.pronunciations,
            "teacher_prons": self.teacher_prons,
            "student_prons": self.student_prons,
            "exact": self.exact,
            "nonce": self.nonce,
            "count": self.count,
            "first_seen_time": self.first_seen_time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LexiconEntry":
        return cls(
            canonical_spelling=d["canonical_spelling"],
            pronunciations=list(d.get("pronunciations", [])),
            teacher_prons=list(d.get("teacher_prons", [])),
            student_prons=list(d.get("student_prons", [])),
            exact=bool(d.get("exact", False)),
            nonce=bool(d.get("nonce", False)),
            count=int(d.get("count", 0)),
            first_seen_time=d.get("first_seen_time"),
        )


class Lexicon:
    """A pronunciation-indexed term store with phonetic retrieval (RAG)."""

    def __init__(self, *, persistent: bool = False) -> None:
        self._entries: dict[str, LexiconEntry] = {}
        self.persistent = persistent

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, term: str) -> bool:
        return term in self._entries

    def get(self, term: str) -> LexiconEntry | None:
        return self._entries.get(term)

    def add_term(
        self,
        term: str,
        *,
        canonical_spelling: str | None = None,
        exact: bool = False,
        nonce: bool = False,
    ) -> LexiconEntry:
        entry = self._entries.get(term)
        if entry is None:
            entry = LexiconEntry(canonical_spelling=canonical_spelling or term,
                                 exact=exact, nonce=nonce)
            self._entries[term] = entry
        if exact:
            # An explicit spelling event upgrades the entry (§10.5).
            entry.exact = True
            if canonical_spelling:
                entry.canonical_spelling = canonical_spelling
        if nonce:
            entry.nonce = True
        return entry

    def observe(
        self,
        term: str,
        ipa: str,
        *,
        role: Role | None = None,
        time: float | None = None,
        canonical_spelling: str | None = None,
        exact: bool = False,
        nonce: bool = False,
    ) -> LexiconEntry:
        """Record a pronunciation observation for ``term`` (grows the index)."""
        entry = self.add_term(term, canonical_spelling=canonical_spelling,
                              exact=exact, nonce=nonce)
        entry.observe(ipa, role, time)
        return entry

    def query(self, ipa: str, *, top_k: int = 5, min_similarity: float = 0.5) -> list[RagMatch]:
        """Retrieve terms whose observed pronunciation matches ``ipa`` (§10.4)."""
        scored: list[tuple[float, LexiconEntry, str]] = []
        for term, entry in self._entries.items():
            sim = entry.best_similarity(ipa)
            if sim >= min_similarity:
                scored.append((sim, entry, term))
        scored.sort(key=lambda t: (-t[0], t[2]))
        return [
            RagMatch(
                term=term,
                similarity=sim,
                canonical_spelling=entry.canonical_spelling,
                persistent=self.persistent,
            )
            for sim, entry, term in scored[:top_k]
        ]

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent": self.persistent,
            "entries": {t: e.to_dict() for t, e in self._entries.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Lexicon":
        lex = cls(persistent=bool(d.get("persistent", False)))
        lex._entries = {t: LexiconEntry.from_dict(e) for t, e in d.get("entries", {}).items()}
        return lex

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def combined_query(
    session: Lexicon,
    persistent: Lexicon | None,
    ipa: str,
    *,
    top_k: int = 5,
    min_similarity: float = 0.5,
) -> list[RagMatch]:
    """Query session and persistent lexicons and merge, session winning ties.

    The session lexicon reflects *this* lesson's evidence and should outrank a
    stale persistent match at equal similarity (§10.4/§15.2 future-context).
    """
    matches = session.query(ipa, top_k=top_k, min_similarity=min_similarity)
    if persistent is not None:
        matches += persistent.query(ipa, top_k=top_k, min_similarity=min_similarity)
    # Dedup by term keeping the highest similarity; session sorts first on ties
    # because it was extended first and sort is stable on equal keys.
    best: dict[str, RagMatch] = {}
    for m in matches:
        cur = best.get(m.term)
        if cur is None or m.similarity > cur.similarity:
            best[m.term] = m
    return sorted(best.values(), key=lambda m: (-m.similarity, m.term))[:top_k]
