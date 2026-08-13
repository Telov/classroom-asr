"""Lightweight phonetic similarity used by the RAG index (§10.4).

The design doc stores *observed pronunciation evidence*, not only dictionary
phonemizations (§10.4), and matches by phonetic similarity. A production system
would embed pronunciations with a learned model; here we provide a dependency-
free baseline: normalized edit distance over IPA symbol sequences, with a small
articulatory feature relaxation so near-classes (e.g. /s/~/θ/, a first-class
Russian-accented-English confusion, Appendix A.2) don't cost a full unit.

Swap :func:`phonetic_similarity` for an embedding cosine when the ML backend is
wired; the RAG index only depends on the function signature.
"""

from __future__ import annotations

import unicodedata

# Substitution cost < 1.0 for phone pairs that are commonly confused in
# Russian-accented English (§1.1 phonological transfer). Symmetric; keyed on
# frozensets. This is intentionally small and hand-picked, not a full feature
# table — it exists so the RAG match for "three" said as /friː/ or /sriː/ still
# scores high (Appendix A.2, A.5).
_NEAR: dict[frozenset[str], float] = {
    frozenset({"θ", "s"}): 0.3,
    frozenset({"θ", "f"}): 0.3,
    frozenset({"θ", "t"}): 0.4,
    frozenset({"ð", "z"}): 0.3,
    frozenset({"ð", "d"}): 0.4,
    frozenset({"w", "v"}): 0.3,
    frozenset({"ŋ", "n"}): 0.4,
    frozenset({"ɪ", "i"}): 0.3,
    frozenset({" æ", "e"}): 0.4,
    frozenset({"ə", "ʌ"}): 0.3,
    frozenset({"ə", "a"}): 0.4,
    frozenset({"ɐ", "ə"}): 0.3,
    frozenset({"ʊ", "u"}): 0.3,
    frozenset({"r", "ɹ"}): 0.2,
}


def strip_ipa(ipa: str) -> list[str]:
    """Split an IPA string into base symbols, dropping delimiters/diacritics.

    We ignore ``/`` ``[`` ``]`` slashes/brackets, stress marks (ˈ ˌ), length
    (ː), and combining diacritics so surface transcription noise doesn't drown
    the phonetic signal.
    """
    out: list[str] = []
    for ch in ipa:
        if ch in "/[]ˈˌː.| \t":
            continue
        if unicodedata.combining(ch):
            continue
        out.append(ch)
    return out


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return _NEAR.get(frozenset({a, b}), 1.0)


def phone_edit_distance(a: list[str], b: list[str]) -> float:
    """Weighted edit distance over phone symbols (near-class subs cost < 1)."""
    n, m = len(a), len(b)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j - 1] + _sub_cost(a[i - 1], b[j - 1]),
                prev[j] + 1.0,
                cur[j - 1] + 1.0,
            )
        prev = cur
    return prev[m]


def phonetic_similarity(ipa_a: str, ipa_b: str) -> float:
    """Similarity in ``[0, 1]``; 1.0 is identical pronunciation.

    ``1 - dist / max(len_a, len_b)``. Empty-vs-empty is defined as 1.0.
    """
    a, b = strip_ipa(ipa_a), strip_ipa(ipa_b)
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    dist = phone_edit_distance(a, b)
    return max(0.0, 1.0 - dist / denom)


def best_phone_subsequence(
    expected_ipa: str, realized_ipa: str, *, flank: int = 4
) -> tuple[float, str]:
    """Align an expected pronunciation to its best local region in a longer phone stream.

    The realized input may cover a whole acoustic window containing neighbouring words. Prefix
    and suffix phones are therefore free, while edits inside the selected region use the same
    accent-aware costs as :func:`phonetic_similarity`. The returned excerpt includes a small
    phone flank for judge context but never returns the full long window.
    """
    expected = strip_ipa(expected_ipa)
    realized = strip_ipa(realized_ipa)
    if not expected or not realized:
        return 0.0, ""

    # Semi-global edit distance: matching may start/end anywhere in ``realized``. Alongside each
    # cost retain the start index so the winning local excerpt can be recovered without a full
    # O(m*n) backpointer matrix.
    prev = [0.0] * (len(realized) + 1)
    prev_start = list(range(len(realized) + 1))
    for i, phone in enumerate(expected, 1):
        cur = [float(i)] + [0.0] * len(realized)
        cur_start = [0] + [0] * len(realized)
        for j, observed in enumerate(realized, 1):
            candidates = (
                (prev[j - 1] + _sub_cost(phone, observed), prev_start[j - 1]),
                (prev[j] + 1.0, prev_start[j]),       # expected phone deleted
                (cur[j - 1] + 1.0, cur_start[j - 1]), # extra realized phone
            )
            cur[j], cur_start[j] = min(candidates, key=lambda item: item[0])
        prev, prev_start = cur, cur_start

    end = min(range(1, len(prev)), key=lambda j: prev[j])
    start = min(prev_start[end], end)
    aligned_len = max(1, end - start)
    similarity = max(0.0, 1.0 - prev[end] / max(len(expected), aligned_len))
    lo, hi = max(0, start - flank), min(len(realized), end + flank)
    return similarity, "".join(realized[lo:hi])
