"""Spoken-number / ordinal / spelling folding for *scoring* only (§18, §29).

The CORAAL error analysis showed a large share of "errors" were pure formatting:
``fifteen`` vs ``15``, ``September tenth`` vs ``September 10th``, ``cause`` vs
``because``, ``OK`` vs ``Okay``. Those are transcription-convention differences,
not acoustic mistakes, and they let the oracle "win" by picking whichever branch
happened to match CORAAL's convention (§18.2 — a measurement artifact).

This module canonicalizes those forms so every system is compared on the same
footing (§29). Crucially it is **verbatim-preserving**: it does NOT drop fillers
(``um``/``uh``/``like``) or function words, because deletion of those is a
first-class metric for this project (§2, §18.1) — unlike the standard Whisper
normalizer, which removes them.

Scope is deliberately bounded (cardinals 0–999,999; ordinals 1st–31st for dates;
a small spelling map). Years like "nineteen seventy-nine" are left alone; folding
them correctly is a follow-up.
"""

from __future__ import annotations

_UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}

_NUMWORDS = set(_UNITS) | set(_TENS) | set(_SCALES)

# Ordinals 1st–31st (covers dates) + the tens ordinals.
_ORDINAL = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th", "thirtieth": "30th",
    "fortieth": "40th", "fiftieth": "50th",
}
for _t, _v in {"twenty": 20, "thirty": 30}.items():
    for _u, _uv in {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9}.items():
        _ORDINAL[f"{_t}-{_u}"] = f"{_v + _uv}th"

# Unambiguous spelling variants (safe to canonicalize; NOT filler removal).
_SPELLING = {
    "ok": "okay", "o.k.": "okay", "'cause": "because", "cause": "because",
    "till": "until", "mm-hm": "mmhm", "mm-hmm": "mmhm", "mhm": "mmhm",
    "uh-huh": "uhhuh", "uh-uh": "uhuh",
}


def _parse_cardinal(words: list[str]) -> int | None:
    total = current = 0
    used = False
    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in ("thousand", "million"):
            total += (current or 1) * _SCALES[w]
            current = 0
        else:
            return None
        used = True
    return (total + current) if used else None


def _split_hyphen_numbers(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        if "-" in t and all(p in _NUMWORDS for p in t.split("-") if p):
            out.extend(p for p in t.split("-") if p)
        else:
            out.append(t)
    return out


def fold(tokens: list[str], *, numbers: bool = True, spelling: bool = True) -> list[str]:
    """Return a normalized copy of ``tokens`` for fair scoring."""
    if spelling:
        tokens = [_SPELLING.get(t, t) for t in tokens]
        tokens = [_ORDINAL.get(t, t) for t in tokens]
    if not numbers:
        return tokens

    tokens = _split_hyphen_numbers(tokens)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _NUMWORDS:
            j = i
            while j < len(tokens) and tokens[j] in _NUMWORDS:
                j += 1
            val = _parse_cardinal(tokens[i:j])
            out.append(str(val) if val is not None else " ".join(tokens[i:j]))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out
