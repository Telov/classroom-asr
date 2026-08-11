"""Reference normalization for scoring (§18, §29).

Two hard rules from the design doc:

* The **canonical transcript is verbatim** (§1.2, §9.2): fillers, repetitions,
  false starts, grammar errors, code-switches, and nonce forms are preserved.
  Normalization such as "twenty twenty six" -> "2026" is a *separate reversible
  display layer*, never part of the canonical text (§9.2).
* Scoring must run every system (ours, Scribe v2, open baselines) through
  **exactly the same** reference normalization/scoring pipeline (§29).

So this normalizer is used only at *scoring time* to make WER comparable across
systems; it never rewrites stored canonical text. It stays deliberately light:
lowercasing and punctuation stripping are opt-in, and it does no ITN, no filler
removal, and no grammar repair.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Punctuation to drop when ``strip_punct`` is on. Apostrophes/hyphens inside a
# word are kept by default so "didn't" / "code-switch" stay one token.
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
_WS = re.compile(r"\s+", re.UNICODE)


@dataclass(frozen=True)
class Normalizer:
    """Configurable, side-effect-free scoring normalizer.

    Defaults are conservative so two systems are compared fairly without
    silently "cleaning" learner errors.
    """

    lowercase: bool = True
    strip_edge_punct: bool = True
    nfc: bool = True                 # canonicalize Unicode (Cyrillic/Latin mix)
    collapse_whitespace: bool = True
    # Verbatim-preserving formatting folds for fair cross-system scoring (§29):
    # numbers/ordinals -> digits, a small spelling map. These do NOT drop fillers
    # or function words (deletion of those stays a metric — §18.1). Off by default
    # so stored/canonical text is untouched; enable only at scoring time.
    fold_numbers: bool = False
    fold_spelling: bool = False

    def word(self, token: str) -> str:
        t = token
        if self.nfc:
            t = unicodedata.normalize("NFC", t)
        if self.lowercase:
            t = t.lower()
        if self.strip_edge_punct:
            t = _EDGE_PUNCT.sub("", t)
        return t

    def tokens(self, text: str) -> list[str]:
        """Whitespace tokenization + per-token normalization, dropping empties."""
        if self.collapse_whitespace:
            text = _WS.sub(" ", text).strip()
        out = []
        for raw in text.split(" "):
            w = self.word(raw)
            if w:
                out.append(w)
        if self.fold_numbers or self.fold_spelling:
            from .numnorm import fold
            out = fold(out, numbers=self.fold_numbers, spelling=self.fold_spelling)
        return out

    def chars(self, text: str) -> list[str]:
        """Character sequence for CER (whitespace collapsed to single spaces)."""
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.lower()
        if self.collapse_whitespace:
            text = _WS.sub(" ", text).strip()
        return list(text)


DEFAULT = Normalizer()
