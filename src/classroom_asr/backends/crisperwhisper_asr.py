"""CrisperWhisper 2.0 backend — a *verbatim*-tuned Whisper branch.

Ordinary Whisper is trained to emit clean, readable text: it deletes ``um``/``uh``,
false starts and repetitions. On a **verbatim** reference (CORAAL, AMI, TED-LIUM)
those deletions dominate the residual WER, and no clean-text branch recovers them.
CrisperWhisper (nyra labs) is a Whisper fine-tune built specifically to transcribe
every spoken word — fillers, stutters, false starts — and tops verbatim benchmarks
where Whisper-large-v3 loses. So it belongs in the candidate graph as the branch
that *keeps* what the others drop (§1.2, §9.2).

CrisperWhisper 2.0 backends: its ``ct2`` runtime needs a **forked** ctranslate2
(``ctranslate2-crisperwhisper``) that cannot coexist with ``faster-whisper`` — they
overwrite each other in site-packages — and we keep faster-whisper as baseline A. So
this backend uses the pure-PyTorch ``transformers`` backend (``pip install
"crisperwhisper[transformers]"``): slower than CT2 but no dependency conflict, and the
verbatim output (the whole point) is identical. It still does its own long-form
windowing internally — no manual chunking. Sizes: ``turbo`` / ``large`` / ``medium`` /
``small`` (append ``_pro`` for the commercial-licensed tier).

Fillers are emitted bracketed (``[um]``, ``[uh]``); non-lexical events (``[laughter]``,
``[noise]`` …) are dropped here so they aren't scored as word insertions, while the
scoring normalizer's edge-punct strip turns ``[um]`` → ``um`` to match the reference.
Everything is imported lazily; the core package needs none of it.
"""

from __future__ import annotations

import re

from ..datamodel import TextCandidate
from ..pipeline.base import AcousticModel, SpeechSegment
from ..types import CandidateSource

# Non-lexical events CrisperWhisper may emit in brackets — not words in the
# reference, so drop them rather than let them score as insertions.
_EVENT = re.compile(
    r"\[(laughter|laugh|noise|music|applause|breath\w*|cough|sigh|sneeze|"
    r"click|silence|inaudible|unk|unknown|crosstalk)\]",
    re.IGNORECASE,
)


class CrisperWhisperV2(AcousticModel):
    def __init__(
        self,
        size: str = "large",                       # turbo | large | medium | small (+ _pro)
        *,
        backend: str = "transformers",             # pure PyTorch; no ctranslate2 conflict
        id_prefix: str = "cw",
        source: CandidateSource = CandidateSource.QWEN,
        language: str | None = "en",
        device: str | None = None,
    ) -> None:
        import torch  # lazy
        from crisperwhisper import CrisperWhisperModel

        self.size = size
        self.id_prefix = id_prefix
        self.source = source
        self.source_name = source.value
        self.language = language

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # CrisperWhisperModel auto-selects GPU; pass device/index when the signature
        # allows pinning (multi-GPU), else fall back to the default constructor.
        idx = int(str(self.device).split(":")[1]) if ":" in str(self.device) else 0
        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        for kwargs in ({"backend": backend, "device": dev, "device_index": idx},
                       {"backend": backend, "device": dev},
                       {"backend": backend}):
            try:
                self.model = CrisperWhisperModel(size, **kwargs)
                break
            except TypeError:
                continue

    @staticmethod
    def _clean(text: str) -> str:
        text = _EVENT.sub(" ", text or "")          # drop non-lexical events
        text = text.replace("[", " ").replace("]", " ")   # unwrap [um] -> um
        return re.sub(r"\s+", " ", text).strip()

    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        if segment.waveform is None:
            raise ValueError("CrisperWhisperV2 needs SpeechSegment.waveform (16 kHz mono float32)")
        text = self.transcribe_full(segment.waveform)
        return [TextCandidate(f"{self.id_prefix}1", text, self.source, score=1.0, beam_rank=0)] if text else []

    def transcribe_full(self, waveform, *, sampling_rate: int = 16_000) -> str:
        """Whole-recording verbatim transcript. CrisperWhisper's CT2 runtime handles
        long audio internally (its own windowing), so we pass the whole recording."""
        kwargs = {"sr": sampling_rate}
        if self.language:
            kwargs["language"] = self.language
        try:
            result = self.model.transcribe(waveform, **kwargs)
        except TypeError:                            # older signature without sr kwarg
            result = self.model.transcribe(waveform)
        return self._clean(getattr(result, "text", "") or "")

    def unload(self) -> None:
        import gc

        self.model = None
        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
