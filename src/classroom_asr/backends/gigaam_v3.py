"""Russian-specialized GigaAM v3 RNNT backend.

The design keeps GigaAM-v3-SSL as a future acoustic feature stream. This adapter serves a
different immediate purpose: it exposes the official Russian RNNT branch as a 1-best word
candidate while preserving the dependency-free core and strict verbatim policy.
"""

from __future__ import annotations

import os
import tempfile
import wave

from ..datamodel import TextCandidate
from ..pipeline.base import AcousticModel, SpeechSegment
from ..types import CandidateSource, Language


GIGAAM_V3_MODEL_ID = "ai-sage/GigaAM-v3"
# Immutable revision of the official ``rnnt`` branch, resolved 2026-08-14.
GIGAAM_V3_RNNT_REVISION = "c7f128b8accdd9624df905e5c2d7b7a48c27c0d8"


class GigaAMV3RNNT(AcousticModel):
    """Official GigaAM-v3 RNNT as a Russian 1-best text-candidate branch."""

    source_name = CandidateSource.GIGAAM.value

    def __init__(
        self,
        model_id: str = GIGAAM_V3_MODEL_ID,
        *,
        revision: str = GIGAAM_V3_RNNT_REVISION,
        device: str | None = None,
        dtype=None,
        id_prefix: str = "g",
        core_s: float = 20.0,
        overlap_s: float = 2.0,
    ) -> None:
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
            raise ValueError("GigaAM trust_remote_code requires a full immutable commit revision")
        if core_s + 2 * overlap_s + 0.9 > 25.0:
            raise ValueError("GigaAM windows, margins, and silence search must stay within 25 s")

        import torch  # lazy optional dependency
        from transformers import AutoModel

        from . import best_dtype, load_pretrained

        self._torch = torch
        self.model_id = model_id
        self.revision = revision
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or best_dtype(torch, self.device)
        self.id_prefix = id_prefix
        self.core_s = core_s
        self.overlap_s = overlap_s
        self.model = load_pretrained(
            AutoModel,
            model_id,
            dtype=self.dtype,
            revision=revision,
            trust_remote_code=True,
        ).to(self.device).eval()

    def _transcribe_window(self, waveform, sampling_rate: int) -> str:
        """Bridge the pinned model's path-only public API via a session-temp WAV."""
        import numpy as np

        handle = tempfile.NamedTemporaryFile(prefix="gigaam_", suffix=".wav", delete=False)
        path = handle.name
        handle.close()
        try:
            pcm = (np.clip(np.asarray(waveform), -1.0, 1.0) * 32767.0).astype("<i2")
            with wave.open(path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sampling_rate)
                wav.writeframes(pcm.tobytes())
            return (self.model.transcribe(path) or "").strip()
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def transcribe_full(self, waveform, *, sampling_rate: int = 16_000) -> str:
        from . import iter_overlapping_silence_chunks, merge_overlapping_text

        parts = []
        windows = iter_overlapping_silence_chunks(
            waveform,
            sampling_rate,
            self.core_s,
            overlap_s=self.overlap_s,
            search_s=0.9,
        )
        for _, _, _, _, chunk in windows:
            if len(chunk) >= 400:
                parts.append(self._transcribe_window(chunk, sampling_rate))
        return merge_overlapping_text(parts)

    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        if segment.waveform is None:
            raise ValueError("GigaAM-v3 needs SpeechSegment.waveform (16 kHz mono float32)")
        text = self.transcribe_full(segment.waveform)
        if not text:
            return []
        return [
            TextCandidate(
                f"{self.id_prefix}1",
                text,
                CandidateSource.GIGAAM,
                score=1.0,
                beam_rank=0,
                language=Language.RU,
            )
        ]

    def unload(self) -> None:
        import gc

        self.model = None
        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
