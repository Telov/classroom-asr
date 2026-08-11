"""Qwen3-ASR backend — the design's actual frozen backbone (§7.1, §26).

``Qwen/Qwen3-ASR-1.7B`` is real and small (released 2026-01), built on Qwen3-Omni.
Unlike the fictional-name assumption earlier, this IS the design's chosen backbone,
so it belongs in the candidate graph as a first-class branch — and being a
multilingual LLM-ASR it fails differently from Whisper/CTC, adding real diversity.

Uses the official ``qwen-asr`` package (``pip install qwen-asr``):
``Qwen3ASRModel.transcribe(audio=(np.ndarray, sr))`` → ``results[0].text``. That's
a 1-best transcription API; one strong, different-architecture hypothesis is what
the oracle needs. Lazy import keeps the core dependency-free.
"""

from __future__ import annotations

from ..datamodel import TextCandidate
from ..pipeline.base import AcousticModel, SpeechSegment
from ..types import CandidateSource


class Qwen3ASR(AcousticModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        *,
        id_prefix: str = "z",
        source: CandidateSource = CandidateSource.QWEN,
        language: str | None = "English",
        device: str | None = None,
        max_inference_batch_size: int = 16,
    ) -> None:
        import torch  # lazy
        from qwen_asr import Qwen3ASRModel

        self.model_id = model_id
        self.id_prefix = id_prefix
        self.source = source
        self.source_name = source.value
        self.language = language

        self._torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self.model = Qwen3ASRModel.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=self.device,
                max_inference_batch_size=max_inference_batch_size,
            )
        except TypeError:  # older/newer signature without the kwarg
            self.model = Qwen3ASRModel.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=self.device
            )

    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        if segment.waveform is None:
            raise ValueError("Qwen3ASR needs SpeechSegment.waveform (16 kHz mono float32)")
        return self.nbest(segment.waveform, n_best=n_best)

    def nbest(self, waveform, *, n_best: int = 1, sampling_rate: int = 16_000) -> list[TextCandidate]:
        return self.nbest_batch([waveform], n_best=n_best, sampling_rate=sampling_rate)[0]

    def nbest_batch(
        self, waveforms, *, n_best: int = 1, sampling_rate: int = 16_000
    ) -> list[list[TextCandidate]]:
        """Transcribe a *batch* of waveforms in one call (qwen-asr batches natively).

        This is the main Qwen3-ASR speedup: one batched ``transcribe`` instead of
        thousands of per-segment calls.
        """
        import logging

        wavs = list(waveforms)
        if not wavs:
            return []
        audio = [(w, sampling_rate) for w in wavs]
        kwargs = {"language": [self.language] * len(wavs)} if self.language else {}
        # qwen-asr's internal HF generate logs "Setting pad_token_id ..." per call;
        # scope-suppress just that logger for just this call (not a blanket filter).
        gen_log = logging.getLogger("transformers.generation.utils")
        prev = gen_log.level
        gen_log.setLevel(logging.ERROR)
        try:
            results = self.model.transcribe(audio=audio, **kwargs)
        finally:
            gen_log.setLevel(prev)

        out: list[list[TextCandidate]] = []
        for r in results:
            text = (getattr(r, "text", "") or "").strip()
            out.append(
                [TextCandidate(f"{self.id_prefix}1", text, self.source, score=1.0, beam_rank=0)]
                if text else []
            )
        return out

    def transcribe_words(self, waveform, *, sampling_rate: int = 16_000):
        """Whole-recording transcription with native timestamps (§6.1).

        Qwen3-ASR supports long audio + ``return_time_stamps``; returns a list of
        ``(start, end, text)`` so the interview can be transcribed once (no
        reference-boundary leakage) and re-segmented by time for scoring.
        """
        import logging

        kwargs = {"language": [self.language]} if self.language else {}
        gen_log = logging.getLogger("transformers.generation.utils")
        prev = gen_log.level
        gen_log.setLevel(logging.ERROR)
        try:
            results = self.model.transcribe(
                audio=[(waveform, sampling_rate)], return_time_stamps=True, **kwargs
            )
        finally:
            gen_log.setLevel(prev)
        if not results:
            return []
        stamps = getattr(results[0], "time_stamps", None) or []
        out: list[tuple[float, float, str]] = []
        for st in stamps:
            text = (getattr(st, "text", "") or "").strip()
            if text:
                out.append((float(st.start_time), float(st.end_time), text))
        return out

    def transcribe_full(self, waveform, *, sampling_rate: int = 16_000, chunk_s: float = 120.0) -> str:
        """Whole-recording transcript, chunked. Feeding a 30–60 min interview in one
        forward pass OOMs a T4 (activations scale with audio length), so we run
        ~chunk_s windows and concatenate."""
        win = int(chunk_s * sampling_rate)
        parts = []
        for start in range(0, len(waveform), win):
            chunk = waveform[start:start + win]
            if len(chunk) < 400:
                continue
            c = self.nbest(chunk, sampling_rate=sampling_rate)
            if c and c[0].text:
                parts.append(c[0].text)
        return " ".join(parts).strip()

    def unload(self) -> None:
        import gc

        del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()
