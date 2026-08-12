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
        dtype=None,
        max_inference_batch_size: int = 16,
    ) -> None:
        import torch  # lazy
        from qwen_asr import Qwen3ASRModel

        from . import best_dtype

        self.model_id = model_id
        self.id_prefix = id_prefix
        self.source = source
        self.source_name = source.value
        self.language = language

        self._torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # fp16 on T4/Turing (no bf16 tensor cores) → big speedup vs bf16
        self.dtype = dtype or best_dtype(torch, self.device)
        try:
            self.model = Qwen3ASRModel.from_pretrained(
                model_id, dtype=self.dtype, device_map=self.device,
                max_inference_batch_size=max_inference_batch_size,
            )
        except TypeError:  # older/newer signature without the kwarg
            self.model = Qwen3ASRModel.from_pretrained(
                model_id, dtype=self.dtype, device_map=self.device
            )

        # Fix the generate-time warnings at the source: set pad_token_id (kills the
        # per-call "Setting pad_token_id …") and drop sampling-only flags since we decode
        # greedily (kills "generation flags not valid: temperature").
        from . import tune_generation_config
        tune_generation_config(self.model)

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
        wavs = list(waveforms)
        if not wavs:
            return []
        audio = [(w, sampling_rate) for w in wavs]
        kwargs = {"language": [self.language] * len(wavs)} if self.language else {}
        results = self.model.transcribe(audio=audio, **kwargs)

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
        kwargs = {"language": [self.language]} if self.language else {}
        results = self.model.transcribe(
            audio=[(waveform, sampling_rate)], return_time_stamps=True, **kwargs
        )
        if not results:
            return []
        stamps = getattr(results[0], "time_stamps", None) or []
        out: list[tuple[float, float, str]] = []
        for st in stamps:
            text = (getattr(st, "text", "") or "").strip()
            if text:
                out.append((float(st.start_time), float(st.end_time), text))
        return out

    def transcribe_chunk_list(self, chunks, *, sampling_rate: int = 16_000,
                              batch_size: int = 16) -> list[str]:
        """Transcribe pre-cut windows → one text **per window** (batched, OOM-backoff).

        Splitting chunking from transcription lets an external runner pool windows from
        several recordings and spread them evenly across GPUs (no idle GPU), then
        reassemble each transcript in order — same output, better utilization."""
        from . import batched_transcribe_list

        def batch(cs):
            res = self.nbest_batch(cs, sampling_rate=sampling_rate)
            return [(r[0].text if r else "") for r in res]

        return batched_transcribe_list(chunks, batch, batch_size=batch_size, torch_mod=self._torch)

    def transcribe_full(self, waveform, *, sampling_rate: int = 16_000,
                        chunk_s: float = 30.0, batch_size: int = 16) -> str:
        """Whole-recording transcript: silence-snapped short windows, **batched**.

        Qwen3-ASR is an audio-*LLM*: one ``transcribe`` call emits a bounded number
        of output tokens, so an over-long window has its transcript **truncated**
        (most words never emitted → massive deletions). Windows are kept near the
        utterance scale (~30 s, like Whisper's internal window) and cut at silence so
        no word is split. qwen-asr batches natively, so the windows go through in
        mini-batches (big speedup vs one call per window), with OOM batch-backoff."""
        from . import iter_silence_chunks

        chunks = [c for _, c in iter_silence_chunks(waveform, sampling_rate, chunk_s)
                  if len(c) >= 400]
        parts = self.transcribe_chunk_list(chunks, sampling_rate=sampling_rate,
                                           batch_size=batch_size)
        return " ".join(p for p in parts if p).strip()

    def unload(self) -> None:
        import gc

        self.model = None
        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
