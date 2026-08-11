"""Mistral Voxtral backend — a third, audio-LLM branch.

Voxtral (Mistral, 2025) is an audio-input LLM: a different family again from
Whisper (attention seq2seq) and wav2vec2 (CTC), so it contributes genuinely
different hypotheses to the candidate graph (§8, §12). Voxtral **Mini 3B**
(~9.5 GB in bf16) fits a T4; Voxtral **Small 24B** (~55 GB) does not.

Uses the transformers transcription API:
``AutoProcessor.apply_transcription_request`` +
``VoxtralForConditionalGeneration.generate`` (transformers >= 4.54,
``mistral-common[audio] >= 1.8.1``). That method takes an audio **file path**, so
for our in-memory segment slices we write a temporary wav per call. Voxtral is
heavier than the CTC/Whisper branches, so run it on a subset of the lesson.

Everything is imported lazily; the core package needs none of it.
"""

from __future__ import annotations

import os
import tempfile

from ..datamodel import TextCandidate
from ..pipeline.base import AcousticModel, SpeechSegment
from ..types import CandidateSource


class VoxtralASR(AcousticModel):
    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        *,
        id_prefix: str = "v",
        source: CandidateSource = CandidateSource.QWEN,
        language: str = "en",
        device: str | None = None,
        max_new_tokens: int = 64,   # CORAAL segments are short; caps decode time
    ) -> None:
        import torch  # lazy
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        from . import load_pretrained

        self.model_id = model_id
        self.id_prefix = id_prefix
        self.source = source
        self.source_name = source.value
        self.language = language
        self.max_new_tokens = max_new_tokens

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = load_pretrained(
            VoxtralForConditionalGeneration, model_id, dtype=self.dtype, device_map=self.device
        ).eval()

    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        if segment.waveform is None:
            raise ValueError("VoxtralASR needs SpeechSegment.waveform (16 kHz mono float32)")
        return self.nbest(segment.waveform, n_best=n_best)

    def nbest(self, waveform, *, n_best: int = 1, sampling_rate: int = 16_000) -> list[TextCandidate]:
        return self.nbest_batch([waveform], n_best=n_best, sampling_rate=sampling_rate)[0]

    def _write_wavs(self, waveforms, sampling_rate):
        import soundfile as sf

        paths = []
        for w in waveforms:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            sf.write(tmp.name, w, sampling_rate)
            paths.append(tmp.name)
        return paths

    def _generate(self, audio, *, max_new_tokens=None):
        torch = self._torch
        inputs = self.processor.apply_transcription_request(
            language=self.language if isinstance(audio, str) else [self.language] * len(audio),
            audio=audio, model_id=self.model_id,
        ).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def nbest_batch(self, waveforms, *, n_best: int = 1,
                    sampling_rate: int = 16_000) -> list[list[TextCandidate]]:
        """Batch several waveforms through one generate call (Voxtral supports
        batched transcription). Falls back to per-item if the batched request
        form isn't accepted by the installed transformers."""
        wavs = list(waveforms)
        if not wavs:
            return []
        paths = self._write_wavs(wavs, sampling_rate)
        try:
            try:
                texts = self._generate(paths)              # one batched call
            except Exception:
                texts = [self._generate(p)[0] for p in paths]  # per-item fallback
        finally:
            for p in paths:
                os.unlink(p)

        results: list[list[TextCandidate]] = []
        for t in texts:
            text = (t or "").strip()
            results.append(
                [TextCandidate(f"{self.id_prefix}1", text, self.source, score=1.0, beam_rank=0)]
                if text else []
            )
        return results

    def transcribe_full(self, waveform, *, sampling_rate: int = 16_000,
                        chunk_s: float = 30.0, min_chunk_s: float = 10.0) -> str:
        """Whole-recording transcript, transcribed in short windows.

        Voxtral is an audio-LLM whose ``generate`` emits a bounded number of output
        tokens, so an over-long window truncates (the segment-tuned ``max_new_tokens``
        of 64 clips anything past ~15 s). The window is kept near the utterance scale
        and the token budget is scaled to it (~8 tok/s + headroom) so nothing is
        dropped; ``chunked_transcribe`` still backs off on OOM as a safety net."""
        from . import chunked_transcribe

        # scale the output-token budget to the window so long chunks aren't clipped
        budget = max(self.max_new_tokens, int(chunk_s * 8) + 32)

        def one(chunk):
            paths = self._write_wavs([chunk], sampling_rate)
            try:
                texts = self._generate(paths, max_new_tokens=budget)
            finally:
                for p in paths:
                    os.unlink(p)
            return (texts[0] or "").strip() if texts else ""

        return chunked_transcribe(waveform, sampling_rate, one,
                                  chunk_s=chunk_s, min_chunk_s=min_chunk_s, torch_mod=self._torch)

    def unload(self) -> None:
        import gc

        self.model = None
        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
