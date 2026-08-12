"""Mistral Voxtral backend — an audio-LLM branch, in two modes.

Voxtral (Mistral, 2025) is an audio-input LLM: a different family again from
Whisper (attention seq2seq) and wav2vec2 (CTC), so it contributes genuinely
different hypotheses to the candidate graph (§8, §12). Voxtral **Mini 3B**
(~9.5 GB in bf16) fits a T4; Voxtral **Small 24B** (~55 GB) does not.

Two decode modes, exposed as ``mode``:

* ``"transcription"`` (default) — ``AutoProcessor.apply_transcription_request`` +
  ``generate``: Voxtral's clean transcription path (drops most disfluencies).
* ``"verbatim"`` — instruct/chat path (``apply_chat_template`` with the audio plus
  a text instruction) telling the model to keep every filler, false start and
  repetition. This is the design's *verbatim* lever (§1.2, §9.2): the same
  checkpoint, prompted to preserve what the transcription mode strips.

Both take audio as a **file path**, so in-memory segment slices are written to a
temp wav per call. Everything is imported lazily; the core package needs none of it.
"""

from __future__ import annotations

import os
import tempfile

from ..datamodel import TextCandidate
from ..pipeline.base import AcousticModel, SpeechSegment
from ..types import CandidateSource

VERBATIM_INSTRUCTION = (
    "Transcribe the audio exactly as spoken, word for word. Include every filler "
    "(um, uh, mm-hmm), false start, repetition and stutter. Do not paraphrase, "
    "summarize, translate, correct grammar, or add commentary or punctuation beyond "
    "what is spoken. Output only the verbatim transcription."
)


class VoxtralASR(AcousticModel):
    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        *,
        id_prefix: str = "v",
        source: CandidateSource = CandidateSource.QWEN,
        language: str = "en",
        device: str | None = None,
        dtype=None,
        max_new_tokens: int = 64,   # CORAAL segments are short; caps decode time
        mode: str = "transcription",           # "transcription" | "verbatim"
        instruction: str | None = None,        # prompt used in "verbatim" mode
    ) -> None:
        import torch  # lazy
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        from . import best_dtype, load_pretrained

        self.model_id = model_id
        self.id_prefix = id_prefix
        self.source = source
        self.source_name = source.value
        self.language = language
        self.max_new_tokens = max_new_tokens
        self.mode = mode
        self.instruction = instruction or VERBATIM_INSTRUCTION

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 on T4/Turing (no bf16 tensor cores) → big speedup vs bf16
        self.dtype = dtype or best_dtype(torch, self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = load_pretrained(
            VoxtralForConditionalGeneration, model_id, dtype=self.dtype, device_map=self.device
        ).eval()

        # We decode greedily (do_sample=False); drop sampling-only flags the checkpoint's
        # generation_config carries so generate() doesn't warn "flags not valid: temperature".
        gcfg = getattr(self.model, "generation_config", None)
        if gcfg is not None and not getattr(gcfg, "do_sample", False):
            for k in ("temperature", "top_p", "top_k"):
                if getattr(gcfg, k, None) is not None:
                    setattr(gcfg, k, None)

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

    def _generate(self, paths, *, max_new_tokens=None):
        """Decode a batch of audio **file paths** → list[str]. Dispatches on mode."""
        torch = self._torch
        mnt = max_new_tokens or self.max_new_tokens
        if self.mode == "verbatim":
            return self._generate_instruct(paths, mnt)
        inputs = self.processor.apply_transcription_request(
            language=[self.language] * len(paths), audio=paths, model_id=self.model_id,
        ).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=mnt)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def _generate_instruct(self, paths, max_new_tokens):
        """Verbatim/instruct path: audio + a 'transcribe verbatim' instruction.

        Voxtral's ``apply_chat_template`` (MistralCommonTokenizer) already tokenizes and
        returns a ready-to-generate batch — it rejects the usual HF kwargs
        (``add_generation_prompt``/``tokenize``/``return_dict``), so we call it bare."""
        torch = self._torch

        def conv(p):
            return [{"role": "user", "content": [
                {"type": "audio", "path": p},
                {"type": "text", "text": self.instruction},
            ]}]

        def run(conversation):
            inputs = self.processor.apply_chat_template(conversation).to(
                self.device, dtype=self.dtype)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.processor.batch_decode(
                out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        try:
            return run([conv(p) for p in paths])       # one batched chat call
        except Exception:
            return [run(conv(p))[0] for p in paths]    # per-item fallback

    def nbest_batch(self, waveforms, *, n_best: int = 1,
                    sampling_rate: int = 16_000) -> list[list[TextCandidate]]:
        """Batch several waveforms through one generate call. Falls back to per-item
        if the batched request form isn't accepted by the installed transformers."""
        wavs = list(waveforms)
        if not wavs:
            return []
        paths = self._write_wavs(wavs, sampling_rate)
        try:
            try:
                texts = self._generate(paths)              # one batched call
            except Exception:
                texts = [self._generate([p])[0] for p in paths]  # per-item fallback
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
                        chunk_s: float = 30.0, batch_size: int = 4) -> str:
        """Whole-recording transcript: silence-snapped short windows, **batched**.

        Voxtral is an audio-LLM whose ``generate`` emits a bounded number of output
        tokens, so an over-long window truncates (the segment-tuned ``max_new_tokens``
        of 64 clips anything past ~15 s). Windows are kept near the utterance scale,
        cut at silence (no split words), and the token budget is scaled to the window
        (~8 tok/s + headroom). Windows go through in mini-batches for speed, with OOM
        batch-backoff."""
        from . import batched_transcribe, iter_silence_chunks

        chunks = [c for _, c in iter_silence_chunks(waveform, sampling_rate, chunk_s)
                  if len(c) >= 400]
        budget = max(self.max_new_tokens, int(chunk_s * 8) + 32)

        def batch(cs):
            paths = self._write_wavs(cs, sampling_rate)
            try:
                return self._generate(paths, max_new_tokens=budget)
            finally:
                for p in paths:
                    os.unlink(p)

        return batched_transcribe(chunks, batch, batch_size=batch_size, torch_mod=self._torch)

    def unload(self) -> None:
        import gc

        self.model = None
        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
