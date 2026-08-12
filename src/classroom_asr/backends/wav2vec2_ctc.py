"""A wav2vec2 CTC backend — a deliberately *different architecture* branch.

Why this exists: the candidate-oracle metric only tests the design's central bet
— that complementary acoustic evidence rescues errors (§8, §12) — if the branches
actually fail differently. Two Whisper sizes share architecture, tokenizer, and
training data, so their errors correlate and the oracle mostly reflects beam
diversity. A CTC acoustic model (no autoregressive attention decoder, no internal
LM) makes genuinely different mistakes, so its hypotheses add real diversity to
the candidate graph.

Bonus: CTC is a single forward pass — no beam decoding — so it is *cheaper* than a
second Whisper, not more expensive.

Greedy CTC gives a 1-best. True N-best would need a CTC beam decoder + LM
(pyctcdecode); we keep it to 1-best here to stay dependency-light — one
different-architecture hypothesis is what the oracle needs. torch/transformers
are imported lazily so the core package stays dependency-free.
"""

from __future__ import annotations

from typing import Sequence

from ..datamodel import TextCandidate
from ..pipeline.base import AcousticModel, SpeechSegment
from ..types import CandidateSource


class Wav2Vec2CTC(AcousticModel):
    """wav2vec2 / HuBERT-style CTC acoustic model producing a greedy 1-best.

    Parameters mirror :class:`~classroom_asr.backends.whisper_asr.WhisperASR` so
    the two are drop-in interchangeable as branches.
    """

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-large-960h-lv60-self",
        *,
        id_prefix: str = "c",
        source: CandidateSource = CandidateSource.GIGAAM,   # generic "branch B" slot
        device: str | None = None,
        fp16: bool = True,
    ) -> None:
        import torch  # lazy
        from transformers import AutoProcessor, AutoModelForCTC

        from . import load_pretrained

        self.model_id = model_id
        self.id_prefix = id_prefix
        self.source = source
        self.source_name = source.value

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)
        # The 960h checkpoint has no `masked_spec_embed` (a SpecAugment/training-only
        # param), so transformers warns it's "newly initialized … probably TRAIN". It is
        # never used at inference — scope-suppress just that one modeling logger for the load.
        import logging
        mlog = logging.getLogger("transformers.modeling_utils")
        prev = mlog.level
        mlog.setLevel(logging.ERROR)
        try:
            self.model = load_pretrained(
                AutoModelForCTC, model_id, dtype=self.dtype
            ).to(self.device).eval()
        finally:
            mlog.setLevel(prev)

    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        if segment.waveform is None:
            raise ValueError("Wav2Vec2CTC needs SpeechSegment.waveform (16 kHz mono float32)")
        return self.nbest_batch([segment.waveform], n_best=n_best)[0]

    def nbest_batch(
        self, waveforms, *, n_best: int = 1, sampling_rate: int = 16_000
    ) -> list[list[TextCandidate]]:
        """Greedy CTC decode for a batch of waveforms (one candidate each).

        ``n_best`` is accepted for interface symmetry but greedy CTC yields a
        single hypothesis per clip.
        """
        torch = self._torch
        wavs = list(waveforms)
        if not wavs:
            return []
        inputs = self.processor(
            wavs, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device, self.dtype)
        attn = getattr(inputs, "attention_mask", None)
        kwargs = {}
        if attn is not None:
            kwargs["attention_mask"] = attn.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values, **kwargs).logits
        pred_ids = logits.argmax(dim=-1)
        texts = self.processor.batch_decode(pred_ids, clean_up_tokenization_spaces=False)

        results: list[list[TextCandidate]] = []
        for t in texts:
            text = (t or "").strip().lower()
            results.append(
                [TextCandidate(f"{self.id_prefix}1", text, self.source, score=1.0, beam_rank=0)]
                if text else []
            )
        return results

    def transcribe_words(self, waveform, *, sampling_rate: int = 16_000, chunk_s: float = 24.0,
                         batch_size: int = 8):
        """Whole-recording CTC transcription with word timestamps (§6.1).

        wav2vec2 can't hold a 30–60 min interview in one forward pass, so we run
        non-overlapping chunks and use HF's built-in ``output_word_offsets`` to
        get per-word frame offsets, converted to absolute time. Chunk boundaries
        are snapped to silence (see :func:`snap_to_silence`) so a word is never
        split across a cut. Returns ``(start, end, word)`` so the model never sees
        reference boundaries.
        """
        from . import iter_silence_chunks

        torch = self._torch
        # seconds per logit frame (e.g. 320 samples / 16 kHz = 0.02 s)
        spf = float(self.model.config.inputs_to_logits_ratio) / sampling_rate
        pieces = [(start, chunk) for start, chunk in
                  iter_silence_chunks(waveform, sampling_rate, chunk_s) if len(chunk) >= 400]
        out: list[tuple[float, float, str]] = []
        # batch several windows through one padded forward pass (big speedup); the
        # attention mask keeps padding from bleeding into each item's word offsets.
        for b in range(0, len(pieces), batch_size):
            group = pieces[b:b + batch_size]
            enc = self.processor([c for _, c in group], sampling_rate=sampling_rate,
                                 return_tensors="pt", padding=True)
            iv = enc.input_values.to(self.device, self.dtype)
            attn = getattr(enc, "attention_mask", None)
            kw = {"attention_mask": attn.to(self.device)} if attn is not None else {}
            with torch.no_grad():
                logits = self.model(iv, **kw).logits
            dec = self.processor.batch_decode(
                logits.argmax(dim=-1), output_word_offsets=True,
                clean_up_tokenization_spaces=False,
            )
            offs = dec.word_offsets if hasattr(dec, "word_offsets") else dec["word_offsets"]
            for (start, _c), word_offsets in zip(group, offs):
                base = start / sampling_rate
                for wo in word_offsets:
                    w = (wo["word"] or "").strip().lower()
                    if w:
                        out.append((base + wo["start_offset"] * spf,
                                    base + wo["end_offset"] * spf, w))
        return out

    def transcribe_full(self, waveform, *, sampling_rate: int = 16_000) -> str:
        """Whole-recording transcript (chunked internally)."""
        return " ".join(t for _, _, t in self.transcribe_words(
            waveform, sampling_rate=sampling_rate)).strip()

    def unload(self) -> None:
        import gc

        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
