"""A real phone-recognition branch (wav2vec2 phoneme CTC) → PhoneEncoder.

This is the first backend on the **phone path** (§7.3, §10.2), the design's
substitute for the named PhoneticXEUS: a wav2vec2 model fine-tuned to emit
eSpeak/IPA phonemes (e.g. ``facebook/wav2vec2-lv-60-espeak-cv-ft``). It produces
the realized-pronunciation lattice that feeds P2G and the phonetic RAG, and the
IPA/phone-error metrics (§18.1).

On ordinary in-vocabulary English words a naive P2G of these phones will rarely
beat a word ASR model, so the phone branch's payoff is **OOV / nonce recovery**
and **pronunciation analysis**, not word-oracle WER on clean speech — exactly the
division of labor the design intends (§10.1, §10.5).

Greedy CTC → a single phone path with a mean-posterior confidence. torch/
transformers imported lazily.
"""

from __future__ import annotations

from typing import Sequence

from ..datamodel import PhonePath
from ..pipeline.base import PhoneEncoder, SpeechSegment


class Wav2Vec2Phone(PhoneEncoder):
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
        *,
        device: str | None = None,
        fp16: bool = True,
    ) -> None:
        import torch  # lazy
        from transformers import AutoModelForCTC, AutoProcessor

        from . import load_pretrained

        self.model_id = model_id
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 on any CUDA device ("cuda:0" included); old `== "cuda"` silently ran float32.
        self.dtype = torch.float16 if (fp16 and str(self.device).startswith("cuda")) else torch.float32

        self.fe, self.tok = self._load_fe_tok(model_id)
        # apply_spec_augment=False: inference-correct and avoids the newly-initialized
        # masked_spec_embed warning at its source (see Wav2Vec2CTC).
        self.model = load_pretrained(
            AutoModelForCTC, model_id, dtype=self.dtype, apply_spec_augment=False
        ).to(self.device).eval()

    @staticmethod
    def _load_fe_tok(model_id):
        """Load the feature extractor and phoneme tokenizer **separately**.

        On transformers 4.57.x the combined ``Wav2Vec2Processor`` is broken for this
        phoneme model, and ``AutoTokenizer`` can even return a *bool* instead of a
        tokenizer. So load the feature extractor and tokenizer directly, trying the
        explicit phoneme-tokenizer class first and **validating** that we actually got a
        decoder (``batch_decode``) — raising clearly if this build can't provide one."""
        from transformers import AutoFeatureExtractor
        import transformers as tf

        fe = AutoFeatureExtractor.from_pretrained(model_id)
        tok = None
        loaders = [
            lambda: tf.Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_id),
            lambda: tf.AutoTokenizer.from_pretrained(model_id, use_fast=False),
            lambda: tf.AutoTokenizer.from_pretrained(model_id),
        ]
        for load in loaders:
            try:
                cand = load()
            except Exception:
                continue
            if hasattr(cand, "batch_decode"):
                tok = cand
                break
        if tok is None:
            raise RuntimeError(
                f"no usable phoneme tokenizer for {model_id} on this transformers build")
        return fe, tok

    def recognize(self, segment: SpeechSegment, *, top_k: int) -> list[PhonePath]:
        if segment.waveform is None:
            raise ValueError("Wav2Vec2Phone needs SpeechSegment.waveform (16 kHz mono float32)")
        return self.recognize_batch([segment.waveform], top_k=top_k)[0]

    def recognize_batch(
        self, waveforms, *, top_k: int = 1, sampling_rate: int = 16_000
    ) -> list[list[PhonePath]]:
        """Greedy phoneme CTC for a batch → one phone path per clip."""
        torch = self._torch
        wavs = list(waveforms)
        if not wavs:
            return []
        inputs = self.fe(
            wavs, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device, self.dtype)
        attn = getattr(inputs, "attention_mask", None)
        kwargs = {"attention_mask": attn.to(self.device)} if attn is not None else {}

        with torch.inference_mode():
            logits = self.model(input_values, **kwargs).logits
        probs = logits.softmax(dim=-1)
        pred_ids = probs.argmax(dim=-1)
        # confidence: mean max-posterior over non-blank frames
        conf = probs.max(dim=-1).values.mean(dim=-1).float().tolist()
        ipas = self.tok.batch_decode(pred_ids, clean_up_tokenization_spaces=False)

        results: list[list[PhonePath]] = []
        for i, ipa in enumerate(ipas):
            ipa = (ipa or "").strip()
            results.append([PhonePath("p1", ipa, float(conf[i]))] if ipa else [])
        return results

    def transcribe_full(self, waveform, *, sampling_rate: int = 16_000, chunk_s: float = 24.0,
                        batch_size: int = 8) -> str:
        """Whole-recording **realized IPA** (chunked). This is the phone branch's
        actual product — a pronunciation transcript — not a word transcript. Its
        value (OOV/nonce recovery, pronunciation/PER metrics — §10, §18.1) needs a
        phonetic reference to score; CORAAL has none, so it's reported as-is.
        Chunk boundaries are snapped to silence so a token isn't split across a cut."""
        from . import iter_silence_chunks

        chunks = [c for _, c in iter_silence_chunks(waveform, sampling_rate, chunk_s)
                  if len(c) >= 400]
        parts = []
        for b in range(0, len(chunks), batch_size):     # batch windows per forward pass
            for r in self.recognize_batch(chunks[b:b + batch_size], top_k=1,
                                          sampling_rate=sampling_rate):
                if r:
                    parts.append(r[0].ipa)
        return " ".join(parts).strip()

    def unload(self) -> None:
        import gc

        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
