"""A real Whisper backend for the :class:`AcousticModel` interface.

This is the first *actual* model wired into the pipeline. The design doc's named
backbone (Qwen3-ASR-1.7B) is a 2026 identifier that is not downloadable today,
so for real-data experiments we substitute an existing open model — Whisper via
`transformers` — behind the same interface. Everything downstream (candidate
graph, oracle metric, orchestrator) is unchanged: it only ever saw the
interface.

Key point for the candidate-oracle experiment: we use **beam search with
``num_return_sequences``** to get a genuine N-best list per segment, not one
greedy hypothesis. The oracle is only meaningful when the branch actually
proposes diverse hypotheses.

torch/transformers are imported lazily inside ``__init__`` so importing this
module (or the wheel) does not require the ML stack until a model is built.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..datamodel import TextCandidate
from ..pipeline.base import AcousticModel, SpeechSegment
from ..types import CandidateSource


class WhisperASR(AcousticModel):
    """Whisper N-best backbone.

    Parameters
    ----------
    model_id:
        HF id, e.g. ``openai/whisper-large-v3`` or ``openai/whisper-medium.en``.
    id_prefix:
        Candidate id prefix (``"q"`` -> ``q1, q2, ...``). Use distinct prefixes
        for distinct branches so ids stay unique when merged into one span.
    source:
        Which :class:`CandidateSource` slot to tag candidates with. For a real
        two-Whisper setup we reuse ``QWEN`` (branch A) and ``GIGAAM`` (branch B)
        as generic branch slots.
    language:
        Force a decoding language (``"en"``) or ``None`` to auto-detect.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        *,
        id_prefix: str = "q",
        source: CandidateSource = CandidateSource.QWEN,
        language: str | None = "en",
        device: str | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        import torch  # lazy
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_id = model_id
        self.id_prefix = id_prefix
        self.source = source
        self.source_name = source.value
        self.language = language
        self.max_new_tokens = max_new_tokens

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        from . import load_pretrained

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = load_pretrained(
            WhisperForConditionalGeneration, model_id, dtype=self.dtype
        ).to(self.device).eval()
        gc = self.model.generation_config
        # Cap with max_new_tokens; clearing max_length avoids the "both set" warning.
        gc.max_length = None
        # Set language/task on the generation config ONCE rather than passing them
        # to every generate() call — passing them per-call makes transformers
        # rebuild suppress-token logits processors and warn about the collision.
        if language:
            gc.language = language
            gc.task = "transcribe"

    # -- interface --------------------------------------------------------- #
    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        if segment.waveform is None:
            raise ValueError("WhisperASR needs SpeechSegment.waveform (16 kHz mono float32)")
        return self.nbest(segment.waveform, n_best=n_best)

    # -- core -------------------------------------------------------------- #
    def nbest(self, waveform, *, n_best: int, sampling_rate: int = 16_000) -> list[TextCandidate]:
        """Beam-search N-best for a single 16 kHz mono waveform."""
        return self.nbest_batch([waveform], n_best=n_best, sampling_rate=sampling_rate)[0]

    def nbest_batch(
        self, waveforms, *, n_best: int, sampling_rate: int = 16_000
    ) -> list[list[TextCandidate]]:
        """Beam-search N-best for a *batch* of waveforms in one forward pass.

        Whisper pads every clip to the same 30 s mel window, so batching many
        short utterances through a single ``generate`` call keeps the GPU busy
        and is several times faster than one-at-a-time decoding — the main lever
        for keeping Colab GPU time down. Returns one candidate list per input.
        """
        torch = self._torch
        wavs = list(waveforms)
        if not wavs:
            return []
        proc = self.processor(
            wavs, sampling_rate=sampling_rate, return_tensors="pt",
            return_attention_mask=True,
        )
        feats = proc.input_features.to(self.device, self.dtype)
        attn = getattr(proc, "attention_mask", None)

        want_scores = n_best > 1     # scores only exist for beam search
        # language/task come from generation_config (set in __init__), not here.
        gen_kwargs = dict(
            num_beams=max(n_best, 1),
            num_return_sequences=n_best,
            max_new_tokens=self.max_new_tokens,
        )
        if attn is not None:
            gen_kwargs["attention_mask"] = attn.to(self.device)
        if want_scores:
            gen_kwargs.update(output_scores=True, return_dict_in_generate=True)

        with torch.no_grad():
            out = self.model.generate(feats, **gen_kwargs)

        seqs = out.sequences if want_scores else out
        # clean_up_tokenization_spaces is destructive for BPE (strips spaces before
        # punctuation); disable it explicitly (also silences the transformers note).
        texts = self.processor.batch_decode(
            seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # sequences_scores: length-normalized log-prob per returned beam, ordered
        # batch-major (item0 beams, item1 beams, ...).
        scores = getattr(out, "sequences_scores", None) if want_scores else None

        results: list[list[TextCandidate]] = []
        for b in range(len(wavs)):
            lo = b * n_best
            chunk = texts[lo: lo + n_best]
            if scores is not None:
                weights = _softmax([float(scores[lo + i]) for i in range(len(chunk))])
            else:
                weights = [1.0 / len(chunk)] * len(chunk) if chunk else []
            cands = [
                TextCandidate(
                    id=f"{self.id_prefix}{i + 1}",
                    text=chunk[i].strip(),
                    source=self.source,
                    score=weights[i],
                    beam_rank=i,
                )
                for i in range(len(chunk))
            ]
            results.append(_dedupe(cands))
        return results

    def unload(self) -> None:
        """Free VRAM so the next model/stage can use the full GPU (§21.2)."""
        import gc

        del self.model
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


def _softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    tot = sum(exps)
    return [e / tot for e in exps] if tot else [1.0 / len(xs)] * len(xs)


def _dedupe(cands: list[TextCandidate]) -> list[TextCandidate]:
    seen: dict[str, TextCandidate] = {}
    for c in cands:
        key = c.text.casefold()
        if key not in seen or c.score > seen[key].score:
            seen[key] = c
    # keep original beam order
    ordered = sorted(seen.values(), key=lambda c: (c.beam_rank if c.beam_rank is not None else 0))
    return ordered
