"""PhoneticXeus — the design doc's named universal phone recognizer (§7.3, §10).

``changelinglab/PhoneticXeus`` is real: a XEUS multilingual speech encoder fine-tuned with
self-conditioned CTC on IPAPack++ (17k h, 100+ languages) to transcribe audio into IPA. It is
SOTA on accented English phone recognition — a much stronger phone path than the
``wav2vec2-lv-60-espeak`` branch, and exactly the model the design names for the OOV/nonce
recovery route (phone lattice -> P2G).

Loaded via ``AutoModel(trust_remote_code=True)`` (the repo ships its own modeling code) and
driven through its ``transcribe(waveform, sampling_rate=16000) -> [{"processed_transcript": ...}]``
API. Product is a realized-IPA string; like the other phone branch it needs a phonetic reference
to score (PER/IPA-CER, §18.1), which CORAAL doesn't provide — so it's shown, not word-scored.
torch/transformers imported lazily so the core package stays dependency-free.
"""

from __future__ import annotations

from ..datamodel import PhonePath
from ..pipeline.base import PhoneEncoder, SpeechSegment


class PhoneticXeus(PhoneEncoder):
    def __init__(
        self,
        model_id: str = "changelinglab/PhoneticXeus",
        *,
        device: str | None = None,
    ) -> None:
        import torch  # lazy
        from transformers import AutoModel

        self.model_id = model_id
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # The repo ships custom modeling code; keep its native dtype (trust_remote_code models
        # don't reliably accept a dtype override). XEUS is ~600M params -> fits a T4 in fp32.
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        ).to(self.device).eval()

    def _transcribe_one(self, waveform, sampling_rate: int = 16_000) -> str:
        """Run the model's own transcribe() on one mono waveform -> IPA string."""
        torch = self._torch
        wav = torch.as_tensor(waveform, dtype=torch.float32)
        if wav.dim() == 2:                       # stereo -> mono, per the model's example
            wav = wav.mean(dim=0)
        with torch.inference_mode():
            res = self.model.transcribe(wav, sampling_rate=sampling_rate)
        if not res:
            return ""
        first = res[0]
        ipa = first.get("processed_transcript", "") if isinstance(first, dict) else str(first)
        return (ipa or "").strip()

    def recognize(self, segment: SpeechSegment, *, top_k: int) -> list[PhonePath]:
        if segment.waveform is None:
            raise ValueError("PhoneticXeus needs SpeechSegment.waveform (16 kHz mono float32)")
        return self.recognize_batch([segment.waveform], top_k=top_k)[0]

    def recognize_batch(
        self, waveforms, *, top_k: int = 1, sampling_rate: int = 16_000
    ) -> list[list[PhonePath]]:
        out: list[list[PhonePath]] = []
        for w in waveforms:
            ipa = self._transcribe_one(w, sampling_rate)
            out.append([PhonePath("p1", ipa, 1.0)] if ipa else [])
        return out

    def transcribe_full(self, waveform, *, sampling_rate: int = 16_000,
                        chunk_s: float = 24.0) -> str:
        """Whole-recording realized IPA. XEUS is a transformer encoder (O(n^2) attention), so a
        30-min interview won't fit one pass — transcribe silence-snapped ~24 s windows (no word
        split across a cut) and concatenate."""
        from . import iter_silence_chunks

        chunks = [c for _, c in iter_silence_chunks(waveform, sampling_rate, chunk_s)
                  if len(c) >= 400]
        parts = [self._transcribe_one(c, sampling_rate) for c in chunks]
        return " ".join(p for p in parts if p).strip()

    def unload(self) -> None:
        import gc

        self.model = None
        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
