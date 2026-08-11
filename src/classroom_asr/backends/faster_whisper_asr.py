"""faster-whisper (CTranslate2) backend — the fast path for branch A.

Whisper's feature extractor pads *every* clip to a 30 s window, so a 1 s
backchannel costs a full 30 s encoder pass; with thousands of short CORAAL
segments that dominates wall-clock. CTranslate2 (faster-whisper) runs the same
model with int8 kernels — several times faster and less VRAM — and we only need
1-best from this branch, so it is a clean drop-in for :class:`WhisperASR`.

Model: ``deepdml/faster-whisper-large-v3-turbo-ct2`` (turbo, CT2 format).
``pip install faster-whisper``. Imported lazily.
"""

from __future__ import annotations

from ..datamodel import TextCandidate
from ..pipeline.base import AcousticModel, SpeechSegment
from ..types import CandidateSource


def _split_device(device: str) -> tuple[str, int]:
    if device is None or device == "cpu":
        return "cpu", 0
    if ":" in device:
        d, i = device.split(":")
        return d, int(i)
    return device, 0


class FasterWhisperASR(AcousticModel):
    def __init__(
        self,
        model_id: str = "deepdml/faster-whisper-large-v3-turbo-ct2",
        *,
        id_prefix: str = "q",
        source: CandidateSource = CandidateSource.QWEN,
        language: str | None = "en",
        device: str | None = None,
        compute_type: str | None = None,
        beam_size: int = 1,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_id = model_id
        self.id_prefix = id_prefix
        self.source = source
        self.source_name = source.value
        self.language = language
        self.beam_size = beam_size

        dev, index = _split_device(device or "cuda:0")
        # int8 kernels: int8_float16 on GPU, int8 on CPU.
        if compute_type is None:
            compute_type = "int8_float16" if dev == "cuda" else "int8"
        self.model = WhisperModel(model_id, device=dev, device_index=index,
                                  compute_type=compute_type)

    def recognize(self, segment: SpeechSegment, *, n_best: int) -> list[TextCandidate]:
        if segment.waveform is None:
            raise ValueError("FasterWhisperASR needs SpeechSegment.waveform (16 kHz mono float32)")
        return self.nbest(segment.waveform, n_best=n_best)

    def nbest(self, waveform, *, n_best: int = 1, sampling_rate: int = 16_000) -> list[TextCandidate]:
        # vad_filter=False: never drop quiet/short words (§6.1 — deletion is a metric).
        segments, _ = self.model.transcribe(
            waveform, language=self.language, beam_size=self.beam_size,
            vad_filter=False, condition_on_previous_text=False, without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return [TextCandidate(f"{self.id_prefix}1", text, self.source, score=1.0, beam_rank=0)] if text else []

    def unload(self) -> None:
        import gc

        del self.model
        gc.collect()
