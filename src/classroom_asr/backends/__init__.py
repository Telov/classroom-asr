"""Real model backends (optional; require the ``ml`` extra).

These implement the interfaces in :mod:`classroom_asr.pipeline.base` against
actual checkpoints. They are imported lazily so the core package stays
dependency-free — importing this subpackage does not pull in torch until a
concrete backend class is constructed.
"""

from __future__ import annotations


def load_pretrained(cls, model_id: str, *, dtype=None, **kwargs):
    """``from_pretrained`` that tolerates the ``torch_dtype`` -> ``dtype`` rename.

    Recent transformers deprecates ``torch_dtype`` in favor of ``dtype``; older
    ones only accept ``torch_dtype``. Try the new spelling first, fall back.
    """
    if dtype is None:
        return cls.from_pretrained(model_id, **kwargs)
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def chunked_transcribe(waveform, sampling_rate, transcribe_chunk, *,
                       chunk_s, min_chunk_s, torch_mod):
    """Transcribe a long recording in the **largest window that fits**.

    Starts at ``chunk_s`` and, on a CUDA OOM, permanently ratchets the window
    down (halving, to a ``min_chunk_s`` floor) and retries — so we use the
    longest context each GPU can hold instead of a fixed tiny chunk, and only the
    recording that actually OOMs pays the shrink. Windows are non-overlapping;
    fewer/larger windows means fewer word-splitting boundaries.
    """
    parts = []
    n = len(waveform)
    ceil = int(chunk_s * sampling_rate)
    floor = int(min_chunk_s * sampling_rate)
    i = 0
    while i < n:
        chunk = waveform[i:i + min(ceil, n - i)]
        if len(chunk) < 400:
            break
        try:
            text = transcribe_chunk(chunk)
            if text:
                parts.append(text)
            i += len(chunk)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and ceil > floor:
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
                ceil = max(floor, ceil // 2)
                continue
            raise
    return " ".join(parts).strip()
