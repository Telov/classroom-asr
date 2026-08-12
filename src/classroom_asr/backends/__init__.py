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


def snap_to_silence(waveform, target, sampling_rate, *, search_s=1.5, frame_s=0.02):
    """Return a cut index near ``target`` that falls in the quietest local frame.

    Cutting a long recording at an arbitrary timestamp can slice through a word,
    which either deletes it (both partials unrecognizable) or splits it into two
    junk tokens — a pure chunking artifact that inflates WER. Instead we search a
    ``±search_s`` region around the intended boundary and cut at the lowest-energy
    (short-frame RMS) point, so boundaries land in the pause *between* words.
    """
    import numpy as np

    n = len(waveform)
    if target >= n:
        return n
    frame = max(1, int(frame_s * sampling_rate))
    lo = max(0, target - int(search_s * sampling_rate))
    hi = min(n, target + int(search_s * sampling_rate))
    region = np.asarray(waveform[lo:hi], dtype=np.float64)
    nf = len(region) // frame
    if nf < 2:
        return target
    rms = np.sqrt((region[:nf * frame].reshape(nf, frame) ** 2).mean(axis=1))
    cut = lo + int(np.argmin(rms)) * frame + frame // 2
    return min(max(cut, lo + frame), hi)   # stay inside the region; always advance


def iter_silence_chunks(waveform, sampling_rate, chunk_s, *, search_s=1.5):
    """Yield ``(start_sample, chunk)`` covering the whole recording with
    non-overlapping, silence-snapped boundaries (contiguous: no gaps, no overlap,
    so no word is lost or duplicated). Windows are ~``chunk_s`` long."""
    n = len(waveform)
    ceil = int(chunk_s * sampling_rate)
    i = 0
    while i < n:
        end = n if (n - i) <= ceil else snap_to_silence(
            waveform, i + ceil, sampling_rate, search_s=search_s)
        if end <= i:                       # snap couldn't advance -> hard cut
            end = min(i + ceil, n)
        yield i, waveform[i:end]
        i = end


def best_dtype(torch_mod, device):
    """Pick the fastest inference dtype for ``device``.

    T4/Turing (compute capability 7.5) has fp16 and int8 tensor cores but **no bf16
    tensor-core path** — running bf16 there falls back to a slow kernel, which is why
    the bf16 LLM branches were ~20x the int8/fp16 Whisper baseline. bf16 tensor cores
    start at Ampere (cc 8.0). So: fp16 on pre-Ampere GPUs, bf16 on Ampere+, fp32 on CPU.
    """
    if not str(device).startswith("cuda") or not torch_mod.cuda.is_available():
        return torch_mod.float32
    try:
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        major = torch_mod.cuda.get_device_capability(idx)[0]
    except Exception:
        major = 0
    return torch_mod.bfloat16 if major >= 8 else torch_mod.float16


def batched_transcribe(chunks, transcribe_batch, *, batch_size, torch_mod, min_batch=1):
    """Transcribe pre-cut chunks in **mini-batches**, backing off on CUDA OOM.

    ``transcribe_batch(list_of_waveforms) -> list_of_str`` (same length/order). This
    is the speed lever for the per-chunk branches: instead of one forward/generate
    per 30 s window (dozens per interview, serial), whole batches go through the GPU
    at once. On OOM the batch size halves (to ``min_batch``) and retries, so a
    memory-tight recording still finishes. Returns the joined non-empty parts.
    """
    texts = []
    i = 0
    bs = max(1, batch_size)
    n = len(chunks)
    while i < n:
        batch = chunks[i:i + bs]
        try:
            res = transcribe_batch(batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > min_batch:
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
                bs = max(min_batch, bs // 2)
                continue
            raise
        texts.extend((t or "").strip() for t in res if t and t.strip())
        i += len(batch)
    return " ".join(texts).strip()


def chunked_transcribe(waveform, sampling_rate, transcribe_chunk, *,
                       chunk_s, min_chunk_s, torch_mod, search_s=1.5):
    """Transcribe a long recording in silence-snapped windows (~``chunk_s``).

    Boundaries are snapped to the quietest nearby frame (see
    :func:`snap_to_silence`) so no word is split across a cut. On a CUDA OOM the
    window ratchets down (halving, to a ``min_chunk_s`` floor) and retries, so a
    memory-tight recording still completes. Windows are contiguous and
    non-overlapping — every sample is covered exactly once.
    """
    parts = []
    n = len(waveform)
    ceil = int(chunk_s * sampling_rate)
    floor = int(min_chunk_s * sampling_rate)
    i = 0
    while i < n:
        end = n if (n - i) <= ceil else snap_to_silence(
            waveform, i + ceil, sampling_rate, search_s=search_s)
        if end <= i:
            end = min(i + ceil, n)
        chunk = waveform[i:end]
        if len(chunk) < 400:
            break
        try:
            text = transcribe_chunk(chunk)
            if text:
                parts.append(text)
            i = end
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and ceil > floor:
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
                ceil = max(floor, ceil // 2)
                continue
            raise
    return " ".join(parts).strip()
