"""Real model backends (optional; require the ``ml`` extra).

These implement the interfaces in :mod:`classroom_asr.pipeline.base` against
actual checkpoints. They are imported lazily so the core package stays
dependency-free — importing this subpackage does not pull in torch until a
concrete backend class is constructed.
"""

from __future__ import annotations


def tune_generation_config(model):
    """Fix at the source the two benign generate-time warnings, on whatever nested module
    actually holds the ``generation_config``:

    * "Setting pad_token_id to eos_token_id …" — set ``pad_token_id`` so generate doesn't
      have to infer it every call.
    * "generation flags are not valid: ['temperature'] …" — we decode greedily
      (``do_sample=False``), so drop the sampling-only flags the checkpoint ships.

    Returns True if it found and adjusted at least one config. Best-effort and side-effect
    free otherwise (some wrappers, e.g. qwen-asr, nest the HF model a level or two down)."""
    seen, found = set(), False

    def eos_scalar(gc):
        eos = getattr(gc, "eos_token_id", None)
        return eos[0] if isinstance(eos, (list, tuple)) and eos else eos

    def visit(obj, depth=0):
        nonlocal found
        if obj is None or depth > 3 or id(obj) in seen:
            return
        seen.add(id(obj))
        gc = getattr(obj, "generation_config", None)
        if gc is not None:
            if getattr(gc, "pad_token_id", None) is None:
                eos = eos_scalar(gc)
                if eos is not None:
                    gc.pad_token_id = eos
            if not getattr(gc, "do_sample", False):
                for k in ("temperature", "top_p", "top_k"):
                    if getattr(gc, k, None) is not None:
                        setattr(gc, k, None)
            found = True
        for attr in ("model", "llm", "language_model", "thinker", "generation_model"):
            visit(getattr(obj, attr, None), depth + 1)

    visit(model)
    return found


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


def batched_transcribe_list(chunks, transcribe_batch, *, batch_size, torch_mod, min_batch=1):
    """Transcribe pre-cut chunks in **mini-batches**, returning one text **per chunk**.

    ``transcribe_batch(list_of_waveforms) -> list_of_str`` (same length/order). Instead
    of one forward/generate per window (serial), whole batches go through the GPU at
    once. On CUDA OOM the batch size halves (to ``min_batch``) and retries. The result
    is aligned 1:1 with ``chunks`` (empty string where a chunk produced nothing), so a
    caller can reassemble per-recording transcripts after distributing windows freely.
    """
    texts = [""] * len(chunks)
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
        for j, t in enumerate(res):
            texts[i + j] = (t or "").strip()
        i += len(batch)
    return texts


def batched_transcribe(chunks, transcribe_batch, *, batch_size, torch_mod, min_batch=1):
    """Like :func:`batched_transcribe_list` but joins the non-empty parts into one string."""
    parts = batched_transcribe_list(chunks, transcribe_batch, batch_size=batch_size,
                                    torch_mod=torch_mod, min_batch=min_batch)
    return " ".join(t for t in parts if t).strip()


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
