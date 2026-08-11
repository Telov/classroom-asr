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
