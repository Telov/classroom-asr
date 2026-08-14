import os
from unittest.mock import patch

import numpy as np

from classroom_asr.backends.gigaam_v3 import GigaAMV3RNNT


class _FakeModel:
    def __init__(self):
        self.paths = []

    def transcribe(self, path):
        self.paths.append(path)
        return "привет на границе"


def test_transcribe_full_uses_overlap_merge_and_removes_session_wavs():
    backend = object.__new__(GigaAMV3RNNT)
    backend.model = _FakeModel()
    backend.core_s = 20.0
    backend.overlap_s = 2.0
    waveform = np.ones(23 * 1_000, dtype=np.float32)

    text = backend.transcribe_full(waveform, sampling_rate=1_000)

    assert text == "привет на границе"
    assert len(backend.model.paths) == 2
    assert all(not os.path.exists(path) for path in backend.model.paths)


def test_rejects_mutable_remote_code_revision_before_importing_ml_stack():
    with patch.dict("sys.modules", {"torch": None, "transformers": None}):
        try:
            GigaAMV3RNNT(revision="rnnt")
        except ValueError as exc:
            assert "immutable commit" in str(exc)
        else:
            raise AssertionError("mutable revision was accepted")
