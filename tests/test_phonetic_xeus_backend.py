from __future__ import annotations

import sys
from types import SimpleNamespace

from classroom_asr.backends.phonetic_xeus import PhoneticXeus


class _LoadedModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def test_remote_code_load_receives_requested_revision(monkeypatch):
    calls = []
    loaded = _LoadedModel()

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return loaded

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=_AutoModel))

    backend = PhoneticXeus("example/model", revision="f" * 40)

    assert calls == [
        ("example/model", {"revision": "f" * 40, "trust_remote_code": True})
    ]
    assert backend.revision == "f" * 40
    assert loaded.device == "cpu"


def test_remote_code_load_rejects_mutable_revision_names(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=object()))

    try:
        PhoneticXeus("example/model", revision="main")
    except ValueError as error:
        assert "full immutable" in str(error)
    else:
        raise AssertionError("mutable remote-code revision was accepted")
