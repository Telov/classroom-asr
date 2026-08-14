from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ctc_backends_require_the_prefetched_safetensors_format():
    """Do not regress to downloading legacy .bin weights alongside safetensors on Kaggle."""

    for relative in (
        "src/classroom_asr/backends/wav2vec2_ctc.py",
        "src/classroom_asr/backends/wav2vec2_phone.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "use_safetensors=True" in source
