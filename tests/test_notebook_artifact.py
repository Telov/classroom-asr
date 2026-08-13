import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "colab" / "CORAAL_candidate_oracle.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_selector_worker_does_not_require_new_helpers_from_installed_package():
    source = _notebook_source()

    assert "from classroom_asr.selector import generated_token_ids" not in source
    assert "def generated_token_ids_compat" in source
    assert "def parse_batch_compat" in source
    assert "selector_module.parse_batch = parse_batch_compat" in source


def test_embedded_selector_worker_is_valid_python():
    source = _notebook_source()
    workers = re.findall(r"f\.write\('''\n(.*?)\n'''\)", source, re.DOTALL)
    matches = [worker for worker in workers if "AutoModelForMultimodalLM" in worker]

    assert len(matches) == 1, "selector worker source was not found uniquely"
    compile(matches[0], "sel_worker.py", "exec")


def test_phone_evidence_reaches_the_selector_worker_payload_and_prompt():
    source = _notebook_source()

    assert "def phone_window_pass" in source
    assert '"phone_evidence": phone_evidence' in source
    assert "def _evidence_by_slot" in source
    assert "selector_module.format_batch = format_batch_evidence" in source
    assert "phone evidence attached to" in source
    assert "lambda m, a: m.transcribe_full(a), \"PhoneticXeus\"" not in source
