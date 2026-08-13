import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "colab" / "CORAAL_candidate_oracle.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_selector_worker_does_not_require_new_helpers_from_installed_package():
    source = _notebook_source()

    assert "from classroom_asr.selector import generated_token_ids" not in source
    assert "def generated_token_ids_compat" in source
    assert "def parse_batch_compat" in source
    assert "selector_module.parse_batch = parse_batch_compat" in source


def test_embedded_selector_worker_is_valid_python():
    source = _notebook_source()
    workers = re.findall(r"f\.write\(r?'''\n(.*?)\n'''\)", source, re.DOTALL)
    worker = next((text for text in workers if "parse_batch_compat" in text), None)

    assert worker, "selector worker source was not found in the generated notebook"
    compile(worker, "sel_worker.py", "exec")


def test_every_generated_code_cell_is_valid_python():
    for index, cell in enumerate(_notebook()["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"notebook_cell_{index}.py", "exec")


def test_notebook_has_quota_friendly_iteration_path():
    source = _notebook_source()

    assert 'SELECTOR_BUNDLE_PATH = next(' in source
    assert '"schema_version": 1' in source
    assert '"references": refs' in source
    assert '"branches": {name: values' in source
    assert "RUN_PHONE_DIAGNOSTICS = False" in source
    assert "SELECTOR_SMOKE_TEST = not SELECTOR_ONLY" in source
    assert 'str(SELECTOR_MAX_LLM_CALLS)' in source
