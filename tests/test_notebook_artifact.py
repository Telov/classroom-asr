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
    match = re.search(r"f\.write\('''\n(.*?)\n'''\)", source, re.DOTALL)

    assert match, "selector worker source was not found in the generated notebook"
    compile(match.group(1), "sel_worker.py", "exec")
