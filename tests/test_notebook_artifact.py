import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_NOTEBOOK = ROOT / "colab" / "CORAAL_candidate_oracle_payload.ipynb"
LAUNCHER_NOTEBOOK = ROOT / "colab" / "CORAAL_candidate_oracle.ipynb"


def _notebook_source(path: Path = PAYLOAD_NOTEBOOK) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_persistent_launcher_fetches_an_immutable_payload_revision():
    notebook = _notebook(LAUNCHER_NOTEBOOK)
    source = _notebook_source(LAUNCHER_NOTEBOOK)

    assert notebook["metadata"]["classroom_asr"] == {"kind": "launcher", "schema": 1}
    assert "api.github.com/repos/{REPO}/commits/main" in source
    assert "CORAAL_candidate_oracle_payload.ipynb" in source
    assert 'os.environ["CLASSROOM_ASR_GIT_REF"] = revision' in source
    assert "shell.run_cell(source" in source
    assert "raise error" in source


def test_payload_is_marked_and_installs_its_exact_revision():
    notebook = _notebook(PAYLOAD_NOTEBOOK)
    source = _notebook_source()

    assert notebook["metadata"]["classroom_asr"] == {"kind": "payload", "schema": 1}
    assert 'ASR_GIT_REF = os.environ.get("CLASSROOM_ASR_GIT_REF", "main")' in source
    assert "classroom-asr.git@{ASR_GIT_REF}" in source


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
