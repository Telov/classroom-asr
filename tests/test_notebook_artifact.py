import ast
import io
import json
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_NOTEBOOK = ROOT / "colab" / "CORAAL_candidate_oracle_payload.ipynb"
LAUNCHER_NOTEBOOK = ROOT / "colab" / "CORAAL_candidate_oracle.ipynb"


def _notebook_source(path: Path = PAYLOAD_NOTEBOOK) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_sources(path: Path) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in _notebook(path)["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_persistent_launcher_fetches_an_immutable_payload_revision():
    notebook = _notebook(LAUNCHER_NOTEBOOK)
    source = _notebook_source(LAUNCHER_NOTEBOOK)

    assert notebook["metadata"]["classroom_asr"] == {"kind": "launcher", "schema": 1}
    assert "api.github.com/repos/{REPO}/commits/main" in source
    assert "CORAAL_candidate_oracle_payload.ipynb" in source
    assert 'os.environ["CLASSROOM_ASR_GIT_REF"] = revision' in source
    assert "shell.run_cell(source" in source
    assert "raise error" in source


def test_launcher_control_state_survives_payload_variable_collision():
    source, = _code_sources(LAUNCHER_NOTEBOOK)
    downloaded = {
        "metadata": {"classroom_asr": {"kind": "payload", "schema": 1}},
        "cells": [
            {"cell_type": "code", "source": ["payload = {'not': 'the notebook'}\n"]},
            {"cell_type": "code", "source": ["ran_after_collision = True\n"]},
        ],
    }
    responses = [
        io.BytesIO(json.dumps({"sha": "a" * 40}).encode()),
        io.BytesIO(json.dumps(downloaded).encode()),
    ]
    namespace = {}

    class Result:
        error_before_exec = None
        error_in_exec = None

    class Shell:
        def run_cell(self, cell_source, **_kwargs):
            exec(cell_source, namespace)
            return Result()

    namespace["get_ipython"] = lambda: Shell()
    with patch("urllib.request.urlopen", side_effect=responses):
        exec(source, namespace)

    assert namespace["ran_after_collision"] is True


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
    workers = []
    for source in _code_sources(PAYLOAD_NOTEBOOK):
        if "AutoModelForMultimodalLM" not in source:
            continue
        tree = ast.parse(source)
        workers.extend(
            ast.literal_eval(call.args[0])
            for call in ast.walk(tree)
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "write" and call.args
                and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str))
        )
    matches = [worker for worker in workers if "AutoModelForMultimodalLM" in worker]

    assert len(matches) == 1, "selector worker source was not found uniquely"
    compile(matches[0], "sel_worker.py", "exec")


def test_crisper_workers_serialize_shared_ct2_cache_initialization():
    source = _notebook_source()

    assert "fcntl.flock(_lock_file, fcntl.LOCK_EX)" in source
    assert 'init_lock = os.path.join(CW_WORK, "ct2_model_init.lock")' in source
    assert "inp, outp, init_lock]" in source


def test_prefetch_excludes_unused_framework_weights_and_prioritizes_first_branch():
    source = _notebook_source()

    assert 'snapshot_download(r, ignore_patterns=["*.h5", "*.msgpack"])' in source
    prefetch_repos = source.index("repos = [m for m, on in [")
    assert source.index("(FW_MODEL, True)", prefetch_repos) < source.index(
        "(VOXTRAL_MODEL, USE_VOXTRAL", prefetch_repos
    )


def test_phone_evidence_reaches_the_selector_worker_payload_and_prompt():
    source = _notebook_source()

    assert "def phone_window_pass" in source
    assert '"phone_evidence": phone_evidence' in source
    assert "def _evidence_by_slot" in source
    assert "selector_module.format_batch = format_batch_evidence" in source
    assert "phone evidence attached to" in source
    assert "lambda m, a: m.transcribe_full(a), \"PhoneticXeus\"" not in source
