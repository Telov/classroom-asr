"""Generate the stable Kaggle launcher notebook.

The launcher is intentionally tiny and should rarely change. Each Run All resolves the current
GitHub ``main`` commit, downloads the canonical payload notebook from that exact revision, and
executes its code cells in the launcher's live kernel. Future experiment changes therefore ship by
commit + push; the user keeps the same Kaggle notebook.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "CORAAL_candidate_oracle.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md(
        """
# Classroom ASR — persistent Kaggle launcher

Keep this notebook and use **Run All** for every experiment. It resolves the latest commit on
`Telov/classroom-asr` `main`, downloads the canonical experiment notebook from that exact commit,
and runs its cells in this kernel. No cell copying, parameter editing, or new Kaggle notebook is
needed when the project changes.

Kaggle settings remain: **GPU T4 x2**, Internet **ON**.
"""
    ),
    code(
        r"""
import json, os, urllib.request

REPO = "Telov/classroom-asr"
API_URL = f"https://api.github.com/repos/{REPO}/commits/main"
PAYLOAD_PATH = "colab/CORAAL_candidate_oracle_payload.ipynb"
HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "classroom-asr-kaggle"}

with urllib.request.urlopen(urllib.request.Request(API_URL, headers=HEADERS), timeout=60) as response:
    revision = json.load(response)["sha"]
payload_url = f"https://raw.githubusercontent.com/{REPO}/{revision}/{PAYLOAD_PATH}"
with urllib.request.urlopen(
        urllib.request.Request(payload_url, headers={**HEADERS, "Cache-Control": "no-cache"}),
        timeout=120) as response:
    payload = json.load(response)

marker = payload.get("metadata", {}).get("classroom_asr", {})
if marker.get("kind") != "payload" or marker.get("schema") != 1:
    raise RuntimeError(f"unexpected classroom-asr payload metadata: {marker!r}")

# Keep package code and notebook structure on the same immutable revision for reproducibility.
os.environ["CLASSROOM_ASR_GIT_REF"] = revision
print(f"classroom-asr payload: {revision[:12]} | {len(payload['cells'])} cells", flush=True)

def _classroom_asr_run_payload(notebook):
    # Payload cells intentionally share the Jupyter user namespace, but launcher control state
    # must not live there: experiment cells commonly use names such as ``payload`` themselves.
    # Function locals keep the immutable cell list and loop counters out of that namespace.
    shell = get_ipython()
    if shell is None:
        raise RuntimeError("the persistent launcher must run inside a Jupyter/Kaggle kernel")
    cells = tuple(notebook["cells"])
    total = len(cells)
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"\n=== payload cell {index + 1}/{total} ===", flush=True)
        result = shell.run_cell(source, store_history=False)
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            raise error

_classroom_asr_run_payload(payload)
"""
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "classroom_asr": {"kind": "launcher", "schema": 1},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, "| persistent GitHub payload launcher")
