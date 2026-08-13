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


def test_selector_worker_uses_the_exact_revision_constrained_choice_api():
    source = _notebook_source()

    assert "format_batch, select_graph_with_chooser" in source
    assert "def generated_token_ids_compat" not in source
    assert "def parse_batch_compat" not in source
    assert "selector_module.parse_batch" not in source


def test_canonical_selector_scores_only_advertised_next_token_ids():
    source = _notebook_source()

    assert 'tokenizer.padding_side = "left"' in source
    assert '_token_probe_text = processor.apply_chat_template(' in source
    assert 'tokenizer.encode(_token_probe_text + _letter, add_special_tokens=False)' in source
    assert 'is not one contextual token' in source
    assert 'prompt = format_batch(decisions) + "\\n1:"' in source
    assert 'model(**inputs, logits_to_keep=1, use_cache=True)' in source
    assert 'past_key_values=cache, logits_to_keep=1, use_cache=True' in source
    assert 'selector attempt {attempt_number}: prefill' in source
    assert 'selector progress: {_stats[\'completed_decisions\']}/{_total_decisions}' in source
    assert 'selector OOM: splitting {len(decisions)} decisions' in source
    assert 'selector plan: {_total_decisions} contested decisions' in source
    assert "conversations = [[" not in source[source.index("# Candidate IDs are deliberately"):]
    assert "model.generate(" not in source[source.index("# Candidate IDs are deliberately"):]
    assert 'graph, score_choices, batch_size=24, evidence_by_slot=evidence' in source
    assert '"llm_selected_wer": (round(llm_selected_wer, 4)' in source
    assert '"llm_selector_stats"' in source
    assert 'thresholds = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]' in source
    assert '"llm_selector_margin_wer"' in source
    assert 'canonical_source = "constrained LLM selector"' in source
    assert "fused_rover_wer" not in source


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


def test_embedded_crisper_worker_is_valid_python_after_literal_decoding():
    workers = []
    for source in _code_sources(PAYLOAD_NOTEBOOK):
        if "CrisperWhisperModel" not in source or "f.write" not in source:
            continue
        tree = ast.parse(source)
        workers.extend(
            ast.literal_eval(call.args[0])
            for call in ast.walk(tree)
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "write" and call.args
                and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str))
        )
    matches = [worker for worker in workers if "CrisperWhisperModel" in worker]

    assert len(matches) == 1, "CrisperWhisper worker source was not found uniquely"
    compile(matches[0], "cw_worker.py", "exec")


def test_crisper_workers_serialize_shared_ct2_cache_initialization():
    source = _notebook_source()

    assert "fcntl.flock(_lock_file, fcntl.LOCK_EX)" in source
    assert 'init_lock = os.path.join(CW_RUN, "ct2_model_init.lock")' in source
    assert "inp, outp, init_lock]" in source
    assert "if main_arg == size:" in source
    assert "Worker 0 may have completed conversion while this worker waited" in source
    assert 'm = CrisperWhisperModel(main_arg, backend="ct2", cache_dir=cache_dir)' in source


def test_completed_crisper_conversion_skips_source_prefetch_and_balances_whole_files():
    source = _notebook_source()

    assert "def _cw_converted_ready():" in source
    assert 'key = f"{slug}_float16_{hashlib.sha256(repo.encode()).hexdigest()[:12]}"' in source
    assert 'and os.path.isfile(os.path.join(candidate, ".conversion_complete"))' in source
    assert "USE_CRISPER and not _cw_converted_ready()" in source
    assert "for path in sorted(paths, key=lambda p: duration_s[p], reverse=True)" in source
    assert "Crisper whole-file GPU plan:" in source


def test_derived_crisper_and_selector_handoffs_are_always_deleted():
    source = _notebook_source()

    assert "shutil.rmtree(CW_RUN, ignore_errors=True)" in source
    assert "shutil.rmtree(SEL_RUN, ignore_errors=True)" in source
    assert "derived inference handoffs are deleted every run" in source


def test_crisper_verbatize_pairs_qwen_text_with_the_same_temporary_audio_window():
    source = _notebook_source()

    assert "USE_CRISPER_VERBATIZE = USE_CRISPER and USE_QWEN3ASR" in source
    assert "qwen_windows_by_path = {" in source
    assert 'WINDOW_PARTS.get("A+B+Qwen3"' in source
    assert "from crisperwhisper.audio import load_audio" in source
    assert 'm.verbatimize(audio[start:end], transcript, language="en", sr=16000)' in source
    assert '("CrisperQwenVerbatize", globals().get("hyp_CWV"))' in source
    assert '"CWV_crisper_qwen_verbatize"' in source


def test_silence_windows_are_cached_only_in_memory_for_the_current_run():
    source = _notebook_source()

    assert "_WINDOW_RECORD_CACHE = {}" in source
    assert "session-only audio views; never persisted or reused across runs" in source
    assert "if key not in _WINDOW_RECORD_CACHE:" in source
    assert "return _WINDOW_RECORD_CACHE[key]" in source


def test_crisper_uses_validated_non_speculative_ct2_and_persists_only_converted_model():
    source = _notebook_source()

    assert 'CRISPER_VERSION      = "2.0.2"' in source
    assert 'CW_ENV_SPEC = f"crisperwhisper[ct2]=={CRISPER_VERSION}"' in source
    assert "CrisperWhisper runtime smoke check failed" in source
    assert 'CW_CT2_CACHE = os.path.join(CW_WORK, "ct2_models")' in source
    assert "CRISPER_SPECULATIVE" not in source
    assert "CRISPER_DRAFT_SIZE" not in source
    assert "draft_model=" not in source
    assert "speculative_decoding" not in source
    assert 'm = CrisperWhisperModel(main_arg, backend="ct2", cache_dir=cache_dir)' in source
    assert 'os.path.isfile(os.path.join(candidate, ".conversion_complete"))' in source
    assert '"transcript"' not in source[source.index("def _cached_model_arg"):source.index(
        "# The first CT2 load", source.index("def _cached_model_arg")
    )]


def test_audio_derived_worker_handoffs_are_session_temporary_not_persisted():
    source = _notebook_source()

    assert 'CW_RUN = tempfile.mkdtemp(prefix="classroom_asr_cw_")' in source
    assert 'SEL_RUN = tempfile.mkdtemp(prefix="classroom_asr_selector_")' in source
    assert 'os.path.join(CW_RUN, f"in_{gi}.json")' in source
    assert 'os.path.join(SEL_RUN, "in.json")' in source
    assert 'os.path.join(CW_WORK, f"in_{gi}.json")' not in source
    assert 'os.path.join(SEL_WORK, "in.json")' not in source
    assert 'glob.glob(os.path.join(CW_WORK, "in_*.json"))' in source
    assert 'for _legacy_name in ("sel_worker.py", "in.json", "out.json")' in source


def test_selector_probe_allows_slow_cold_imports_and_surfaces_the_real_failure():
    source = _notebook_source()

    assert 'text=True, timeout=300' in source
    assert 'result.stderr or result.stdout' in source
    assert '[-2400:]' in source
    assert 'selector venv prewarm failed:' in source


def test_prefetch_excludes_unused_framework_weights_and_prioritizes_first_branch():
    source = _notebook_source()

    assert 'snapshot_download(r, ignore_patterns=["*.h5", "*.msgpack"])' in source
    prefetch_repos = source.index("repos = [m for m, on in [")
    assert source.index("(FW_MODEL, True)", prefetch_repos) < source.index(
        "(VOXTRAL_MODEL, USE_VOXTRAL", prefetch_repos
    )


def test_accuracy_shadow_uses_auto_language_known_good_qwen_windows_and_full_whisper():
    source = _notebook_source()

    assert 'QWEN_CHUNK_S = 30' in source
    assert 'QWEN_MAX_NEW_TOKENS = 512' in source
    assert 'VOXTRAL_CHUNK_S = 30' in source
    assert 'Qwen3ASR(\n                            QWEN3ASR_MODEL, language=None' in source
    assert 'max_new_tokens=QWEN_MAX_NEW_TOKENS' in source
    assert 'FW_QUALITY_MODEL = "Systran/faster-whisper-large-v3"' in source
    assert 'compute_type="float16", beam_size=5' in source
    assert '("WhisperLargeV3", globals().get("hyp_A3"))' in source


def test_no_vad_whisper_shadow_reuses_baseline_models_and_stays_a_separate_branch():
    source = _notebook_source()

    assert "USE_WHISPER_NO_VAD_SHADOW = True" in source
    assert 'with stage("Whisper turbo load (shared)")' in source
    assert 'hyp_A = _turbo_pass(WHISPER_VAD, "A whisper")' in source
    assert 'hyp_ANV = _turbo_pass(False, "Whisper no-VAD shadow")' in source
    assert 'add_branch("+WhisperNoVAD", hyp_ANV)' in source
    assert '("WhisperNoVAD", globals().get("hyp_ANV"))' in source
    assert '"WhisperNoVAD": ("Whisper turbo load (shared)",' in source
    assert '"ANV_whisper_turbo_no_vad"' in source


def test_window_batches_emit_progress_and_fail_independently():
    source = _notebook_source()

    assert 'progress = tqdm(total=len(shard), desc=f"{desc}:{pos}"' in source
    assert "for offset in range(0, len(shard), batch_size):" in source
    assert "backend returned {len(texts)} texts for {len(batch_tasks)} windows" in source
    assert 'texts = [""] * len(batch_tasks)' in source
    assert "progress.update(len(batch_tasks))" in source
    assert '[Voxtral shared] FAILED:' in source
    assert "if vmodels is not None: _free_models(vmodels)" in source


def test_partial_multi_gpu_loads_are_released_and_crisper_emits_heartbeats():
    source = _notebook_source()

    load_models = source[source.index("def load_models(make_model):"):source.index(
        "def whole_rec", source.index("def load_models(make_model):")
    )]
    assert "models.append(make_model(device))" in load_models
    assert 'model load {index + 1}/{len(GPUS)} on {device}: started' in load_models
    assert "for model in models:" in load_models
    assert "with torch.cuda.device(g): torch.cuda.empty_cache()" in load_models
    assert 'print("CW independent:", os.path.basename(p), "started"' in source
    assert 'f"{wi + 1}/{len(rows)} windows"' in source


def test_voxtral_verbatim_prompt_uses_aae_as_context_without_identity_guessing():
    source = (ROOT / "src" / "classroom_asr" / "backends" / "voxtral_asr.py").read_text(
        encoding="utf-8"
    )

    assert "African American English (AAE)" in source
    assert "never infer or invent a word from identity" in source
    assert "correct grammar, normalize dialect" in source


def test_overlap_mode_skips_selector_and_its_now_unused_phone_branches():
    source = _notebook_source()

    assert 'USE_LLM_SELECTOR = False; SELECTOR_MODEL = "Qwen/Qwen3.5-9B"' in source
    assert "USE_PHONE      = USE_LLM_SELECTOR" in source
    assert "USE_PHONETIC_XEUS = USE_LLM_SELECTOR" in source
    assert "(SELECTOR_MODEL, USE_LLM_SELECTOR)" in source


def test_word_branch_overlap_ablation_is_reported_without_rerunning_asr():
    source = _notebook_source()

    assert "=== word-branch overlap: exact leave-one-out floor + graph-oracle effect ===" in source
    assert '"unique_reference_hits": len(_unique)' in source
    assert '"recall_floor_increase_if_removed"' in source
    assert '"realizable_oracle_without"' in source
    assert '"realizable_oracle_increase_if_removed"' in source
    assert "_oracle_without = realizable_oracle_wer(_other_branches)" in source
    assert '"branch_overlap_ablation"' in source
    assert "=== pairwise overlap of correctly recovered reference occurrences ===" in source
    assert '"smaller_hit_set_covered_fraction"' in source
    assert '"branch_pair_overlap"' in source


def test_all_qwen_anchored_branch_subsets_emit_an_accuracy_runtime_pareto_frontier():
    source = _notebook_source()

    assert "_optional_indices = list(range(1, len(_wb_named)))" in source
    assert "for _chosen_optional in combinations(_optional_indices, _count):" in source
    assert '"Qwen3-ASR": ("A+B+Qwen3",)' in source
    assert '"Voxtral": ("Voxtral load (shared)", "+Voxtral")' in source
    assert '"CrisperQwenVerbatize": ("+CrisperWhisper",)' in source
    assert '"realizable_oracle_wer": realizable_oracle_wer(_subset_pool)' in source
    assert 'round(_candidate["realizable_oracle_wer"], 4)' in source
    assert "branch_subset_pareto.append({" in source
    assert '"branch_subset_pareto"' in source


def test_whole_recording_candidate_graphs_use_cached_compiled_alignment():
    source = _notebook_source()

    assert "def fast_graph_opcodes(pivot, hypothesis):" in source
    assert "_GRAPH_OPCODE_CACHE[key] = Levenshtein.opcodes(pivot, hypothesis).as_list()" in source
    assert "g = _bg(tls, pivot_index=0, opcodes_fn=fast_graph_opcodes)" in source
    assert "opcodes_fn=fast_graph_opcodes" in source
    assert "opcodes_fn=lambda pivot, hyp: Levenshtein.opcodes(pivot, hyp).as_list()" in source


def test_summary_identifies_run_and_reports_each_branch_error_shape():
    source = _notebook_source()

    assert "RUN_STARTED_EPOCH = _run_time.time()" in source
    assert "def error_counts_of(hyps):" in source
    assert '"substitutions": S, "deletions": D, "insertions": I' in source
    assert '"interviews_with_text"' in source
    assert '"source_git_ref": ASR_GIT_REF' in source
    assert '"overall_wall_seconds"' in source
    assert '"resolved_hf_revisions"' in source
    assert '"branch_metrics": branch_metrics' in source
    assert '"run_fingerprint": run_fingerprint' in source


def test_phone_evidence_reaches_the_selector_worker_payload_and_prompt():
    source = _notebook_source()

    assert "def phone_window_pass" in source
    assert '"phone_evidence": phone_evidence' in source
    assert "def _evidence_by_slot" in source
    assert "best_phone_subsequence" in source
    assert "selector compact phone evidence:" in source
    assert "candidate-local chars=" in source
    assert "format_batch(decisions)" in source
    assert "evidence_by_slot=evidence" in source
    assert "phone evidence attached to" in source
    assert "lambda m, a: m.transcribe_full(a), \"PhoneticXeus\"" not in source


def test_selector_threshold_curve_reuses_the_scored_graph_without_realignment():
    source = _notebook_source()

    assert "prepared.append((graph, evidence))" in source
    assert "selector_module.assemble(graph, gated)" in source
    assert "threshold variants reassembled from the existing graph" in source
    threshold_loop = source[source.index("for threshold in thresholds:"):]
    assert "select_transcript_with_chooser(" not in threshold_loop
