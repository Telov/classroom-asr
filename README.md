# RU↔EN Classroom ASR

Offline transcription and pronunciation-analysis pipeline for one-to-one online
English lessons taught to Russian-native beginners. Implements **Design v1.0**
(`RU_EN_Classroom_ASR_Design_Document`).

> System principle — use specialists to hear different aspects of the signal,
> preserve their uncertainty, reconstruct unknown spellings from pronunciation,
> and use a constrained whole-lesson LLM only as an evidence-aware *selector* —
> never as an unconstrained transcript writer.

This is **not** a streaming ASR product. It is an offline pipeline that:

1. keeps two logical speaker streams (teacher / student) on one lesson timeline;
2. runs complementary acoustic models and preserves their uncertainty;
3. reconstructs novel/OOV words from pronunciation (phone lattice → P2G);
4. builds a rich per-span **candidate graph**;
5. lets a role-aware LLM **select** among acoustically supported candidates using
   whole-lesson (past *and* future) context;
6. assembles a strictly **verbatim** transcript deterministically.

## Status of this repository

**Stage 0 foundation** (design-doc §20 step 1–3, §24.1, §28). The core is
implemented and tested with **zero ML dependencies** so the data model, metrics,
candidate graph, and orchestration are runnable today. Real acoustic/LLM models
plug in behind the interfaces in `pipeline/base.py` — the doc treats model names
as replaceable and the architecture as modular.

What runs now:

- immutable evidence data model + lesson-package IO (§1.2, §22);
- two-channel timeline with overlap intervals (§5);
- evaluation metrics incl. **candidate-oracle WER**, the key development gate (§18);
- session / persistent lexicon with a pronunciation index + phonetic RAG (§10.4);
- candidate-graph builder incl. MBR/consensus candidate (§12);
- pluggable pipeline stages with reference **stub** backends;
- whole-lesson global→local→global orchestrator (§15) end-to-end on synthetic data.

What is stubbed (interfaces defined, real models TODO): Qwen3-ASR backbone,
GigaAM-v3-SSL, PhoneticXEUS phone encoder, robust P2G, the 9B/12B conversation
selector, and the fusion/adapter training (§7, §8, §14, §17).

## Layout → design-doc section map

| Path | Design doc |
|---|---|
| `src/classroom_asr/types.py` | roles, audio sources, language tags (§5, §11) |
| `src/classroom_asr/timeline.py` | lesson timeline, overlap intervals (§5.4–5.5) |
| `src/classroom_asr/datamodel.py` | immutable evidence model, candidate record (§1.2, §22) |
| `src/classroom_asr/lexicon.py` | session/persistent lexicon, phonetic RAG (§10.4–10.5) |
| `src/classroom_asr/candidates.py` | candidate-graph builder, MBR/consensus (§12) |
| `src/classroom_asr/metrics.py` | WER, deletion slices, candidate-oracle WER (§18) |
| `src/classroom_asr/config.py` | frozen model/param choices as config (§26) |
| `src/classroom_asr/io.py` | lesson-package on-disk layout (§22.1) |
| `src/classroom_asr/pipeline/base.py` | stage interfaces: VAD, acoustic, phone, P2G, selector |
| `src/classroom_asr/pipeline/stubs.py` | dependency-free reference backends |
| `src/classroom_asr/pipeline/orchestrator.py` | whole-lesson inference workflow (§15.1) |
| `scripts/run_demo.py` | end-to-end run on synthetic two-channel data |

## Quick start

```bash
python -m pip install -e .            # core only, no ML deps
python scripts/run_demo.py            # run the stub pipeline end-to-end
python -m classroom_asr demo          # same, via the installed package
python -m classroom_asr demo --json   # machine-readable output
python -m pytest                      # run the test suite
```

(The `classroom-asr` console script is also installed; it needs your Python
`Scripts/` directory on `PATH`. `python -m classroom_asr` always works.)

To wire real models later: `python -m pip install -e ".[ml,data]"` and implement
the `pipeline/base.py` interfaces against the chosen checkpoints.

## Real-data oracle run (Colab, CORAAL)

`colab/CORAAL_candidate_oracle.ipynb` runs the **candidate-oracle WER gate (§18.2)**
on ~1 h of professionally transcribed, verbatim, real-world conversation
([CORAAL](https://oraal.uoregon.edu/coraal)) using a **real Whisper backend**
([backends/whisper_asr.py](src/classroom_asr/backends/whisper_asr.py)) behind our
`AcousticModel` interface — the candidate graph and oracle metric are our code, only
the fictional 2026 backbone is swapped for an existing open model. Branch A is
Whisper (attention seq2seq) and branch B is a **wav2vec2 CTC** model — a different
architecture that fails differently — so the A→A+B headroom measures genuine
complementary evidence, not just beam diversity (the multi-encoder bet, §8).

Runs on **Colab or Kaggle** (Kaggle 2×T4 recommended — the notebook shards across
all GPUs). It installs the package from GitHub; regenerate it with
`python scripts/build_colab_notebook.py`. Branches: Whisper turbo (A, 1-best),
wav2vec2 CTC (B), **Qwen3-ASR-1.7B (Z,
the design's real backbone)**, Voxtral Mini (C, subset), and a wav2vec2 phoneme path.
Includes an **error-analysis** section (S/D/I, most-deleted words, worst utterances).
CORAAL is spontaneous **English**, so read the baseline→oracle *gap*, not absolute
WER (§1.3, §18).

### Quota-friendly Kaggle iteration

A full acoustic run writes `selector_iteration_bundle.json` before starting the LLM
judge. Preserve that small file with Kaggle Files persistence, download it, or attach
it as a private dataset on the next run. The same notebook auto-detects it and then:

1. installs only the selector/scoring dependencies;
2. skips CORAAL download and every acoustic model;
3. reconstructs ROVER and oracle metrics from the saved hypotheses; and
4. runs the full Qwen3.5-9B selector.

By default, the expensive acoustic session runs only a one-batch Qwen3.5-0.8B smoke
test. This validates model loading, prompt formatting, generation, parsing, and output
plumbing without spending the full judge budget. Set
`RUN_FULL_SELECTOR_DURING_ACOUSTIC_RUN=True` for a one-session acoustics + 9B run, or
`FORCE_FULL_RUN=True` to ignore an attached/preserved bundle. Phone/IPA diagnostics are
off by default because CORAAL cannot score them and they do not feed the word selector;
enable `RUN_PHONE_DIAGNOSTICS` only for a dedicated pronunciation-path run.

## Non-goals (§2)

No streaming; no grammar/filler cleanup in the canonical transcript; no generic
diarization (channels identify speakers); no commitment to full end-to-end
fine-tuning before adapters are proven.
