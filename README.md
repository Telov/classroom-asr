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

The dependency-free core remains usable with **zero ML dependencies**, while the
Kaggle benchmark now exercises real acoustic backends behind the same interfaces.
The architecture is deliberately modular: benchmark branches measure candidate
quality and complementarity before any model is promoted into the production stack.

What runs now:

- immutable evidence data model + lesson-package IO (§1.2, §22);
- two-channel timeline with overlap intervals (§5);
- evaluation metrics incl. **candidate-oracle WER**, the key development gate (§18);
- session / persistent lexicon with a pronunciation index + phonetic RAG (§10.4);
- candidate-graph builder incl. MBR/consensus candidate (§12);
- pluggable pipeline stages with reference **stub** backends;
- whole-lesson global→local→global orchestrator (§15) end-to-end on synthetic data;
- real Qwen3-ASR, faster-whisper, wav2vec2 CTC/phone, Voxtral, CrisperWhisper,
  and PhoneticXEUS backend adapters;
- a reproducible Kaggle CORAAL benchmark with exact branch error counts,
  Qwen-anchored candidate graphs, exact lattice realizable-oracle WER, architecture-aware
  leave-one-out overlap, per-interview error shapes, and marginal runtime attribution.

What remains research/integration work: robust lattice-aware P2G, calibrated
confidence, the production whole-lesson constrained selector, learned multi-encoder
fusion/adapters, and training on the actual RU↔EN classroom distribution. GigaAM is
intentionally excluded from the current English benchmark.

## Layout → design-doc section map

| Path | Design doc |
|---|---|
| `src/classroom_asr/types.py` | roles, audio sources, language tags (§5, §11) |
| `src/classroom_asr/timeline.py` | lesson timeline, overlap intervals (§5.4–5.5) |
| `src/classroom_asr/datamodel.py` | immutable evidence model, candidate record (§1.2, §22) |
| `src/classroom_asr/lexicon.py` | session/persistent lexicon, phonetic RAG (§10.4–10.5) |
| `src/classroom_asr/candidates.py` | candidate-graph builder, MBR/consensus (§12) |
| `src/classroom_asr/candidate_graph.py` | Qwen-pivoted whole-transcript lattice + exact oracle (§12, §18) |
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
python -m pytest                      # run the full test suite
```

(The `classroom-asr` console script is also installed; it needs your Python
`Scripts/` directory on `PATH`. `python -m classroom_asr` always works.)

To wire real models later: `python -m pip install -e ".[ml,data]"` and implement
the `pipeline/base.py` interfaces against the chosen checkpoints.

## Real-data oracle run (Kaggle, CORAAL)

Import [`colab/CORAAL_candidate_oracle.ipynb`](colab/CORAAL_candidate_oracle.ipynb)
into Kaggle once and keep using that same notebook. It is a small persistent launcher:
every **Run All** resolves the current `main` commit, downloads the canonical payload
from that immutable revision, pins the package install to the same revision, and runs
all payload cells. Do not create a new Kaggle notebook or copy cells after project
updates.

Recommended Kaggle settings are **GPU T4 ×2**, Internet **On**, and file persistence
**On**. Persistence reuses only dependency environments, downloaded CORAAL archives,
and the bounded Crisper CT2 conversion; transcripts, timestamps, IPA/phone evidence,
selector inputs, and other inference products are regenerated every run.

The benchmark evaluates roughly one hour of professionally transcribed, spontaneous
English conversation from [CORAAL](https://oraal.uoregon.edu/coraal). Active branches
are:

- Whisper large-v3-turbo with VAD (baseline) and a same-load no-VAD quiet-word shadow;
- full Whisper large-v3 FP16 beam-5 quality shadow;
- wav2vec2 CTC;
- **Qwen3-ASR-1.7B**, the primary multilingual backbone, with automatic language detection;
- Voxtral Mini clean and AAE-aware verbatim modes on one shared model load;
- independent CrisperWhisper and Qwen-conditioned Crisper Verbatimize.

The selector and its phone/IPA inputs are temporarily paused while upstream branches
undergo exact overlap and leave-one-out oracle analysis. The final JSON includes corpus and
per-interview S/D/I, deletion rates, model/runtime fingerprints, observed loaded Hub revisions
plus a separately labeled cache inventory,
unique recovered-word categories, pairwise overlap, exact optional-branch leave-one-out oracle
deltas, and the marginal runtime each removal would actually save after shared loads are accounted
for. Qwen remains the required graph pivot rather than being presented as a removable peer. Read
the Qwen→realizable-oracle gap and deletion slices, not just absolute WER (§18).

## Non-goals (§2)

No streaming; no grammar/filler cleanup in the canonical transcript; no generic
diarization (channels identify speakers); no commitment to full end-to-end
fine-tuning before adapters are proven.
