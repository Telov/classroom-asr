# ASR Project Guidance

## Sources of truth

- The authoritative architecture is `docs/design/RU_EN_Classroom_ASR_Design_Document.docx`.
- The design document is private and local-only. It must remain ignored by Git and must never be
  committed, pushed, uploaded, or otherwise published.
- The project implements an offline, two-channel teacher/student RU<->EN classroom transcription
  and pronunciation-analysis pipeline.
- Preserve complementary acoustic evidence until late in the pipeline. The canonical transcript
  is strictly verbatim: do not clean grammar, fillers, repetitions, false starts, slang,
  code-switches, learner errors, or nonce forms.
- The phone path is functional evidence, not a diagnostic printout. Preserve phone uncertainty,
  feed phone/IPA evidence into P2G, phonetic retrieval, and the constrained conversation selector,
  and keep realized pronunciation separate from selected orthography.
- The final LLM is a constrained judge over candidate IDs. It must not freely rewrite a plausible
  transcript. The `NEW` escape hatch remains gated as described in the design document.
- Candidate-oracle WER and the candidate recall floor are different metrics. Do not label the
  recall floor as WER or as a realizable transcript.
- Qwen3-ASR is the primary word-transcript backbone. Align other word candidates to Qwen and keep
  the Qwen token whenever the constrained selector is not asked, abstains, or fails. Do not create
  or score a majority-vote/ROVER transcript variant; branch agreement is only an uncertainty
  signal. On a successful run, the constrained selector output is the canonical transcript.
- Feed the selector compact candidate-local phonetic evidence: expected pronunciation in local
  context, accent-aware match scores, and localized realized-phone excerpts. Do not place entire
  multi-second IPA windows in the LLM prompt.

## Document reading and project notes

- Prefer the globally installed Docling CLI for extracting structured content from DOCX, PDF,
  PPTX, HTML, and similar documents. A normal local conversion is:

  `docling convert <source> --to md --output <temporary-output-directory> --abort-on-error`

- Use temporary output for exploratory conversions. Do not commit generated Docling Markdown or
  JSON unless it is intentionally part of the project.
- Docling extraction is not visual-layout validation. When layout, figures, page breaks, comments,
  or tracked changes matter, also render and inspect the original document with the appropriate
  document/PDF workflow.
- Keep durable project instructions and decisions in this `AGENTS.md` or a clearly linked file
  under `docs/`. Do not rely on private assistant memory directories as the only copy.
- Update notes when a decision materially changes. Prefer current verified facts over stale session
  history, and identify unresolved assumptions explicitly.

## Collaboration and decisions

- Ask the user before making a choice that could materially change project development. This
  includes architecture, model families or versions, dependency/runtime strategy, evaluation or
  normalization semantics, benchmark scope, quality/speed tradeoffs, data retention, deployment
  workflow, or new manual steps for the user.
- Ask when competing interpretations could lead to meaningfully different implementations. State
  the evidence, alternatives, and likely consequences so the user can decide.
- Do not ask about ordinary low-risk implementation details that can be discovered locally and do
  not alter the project direction.
- Do not introduce user parameters, manual artifact transfer, intermediate-save workflows, or
  repeated setup steps without prior approval. The intended Kaggle workflow is zero-touch after
  starting the notebook, and the user turns sessions off after runs.
- Do not persist or reuse transcripts, branch hypotheses, timestamps, IPA/phone outputs, selector
  inputs, or other derived inference results across runs. Dependency environments and model files
  may be cached; transcription and phonetic evidence must be regenerated from audio every run.
- Optimize execution and iteration time without silently removing design-required evidence or
  lowering the intended acceptance test.

## Repository workflow

- Make coherent Git commits habitually after changes are verified. Keep generated notebook changes
  in the same commit as their generator source and tests.
- Push verified commits to `origin/main` routinely without asking for separate authorization, so
  fresh Kaggle runtimes receive current package and notebook code. Still ask before force-pushing,
  rewriting published history, deleting remote refs, or making other destructive remote changes.
- After changing notebook structure, regenerate
  `colab/CORAAL_candidate_oracle_payload.ipynb` with `python scripts/build_colab_notebook.py`.
- Run `python -m pytest -q` for the full suite. Also compile embedded/generated worker code when it
  changes and verify the notebook contains no stale execution output.
- Preserve unrelated user changes in a dirty worktree.

## Search and discovery

- Prefer `rg` for content, `fd` for filenames, and `sg`/ast-grep for structural code searches.
- These tools can honor ignore rules. When a requested file may be hidden, ignored, untracked,
  binary, outside the repository, or stored by another application, use raw filesystem discovery
  such as PowerShell `Get-ChildItem -Force` and the relevant application storage as well.
- Do not conclude that a document is absent solely because `rg --files` did not list it.

## Runtime constraints worth preserving

- Keep heavy ML dependencies lazy/isolated so the package's core remains usable without loading
  model stacks.
- The main Kaggle runtime and the selector may require incompatible Transformers versions; preserve
  the selector's isolated environment unless a verified replacement is agreed upon.
- One failed optional branch should not destroy an expensive benchmark run. Fail visibly and fall
  back only where the result remains honest and clearly labeled.
- Never use reference transcript boundaries or other gold information to improve model inference in
  the whole-recording benchmark.
