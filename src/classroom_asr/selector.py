"""The constrained conversation-LLM selector over the candidate graph (§14).

Design §14.1: the LLM is a *judge over a constrained candidate graph* — it selects an
evidence-supported candidate per uncertain span, it does not compose a transcript from scratch.
This module is the reference-free realization of that:

* :func:`build_decisions` — from the :mod:`rover` graph, take only the **contested** slots (no
  clear majority; the confident/agreeing ones are frozen, §12.5), and package each as a
  :class:`Decision`: the labelled candidate options every branch offered (``∅`` = drop the word)
  plus a few words of surrounding context.
* :func:`format_batch` / :func:`parse_batch` — render a batch of decisions as one prompt and
  read the model's ``N:LETTER`` answers back to candidate tokens. Deterministic, unit-tested,
  and model-agnostic (the LLM call is injected as ``llm_fn``).
* :func:`select_transcript` — glue: graph → contested decisions → LLM choices → assembled
  transcript, with the ROVER majority vote as the default for every slot the LLM didn't (or
  couldn't) decide. So a bad/blank LLM answer degrades to the deterministic fusion, never worse.

The design's ``NEW`` free-form escape hatch (§14.5) is intentionally **off** here — enabled later
only for OOV/nonce spans with strong phone/P2G support.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .normalize import DEFAULT, Normalizer
from .rover import NULL, Slot, build_graph

DROP = "∅"  # shown to the model as the "drop this word" candidate
_ANSWER = re.compile(
    r"^\s*(?:[-*]\s*)?[\"']?(\d+)[\"']?\s*(?:[:.)=\-]|->)\s*"
    r"(?:[\"']?\s*)?(?:option\s+)?([A-Za-z])\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Decision:
    slot: int                                   # index into the graph
    before: list[str]                           # context words before the slot
    after: list[str]                            # context words after the slot
    options: list[tuple[str, object]]           # (letter, token); token may be NULL
    default: object                             # ROVER winner — used if the LLM abstains


def _contested(slot: Slot) -> bool:
    """A slot worth asking the LLM about: branches disagree and none holds a strict majority."""
    if len(slot.votes) <= 1:
        return False
    total = sum(slot.votes.values())
    return max(slot.votes.values()) * 2 <= total


def _labelled(slot: Slot) -> list[tuple[str, object]]:
    """Candidate options as (letter, token), most-voted first, ``NULL`` last."""
    items = sorted(slot.votes.items(), key=lambda kv: (-kv[1], kv[0] is NULL, str(kv[0])))
    return [(chr(ord("A") + i), tok) for i, (tok, _n) in enumerate(items)]


def build_decisions(graph: list[Slot], *, context: int = 8) -> list[Decision]:
    """One :class:`Decision` per contested slot; context words come from the neighbouring pivot
    ("word") slots (the interleaved "ins" slots carry no pivot word)."""
    words = [(j, s.pivot) for j, s in enumerate(graph) if s.kind == "word"]
    out: list[Decision] = []
    for i, s in enumerate(graph):
        if not _contested(s):
            continue
        before = [w for j, w in words if j < i][-context:]
        after = [w for j, w in words if j > i][:context]
        out.append(Decision(i, before, after, _labelled(s), s.winner()))
    return out


def _opt_text(tok) -> str:
    return f"{DROP}(drop)" if tok is NULL else str(tok)


def format_batch(decisions: list[Decision]) -> str:
    """Render a batch of decisions as a single instruction prompt."""
    lines = [
        "You are a transcription judge. For each item, several speech recognizers proposed a",
        "different word for one position (marked [?]). Using the surrounding context, pick the",
        "option that reads as the correct VERBATIM transcription — keep real fillers (um, uh,",
        f"yeah); {DROP}(drop) means no word belongs there. Do not invent words; choose only from the",
        "options. Answer each item on its own line as  N:LETTER  and nothing else.",
        "",
    ]
    for n, d in enumerate(decisions, 1):
        ctx = " ".join([*d.before, "[?]", *d.after]).strip()
        opts = "  ".join(f"{L}={_opt_text(tok)}" for L, tok in d.options)
        lines.append(f"{n}. context: ...{ctx}...")
        lines.append(f"   options: {opts}")
    lines += ["", "Answers:"]
    return "\n".join(lines)


def parse_batch(text: str, decisions: list[Decision]) -> dict[int, object]:
    """Map model answers back to ``{slot_index: chosen_token}``.

    The requested wire format is ``N:LETTER``, but instruction-tuned models commonly wrap that
    in Markdown or JSON, or spell it as ``N: option LETTER``.  Accept those harmless variations
    while keeping the parser constrained to a numbered item and one of that item's advertised
    letters.  Unknown, duplicated, and missing answers are skipped or overwritten by the last
    valid answer for that item.
    """
    choices: dict[int, object] = {}
    raw = text or ""
    answers: list[tuple[object, object]] = []
    try:
        payload = json.loads(raw.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        answers.extend(payload.items())
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                number = item.get("item", item.get("number", item.get("n")))
                letter = item.get("answer", item.get("choice", item.get("letter")))
                answers.append((number, letter))
    answers.extend((m.group(1), m.group(2)) for m in _ANSWER.finditer(raw))

    for number, answer in answers:
        try:
            n = int(number)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(decisions):
            d = decisions[n - 1]
            opts = dict(d.options)
            match = re.search(r"\b([A-Za-z])\b", str(answer or ""))
            if not match:
                continue
            letter = match.group(1).upper()
            if letter in opts:
                choices[d.slot] = opts[letter]
    return choices


def generated_token_ids(generated_ids, input_ids) -> list[int]:
    """Return only newly generated token IDs without depending on a Transformers convention.

    Decoder-only ``generate`` normally returns ``prompt + completion``.  Some wrappers and
    generation backends return only the completion.  Blindly slicing either form by the prompt
    length can therefore turn a valid response into an empty string.  Inputs may be tensors or
    ordinary (nested) sequences; the return value is always a plain list for easy decoding.
    """
    generated = generated_ids.tolist() if hasattr(generated_ids, "tolist") else generated_ids
    prompt = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    if generated and isinstance(generated[0], (list, tuple)):
        generated = generated[0]
    if prompt and isinstance(prompt[0], (list, tuple)):
        prompt = prompt[0]
    generated = list(generated or [])
    prompt = list(prompt or [])
    if len(generated) >= len(prompt) and generated[:len(prompt)] == prompt:
        return generated[len(prompt):]
    return generated


def assemble(graph: list[Slot], choices: dict[int, object]) -> list[str]:
    """Final tokens: the LLM's choice where it decided, the ROVER winner everywhere else."""
    out: list[str] = []
    for i, s in enumerate(graph):
        tok = choices.get(i, s.winner())
        if tok is not NULL:
            out.extend(str(tok).split())          # an "ins" candidate may be a multi-word phrase
    return out


def select_transcript(
    transcripts,
    llm_fn,
    *,
    norm: Normalizer = DEFAULT,
    batch_size: int = 24,
    context: int = 8,
) -> tuple[str, int, int]:
    """Fuse branch transcripts with the LLM judge over contested slots.

    ``llm_fn(prompt) -> str`` runs the model. Returns ``(transcript, n_decisions, n_chosen)`` —
    ``n_chosen`` is how many contested slots the LLM actually decided (the rest fell back to the
    ROVER vote, so the result is never worse than :func:`classroom_asr.rover.fuse`).
    """
    token_lists = [norm.tokens(t) for t in transcripts if t and t.strip()]
    graph = build_graph(token_lists)
    decisions = build_decisions(graph, context=context)
    choices: dict[int, object] = {}
    for b in range(0, len(decisions), batch_size):
        batch = decisions[b:b + batch_size]
        try:
            choices.update(parse_batch(llm_fn(format_batch(batch)), batch))
        except Exception:
            pass                                # abstain -> ROVER default for those slots
    return " ".join(assemble(graph, choices)).strip(), len(decisions), len(choices)
