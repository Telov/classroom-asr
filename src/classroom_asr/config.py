"""Frozen model choices and pipeline hyperparameters as data (§26, §3).

This encodes the *decisions* from the model decision matrix (§26) and the
tunable knobs scattered through the doc, so experiments change config rather
than code. Model identifiers are the doc's 2026-era names; they are pinned to
concrete checkpoint revisions only when a real backend is wired (§7, §21.1).

Everything is a plain dataclass with ``to_dict``/``from_dict`` so a config can
be round-tripped to JSON without a YAML dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelChoices:
    """The frozen-for-v1 backbone plus the A/B challengers (§26)."""

    asr_backbone: str = "Qwen3-ASR-1.7B"                 # §7.1, frozen
    russian_specialist: str = "GigaAM-v3-SSL"            # §7.2, frozen; A/B multilingual SSL
    russian_word_branch: str = "GigaAM-v3-RNNT"          # production 1-best Russian transcript
    russian_word_revision: str = "c7f128b8accdd9624df905e5c2d7b7a48c27c0d8"
    phone_encoder: str = "PhoneticXEUS"                  # §7.3, default; A/B ZIPA/POWSM
    phone_encoder_challengers: tuple[str, ...] = ("ZIPA-CTC-NS", "POWSM-CTC")
    selector: str = "Qwen3.5-9B"                         # §14.2, default judge
    selector_challenger: str = "Gemma-4-12B"             # §14.2, A/B
    baselines: tuple[str, ...] = (
        "Qwen3-ASR-1.7B",
        "GigaAM-v3-RNNT",
        "Whisper-large-v3",
        "Scribe-v2-batch",
        "Scribe-v2-batch+keyterms",
    )


@dataclass
class VADConfig:
    """Non-destructive VAD (§6.1): padding is generous, nothing is discarded."""

    pre_pad_s: float = 0.75          # §6.1 recommends ~0.5–1.0 s
    post_pad_s: float = 0.75
    merge_gap_s: float = 0.2
    keep_full_waveform: bool = True  # §6.1: never permanently discard audio


@dataclass
class CandidateConfig:
    """N-best breadth and expansion policy (§12)."""

    qwen_nbest: int = 12             # §12.1: 8–16, tune by oracle-WER gain
    gigaam_nbest: int = 1            # official v3 RNNT public API is 1-best
    phone_topk: int = 4              # §10.2 top-K phone paths
    p2g_nbest: int = 4
    rag_top_k: int = 5
    rag_min_similarity: float = 0.5
    keep_no_bias_diagnostic: bool = True   # §12.1 acoustic-faithfulness candidate
    expand_confidence_floor: float = 0.85  # §12.5
    expand_agreement_floor: float = 0.99   # §12.5


@dataclass
class SelectorConfig:
    """Conversation LLM selector behavior (§14)."""

    context_tokens: int = 262_144    # §14.2/§14.3: 262k comfortable for a lesson
    allow_novel: bool = False        # §14.5: NEW output disabled initially
    retrieve_before_turns: int = 4   # compact retrieval package, not full history
    retrieve_after_turns: int = 4    # future context is a deliberate advantage (§15.2)


@dataclass
class LossWeights:
    """Deletion-weighted, multi-task training weights (§17.2, §17.4).

    Initial values are hyperparameters, not doctrine (§17.4).
    """

    lambda_phone: float = 0.3
    lambda_switch: float = 0.3
    lambda_nonce: float = 0.3
    lambda_deletion: float = 0.5
    lambda_verbatim: float = 0.3
    lambda_rank: float = 0.5
    # deletion up-weights (§17.4)
    w_short_word_del: float = 2.5
    w_low_energy_del: float = 2.5
    w_switch_boundary_del: float = 2.0
    w_nonce_del: float = 3.5


@dataclass
class EvalConfig:
    """Slice thresholds for the evaluation spec (§18.1/§18.3)."""

    short_word_ms: float = 300.0     # §18.1 duration bucket
    switch_window: int = 2           # §18.1 ±1–3 words around a switch
    bootstrap_resamples: int = 1000  # §18.4 paired bootstrap CIs


@dataclass
class Config:
    models: ModelChoices = field(default_factory=ModelChoices)
    vad: VADConfig = field(default_factory=VADConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sample_rate: int = 16_000        # normalized model input (§5.1)
    vram_budget_gb: float = 40.0     # §3 hard constraint

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        def sub(key: str, klass):
            return klass(**d[key]) if key in d and isinstance(d[key], dict) else klass()

        cfg = cls(
            models=sub("models", ModelChoices),
            vad=sub("vad", VADConfig),
            candidates=sub("candidates", CandidateConfig),
            selector=sub("selector", SelectorConfig),
            loss=sub("loss", LossWeights),
            eval=sub("eval", EvalConfig),
        )
        cfg.sample_rate = int(d.get("sample_rate", cfg.sample_rate))
        cfg.vram_budget_gb = float(d.get("vram_budget_gb", cfg.vram_budget_gb))
        return cfg


DEFAULT_CONFIG = Config()
