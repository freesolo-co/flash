"""Strict benchmark plans and timestamped evidence, independent of a GPU provider."""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

Name = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
Text = Annotated[str, Field(min_length=1)]
SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Seconds = Annotated[float, Field(ge=0)]
Phase = Literal[
    "allocation",
    "startup",
    "rollout",
    "environment_reward",
    "training",
    "weight_publication",
    "checkpointing",
    "evaluation",
]
PHASES = get_args(Phase)
Workload = Literal["short_sft", "single_turn_grpo", "long_multi_turn", "concurrent_adapters"]
Ledger = Literal["infrastructure_actual", "infrastructure_estimate", "customer_charge"]
LEDGERS = get_args(Ledger)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Revisions(StrictModel):
    model: Text
    model_revision: SHA
    image: Digest
    code: SHA
    environment: Digest
    dataset: Digest
    evaluator: Digest
    adapters: dict[Text, Digest] = Field(default_factory=dict)


class Target(StrictModel):
    metric: Text
    value: float
    direction: Literal["at_least", "at_most"] = "at_least"
    min_samples: int = Field(default=1, ge=1)


class Configuration(StrictModel):
    parameters: dict[str, Any]
    seed: int = Field(ge=0, le=2**32 - 1)
    cache: Literal["cold", "warm"]


class Case(StrictModel):
    name: Name
    workload: Workload
    command: list[Text] = Field(min_length=1)
    revisions: Revisions
    configuration: dict[str, Any]
    target: Target
    timeout_s: float = Field(default=3600.0, gt=0)
    cleanup_timeout_s: float = Field(default=180.0, gt=0)
    required_phases: list[Phase] = Field(default_factory=lambda: list(PHASES), min_length=1)

    @model_validator(mode="after")
    def concurrent_revisions(self):
        if self.workload == "concurrent_adapters" and len(self.revisions.adapters) < 2:
            raise ValueError("concurrent adapters require at least two pinned adapter digests")
        if len(set(self.required_phases)) != len(self.required_phases):
            raise ValueError("required_phases must be unique")
        return self


class Suite(StrictModel):
    schema_version: Literal[1] = 1
    name: Name
    seeds: list[int] = Field(default_factory=lambda: [17, 29, 43], min_length=1)
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases_and_seeds(self):
        if any(seed < 0 or seed > 2**32 - 1 for seed in self.seeds):
            raise ValueError("seeds must be unsigned 32-bit integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if len({case.name for case in self.cases}) != len(self.cases):
            raise ValueError("case names must be unique")
        return self


class Interval(StrictModel):
    start_s: Seconds
    end_s: Seconds

    @model_validator(mode="after")
    def ordered(self):
        if self.end_s < self.start_s:
            raise ValueError("end_s must be >= start_s")
        return self


class Span(Interval):
    kind: Literal["span"] = "span"
    phase: Phase
    source: Text
    attempt: int = Field(default=0, ge=0)


class Quality(StrictModel):
    kind: Literal["quality"] = "quality"
    at_s: Seconds
    metric: Text
    value: float
    samples: int = Field(ge=1)
    checkpoint: Text
    # Held-out evaluation only. Training reward/loss must not cross a quality target.
    split: Literal["held_out"] = "held_out"


class Request(Interval):
    kind: Literal["request"] = "request"
    customer: Name
    adapter: Text
    success: bool
    ttft_s: Seconds | None = None

    @model_validator(mode="after")
    def ttft_within_request(self):
        if self.ttft_s is not None and self.ttft_s > self.end_s - self.start_s:
            raise ValueError("ttft_s exceeds request latency")
        return self


class Money(Interval):
    kind: Literal["money"] = "money"
    ledger: Ledger
    amount_usd: float = Field(ge=0)
    receipt: Text
    resource: Text
    # One receipt line must appear once per ledger. Intervals may overlap for distinct resources.
    line_item: Text


class Provenance(StrictModel):
    kind: Literal["provenance"] = "provenance"
    revisions: Revisions
    configuration_digest: Digest
    resolved_configuration: Configuration
    evidence: Text


class Cache(StrictModel):
    kind: Literal["cache"] = "cache"
    observed: Literal["cold", "warm", "unknown"]
    evidence: Text


class Complete(StrictModel):
    kind: Literal["complete"] = "complete"
    at_s: Seconds
    outcome: Literal["succeeded", "failed", "cancelled"]
    complete_ledgers: list[Ledger] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ledgers(self):
        if len(set(self.complete_ledgers)) != len(self.complete_ledgers):
            raise ValueError("complete_ledgers must be unique")
        return self


Event = Annotated[
    Span | Quality | Request | Money | Provenance | Cache | Complete,
    Field(discriminator="kind"),
]
EVENT = TypeAdapter(Event)
