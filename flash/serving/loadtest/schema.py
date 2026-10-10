"""strict authored and resolved schemas for hosted inference load tests."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    StrictBool,
    field_validator,
    model_validator,
)

_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DeploymentExpectation(StrictModel):
    sha: str
    deployment_id: str

    @field_validator("sha")
    @classmethod
    def validate_sha(cls, value: str) -> str:
        value = value.strip()
        if not _SHA_RE.fullmatch(value):
            raise ValueError("sha must be a lowercase hexadecimal deployment revision")
        return value

    @field_validator("deployment_id")
    @classmethod
    def validate_deployment_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("deployment_id must not be empty")
        return value


class Discovery(StrictModel):
    enabled: StrictBool = False
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    require: list[str] = Field(default_factory=list)

    @field_validator("include", "exclude", "require")
    @classmethod
    def validate_models(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("model selectors must not be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("model selectors must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_sets(self) -> Discovery:
        if set(self.require) & set(self.exclude):
            raise ValueError("required models must not be excluded")
        if self.include and not set(self.require).issubset(self.include):
            raise ValueError("required models must be included when include is set")
        return self


class ClientLimits(StrictModel):
    connect_timeout_seconds: PositiveFloat = 5.0
    read_timeout_seconds: PositiveFloat = 120.0
    write_timeout_seconds: PositiveFloat = 10.0
    pool_timeout_seconds: PositiveFloat = 5.0
    max_in_flight: PositiveInt = 64
    max_scheduling_lag_ms: PositiveFloat = 250.0


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value:
            raise ValueError("message content must not be empty")
        return value


class RequestProfile(StrictModel):
    name: str
    weight: PositiveFloat = 1.0
    messages: list[ChatMessage]
    max_tokens: PositiveInt = 32
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("profile name must not be empty")
        return value

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        return value


class BaseTarget(StrictModel):
    name: str
    kind: Literal["base_model"] = "base_model"
    model: str
    weight: PositiveFloat = 1.0

    @field_validator("name", "model")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target fields must not be empty")
        return value


class AdapterTarget(StrictModel):
    """an immutable adapter target and the provenance headers it requires.

    ``checkpoint`` is always verified because ``X-Freesolo-Checkpoint`` is part of the hosted
    contract on every deployment this harness targets. ``adapter_revision`` and ``hf_revision``
    are optional because they are not emitted by every hosted deployment: a scenario that omits
    them verifies checkpoint identity alone rather than silently passing a header check that the
    deployment never answered. Supplying one asserts the deployment does emit it, and a response
    that omits it is then a provenance mismatch rather than a skipped assertion.
    """

    name: str
    kind: Literal["adapter"]
    model: str
    base_model: str
    checkpoint: str
    adapter_revision: str | None = None
    hf_revision: str | None = None
    weight: PositiveFloat = 1.0

    @field_validator("name", "model", "base_model", "checkpoint")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target fields must not be empty")
        return value

    @field_validator("adapter_revision")
    @classmethod
    def validate_adapter_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("adapter_revision must not be empty when it is supplied")
        return value

    @field_validator("hf_revision")
    @classmethod
    def validate_hf_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not _HF_REVISION_RE.fullmatch(value):
            raise ValueError("hf_revision must be a canonical 40-character hub commit sha")
        return value


Target = Annotated[BaseTarget | AdapterTarget, Field(discriminator="kind")]


class PhaseBase(StrictModel):
    name: str
    target_names: list[str] = Field(default_factory=list)
    profile_names: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("phase name must not be empty")
        return value

    @field_validator("target_names", "profile_names")
    @classmethod
    def validate_names(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("phase selectors must be unique")
        return value


class WarmPhase(PhaseBase):
    kind: Literal["warm"]
    requests: PositiveInt
    concurrency: PositiveInt


class ColdBurstPhase(PhaseBase):
    kind: Literal["cold_burst"]
    requests: PositiveInt
    burst_window_seconds: float = Field(default=0.0, ge=0.0)
    cold_intent: Literal["cold_scale_out", "true_scale_zero"]

    @property
    def cold_attestation(self) -> str:
        if self.cold_intent == "cold_scale_out":
            return "cold_scale_out_intent_http_unattested"
        return "true_scale_zero_intent_http_unattested"


class SustainedPhase(PhaseBase):
    kind: Literal["sustained"]
    duration_seconds: PositiveFloat
    rate_rps: PositiveFloat


class MixedPhase(PhaseBase):
    kind: Literal["mixed"]
    duration_seconds: PositiveFloat
    rate_rps: PositiveFloat


class OverloadStage(StrictModel):
    duration_seconds: PositiveFloat
    rate_rps: PositiveFloat


class OverloadPhase(PhaseBase):
    """ordered open-loop rate stages that push a deployment past its serving capacity.

    ``expects_capacity_contract`` records whether the deployment under test is known to
    implement a retryable capacity rejection (503 with ``serving_capacity_unavailable`` and
    ``Retry-After: 1``). It defaults to ``False`` because a deployment without that contract
    sheds load in ways this harness cannot distinguish from ordinary failure over public HTTP.
    The flag never changes how a response is classified; it only decides whether an absence of
    capacity rejections is a failed expectation or an accurate description of the deployment.
    Either way the absence is reported as ``overload_not_demonstrated``, never as success.
    """

    kind: Literal["overload"]
    stages: list[OverloadStage]
    expects_capacity_contract: StrictBool = False

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, value: list[OverloadStage]) -> list[OverloadStage]:
        if not value:
            raise ValueError("overload requires at least one stage")
        return value


Phase = Annotated[
    WarmPhase | ColdBurstPhase | SustainedPhase | MixedPhase | OverloadPhase,
    Field(discriminator="kind"),
]


class Scenario(StrictModel):
    schema_version: Literal[1] = 1
    name: str
    endpoint: HttpUrl
    expected_deployment: DeploymentExpectation
    credential_env: str
    required_capabilities: list[str]
    discovery: Discovery = Field(default_factory=Discovery)
    targets: list[Target] = Field(default_factory=list)
    profiles: list[RequestProfile]
    client: ClientLimits = Field(default_factory=ClientLimits)
    seed: int = Field(default=1, ge=0)
    phases: list[Phase]
    fake: StrictBool = False

    @property
    def origin(self) -> str:
        """the endpoint as a request prefix, without the trailing slash pydantic appends.

        every caller builds urls as ``f"{origin}/healthz"``, so the trim belongs here once rather
        than at each call site where one omission would silently produce a double slash.
        """
        return str(self.endpoint).rstrip("/")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("scenario name must not be empty")
        return value

    @field_validator("credential_env")
    @classmethod
    def validate_credential_env(cls, value: str) -> str:
        value = value.strip()
        if not _ENV_RE.fullmatch(value):
            raise ValueError("credential_env must be an uppercase environment variable name")
        return value

    @field_validator("required_capabilities")
    @classmethod
    def validate_capabilities(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("required_capabilities must contain nonempty values")
        if len(set(value)) != len(value):
            raise ValueError("required_capabilities must be unique")
        return value

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: HttpUrl) -> HttpUrl:
        parts = urlsplit(str(value))
        if parts.query or parts.fragment or parts.path not in ("", "/"):
            raise ValueError("endpoint must be an origin without path, query, or fragment")
        normalized = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        return HttpUrl(normalized)

    @field_validator("profiles")
    @classmethod
    def validate_profiles(cls, value: list[RequestProfile]) -> list[RequestProfile]:
        if not value:
            raise ValueError("at least one request profile is required")
        return value

    @field_validator("phases")
    @classmethod
    def validate_phases(cls, value: list[Phase]) -> list[Phase]:
        if not value:
            raise ValueError("at least one phase is required")
        cold_indexes = [i for i, phase in enumerate(value) if phase.kind == "cold_burst"]
        if cold_indexes and cold_indexes != [0]:
            raise ValueError(
                "cold_burst must be the first inference-producing phase and appear once"
            )
        return value

    @model_validator(mode="after")
    def validate_references(self) -> Scenario:
        target_names = [target.name for target in self.targets]
        profile_names = [profile.name for profile in self.profiles]
        phase_names = [phase.name for phase in self.phases]
        for label, names in (
            ("target", target_names),
            ("profile", profile_names),
            ("phase", phase_names),
        ):
            if len(set(names)) != len(names):
                raise ValueError(f"{label} names must be unique")
        known_targets = (
            set(target_names) | set(self.discovery.include) | set(self.discovery.require)
        )
        known_profiles = set(profile_names)
        for phase in self.phases:
            unknown_targets = set(phase.target_names) - known_targets
            unknown_profiles = set(phase.profile_names) - known_profiles
            if unknown_targets and not self.discovery.enabled:
                raise ValueError(f"phase {phase.name!r} references unknown targets")
            if unknown_profiles:
                raise ValueError(f"phase {phase.name!r} references unknown profiles")
        if not self.targets and not self.discovery.enabled:
            raise ValueError("targets or enabled discovery are required")
        return self


class HealthSnapshot(StrictModel):
    """the subset of ``/healthz`` this harness depends on.

    ``accounting_ok`` is optional because it is not part of every hosted deployment's health
    body. ``None`` means the deployment did not report it, which is not the same as reporting
    ``False``: a missing field cannot be read as an accounting failure, and an explicit ``False``
    must still stop the run.
    """

    ok: StrictBool
    accounting_ok: StrictBool | None = None
    deployment_sha: str
    deployment_id: str
    capabilities: list[str]
    base_models: list[str]


class ResolvedScenario(StrictModel):
    authored: Scenario
    health: HealthSnapshot
    targets: list[Target]
    phase_cold_attestations: dict[str, str]
    phase_capacity_expectations: dict[str, bool]
    claim_limitations: list[str]


CLAIM_LIMITATIONS = [
    "http-only evidence cannot prove live replica scale-out",
    "http-only evidence cannot prove a target was in a cold state",
    "http-only evidence cannot distinguish deadline pressure from resource exhaustion",
    "a bounded load test cannot establish an availability sla",
]

NO_CAPACITY_CONTRACT_LIMITATION = (
    "this deployment declares no retryable capacity contract, so an absence of capacity "
    "rejections is not evidence of headroom"
)

FAKE_RUN_LIMITATION = "fake or test runs cannot support production claims"


def capacity_expectations(scenario: Scenario) -> dict[str, bool]:
    """per-overload-phase capacity contract declarations, keyed by phase name.

    the live run and the offline ``summarize`` must reach the same overload verdict, so both read
    this one function over the authored phases rather than keeping parallel comprehensions that
    could drift apart and disagree about the same evidence.
    """
    return {
        phase.name: phase.expects_capacity_contract
        for phase in scenario.phases
        if isinstance(phase, OverloadPhase)
    }


def claim_limitations(capacity: dict[str, bool], *, fake: bool) -> list[str]:
    """what this run's evidence cannot establish, derived from the authored scenario alone.

    ``scenario.resolved.json`` and ``summary.json`` both publish this list, so it is derived here
    once from authored declarations rather than separately at each writer. deriving the
    no-capacity entry from observed rows instead would drop it whenever an overload phase
    produced none, which is exactly the interrupted run where the caveat matters most: incomplete
    evidence would advertise itself as less limited than complete evidence.
    """
    limitations = list(CLAIM_LIMITATIONS)
    if fake:
        limitations.append(FAKE_RUN_LIMITATION)
    if capacity and not all(capacity.values()):
        limitations.append(NO_CAPACITY_CONTRACT_LIMITATION)
    return limitations


def public_scenario_dict(scenario: Scenario) -> dict[str, Any]:
    """return an artifact-safe authored scenario with prompt content redacted."""
    value = scenario.model_dump(mode="json")
    for profile in value["profiles"]:
        profile["messages"] = [
            {"role": message["role"], "content": "[redacted]"} for message in profile["messages"]
        ]
    return value
