from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Literal

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from flash.serving.src import settings as cfg
from flash.serving.src.model_config import (
    engine_overrides_for,
    gpu_for,
    hosted_traffic_policy_for,
    image_limit_for,
    is_supported_base_model,
)
from flash.serving.src.persistence import load_adapters
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.settings import Settings
from flash.serving.src.supabase_rest import (
    postgrest_error,
    raise_for_supabase,
    supabase_headers,
    supabase_table_url,
)

READINESS_TABLE = "hosted_model_readiness_passes"
READINESS_EVIDENCE_VERSION = 1
READINESS_SELECT = (
    "deployment_mode,deployment_sha,deployment_id,model_id,engine_contract_sha256,"
    "evidence_version,evidence,evidence_sha256,passed_at"
)
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PUBLICATION_RETRY_DELAYS_SECONDS = (0.1, 0.25, 0.5)
_UNIQUE_VIOLATION = "23505"


def _canonical_json(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def evidence_sha256(evidence: dict[str, Any] | BaseModel) -> str:
    return hashlib.sha256(_canonical_json(evidence)).hexdigest()


def engine_contract(model_id: str) -> dict[str, Any]:
    if not is_supported_base_model(model_id):
        raise ValueError(f"unsupported readiness model: {model_id}")
    policy = hosted_traffic_policy_for(model_id)
    return {
        "version": 2,
        "model_id": model_id,
        "gpu": gpu_for(model_id),
        "image_input_limit": image_limit_for(model_id),
        "engine_overrides": engine_overrides_for(model_id),
        "hosted_traffic_policy": {
            "min_containers": policy.min_containers,
            "max_containers": policy.max_containers,
            "buffer_containers": policy.buffer_containers,
            "queue_capacity": policy.queue_capacity,
            "retry_after_seconds": policy.retry_after_seconds,
            "max_num_seqs": policy.max_num_seqs,
            "max_inputs": policy.max_inputs,
            "target_inputs": policy.target_inputs,
        },
        "runtime": {
            "trust_remote_code": cfg.TRUST_REMOTE_CODE,
            "dtype": cfg.DTYPE,
            "quantization": cfg.QUANTIZATION,
            "kv_cache_dtype": cfg.KV_CACHE_DTYPE,
            "tensor_parallel_size": cfg.TENSOR_PARALLEL_SIZE,
            "gpu_memory_utilization": cfg.GPU_MEMORY_UTILIZATION,
            "max_model_len": cfg.MAX_MODEL_LEN,
            "max_loras": cfg.MAX_LORAS,
            "max_lora_rank": cfg.MAX_LORA_RANK,
            "max_cpu_loras": cfg.MAX_CPU_LORAS,
            **cfg.vllm_engine_kwargs(),
        },
    }


def engine_contract_sha256(model_id: str) -> str:
    return hashlib.sha256(_canonical_json(engine_contract(model_id))).hexdigest()


class ReadinessCheckEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: Literal[True]
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReadinessGenerationEvidence(ReadinessCheckEvidence):
    request_id: str = Field(min_length=1)
    engine_replica_id: str = Field(min_length=1)
    prompt_tokens: int = Field(ge=1)
    completion_tokens: int = Field(ge=1)
    finish_reason: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)


class ReadinessPassEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    outcome: Literal["passed"]
    deployment_mode: Literal["production", "development"]
    deployment_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    deployment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    engine_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    health: ReadinessCheckEvidence
    non_streaming: ReadinessGenerationEvidence
    streaming: ReadinessGenerationEvidence
    provenance: ReadinessCheckEvidence
    runtime_attestation: ReadinessCheckEvidence

    @field_validator("deployment_id", "model_id")
    @classmethod
    def validate_unpadded_identity(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("identity values must be unpadded")
        return value

    @model_validator(mode="after")
    def validate_generation_checkpoints(self) -> ReadinessPassEvidence:
        if self.non_streaming.checkpoint != self.model_id:
            raise ValueError("non-streaming readiness checkpoint must match model_id")
        if self.streaming.checkpoint != self.model_id:
            raise ValueError("streaming readiness checkpoint must match model_id")
        return self


class HostedModelReadinessPass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_mode: Literal["production", "development"]
    deployment_sha: str
    deployment_id: str
    model_id: str
    engine_contract_sha256: str
    evidence_version: Literal[1]
    evidence: ReadinessPassEvidence
    evidence_sha256: str
    passed_at: AwareDatetime

    @field_validator("deployment_sha")
    @classmethod
    def validate_deployment_sha(cls, value: str) -> str:
        if _SHA40_RE.fullmatch(value) is None:
            raise ValueError("deployment_sha must be a lowercase 40-character commit SHA")
        return value

    @field_validator("engine_contract_sha256", "evidence_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("digest must be a lowercase 64-character sha256")
        return value

    @field_validator("deployment_id", "model_id")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("identity values must be nonempty and unpadded")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> HostedModelReadinessPass:
        if evidence_sha256(self.evidence) != self.evidence_sha256:
            raise ValueError("evidence_sha256 does not match evidence")
        expected_identity = (
            self.deployment_mode,
            self.deployment_sha,
            self.deployment_id,
            self.model_id,
            self.engine_contract_sha256,
        )
        evidence_identity = (
            self.evidence.deployment_mode,
            self.evidence.deployment_sha,
            self.evidence.deployment_id,
            self.evidence.model_id,
            self.evidence.engine_contract_sha256,
        )
        if evidence_identity != expected_identity:
            raise ValueError("readiness evidence identity does not match its storage row")
        return self


class ReadinessPublication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_mode: Literal["production", "development"]
    deployment_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    deployment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    engine_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_version: Literal[1] = READINESS_EVIDENCE_VERSION
    evidence: ReadinessPassEvidence
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("deployment_id", "model_id")
    @classmethod
    def validate_unpadded_identity(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("identity values must be unpadded")
        return value

    @model_validator(mode="after")
    def validate_digests(self) -> ReadinessPublication:
        if self.engine_contract_sha256 != engine_contract_sha256(self.model_id):
            raise ValueError("engine_contract_sha256 does not match the current engine contract")
        if self.evidence_sha256 != evidence_sha256(self.evidence):
            raise ValueError("evidence_sha256 does not match evidence")
        expected_identity = (
            self.deployment_mode,
            self.deployment_sha,
            self.deployment_id,
            self.model_id,
            self.engine_contract_sha256,
        )
        evidence_identity = (
            self.evidence.deployment_mode,
            self.evidence.deployment_sha,
            self.evidence.deployment_id,
            self.evidence.model_id,
            self.evidence.engine_contract_sha256,
        )
        if evidence_identity != expected_identity:
            raise ValueError("readiness evidence identity does not match its publication")
        return self


def build_runtime_readiness_evidence(
    settings: Settings,
    model_id: str,
    *,
    engine_health: dict[str, Any],
    non_streaming: dict[str, Any],
    streaming: dict[str, Any],
    deployment_health: dict[str, Any],
) -> dict[str, Any]:
    if (
        engine_health.get("ok") is not True
        or engine_health.get("engine_dead") is not False
        or engine_health.get("base_model") != model_id
    ):
        raise ValueError("engine health does not attest the exact readiness model")
    if (
        deployment_health.get("ok") is not True
        or deployment_health.get("deployment_sha") != settings.deployment_sha
        or deployment_health.get("deployment_id") != settings.deployment_id
    ):
        raise ValueError("router health does not attest the exact deployment identity")

    def generation_evidence(result: dict[str, Any]) -> dict[str, Any]:
        checkpoint = result.get("checkpoint")
        if checkpoint != model_id:
            raise ValueError("generation checkpoint does not attest the exact readiness model")
        return {
            "passed": True,
            "evidence_sha256": evidence_sha256(result),
            "request_id": result.get("request_id"),
            "engine_replica_id": result.get("engine_replica_id"),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "finish_reason": result.get("finish_reason"),
            "checkpoint": checkpoint,
        }

    evidence = ReadinessPassEvidence(
        schema_version=1,
        outcome="passed",
        deployment_mode=settings.deployment_mode,
        deployment_sha=settings.deployment_sha,
        deployment_id=settings.deployment_id,
        model_id=model_id,
        engine_contract_sha256=engine_contract_sha256(model_id),
        health={"passed": True, "evidence_sha256": evidence_sha256(engine_health)},
        non_streaming=generation_evidence(non_streaming),
        streaming=generation_evidence(streaming),
        provenance={"passed": True, "evidence_sha256": evidence_sha256(deployment_health)},
        runtime_attestation={
            "passed": True,
            "evidence_sha256": evidence_sha256(
                {
                    "engine": engine_health,
                    "deployment": deployment_health,
                    "model_id": model_id,
                }
            ),
        },
    )
    return evidence.model_dump(mode="json")


def build_readiness_publication(
    settings: Settings,
    model_id: str,
    evidence: dict[str, Any],
) -> ReadinessPublication:
    return ReadinessPublication(
        deployment_mode=settings.deployment_mode,
        deployment_sha=settings.deployment_sha,
        deployment_id=settings.deployment_id,
        model_id=model_id,
        engine_contract_sha256=engine_contract_sha256(model_id),
        evidence=evidence,
        evidence_sha256=evidence_sha256(evidence),
    )


def _has_readiness_identity(settings: Settings) -> bool:
    return bool(
        settings.has_supabase
        and settings.deployment_mode in {"production", "development"}
        and _SHA40_RE.fullmatch(settings.deployment_sha)
        and bool(settings.deployment_id.strip())
        and settings.deployment_id == settings.deployment_id.strip()
    )


def _readiness_params(settings: Settings) -> dict[str, str]:
    return {
        "select": READINESS_SELECT,
        "deployment_mode": f"eq.{settings.deployment_mode}",
        "deployment_sha": f"eq.{settings.deployment_sha}",
        "deployment_id": f"eq.{settings.deployment_id}",
        "order": "model_id.asc",
    }


def _parse_readiness_rows(
    response: httpx.Response,
    *,
    expected_identity: tuple[str, str, str] | None = None,
) -> list[HostedModelReadinessPass]:
    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError("Supabase readiness response must be a list")
    parsed: list[HostedModelReadinessPass] = []
    for row in rows:
        try:
            readiness = HostedModelReadinessPass.model_validate(row)
        except ValueError:
            continue
        if not is_supported_base_model(readiness.model_id):
            continue
        if (
            expected_identity is not None
            and (
                readiness.deployment_mode,
                readiness.deployment_sha,
                readiness.deployment_id,
            )
            != expected_identity
        ):
            continue
        expected_digest = engine_contract_sha256(readiness.model_id)
        if readiness.engine_contract_sha256 != expected_digest:
            continue
        parsed.append(readiness)
    return parsed


def load_readiness_passes(settings: Settings) -> list[HostedModelReadinessPass]:
    if not _has_readiness_identity(settings):
        return []
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            supabase_table_url(settings, READINESS_TABLE),
            params=_readiness_params(settings),
            headers=supabase_headers(settings, "flash"),
        )
    raise_for_supabase(response, "load hosted model readiness passes")
    return _parse_readiness_rows(
        response,
        expected_identity=(
            settings.deployment_mode,
            settings.deployment_sha,
            settings.deployment_id,
        ),
    )


def qualified_base_records(settings: Settings) -> list[AdapterRecord]:
    return [
        AdapterRecord(
            adapter_id=readiness.model_id,
            repo_id=readiness.model_id,
            base_model=readiness.model_id,
            serve_base_model=True,
            thinking=True,
            org_id=None,
            status="ready",
            metadata={
                "readiness": {
                    "deployment_mode": readiness.deployment_mode,
                    "deployment_sha": readiness.deployment_sha,
                    "deployment_id": readiness.deployment_id,
                    "engine_contract_sha256": readiness.engine_contract_sha256,
                    "evidence_version": readiness.evidence_version,
                    "evidence_sha256": readiness.evidence_sha256,
                    "passed_at": readiness.passed_at.isoformat(),
                }
            },
        )
        for readiness in load_readiness_passes(settings)
    ]


def load_routing_snapshot(settings: Settings) -> list[AdapterRecord]:
    adapters = load_adapters(settings)
    bases = qualified_base_records(settings)
    return adapters + bases


def _publication_matches(
    publication: ReadinessPublication,
    row: HostedModelReadinessPass,
) -> bool:
    return publication.model_dump(mode="json") == row.model_dump(mode="json", exclude={"passed_at"})


def _read_published(
    client: httpx.Client,
    settings: Settings,
    publication: ReadinessPublication,
) -> HostedModelReadinessPass | None:
    response = client.get(
        supabase_table_url(settings, READINESS_TABLE),
        params={
            "select": READINESS_SELECT,
            "deployment_mode": f"eq.{publication.deployment_mode}",
            "deployment_sha": f"eq.{publication.deployment_sha}",
            "deployment_id": f"eq.{publication.deployment_id}",
            "model_id": f"eq.{publication.model_id}",
            "limit": "1",
        },
        headers=supabase_headers(settings, "flash"),
    )
    raise_for_supabase(response, "read hosted model readiness pass")
    rows = _parse_readiness_rows(response)
    if len(rows) > 1:
        raise RuntimeError("read hosted model readiness pass returned multiple rows")
    return rows[0] if rows else None


def publish_readiness_pass(
    settings: Settings,
    model_id: str,
    evidence: dict[str, Any],
    *,
    retry_delays_seconds: tuple[float, ...] = _PUBLICATION_RETRY_DELAYS_SECONDS,
) -> HostedModelReadinessPass:
    if not _has_readiness_identity(settings):
        raise RuntimeError("readiness publication requires Supabase and exact deployment identity")
    publication = build_readiness_publication(settings, model_id, evidence)
    payload = publication.model_dump(mode="json")
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(len(retry_delays_seconds) + 1):
            try:
                response = client.post(
                    supabase_table_url(settings, READINESS_TABLE),
                    params={"select": READINESS_SELECT},
                    headers={
                        **supabase_headers(settings, "flash"),
                        "Prefer": "return=representation",
                    },
                    json=payload,
                )
            except httpx.TransportError:
                if attempt >= len(retry_delays_seconds):
                    raise
            else:
                if response.status_code == 409:
                    code, _ = postgrest_error(response)
                    if code != _UNIQUE_VIOLATION:
                        raise_for_supabase(response, "publish hosted model readiness pass")
                    existing = _read_published(client, settings, publication)
                    if existing is None or not _publication_matches(publication, existing):
                        raise RuntimeError(
                            "hosted model readiness pass conflicts with existing evidence"
                        )
                    return existing
                if not response.is_error:
                    rows = _parse_readiness_rows(response)
                    if len(rows) != 1 or not _publication_matches(publication, rows[0]):
                        raise RuntimeError("readiness publication returned an unexpected row")
                    return rows[0]
                if response.status_code not in {408, 429} and response.status_code < 500:
                    raise_for_supabase(response, "publish hosted model readiness pass")
                if attempt >= len(retry_delays_seconds):
                    raise_for_supabase(response, "publish hosted model readiness pass")
            time.sleep(retry_delays_seconds[attempt])
    raise AssertionError("readiness publication retry loop did not return")
