"""Only curated catalog models are trainable + VRAM estimator unit tests (CPU-only, no network).

Adding a model means forking Flash and adding a ModelInfo entry to flash/catalog.py. There is no
config key that accepts an uncataloged model: the rejection is catalog membership itself, so it
holds identically on a managed and a self-hosted plane.
"""

from __future__ import annotations

import pytest

from flash.catalog import resolve_model
from flash.engine.vram import GPU_VRAM_GB, estimate_vram_gb
from flash.schema import ConfigError, spec_from_dict
from tests._helpers.specs import raw_spec as _raw


def test_an_uncataloged_model_is_rejected_with_the_fork_instruction():
    # The error is the primary documentation for the new workflow -- it is what every rejected
    # user sees, so it must name the concrete next step, not just refuse.
    with pytest.raises(ValueError, match="unsupported model") as ei:
        resolve_model("some-org/some-model", "sft")
    message = str(ei.value)
    assert "fork" in message.lower()
    assert "flash/catalog.py" in message


def test_catalog_model_resolves_normally():
    info = resolve_model("Qwen/Qwen3.5-4B", "grpo")
    assert info.id == "Qwen/Qwen3.5-4B"


def test_the_parser_rejects_an_uncataloged_model_before_submit():
    # Rejecting client-side means a fork with a new catalog entry works without touching the
    # plane, and a typo fails locally instead of after a network round trip.
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="acme/unlisted"))
    assert "unsupported model" in str(ei.value)


def test_model_policy_is_no_longer_an_accepted_config_key():
    # The key is gone, not merely ignored: a config carrying it must fail loudly rather than
    # silently training something other than what it asked for.
    with pytest.raises(ConfigError, match="unknown config key"):
        spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", model_policy="allow"))


def test_a_catalog_spec_round_trips_without_a_policy_field():
    spec = spec_from_dict(_raw(model="Qwen/Qwen3.5-4B"))
    payload = spec.to_dict()
    assert "model_policy" not in payload
    assert spec_from_dict(payload).model == "Qwen/Qwen3.5-4B"


def test_a_run_persisted_before_the_upgrade_still_reloads():
    """A record written by the OLD plane must survive the upgrade that drops the field.

    effective_preparation.worker_spec is written with to_internal_dict() (asdict), so EVERY record
    the current plane has written names model_policy -- including defaulted ones, which to_dict()
    popped but asdict() kept. Stored records are never rewritten and from_dict is strict, so without
    the dropped-key tolerance the first reload after deploy raises and a still-running job loses its
    recovery, deploy, and serving paths (cursor[bot]).
    """
    from flash.spec import JobSpec

    spec = spec_from_dict(_raw(model="Qwen/Qwen3.5-4B"))
    persisted = spec.to_internal_dict()
    persisted["model_policy"] = "catalog"  # what the pre-upgrade plane wrote

    assert JobSpec.from_dict(persisted).model == "Qwen/Qwen3.5-4B"


def test_the_dropped_key_is_tolerated_on_read_only_never_authored():
    """Tolerance must not quietly re-open the flag as an authorable key.

    from_dict ignores it so old RECORDS load; the schema layer still rejects it so a CONFIG naming
    it fails loudly rather than silently training under a policy that no longer exists.
    """
    with pytest.raises(ConfigError, match="unknown config key"):
        spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", model_policy="allow"))


def test_a_pre_upgrade_snapshot_still_passes_its_integrity_digest(tmp_path, monkeypatch):
    """Parsing an old record is not enough -- its stored DIGEST has to reproduce too.

    Letting the key through from_dict fixes deserialization, but _preparation_digest rehashes the
    spec, and to_internal_dict no longer emits model_policy. A pre-upgrade snapshot hashed it in, so
    a workload-profile or warm-start run parses and THEN fails "persisted effective preparation
    failed integrity validation" -- still blocking recovery, deploy and serving (cursor[bot],
    codex[bot]). The digest must be reproduced from the bytes that were hashed.
    """
    import hashlib
    import json

    monkeypatch.setenv("FLASH_HOME", str(tmp_path))
    import flash.runner as runner

    spec = spec_from_dict(_raw(model="Qwen/Qwen3.5-4B"))
    spec = type(spec).from_dict(
        {
            **spec.to_internal_dict(),
            "workload_profile_kind": "sft",
            "workload_profile": {"steps": 10},
        }
    )
    public, worker = spec.to_dict(), spec.to_internal_dict()

    # Recompute the digest exactly as the OLD plane did: its worker payload carried model_policy,
    # and it dropped empty workload_profile_* keys (the version-1 omission rule) before hashing.
    old_worker = {**worker, "model_policy": "catalog"}
    hashed_worker = {
        k: v
        for k, v in old_worker.items()
        if v
        or k
        not in (
            "workload_profile_kind",
            "workload_profile_input_digest",
            "workload_profile_producer_version",
            "workload_profile",
        )
    }
    payload = {
        "version": 1,
        "public_spec": public,
        "worker_spec": hashed_worker,
        "adapter_identity": None,
    }
    old_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()

    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=public,
            effective_preparation={
                "worker_spec": old_worker,
                "workload_profile": {"steps": 10},
                "preparation_digest": old_digest,
            },
        )
    )
    loaded = runner.effective_spec_from_status(runner.get_status(spec.run_id))
    assert loaded.model == "Qwen/Qwen3.5-4B"


# ---------------------------------------------------------------------------
# Estimator sanity: calibrated against catalog anchors
# ---------------------------------------------------------------------------
def _headroom(params_b: float, algo: str, quant: str, gpu: str) -> str:
    """The advisory fits/tight/too_big banding, applied straight to the sizing equations.

    This used to be `check_fit`, an estimator wrapper that fetched the model's size from HF. Only the
    open-model path consulted its verdict, so the wrapper went with it -- but the BANDS are what the
    anchors below calibrate, and those still gate real sizing decisions.
    """
    est = estimate_vram_gb(params_b, algo, quant)
    gpu_gb = GPU_VRAM_GB[gpu]
    if est > gpu_gb * 1.15:
        return "too_big"
    return "tight" if est > gpu_gb * 0.85 else "fits"


@pytest.mark.parametrize(
    ("params_b", "algo", "quant", "gpu", "expected"),
    [
        (4.0, "grpo", "bf16", "RTX 5090", "fits"),  # Qwen3-4B colocate on 32 GB (measured)
        (4.0, "sft", "bf16", "RTX 4090", "fits"),
        (9.65, "sft", "bf16", "RTX 5090", "tight"),  # Qwen3.5-9B SFT real logits peak
        (36.0, "sft", "bf16", "RTX 5090", "too_big"),  # 72 GB of weights
        (36.0, "grpo", "bf16", "RTX 5090", "too_big"),  # 2 bf16 copies + KV >> 32 GB
    ],
)
def test_estimator_anchors(params_b, algo, quant, gpu, expected):
    # params_b is supplied directly, so this calibrates the SIZING EQUATIONS against known anchors
    # and never resolves the id through the catalog.
    assert _headroom(params_b, algo, quant, gpu) == expected


def test_grpo_needs_more_than_sft():
    assert estimate_vram_gb(4.0, "grpo") > estimate_vram_gb(4.0, "sft")
