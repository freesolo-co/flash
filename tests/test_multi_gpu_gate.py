"""CPU tests for the multi-gpu submit gate: gpu.count > 1 needs a provider that actually rents n
cards on one machine.

Every phase shards now -- sft, grpo and opd all delegate to verl, whose workers launch
nproc-per-node == gpu.count ranks -- so the backend half of this gate is gone and only the provider
half can reject.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# the keys each phase used to be selected by. all dead: no worker reads any of them now.
_STALE_BACKEND_ENV = {
    "sft": "FLASH_SFT_BACKEND",
    "grpo": "FLASH_RL_BACKEND",
    "opd": "FLASH_OPD_BACKEND",
}


def _spec(count: int, algorithm: str = "grpo", backend: str = "", provider: str = "runpod"):
    from flash.spec import JobSpec

    gpu: dict = {"type": "RTX 5090", "count": count}
    if provider:
        gpu["provider"] = provider
    body: dict = {"model": "test/gate", "algorithm": algorithm, "gpu": gpu}
    if backend:
        key = _STALE_BACKEND_ENV[algorithm]
        body["worker_env"] = {key: backend}
    return JobSpec.from_dict(body)


def test_gate_helper_allows_single_gpu():
    from flash import runner

    # count == 1 is the default single-gpu path; must not raise.
    assert runner._require_supported_gpu_count(_spec(1)) is None


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_gate_allows_multi_gpu_for_verl_backend(algorithm):
    from flash import runner

    # every verl worker launches nproc-per-node == gpu.count ranks, so all three shard.
    assert runner._require_supported_gpu_count(_spec(4, algorithm, backend="verl")) is None


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_gate_allows_multi_gpu_with_no_key_at_all(algorithm):
    from flash import runner

    # no phase has a selector: run_sft, run_rl and run_opd all delegate to verl, so multi-gpu needs
    # no opt-in anywhere.
    assert runner._require_supported_gpu_count(_spec(4, algorithm)) is None


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_a_stale_backend_key_cannot_downgrade_any_phase(algorithm):
    from flash import runner
    from flash.spec import effective_backend

    # a config carried over from when these phases had selectors must not resolve a backend that no
    # longer exists: the worker would still run verl, and the gate would refuse it multi-gpu while
    # the launcher handed it trl's expandable_segments alloc conf (which kills verl's CuMemAllocator).
    spec = _spec(4, algorithm, backend="trl")
    assert effective_backend(spec) == "verl"
    assert runner._require_supported_gpu_count(spec) is None


def test_gate_rejects_multi_gpu_on_providers_that_ignore_count():
    from flash import runner

    # vast's num_gpus filter has no caller and lambda instance types are hardcoded gpu_1x_*, so a
    # 4-rank verl trainer would land on ONE rented card. unset is rejected for the same reason.
    for provider in ("vast", "lambda", ""):
        with pytest.raises(ValueError, match=r"gpu\.provider"):
            runner._require_supported_gpu_count(_spec(4, backend="verl", provider=provider))


def test_effective_backend_is_verl_for_every_phase():
    from flash.spec import effective_backend

    # verl is the only backend. this stays pinned because the answer is load-bearing: it picks the
    # allocator conf and gates multi-gpu, and a spec that resolved anything else would take
    # expandable_segments and crash its vllm rollout on the CuMemAllocator assert.
    for algorithm in ("sft", "grpo", "opd"):
        assert effective_backend(_spec(1, algorithm)) == "verl"
        assert effective_backend(_spec(1, algorithm, backend="trl")) == "verl"


def test_submit_job_rejects_multi_gpu_at_boundary(monkeypatch):
    from flash import runner

    # the gate fires at the top of submit_job, before any provisioning or billing (even in dry-run).
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        with pytest.raises(ValueError, match=r"gpu\.provider"):
            runner.submit_job(_spec(2, provider="vast"), dry_run=True)


def test_submit_job_rejects_multi_gpu_prepared_worker_spec(monkeypatch):
    from flash import runner
    from flash.runner import PreparedJob

    # a single-gpu public spec paired with a prepared_job whose EFFECTIVE worker_spec is multi-gpu must
    # still be rejected: allocation and training provision the worker_spec, not the public spec.
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        prepared = PreparedJob(
            public_spec=_spec(1),
            worker_spec=_spec(2, provider="vast"),
            estimated_cost_usd=0.0,
        )
        with pytest.raises(ValueError, match=r"gpu\.provider"):
            runner.submit_job(_spec(1), dry_run=True, prepared_job=prepared)


def test_run_training_gates_effective_spec_on_recovery():
    import io

    from flash.runner.lifecycle import _run_training

    # recovery rebuilds the worker spec from a persisted snapshot and calls _run_training directly,
    # bypassing submit_job's gate; the shared submit+recovery path must fail closed on a multi-gpu spec
    # before any provisioning (the gate precedes the first get_status/allocation touch).
    with pytest.raises(ValueError, match=r"gpu\.provider"):
        _run_training(_spec(2, provider="vast"), io.StringIO(), prior_cost=0.0)


def test_gate_message_names_the_field_that_enables_sharding():
    from flash import runner

    # the remaining rejection is the provider half, and gpu.provider is the field that fixes it. a
    # message offering only "set gpu.count to 1" would leave multi-gpu undiscoverable, so assert the
    # remedy travels with the rejection.
    with pytest.raises(ValueError, match=r"gpu\.provider") as excinfo:
        runner._require_supported_gpu_count(_spec(4, "grpo", provider="vast"))

    message = str(excinfo.value)
    assert "gpu.provider" in message
    assert "runpod" in message, "the message must name a provider that actually rents n cards"


def _submittable(algorithm: str, backend: str = ""):
    """a spec that survives a full submit, unlike _spec (which only needs to reach the gate).

    the gate raises before model/train validation, so the gate helper can use a fake model id and no
    train table. recording the backend happens at the END of submit, so this one must be real.
    """
    from flash.spec import JobSpec

    train: dict = {"max_examples": 4}
    if algorithm == "opd":
        train["teacher_model"] = "kimi-k2.6"
    body: dict = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": algorithm,
        "gpu": {"type": "H200", "count": 1, "provider": "runpod"},
        "train": train,
    }
    if backend:
        body["worker_env"] = {_STALE_BACKEND_ENV[algorithm]: backend}
    return JobSpec.from_dict(body)


@pytest.mark.parametrize(
    ("algorithm", "backend"),
    [("grpo", "verl"), ("grpo", ""), ("grpo", "trl"), ("sft", ""), ("opd", "")],
)
def test_submit_records_the_resolved_backend(monkeypatch, algorithm, backend):
    from flash import runner

    # the run records which trainer actually ran, so it stays auditable from the run itself rather
    # than from what the config appeared to ask for. a stale key must not change the record.
    expected = "verl"
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        status = runner.submit_job(_submittable(algorithm, backend), dry_run=True)
        assert (status.effective_preparation or {}).get("backend") == expected
