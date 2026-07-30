"""CPU tests for the multi-gpu submit gate: gpu.count > 1 needs a sharding backend AND a provider
that actually rents n cards on one machine.

GRPO is the only phase with a backend selector left. sft and opd delegate to verl unconditionally,
so they shard by construction and the gate can only ever reject them on the provider half.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# grpo's key is FLASH_RL_BACKEND (spec.phase maps grpo -> rl), and it is the only one a worker reads.
_RL_BACKEND_ENV = "FLASH_RL_BACKEND"

# the keys sft/opd used to be selected by. they are dead: no worker reads them any more.
_STALE_BACKEND_ENV = {"sft": "FLASH_SFT_BACKEND", "opd": "FLASH_OPD_BACKEND"}


def _spec(count: int, algorithm: str = "grpo", backend: str = "", provider: str = "runpod"):
    from flash.spec import JobSpec

    gpu: dict = {"type": "RTX 5090", "count": count}
    if provider:
        gpu["provider"] = provider
    body: dict = {"model": "test/gate", "algorithm": algorithm, "gpu": gpu}
    if backend:
        key = _RL_BACKEND_ENV if algorithm == "grpo" else _STALE_BACKEND_ENV[algorithm]
        body["worker_env"] = {key: backend}
    return JobSpec.from_dict(body)


def test_gate_helper_rejects_multi_gpu_on_default_trl_backend():
    from flash import runner

    # grpo with no [worker_env] override resolves trl, which trains in one process.
    with pytest.raises(ValueError, match="single process"):
        runner._require_supported_gpu_count(_spec(2))


def test_gate_helper_allows_single_gpu():
    from flash import runner

    # count == 1 is the default single-gpu path; must not raise.
    assert runner._require_supported_gpu_count(_spec(1)) is None


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_gate_allows_multi_gpu_for_verl_backend(algorithm):
    from flash import runner

    # every verl worker launches nproc-per-node == gpu.count ranks, so all three shard.
    assert runner._require_supported_gpu_count(_spec(4, algorithm, backend="verl")) is None


@pytest.mark.parametrize("algorithm", ["sft", "opd"])
def test_gate_allows_multi_gpu_for_verl_only_phases_with_no_key(algorithm):
    from flash import runner

    # sft/opd have no selector: run_sft and run_opd delegate to verl, so multi-gpu needs no opt-in.
    assert runner._require_supported_gpu_count(_spec(4, algorithm)) is None


@pytest.mark.parametrize("algorithm", ["sft", "opd"])
def test_a_stale_backend_key_cannot_downgrade_a_verl_only_phase(algorithm):
    from flash import runner
    from flash.spec import effective_backend

    # a config carried over from when these phases had selectors must not resolve a backend that no
    # longer exists: the worker would still run verl, and the gate would refuse it multi-gpu while
    # the launcher handed it trl's expandable_segments alloc conf (which kills verl's CuMemAllocator).
    spec = _spec(4, algorithm, backend="trl")
    assert effective_backend(spec) == "verl"
    assert runner._require_supported_gpu_count(spec) is None


def test_gate_rejects_multi_gpu_for_explicit_trl_backend():
    from flash import runner

    # grpo is the one phase where the key still selects, so trl must still be refused n cards.
    with pytest.raises(ValueError, match="single process"):
        runner._require_supported_gpu_count(_spec(4, "grpo", backend="trl"))


def test_gate_rejects_multi_gpu_on_providers_that_ignore_count():
    from flash import runner

    # vast's num_gpus filter has no caller and lambda instance types are hardcoded gpu_1x_*, so a
    # 4-rank verl trainer would land on ONE rented card. unset is rejected for the same reason.
    for provider in ("vast", "lambda", ""):
        with pytest.raises(ValueError, match=r"gpu\.provider"):
            runner._require_supported_gpu_count(_spec(4, backend="verl", provider=provider))


def test_effective_backend_reads_the_phase_env_key():
    from flash.spec import effective_backend

    # grpo's worker env key is FLASH_RL_BACKEND (spec.phase maps grpo -> rl), not FLASH_GRPO_BACKEND.
    assert effective_backend(_spec(1, "grpo", backend="verl")) == "verl"
    assert effective_backend(_spec(1, "grpo")) == "trl"
    # sft/opd resolve verl with no key at all.
    assert effective_backend(_spec(1, "sft")) == "verl"
    assert effective_backend(_spec(1, "opd")) == "verl"


def test_submit_job_rejects_multi_gpu_at_boundary(monkeypatch):
    from flash import runner

    # the gate fires at the top of submit_job, before any provisioning or billing (even in dry-run).
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        with pytest.raises(ValueError, match="multi-gpu training"):
            runner.submit_job(_spec(2), dry_run=True)


def test_submit_job_rejects_multi_gpu_prepared_worker_spec(monkeypatch):
    from flash import runner
    from flash.runner import PreparedJob

    # a single-gpu public spec paired with a prepared_job whose EFFECTIVE worker_spec is multi-gpu must
    # still be rejected: allocation and training provision the worker_spec, not the public spec.
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        prepared = PreparedJob(
            public_spec=_spec(1),
            worker_spec=_spec(2),
            estimated_cost_usd=0.0,
        )
        with pytest.raises(ValueError, match="multi-gpu training"):
            runner.submit_job(_spec(1), dry_run=True, prepared_job=prepared)


def test_run_training_gates_effective_spec_on_recovery():
    import io

    from flash.runner.lifecycle import _run_training

    # recovery rebuilds the worker spec from a persisted snapshot and calls _run_training directly,
    # bypassing submit_job's gate; the shared submit+recovery path must fail closed on a multi-gpu spec
    # before any provisioning (the gate precedes the first get_status/allocation touch).
    with pytest.raises(ValueError, match="multi-gpu training"):
        _run_training(_spec(2), io.StringIO(), prior_cost=0.0)


def test_gate_message_names_the_key_that_enables_sharding():
    from flash import runner

    # [worker_env] is the ONLY route to a sharding backend for grpo, and the key is not guessable from
    # the algorithm (grpo -> FLASH_RL_BACKEND). a message offering only "set gpu.count to 1" leaves
    # multi-gpu undiscoverable, so assert the remedy travels with the rejection.
    # match the backend rejection specifically: the gate also raises for gpu.provider, and that
    # path carries no backend key, so a bare raises() would pass while checking nothing.
    with pytest.raises(ValueError, match="multi-gpu training") as excinfo:
        runner._require_supported_gpu_count(_spec(4, "grpo"))

    message = str(excinfo.value)
    assert _RL_BACKEND_ENV in message
    assert "verl" in message
    assert "worker_env" in message


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
        body["worker_env"] = {_RL_BACKEND_ENV: backend}
    return JobSpec.from_dict(body)


@pytest.mark.parametrize(
    ("algorithm", "backend", "expected"),
    [("grpo", "verl", "verl"), ("grpo", "", "trl"), ("sft", "", "verl"), ("opd", "", "verl")],
)
def test_submit_records_the_resolved_backend(monkeypatch, algorithm, backend, expected):
    from flash import runner

    # grpo's backend is resolved ONLY from the spec's [worker_env] and defaults to trl SILENTLY, so a
    # spec that forgot the key runs a different trainer than intended with no error. record the
    # resolution on the run so which trainer actually ran is auditable from the run itself.
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        status = runner.submit_job(_submittable(algorithm, backend), dry_run=True)
        assert (status.effective_preparation or {}).get("backend") == expected
