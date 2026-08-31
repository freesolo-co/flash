from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import flash.engine.worker.io.hf as worker_hf
import flash.engine.worker.runtime.state as worker_state
import flash.runner.lifecycle.attempts as runner_attempts
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.supervise.attach as runner_attach
import flash.runner.supervise.lifecycle as runner_lifecycle
import flash.runner.supervise.recovery as runner_recovery
import flash.teacher.retry_contract as retry_contract
from flash.teacher.retry_contract import (
    OPD_RESUME_REVISION_ENV,
    OPD_RESUME_STATE_VERSION,
    OPD_RETRY_CONTRACT_STATUS_KEY,
    OPD_RETRY_CONTRACT_VERSION,
    canonical_opd_optimizer_start_json,
    decode_opd_optimizer_start_json,
    opd_optimizer_start_marker_path,
    validate_opd_resume_state_metadata,
)
from tests._helpers.runner import provisioned_status
from tests._helpers.source_snapshot import valid_source_snapshot

_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64
_SOURCE_SNAPSHOT = valid_source_snapshot()


def _patch_worker_marker_context(
    monkeypatch,
    *,
    repo: str,
    run_id: str,
    attempt: int,
    seed: int,
    api=None,
    remaining: float | None = None,
) -> None:
    monkeypatch.setattr(worker_state, "HF_REPO", repo)
    monkeypatch.setattr(worker_state, "RUN_ID", run_id)
    monkeypatch.setattr(worker_state, "ATTEMPT", attempt)
    monkeypatch.setattr(worker_state, "SEED", seed)
    monkeypatch.setattr(worker_state, "_remaining_worker_wall_seconds", lambda: remaining)
    monkeypatch.setattr(worker_hf, "hf_api", lambda: api)


@pytest.fixture(autouse=True)
def _stub_teacher_broker_transport(monkeypatch):
    import flash.server.domain.teacher.broker as teacher_broker

    monkeypatch.setattr(
        teacher_broker,
        "require_teacher_broker_configuration",
        lambda _spec, **_kwargs: "https://broker.example",
    )

    @contextlib.contextmanager
    def teacher_transport(_spec, **_kwargs):
        yield {
            "FLASH_PUBLIC_URL": "https://broker.example",
            "FLASH_TEACHER_CAPABILITY": "capability-test-value",
        }

    monkeypatch.setattr(teacher_broker, "teacher_attempt_transport", teacher_transport)


def _valid_resume_state(step: int, *, seed: int = 42, **overrides) -> dict:
    state = {
        "contract_version": OPD_RESUME_STATE_VERSION,
        "seed": seed,
        "opt_steps": step,
        "step": step,
        "rollout_seed_ordinal": step,
        "prompt_pool_fingerprint": "a" * 64,
        "loss_curve": [0.5] * step,
        "coverage_curve": [1.0] * step,
        "generated_tokens": step,
        "teacher_input_tokens": step,
        "teacher_output_tokens": step,
        "truncated_rollouts": 0,
        "coverage_sum": 0.0,
        "aligned_sequences": 0,
        "samples_seen": step,
        "teacher_ok": step,
        "teacher_transient": 0,
        "teacher_error": 0,
        "skip_counts": {},
        "no_signal_resamples": 0,
        "no_signal_skipped_steps": 0,
        "episodes_seen": 0,
        "mt_turn_records": 0,
        "opd_phase_seconds": {},
        "opd_phase_counts": {},
        "train_wall_seconds": 0.0,
        # `accounting_snapshot` emits both unconditionally, so every real version-4 state carries
        # the pair and the contract requires it.
        "align_group_sum": 0.0,
        "align_group_n": 0,
    }
    state.update(overrides)
    return state


def _remote(*, attempt: int = 0) -> dict:
    return {
        "provider": "runpod",
        "endpoint_id": f"endpoint-{attempt}",
        "endpoint_name": f"endpoint-{attempt}-name",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "job_id": f"job-{attempt}",
        "attempt": attempt,
        "launch_claim_token": f"claim-{attempt}",
        "started_ts": float(attempt + 1),
        "allocated_gpu": "RTX 4090",
        "allocated_gpu_count": 1,
        "allocated_usable_vram_gb": 24.0,
    }


def _opd_spec(run_id: str, *, max_retries: int = 1, seed: int = 42):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    return JobSpec(
        run_id=run_id,
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        seed=seed,
        train=TrainSpec(hf_repo="private/runs", max_examples=1, epochs=1),
        gpu=GpuSpec(type="RTX 4090", max_retries=max_retries),
    )


def _save_status(
    spec,
    *,
    state="running",
    next_attempt=0,
    remote=None,
    contracted=True,
    source_snapshot=None,
):
    kwargs = {"_next_attempt": next_attempt}
    if contracted:
        kwargs["_opd_retry_contract_version"] = OPD_RETRY_CONTRACT_VERSION
    status = provisioned_status(spec, state=state, remote=remote)
    status.source_snapshot = source_snapshot
    runner_state._save_status(status, **kwargs)


def test_status_initialization_stamps_opd_contract_only_when_explicit(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    contracted = _opd_spec("contract-opd")
    uncontracted = _opd_spec("uncontracted-opd")
    _save_status(contracted)
    _save_status(uncontracted, contracted=False)
    for spec in (
        JobSpec(run_id="contract-sft", algorithm="sft"),
        JobSpec(run_id="contract-grpo", algorithm="grpo"),
    ):
        runner_state._save_status(
            runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
        )

    opd_raw = runner_status._load_status_json("contract-opd")
    assert opd_raw[OPD_RETRY_CONTRACT_STATUS_KEY] == OPD_RETRY_CONTRACT_VERSION
    assert OPD_RETRY_CONTRACT_STATUS_KEY not in runner_status.get_status("contract-opd").__dict__
    assert OPD_RETRY_CONTRACT_STATUS_KEY not in runner_status._load_status_json("uncontracted-opd")
    assert OPD_RETRY_CONTRACT_STATUS_KEY not in runner_status._load_status_json("contract-sft")
    assert OPD_RETRY_CONTRACT_STATUS_KEY not in runner_status._load_status_json("contract-grpo")
    with pytest.raises(ValueError, match="cannot be stored for a non-opd run"):
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="invalid-contract-sft",
                state="running",
                spec=JobSpec(run_id="invalid-contract-sft", algorithm="sft").to_dict(),
            ),
            _opd_retry_contract_version=OPD_RETRY_CONTRACT_VERSION,
        )
    sft_claim = runner_attempts.reserve_verified_attempt_launch("contract-sft")
    grpo_claim = runner_attempts.reserve_verified_attempt_launch("contract-grpo")
    assert sft_claim is not None
    assert sft_claim.attempt == 0
    assert grpo_claim is not None
    assert grpo_claim.attempt == 0


def test_marker_path_and_canonical_exact_schema():
    raw = canonical_opd_optimizer_start_json(run_id="run-1", attempt=3, seed=42)
    assert raw == (
        b'{"attempt":3,"contract":"flash.opd.optimizer-start","phase":"opd",'
        b'"run_id":"run-1","seed":42,"version":1}'
    )
    assert opd_optimizer_start_marker_path("run-1", 3) == (
        "_opd_retry/run-1/attempts/attempt-3/optimizer-start.v1.json"
    )
    assert decode_opd_optimizer_start_json(raw, run_id="run-1", attempt=3, seed=42) == json.loads(
        raw
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        (
            b'{"attempt":3,"contract":"flash.opd.optimizer-start","phase":"opd",'
            b'"run_id":"run-1","seed":42,"version":1,"extra":0}'
        ),
        (
            b'{"attempt":true,"contract":"flash.opd.optimizer-start","phase":"opd",'
            b'"run_id":"run-1","seed":42,"version":1}'
        ),
        b'{"attempt":3,"contract":"wrong","phase":"opd","run_id":"run-1","seed":42,"version":1}',
        (
            b'{"attempt":3,"attempt":3,"contract":"flash.opd.optimizer-start",'
            b'"phase":"opd","run_id":"run-1","seed":42,"version":1}'
        ),
        (
            b'{"attempt": 3, "contract": "flash.opd.optimizer-start", "phase": "opd", '
            b'"run_id": "run-1", "seed": 42, "version": 1}'
        ),
        b"\xff",
    ],
)
def test_marker_decode_rejects_malformed_or_noncanonical_evidence(raw):
    with pytest.raises(ValueError, match="marker"):
        decode_opd_optimizer_start_json(raw, run_id="run-1", attempt=3, seed=42)


def test_marker_upload_failure_is_retriable_and_attempted_exactly_once(monkeypatch):
    """A failed marker upload is retriable, preserves its cause, and is attempted once.

    Retrying in place could double-commit an ambiguous marker. The optimizer wrapper's pre-step
    ordering is covered by ``test_optimizer_step_is_blocked_when_marker_publication_fails``.
    """
    from flash.engine.worker.io import hf
    from flash.engine.worker.perf import RetriableInfraError

    uploads = []
    failure = RetriableInfraError("required upload failed")

    def upload_file(**kwargs):
        uploads.append(kwargs)
        raise failure

    api = SimpleNamespace(upload_file=upload_file)
    _patch_worker_marker_context(
        monkeypatch,
        repo="private/runs",
        run_id="run-1",
        attempt=0,
        seed=42,
        api=api,
    )

    with pytest.raises(RetriableInfraError) as caught:
        hf.publish_opd_optimizer_start_marker()

    assert caught.value.__cause__ is failure
    assert len(uploads) == 1


def test_ambiguous_marker_upload_is_not_retried_and_leaks_no_token(monkeypatch):
    """A lost commit response must not be retried, and its diagnostic must not carry the HF token.

    The dangerous case is a commit that landed while the response was lost: retrying could publish a
    second marker for the same attempt, and the underlying error text embeds the Authorization header.
    Both guarantees live in `publish_opd_optimizer_start_marker` (single attempt, `sanitize_diagnostic`
    at limit 500), which is why this survives the trl deletion unchanged -- only the caller moved.
    """
    from flash.engine.worker.io import hf
    from flash.engine.worker.perf import RetriableInfraError

    class CommitThenLoseFirstResponse:
        def __init__(self):
            self.uploads = []
            self.committed = None

        def upload_file(self, **kwargs):
            self.uploads.append(kwargs)
            self.committed = Path(kwargs["path_or_fileobj"]).read_bytes()
            if len(self.uploads) == 1:
                raise TimeoutError(
                    "commit response was lost Authorization: Bearer marker-secret " + "x" * 1000
                )

    api = CommitThenLoseFirstResponse()
    monkeypatch.setenv("HF_TOKEN", "marker-secret")
    _patch_worker_marker_context(
        monkeypatch,
        repo="private/runs",
        run_id="run-ambiguous",
        attempt=2,
        seed=42,
        api=api,
    )
    with pytest.raises(RetriableInfraError) as caught:
        hf.publish_opd_optimizer_start_marker()

    assert isinstance(caught.value.__cause__, TimeoutError)
    assert "marker-secret" not in str(caught.value)
    assert (
        len(str(caught.value))
        <= len("RETRIABLE_INFRA_GPU: required upload of OPD optimizer-start marker failed: ") + 500
    )
    assert len(api.uploads) == 1
    assert api.committed == canonical_opd_optimizer_start_json(
        run_id="run-ambiguous", attempt=2, seed=42
    )


def test_worker_marker_rejects_empty_repo(monkeypatch):
    from flash.engine.worker.io import hf

    _patch_worker_marker_context(
        monkeypatch,
        repo="",
        run_id="run-1",
        attempt=0,
        seed=42,
    )
    with pytest.raises(RuntimeError, match="requires a private HF repository"):
        hf.publish_opd_optimizer_start_marker()


def test_worker_marker_starts_no_hf_call_at_deadline(monkeypatch):
    from flash.engine.worker.io import hf
    from flash.engine.worker.perf import RetriableInfraError

    calls = []
    _patch_worker_marker_context(
        monkeypatch,
        repo="private/runs",
        run_id="run-1",
        attempt=0,
        seed=42,
        api=SimpleNamespace(upload_file=lambda **_kwargs: calls.append("hf_api")),
        remaining=0.0,
    )

    with pytest.raises(RetriableInfraError, match="run wall deadline exceeded"):
        hf.publish_opd_optimizer_start_marker()

    assert calls == []


def test_worker_marker_writes_fsync_and_required_upload(monkeypatch):
    from flash.engine.worker.io import hf

    fsync_calls = []
    uploads = []

    def upload_file(**kwargs):
        uploads.append(
            (
                kwargs["path_in_repo"],
                kwargs["repo_id"],
                kwargs["repo_type"],
                Path(kwargs["path_or_fileobj"]).read_bytes(),
            )
        )

    monkeypatch.setattr(hf.os, "fsync", lambda fd: fsync_calls.append(fd))
    _patch_worker_marker_context(
        monkeypatch,
        repo="private/runs",
        run_id="run-1",
        attempt=4,
        seed=42,
        api=SimpleNamespace(upload_file=upload_file),
    )
    hf.publish_opd_optimizer_start_marker()

    assert len(fsync_calls) == 1
    assert uploads == [
        (
            "_opd_retry/run-1/attempts/attempt-4/optimizer-start.v1.json",
            "private/runs",
            "dataset",
            canonical_opd_optimizer_start_json(run_id="run-1", attempt=4, seed=42),
        )
    ]


def _install_hf_reader(monkeypatch, api, download):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)


class _FakePrivateHf:
    def __init__(self, root: Path, *, raise_after_upload=False):
        self.root = root
        self.raise_after_upload = raise_after_upload
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, object]] = []

    def repo_info(self, **kwargs):
        self.calls.append(("repo_info", kwargs))
        return SimpleNamespace(sha="private-pinned-sha")

    def get_paths_info(self, **kwargs):
        self.calls.append(("get_paths_info", kwargs))
        return [
            SimpleNamespace(path=path, size=len(self.files[path]))
            for path in kwargs["paths"]
            if path in self.files
        ]

    def upload_file(self, **kwargs):
        path = kwargs["path_in_repo"]
        self.calls.append(("upload_file", kwargs))
        self.files[path] = Path(kwargs["path_or_fileobj"]).read_bytes()
        if self.raise_after_upload:
            raise TimeoutError("commit response was lost")

    def list_repo_files(self, **kwargs):
        self.calls.append(("list_repo_files", kwargs))
        return list(self.files)

    def download(self, **kwargs):
        self.calls.append(("download", kwargs))
        path = kwargs["filename"]
        local_path = self.root / path.replace("/", "-")
        local_path.write_bytes(self.files[path])
        return str(local_path)

    def install(self, monkeypatch):
        _install_hf_reader(monkeypatch, self, self.download)


def test_strict_reader_pins_one_sha_and_any_present_marker_blocks(monkeypatch, tmp_path):
    from flash.providers.artifacts.hf import verify_opd_replacement_safe

    calls = []
    present_path = opd_optimizer_start_marker_path("run-1", 1)

    class Api:
        def repo_info(self, **kwargs):
            calls.append(("repo_info", kwargs))
            return SimpleNamespace(sha="pinned-sha")

        def get_paths_info(self, **kwargs):
            calls.append(("get_paths_info", kwargs))
            return [SimpleNamespace(path=present_path)]

    def download(**kwargs):
        calls.append(("download", kwargs))
        path = tmp_path / "marker.json"
        path.write_bytes(canonical_opd_optimizer_start_json(run_id="run-1", attempt=1, seed=42))
        return str(path)

    _install_hf_reader(monkeypatch, Api(), download)
    # a validated marker proves mutation; this api exposes no list_repo_files, so the resume-checkpoint
    # presence check fails closed and replacement stays blocked (now for lack of a usable checkpoint).
    with pytest.raises(RuntimeError, match="no complete resume checkpoint is available"):
        verify_opd_replacement_safe(
            hf_repo="private/runs",
            run_id="run-1",
            seed=42,
            next_attempt=2,
            contract_version=1,
            phase="opd",
            expected_fsdp_generation=1,
        )

    assert [name for name, _kwargs in calls] == ["repo_info", "get_paths_info", "download"]
    assert calls[1][1]["revision"] == "pinned-sha"
    assert calls[1][1]["paths"] == [
        opd_optimizer_start_marker_path("run-1", 0),
        present_path,
    ]
    assert calls[2][1]["revision"] == "pinned-sha"
    assert calls[2][1]["filename"] == present_path


def test_strict_reader_all_absent_is_safe(monkeypatch):
    from flash.providers.artifacts.hf import verify_opd_replacement_safe

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="pinned-sha")

        def get_paths_info(self, **_kwargs):
            return []

    _install_hf_reader(
        monkeypatch,
        Api(),
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    verify_opd_replacement_safe(
        hf_repo="private/runs",
        run_id="run-1",
        seed=42,
        next_attempt=2,
        contract_version=1,
        phase="opd",
        expected_fsdp_generation=1,
    )


@pytest.mark.parametrize("mode", ["malformed", "timeout", "listing"])
def test_strict_reader_malformed_or_outage_blocks(monkeypatch, tmp_path, mode):
    from flash.providers.artifacts.hf import verify_opd_replacement_safe

    marker_path = opd_optimizer_start_marker_path("run-1", 0)

    class Api:
        def repo_info(self, **_kwargs):
            if mode == "listing":
                raise PermissionError("auth unavailable")
            return SimpleNamespace(sha="pinned-sha")

        def get_paths_info(self, **_kwargs):
            return [SimpleNamespace(path=marker_path)]

    def download(**_kwargs):
        if mode == "timeout":
            raise TimeoutError("download timed out")
        path = tmp_path / "bad-marker.json"
        path.write_text("{}")
        return str(path)

    _install_hf_reader(monkeypatch, Api(), download)
    with pytest.raises(RuntimeError, match="replacement is blocked"):
        verify_opd_replacement_safe(
            hf_repo="private/runs",
            run_id="run-1",
            seed=42,
            next_attempt=1,
            contract_version=1,
            phase="opd",
            expected_fsdp_generation=1,
        )


@pytest.mark.parametrize(
    ("repo", "version"),
    [("", 1), ("private/runs", 2), ("private/runs", True), ("private/runs", "1")],
)
def test_strict_reader_missing_repo_or_unsupported_contract_blocks(repo, version):
    from flash.providers.artifacts.hf import verify_opd_replacement_safe

    with pytest.raises(RuntimeError, match="replacement is blocked"):
        verify_opd_replacement_safe(
            hf_repo=repo,
            run_id="run-1",
            seed=42,
            next_attempt=1,
            contract_version=version,
            phase="opd",
            expected_fsdp_generation=1,
        )


def test_initial_contracted_opd_reservation_skips_empty_attempt_query(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("initial-opd")
    _save_status(spec, next_attempt=0)
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not query HF")),
    )

    claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
    assert claim is not None
    assert claim.attempt == 0
    assert claim.resume_revision is None
    assert claim.resume_world_size is None


def test_opd_launch_claim_persists_verified_resume_revision_and_world_size(monkeypatch, tmp_path):
    from flash.providers.artifacts import hf as hf_artifacts

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("opd-resume-claim")
    _save_status(spec, next_attempt=0)
    monkeypatch.setattr(
        hf_artifacts,
        "verify_opd_replacement_safe",
        lambda **_kwargs: ("checkpoint-revision", 2),
    )

    claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)

    assert claim is not None
    assert claim.resume_revision == "checkpoint-revision"
    assert claim.resume_world_size == 2
    raw = runner_status._load_status_json(spec.run_id)
    assert raw[runner_state._ACTIVE_LAUNCH_CLAIM_KEY] == claim.to_dict()


def test_retry_gate_uses_authoritative_jobspec_seed(monkeypatch, tmp_path):
    from flash.providers.artifacts import hf as _hf_artifacts

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("custom-seed-opd", seed=987)
    _save_status(spec, next_attempt=1)
    seen = {}

    def verify(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(_hf_artifacts, "verify_opd_replacement_safe", verify)

    assert runner_attempts._verified_opd_retry_state(spec.run_id) == (1, None, None)
    assert seen["seed"] == 987


def test_precontract_opd_fails_closed(monkeypatch, tmp_path):

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("precontract-opd")
    _save_status(spec, contracted=False)

    with pytest.raises(RuntimeError, match="contract is missing or invalid"):
        runner_attempts._verified_opd_retry_state(spec.run_id)
    with pytest.raises(RuntimeError, match="contract is missing or invalid"):
        runner_attempts.reserve_verified_attempt_launch(spec.run_id)


def test_next_attempt_cas_race_blocks_opd_reservation(monkeypatch, tmp_path):
    from flash.providers.artifacts import hf as hf_artifacts

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("opd-cas")
    _save_status(spec, next_attempt=0)

    def advance_attempt(**_kwargs):
        runner_state._save_status(runner_status.get_status(spec.run_id), _next_attempt=1)
        return

    monkeypatch.setattr(hf_artifacts, "verify_opd_replacement_safe", advance_attempt)

    assert runner_attempts.reserve_verified_attempt_launch(spec.run_id) is None
    assert runner_status._load_status_json(spec.run_id)[runner_state._NEXT_ATTEMPT_KEY] == 1


def test_opd_automatic_retry_after_teardown_requires_all_markers_absent(monkeypatch, tmp_path):
    import flash.providers.core.allocator as allocator
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate, PollResult
    from flash.runner.supervise import lifecycle

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    private_hf = _FakePrivateHf(tmp_path)
    private_hf.install(monkeypatch)
    spec = _opd_spec("automatic-retry-absent")
    _save_status(spec, next_attempt=0, source_snapshot=_SOURCE_SNAPSHOT)
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    larger = Candidate("runpod", "H100", 3.29, 80)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=(candidate, larger),
        ),
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda *_args: None)

    class Provider:
        supports_weight_cache = False

        def __init__(self):
            self.attempts = []
            self.teardown = []
            self.events = []

        def submit_attempt(self, _spec, *, attempt, on_handle, **_kwargs):
            self.attempts.append(attempt)
            self.events.append(("submit", attempt))
            on_handle(_remote(attempt=attempt))
            if attempt == 0:
                return PollResult(False, failure="poll_error", detail="transient")
            return PollResult(True, metrics={"train_tokens": 1})

        def cancel(self, handle):
            event = ("cancel", handle.data["attempt"])
            self.teardown.append(event)
            self.events.append(event)

        def destroy(self, handle):
            event = ("destroy", handle.data["attempt"])
            self.teardown.append(event)
            self.events.append(event)

    provider = Provider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)

    metrics = lifecycle._run_attempts_supervised(spec, io.StringIO())

    assert metrics == {
        "train_tokens": 1,
        "allocated_gpu": "H100",
        # stamped alongside the gpu so cost attribution prices the class on the substrate
        # that actually billed it.
        "allocated_provider": "runpod",
        # and the card count, since the hourly rate is per card.
        "allocated_gpu_count": 1,
    }
    assert provider.attempts == [0, 1]
    retry_submit = provider.events.index(("submit", 1))
    assert provider.events.index(("cancel", 0)) < retry_submit
    assert provider.events.index(("destroy", 0)) < retry_submit
    assert [name for name, _kwargs in private_hf.calls] == ["repo_info", "get_paths_info"]
    assert private_hf.calls[1][1]["paths"] == [opd_optimizer_start_marker_path(spec.run_id, 0)]


def test_opd_retry_passes_gate_revision_and_overwrites_spoofed_value(monkeypatch, tmp_path):
    import flash.providers.core.allocator as allocator
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate, PollResult
    from flash.runner.supervise import lifecycle

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    private_hf = _FakePrivateHf(tmp_path)
    private_hf.install(monkeypatch)
    spec = _opd_spec("automatic-retry-pinned")
    _save_status(spec, next_attempt=0, source_snapshot=_SOURCE_SNAPSHOT)
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    larger = Candidate("runpod", "H100", 3.29, 80)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=(candidate, larger),
        ),
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda *_args: None)

    class Provider:
        supports_weight_cache = False

        def __init__(self):
            self.runtime_secrets = []
            self.worker_envs = []

        def submit_attempt(self, _spec, *, attempt, on_handle, **kwargs):
            from flash.providers._lifecycle.net.worker import build_worker_env

            secrets = kwargs.get("runtime_secrets")
            self.runtime_secrets.append(secrets)
            self.worker_envs.append(build_worker_env(_spec, runtime_secrets=secrets))
            on_handle(_remote(attempt=attempt))
            if attempt == 0:
                marker_path = opd_optimizer_start_marker_path(spec.run_id, 0)
                private_hf.files[marker_path] = canonical_opd_optimizer_start_json(
                    run_id=spec.run_id, attempt=0, seed=42
                )
                checkpoint_names = (
                    *_COMPLETE_CKPT[:6],
                    "actor/fsdp_config.json",
                    "actor/model_world_size_1_rank_0.pt",
                    "actor/optim_world_size_1_rank_0.pt",
                    "actor/extra_state_world_size_1_rank_0.pt",
                )
                for path in _ckpt_files("opd", spec.run_id, 20, checkpoint_names):
                    if path.endswith("/opd_state.json"):
                        payload = json.dumps(_valid_resume_state(20)).encode()
                    elif path.endswith("/actor/fsdp_config.json"):
                        payload = json.dumps({"FSDP_version": 1, "world_size": 1}).encode()
                    else:
                        payload = b"checkpoint"
                    private_hf.files[path] = payload
                return PollResult(False, failure="poll_error", detail="transient")
            return PollResult(True, metrics={"train_tokens": 1})

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            return None

    provider = Provider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)

    metrics = lifecycle._run_attempts_supervised(
        spec,
        io.StringIO(),
        runtime_secrets={
            "WANDB_API_KEY": "real-secret",
            OPD_RESUME_REVISION_ENV: "spoofed-sha",
        },
    )

    assert metrics == {
        "train_tokens": 1,
        "allocated_gpu": "H100",
        # stamped alongside the gpu so cost attribution prices the class on the substrate
        # that actually billed it.
        "allocated_provider": "runpod",
        # and the card count, since the hourly rate is per card.
        "allocated_gpu_count": 1,
    }
    broker_transport = {
        "FLASH_PUBLIC_URL": "https://broker.example",
        "FLASH_TEACHER_CAPABILITY": "capability-test-value",
    }
    assert provider.runtime_secrets == [
        {"WANDB_API_KEY": "real-secret", **broker_transport},
        {
            "WANDB_API_KEY": "real-secret",
            OPD_RESUME_REVISION_ENV: "private-pinned-sha",
            **broker_transport,
        },
    ]
    assert OPD_RESUME_REVISION_ENV not in provider.worker_envs[0]
    assert provider.worker_envs[1][OPD_RESUME_REVISION_ENV] == "private-pinned-sha"


def test_failed_attached_opd_worker_decodes_present_marker_after_teardown(monkeypatch, tmp_path):
    from flash.providers.core import registry as providers
    from flash.providers.core.base import PollResult

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = _opd_spec("attach-opd-block")
    marker_path = opd_optimizer_start_marker_path(spec.run_id, 0)
    private_hf = _FakePrivateHf(tmp_path)
    private_hf.files[marker_path] = canonical_opd_optimizer_start_json(
        run_id=spec.run_id, attempt=0, seed=42
    )
    private_hf.install(monkeypatch)
    _save_status(
        spec,
        next_attempt=1,
        remote=_remote(attempt=0),
        source_snapshot=_SOURCE_SNAPSHOT,
    )
    events = []

    class Provider:
        def poll_attempt(self, *_args, **_kwargs):
            return PollResult(False, failure="stalled", detail="worker stopped")

        def cancel(self, _handle):
            events.append("cancel")

        def destroy(self, _handle):
            events.append("destroy")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    resumed = []
    monkeypatch.setattr(
        runner_lifecycle, "_run_training", lambda *_args, **_kwargs: resumed.append(True)
    )

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert events == ["cancel", "destroy"]
    assert resumed == []
    assert status.state == "failed"
    assert status.remote is None
    assert status.cleanup_confirmed_remote == _remote(attempt=0)
    assert status.realized_cost_remote == _remote(attempt=0)
    assert "replacement is blocked" in status.error
    assert [name for name, _kwargs in private_hf.calls] == [
        "repo_info",
        "get_paths_info",
        "download",
        "list_repo_files",
    ]
    assert private_hf.calls[2][1]["revision"] == "private-pinned-sha"


def test_handleless_opd_recovery_blocks_through_recover_runs(monkeypatch, tmp_path):
    from flash.providers.core import registry as providers
    from flash.server.platform import runtime as _runtime

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("handleless-opd-block")
    marker_path = opd_optimizer_start_marker_path(spec.run_id, 0)
    private_hf = _FakePrivateHf(tmp_path)
    private_hf.files[marker_path] = canonical_opd_optimizer_start_json(
        run_id=spec.run_id, attempt=0, seed=42
    )
    private_hf.install(monkeypatch)
    _save_status(
        spec,
        state="provisioning",
        next_attempt=1,
        source_snapshot=_SOURCE_SNAPSHOT,
    )
    started = []
    monkeypatch.setattr(
        runner_lifecycle, "_run_job_background", lambda *_args: started.append(True)
    )
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(_runtime.db, "all_runs", lambda: [{"run_id": spec.run_id}])
    monkeypatch.setattr(providers, "configured_providers", list)

    _runtime.recover_runs()

    assert started == []
    status = runner_status.get_status(spec.run_id)
    assert status.state == "failed"
    assert "replacement is blocked" in status.error
    assert [name for name, _kwargs in private_hf.calls] == [
        "repo_info",
        "get_paths_info",
        "download",
        "list_repo_files",
    ]


def test_ambiguous_marker_upload_lands_evidence_and_blocks_replacement(monkeypatch, tmp_path):
    """An ambiguous marker upload still leaves evidence that blocks replacement.

    A lost response cannot prove the commit failed, so the published marker must fail closed before
    another worker allocates a GPU.
    """
    import flash.providers.core.allocator as allocator
    from flash.engine.worker.io import hf
    from flash.engine.worker.perf import RetriableInfraError
    from flash.runner.supervise import lifecycle

    private_hf = _FakePrivateHf(tmp_path, raise_after_upload=True)
    monkeypatch.setattr(hf, "_sleep_with_hf_deadline", lambda _delay: True)
    _patch_worker_marker_context(
        monkeypatch,
        repo="private/runs",
        run_id="ambiguous-upload",
        attempt=0,
        seed=42,
        api=private_hf,
    )
    with pytest.raises(RetriableInfraError, match="required upload"):
        hf.publish_opd_optimizer_start_marker()

    marker_path = opd_optimizer_start_marker_path("ambiguous-upload", 0)
    assert private_hf.files[marker_path] == canonical_opd_optimizer_start_json(
        run_id="ambiguous-upload", attempt=0, seed=42
    )

    private_hf.raise_after_upload = False
    private_hf.install(monkeypatch)
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("ambiguous-upload")
    _save_status(spec, next_attempt=1, source_snapshot=_SOURCE_SNAPSHOT)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replacement must be blocked before allocation")
        ),
    )

    # the fake private hf exposes no list_repo_files, so the resume-checkpoint presence check fails
    # closed: a proven mutation with no usable checkpoint keeps replacement blocked before allocation.
    with pytest.raises(RuntimeError, match="no complete resume checkpoint is available"):
        lifecycle._run_attempts_supervised(spec, io.StringIO())


# --- gate matrix: a proven mutation marker is safe to replace only when paired with a complete
# full-state resume checkpoint at the same pinned revision. these drive verify_opd_replacement_safe
# directly with a validated attempt-1 marker present, varying only the checkpoint listing.

_MISSING_SIZE = object()
_COMPLETE_CKPT = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "opd_state.json",
    "tokenizer.json",
    "actor/fsdp_config.json",
    "actor/model_world_size_2_rank_0.pt",
    "actor/model_world_size_2_rank_1.pt",
    "actor/optim_world_size_2_rank_0.pt",
    "actor/optim_world_size_2_rank_1.pt",
    "actor/extra_state_world_size_2_rank_0.pt",
    "actor/extra_state_world_size_2_rank_1.pt",
)
# adapter written but optimizer.pt missing: a torn/partial upload, not resumable.
_INCOMPLETE_CKPT = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "rng_state.pth",
    "opd_state.json",
)


def _ckpt_files(phase, run_id, step, names):
    return [f"{phase}/{run_id}/checkpoint/checkpoint-{step}/{name}" for name in names]


def _install_marker_gate(
    monkeypatch,
    tmp_path,
    *,
    checkpoint_files,
    checkpoint_state=None,
    state_download_error=None,
    fsdp_generation=1,
    fsdp_world_size=2,
    shard_size_overrides=None,
):
    """Install one validated marker plus pinned checkpoint listing and metadata reads."""
    present_path = opd_optimizer_start_marker_path("run-1", 1)
    seen = {"list_revision": None, "downloads": [], "path_info_requests": []}

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="pinned-sha")

        def get_paths_info(self, **kwargs):
            requested = kwargs.get("paths", [])
            seen["path_info_requests"].append(requested)
            if present_path in requested:
                return [SimpleNamespace(path=present_path, size=1)]
            available = (
                set(checkpoint_files) if not isinstance(checkpoint_files, Exception) else set()
            )
            infos = []
            for path in requested:
                if path not in available:
                    continue
                size = (shard_size_overrides or {}).get(path, 1)
                infos.append(
                    SimpleNamespace(path=path)
                    if size is _MISSING_SIZE
                    else SimpleNamespace(path=path, size=size)
                )
            return infos

        def list_repo_files(self, **kwargs):
            seen["list_revision"] = kwargs.get("revision")
            if isinstance(checkpoint_files, Exception):
                raise checkpoint_files
            return list(checkpoint_files)

    def download(**kwargs):
        seen["downloads"].append(kwargs)
        if kwargs["filename"] == present_path:
            path = tmp_path / "marker.json"
            path.write_bytes(canonical_opd_optimizer_start_json(run_id="run-1", attempt=1, seed=42))
            return str(path)
        if kwargs["filename"].endswith("/actor/fsdp_config.json"):
            path = tmp_path / "fsdp_config.json"
            path.write_text(
                json.dumps({"FSDP_version": fsdp_generation, "world_size": fsdp_world_size})
            )
            return str(path)
        if state_download_error is not None:
            raise state_download_error
        path = tmp_path / "opd_state.json"
        state = _valid_resume_state(40) if checkpoint_state is None else checkpoint_state
        path.write_text(state if isinstance(state, str) else json.dumps(state))
        return str(path)

    _install_hf_reader(monkeypatch, Api(), download)
    return seen


def _run_gate():
    from flash.providers.artifacts.hf import verify_opd_replacement_safe

    return verify_opd_replacement_safe(
        hf_repo="private/runs",
        run_id="run-1",
        seed=42,
        next_attempt=2,
        contract_version=1,
        phase="opd",
        expected_fsdp_generation=1,
    )


def test_gate_allows_replacement_when_marker_paired_with_valid_metadata(monkeypatch, tmp_path):
    seen = _install_marker_gate(
        monkeypatch, tmp_path, checkpoint_files=_ckpt_files("opd", "run-1", 40, _COMPLETE_CKPT)
    )
    assert _run_gate() == ("pinned-sha", 2)
    assert seen["list_revision"] == "pinned-sha"
    metadata_download = seen["downloads"][-1]
    assert metadata_download["revision"] == "pinned-sha"
    assert metadata_download["filename"] == ("opd/run-1/checkpoint/checkpoint-40/opd_state.json")


@pytest.mark.parametrize(
    "checkpoint_state",
    [
        _valid_resume_state(40, contract_version=1),
        _valid_resume_state(40, opt_steps=39),
        _valid_resume_state(40, seed=99),
    ],
    ids=["schema-v1", "checkpoint-step-mismatch", "seed-mismatch"],
)
def test_gate_blocks_invalid_checkpoint_metadata(monkeypatch, tmp_path, checkpoint_state):
    _install_marker_gate(
        monkeypatch,
        tmp_path,
        checkpoint_files=_ckpt_files("opd", "run-1", 40, _COMPLETE_CKPT),
        checkpoint_state=checkpoint_state,
    )

    with pytest.raises(RuntimeError, match="metadata is unverifiable"):
        _run_gate()


def test_gate_blocks_malformed_checkpoint_metadata_json(monkeypatch, tmp_path):
    _install_marker_gate(
        monkeypatch,
        tmp_path,
        checkpoint_files=_ckpt_files("opd", "run-1", 40, _COMPLETE_CKPT),
        checkpoint_state="{not json",
    )

    with pytest.raises(RuntimeError, match="metadata is unverifiable"):
        _run_gate()


def test_gate_blocks_pinned_checkpoint_metadata_download_failure(monkeypatch, tmp_path):
    _install_marker_gate(
        monkeypatch,
        tmp_path,
        checkpoint_files=_ckpt_files("opd", "run-1", 40, _COMPLETE_CKPT),
        state_download_error=TimeoutError("pinned download failed"),
    )

    with pytest.raises(RuntimeError, match="metadata is unverifiable"):
        _run_gate()


def _sharded_ckpt(step: int, world_size: int) -> list[str]:
    """A complete checkpoint listing with the canonical native shard set."""
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "opd_state.json",
        "tokenizer.json",
        "actor/fsdp_config.json",
        *(
            f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
            for kind in ("model", "optim", "extra_state")
            for rank in range(world_size)
        ),
    )
    return _ckpt_files("opd", "run-1", step, names)


def test_gate_reports_the_width_the_checkpoint_was_written_at(monkeypatch, tmp_path):
    """the retry has to be allocated at the validated native checkpoint width.

    the gate reads the width from pinned fsdp_config.json, expands the canonical expected shard names,
    and validates their pinned path metadata before reporting that width. this matters because the
    control plane picks the retry's card count before any worker exists to open the checkpoint.
    """
    _install_marker_gate(monkeypatch, tmp_path, checkpoint_files=_sharded_ckpt(40, 2))
    assert _run_gate() == ("pinned-sha", 2)


def test_gate_blocks_torn_shards_instead_of_guessing_a_width(monkeypatch, tmp_path):
    _install_marker_gate(
        monkeypatch,
        tmp_path,
        checkpoint_files=_sharded_ckpt(40, 2)
        + _ckpt_files("opd", "run-1", 40, ("actor/model_world_size_4_rank_0.pt",)),
    )
    with pytest.raises(RuntimeError, match="malformed shard name"):
        _run_gate()


@pytest.mark.parametrize("missing_kind", ["optim", "extra_state"])
def test_gate_blocks_missing_native_state_class(monkeypatch, tmp_path, missing_kind):
    files = [
        path for path in _sharded_ckpt(40, 2) if f"/actor/{missing_kind}_world_size_" not in path
    ]
    _install_marker_gate(monkeypatch, tmp_path, checkpoint_files=files)

    with pytest.raises(RuntimeError, match=rf"missing shard.*{missing_kind}_world_size_2_rank_0"):
        _run_gate()


def test_gate_blocks_wrong_private_fsdp_generation(monkeypatch, tmp_path):
    _install_marker_gate(
        monkeypatch,
        tmp_path,
        checkpoint_files=_sharded_ckpt(40, 2),
        fsdp_generation=2,
    )

    with pytest.raises(RuntimeError, match="fsdp generation mismatch"):
        _run_gate()


def test_gate_rejects_oversized_fsdp_width_before_expanding_shard_paths(monkeypatch, tmp_path):
    seen = _install_marker_gate(
        monkeypatch,
        tmp_path,
        checkpoint_files=_sharded_ckpt(40, 2),
        fsdp_world_size=9,
    )

    with pytest.raises(RuntimeError, match="invalid or missing fsdp stamp"):
        _run_gate()

    assert len(seen["path_info_requests"]) == 1


@pytest.mark.parametrize(
    "bad_name",
    [
        "actor/model_world_size_2_rank_00.pt",
        "actor/model_world_size_02_rank_0.pt",
    ],
)
def test_gate_blocks_noncanonical_native_rank_alias(monkeypatch, tmp_path, bad_name):
    files = _sharded_ckpt(40, 2) + _ckpt_files("opd", "run-1", 40, (bad_name,))
    _install_marker_gate(monkeypatch, tmp_path, checkpoint_files=files)

    with pytest.raises(RuntimeError, match="malformed shard name"):
        _run_gate()


@pytest.mark.parametrize(
    "invalid_size",
    [_MISSING_SIZE, True, "1", 1.5, 0, -1],
    ids=["missing", "boolean", "string", "float", "zero", "negative"],
)
def test_gate_blocks_invalid_expected_native_file_size(monkeypatch, tmp_path, invalid_size):
    invalid_path = "opd/run-1/checkpoint/checkpoint-40/actor/optim_world_size_2_rank_1.pt"
    _install_marker_gate(
        monkeypatch,
        tmp_path,
        checkpoint_files=_sharded_ckpt(40, 2),
        shard_size_overrides={invalid_path: invalid_size},
    )

    with pytest.raises(RuntimeError, match=r"empty or unreadable shard.*optim_world_size_2_rank_1"):
        _run_gate()


def test_nested_shards_do_not_count_toward_checkpoint_completeness(monkeypatch, tmp_path):
    """Collecting the nested shards must not let them stand in for a missing required file.

    The completeness rule is about DIRECT children (adapter, optimizer.pt, rng_state.pth,
    opd_state.json). Shards live one level down, so a torn upload that has shards but no optimizer
    is still not resumable, and the gate must still fail closed.
    """
    _install_marker_gate(
        monkeypatch,
        tmp_path,
        checkpoint_files=_ckpt_files("opd", "run-1", 40, _INCOMPLETE_CKPT)
        + _ckpt_files("opd", "run-1", 40, ("actor/model_world_size_2_rank_0.pt",)),
    )
    with pytest.raises(RuntimeError, match="no complete resume checkpoint is available"):
        _run_gate()


@pytest.mark.parametrize(
    ("resume_world_size", "survivor_indexes", "headline"),
    [
        pytest.param(2, (0, 1), ("runpod", "h100", 1.0, 4), id="accept-four-cards-two-ranks"),
        pytest.param(4, (2,), ("modal", "b200", 3.0, 4), id="reject-four-cards-two-ranks"),
    ],
)
def test_retry_allocation_is_pinned_to_the_resume_checkpoint_width(
    resume_world_size, survivor_indexes, headline
):
    """a pinned resume admits only candidates that execute at the checkpoint width."""
    from flash.providers.core.base import Allocation, Candidate
    from flash.runner.supervise.attempt_supervision import _pinned_to_resume_width

    candidates = (
        Candidate("runpod", "h100", 1.0, 80, 4, 2),
        Candidate("vast", "h200", 2.0, 141, 2, 2),
        Candidate("modal", "b200", 3.0, 180, 4, 4),
    )
    allocation = Allocation(
        provider="runpod",
        gpu="h100",
        hourly_usd=1.0,
        min_vram_gb=80,
        candidates=candidates,
        gpu_count=4,
    )

    pinned = _pinned_to_resume_width(allocation, resume_world_size)
    expected = tuple(candidates[index] for index in survivor_indexes)
    assert pinned.candidates == expected
    assert all(
        survivor is candidate
        for survivor, candidate in zip(pinned.candidates, expected, strict=True)
    )
    assert (pinned.provider, pinned.gpu, pinned.hourly_usd, pinned.gpu_count) == headline
    assert [(candidate.gpu_count, candidate.executed_gpu_count) for candidate in candidates] == [
        (4, 2),
        (2, 2),
        (4, 4),
    ]

    assert _pinned_to_resume_width(allocation, None) is allocation
    assert _pinned_to_resume_width(allocation, 0) is allocation
    assert _pinned_to_resume_width(allocation, 8).candidates == ()


def test_pinned_resume_stop_diagnostic_names_executed_checkpoint_width():
    """a filtered rental reports the incompatible execution width, not its card count."""
    from flash.core.spec import JobSpec
    from flash.providers.core.base import Allocation, Candidate
    from flash.runner.lifecycle.attempts import AttemptLaunchClaim
    from flash.runner.supervise.attempt_supervision import (
        _build_candidate_plan,
        _pinned_to_resume_width,
    )
    from flash.runner.supervise.retry_decision import RetryState

    rented_two_executes_one = Candidate("runpod", "h100", 1.0, 80, 2, 1)
    allocation = Allocation("runpod", "h100", 1.0, 80, (rented_two_executes_one,), gpu_count=2)
    filtered = _pinned_to_resume_width(allocation, 2)

    ctx = SimpleNamespace(
        last_detail=None,
        seed=42,
        log=io.StringIO(),
    )
    spec = JobSpec(run_id="opd-width-diagnostic", model="Qwen/Qwen3.5-9B", algorithm="opd")
    prepared = (
        AttemptLaunchClaim(0, "opd-width", "revision", 2),
        spec,
        {},
        RetryState.initial_for_spec(spec),
    )

    assert filtered.candidates == ()
    assert _build_candidate_plan(ctx, prepared, filtered) is None
    assert ctx.last_detail == (
        "no candidate executing at checkpoint world size 2 is available, and this retry must "
        "preserve that executed rank width"
    )
    assert ctx.log.getvalue() == ""


class _IntSubclass(int):
    pass


@pytest.mark.parametrize(
    ("executed_present", "executed_gpu_count", "wrong_width"),
    [
        pytest.param(False, None, 2, id="absent"),
        pytest.param(True, None, 2, id="none"),
        pytest.param(True, 0, 2, id="zero"),
        pytest.param(True, -1, -1, id="negative"),
        pytest.param(True, True, 1, id="boolean-true"),
        pytest.param(True, False, 2, id="boolean-false"),
        pytest.param(True, _IntSubclass(2), 2, id="int-subclass"),
        pytest.param(True, 2.0, 2, id="float"),
        pytest.param(True, "2", 2, id="string"),
        pytest.param(True, object(), 2, id="opaque"),
    ],
)
def test_retry_allocation_falls_back_for_unusable_executed_width(
    executed_present, executed_gpu_count, wrong_width
):
    """absent and malformed executed widths fall back to the rented card count."""
    from flash.providers.core.base import Allocation, Candidate
    from flash.runner.supervise.attempt_supervision import _pinned_to_resume_width

    if executed_present:
        candidate = Candidate("runpod", "h100", 1.0, 80, 4, executed_gpu_count)
    else:
        candidate = SimpleNamespace(
            provider="runpod", gpu="h100", hourly_usd=1.0, vram_gb=80, gpu_count=4
        )
    allocation = Allocation("runpod", "h100", 1.0, 80, (candidate,), gpu_count=4)

    fallback = _pinned_to_resume_width(allocation, 4)
    assert fallback.candidates == (candidate,)
    assert fallback.candidates[0] is candidate
    assert fallback.gpu_count == 4
    assert _pinned_to_resume_width(allocation, wrong_width).candidates == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"align_group_sum": float("nan")},
        {"align_group_sum": float("inf")},
        {"align_group_sum": -1.0},
        {"align_group_n": -1},
        {"align_group_n": 1.5},
        {"align_group_n": True},
    ],
)
def test_resume_validator_rejects_corrupt_alignment_granularity(overrides):
    # the verl worker resumes its granularity accumulators from these fields and publishes their
    # ratio into train_meta. a tampered or corrupt checkpoint must fail the fail-closed contract
    # here, not coerce past it and propagate a non-finite or negative mean into every later
    # checkpoint and the run's reported alignment health.
    state = _valid_resume_state(2, **overrides)
    # match on the corrupt field itself: a validator that rejected for some unrelated reason would
    # pass a bare raises() while leaving this pair entirely unchecked.
    (field,) = overrides

    with pytest.raises(ValueError, match=field):
        validate_opd_resume_state_metadata(state, expected_seed=42, checkpoint_step=2)


def test_resume_state_version_rejects_states_predating_alignment_granularity():
    # the verl worker resumes align_group_sum/align_group_n and publishes their ratio as
    # mean_align_granularity. a state written before those accumulators existed carries neither, so
    # resuming one restarts the accumulator at zero and the ratio would describe only the
    # post-resume samples while being reported as the whole run's alignment health. the contract
    # version is the enforcement point: it must have moved past 2 so the existing fail-closed check
    # refuses those states outright.
    assert OPD_RESUME_STATE_VERSION > 2

    stale = _valid_resume_state(2)
    stale["contract_version"] = 2

    with pytest.raises(ValueError, match="contract_version"):
        validate_opd_resume_state_metadata(stale, expected_seed=42, checkpoint_step=2)


def test_resume_validator_requires_the_alignment_granularity_pair():
    # `accounting_snapshot` is the ONLY producer of a version-4 state and always emits both fields,
    # so an absent one means a corrupt record, not an older writer: the trl backend that wrote
    # neither was deleted (8421a240, 2026-07-31) before this contract version existed (975ddbdc,
    # 2026-08-05), so no version-4 state can lack them. required rather than optional because the
    # reader defaults an absent field to 0, and the published mean_align_granularity would then
    # divide a real sum by a zeroed count -- or report 0.0 for a run that measured every group.
    with_granularity = _valid_resume_state(2, align_group_sum=3.0, align_group_n=2)
    validated = validate_opd_resume_state_metadata(
        with_granularity, expected_seed=42, checkpoint_step=2
    )
    assert validated["align_group_sum"] == 3.0
    assert validated["align_group_n"] == 2


@pytest.mark.parametrize(
    ("dropped", "missing"),
    [
        (("align_group_n",), "align_group_n"),
        (("align_group_sum",), "align_group_sum"),
        (("align_group_sum", "align_group_n"), "align_group_"),
    ],
)
def test_resume_validator_rejects_an_incomplete_alignment_pair(dropped, missing):
    # both a half-present pair and an entirely absent one are corrupt: the reader defaults each
    # missing field to 0 independently, so the published mean_align_granularity divides a real sum
    # by a zeroed count, or reports 0.0 for a run that measured alignment on every group.
    state = _valid_resume_state(2)
    for field in dropped:
        del state[field]

    with pytest.raises(ValueError, match=missing):
        validate_opd_resume_state_metadata(state, expected_seed=42, checkpoint_step=2)


def test_discarded_rollouts_is_not_part_of_the_resume_accounting_contract():
    required = set(retry_contract._OPD_RESUME_ACCOUNTING_SCHEMA)

    assert "truncated_rollouts" in required
    assert "discarded_rollouts" not in required


def test_shared_resume_metadata_validator_returns_a_copy():
    state = _valid_resume_state(3)

    validated = validate_opd_resume_state_metadata(
        state,
        expected_seed=42,
        checkpoint_step=3,
    )

    assert validated == state
    assert validated is not state


def test_gate_validates_every_present_marker(monkeypatch, tmp_path):
    paths = [opd_optimizer_start_marker_path("run-1", attempt) for attempt in range(2)]

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="pinned-sha")

        def get_paths_info(self, **_kwargs):
            return [SimpleNamespace(path=path) for path in paths]

        def list_repo_files(self, **_kwargs):
            return _ckpt_files("opd", "run-1", 40, _COMPLETE_CKPT)

    def download(**kwargs):
        path = tmp_path / f"marker-{kwargs['filename'].split('attempt-')[1].split('/')[0]}.json"
        if kwargs["filename"] == paths[0]:
            path.write_bytes(canonical_opd_optimizer_start_json(run_id="run-1", attempt=0, seed=42))
        else:
            path.write_text("{}")
        return str(path)

    _install_hf_reader(monkeypatch, Api(), download)
    with pytest.raises(RuntimeError, match="evidence is unverifiable"):
        _run_gate()


def test_gate_blocks_when_newest_checkpoint_is_incomplete(monkeypatch, tmp_path):
    # an older complete dir plus a newer incomplete one: the reader keys off the newest step (what
    # hf_resume_checkpoint downloads), so a torn latest upload fails closed even with an older complete dir.
    files = _ckpt_files("opd", "run-1", 8, _COMPLETE_CKPT) + _ckpt_files(
        "opd", "run-1", 40, _INCOMPLETE_CKPT
    )
    _install_marker_gate(monkeypatch, tmp_path, checkpoint_files=files)
    with pytest.raises(RuntimeError, match="no complete resume checkpoint is available"):
        _run_gate()


def test_gate_blocks_when_checkpoint_listing_fails(monkeypatch, tmp_path):
    _install_marker_gate(
        monkeypatch, tmp_path, checkpoint_files=PermissionError("auth unavailable")
    )
    with pytest.raises(RuntimeError, match="no complete resume checkpoint is available"):
        _run_gate()


def test_gate_blocks_when_no_checkpoint_dir_present(monkeypatch, tmp_path):
    # marker proves mutation but the checkpoint prefix is empty (no save_every boundary reached yet).
    _install_marker_gate(monkeypatch, tmp_path, checkpoint_files=[])
    with pytest.raises(RuntimeError, match="no complete resume checkpoint is available"):
        _run_gate()


@pytest.mark.parametrize("step", [0, "040"])
def test_gate_ignores_noncanonical_checkpoint_steps(monkeypatch, tmp_path, step):
    _install_marker_gate(
        monkeypatch,
        tmp_path,
        checkpoint_files=_ckpt_files("opd", "run-1", step, _COMPLETE_CKPT),
    )
    with pytest.raises(RuntimeError, match="no complete resume checkpoint is available"):
        _run_gate()


def test_gate_ignores_checkpoint_under_other_phase(monkeypatch, tmp_path):
    # a complete checkpoint under a different phase prefix must not unblock an opd replacement.
    _install_marker_gate(
        monkeypatch, tmp_path, checkpoint_files=_ckpt_files("rl", "run-1", 40, _COMPLETE_CKPT)
    )
    with pytest.raises(RuntimeError, match="no complete resume checkpoint is available"):
        _run_gate()


def test_gate_ignores_nested_files_when_checking_completeness(monkeypatch, tmp_path):
    # optimizer.pt nested one dir deeper is not a direct child of checkpoint-40, so the flat state set
    # is incomplete and replacement stays blocked.
    files = _ckpt_files("opd", "run-1", 40, _INCOMPLETE_CKPT)
    files.append("opd/run-1/checkpoint/checkpoint-40/nested/optimizer.pt")
    _install_marker_gate(monkeypatch, tmp_path, checkpoint_files=files)
    with pytest.raises(RuntimeError, match="no complete resume checkpoint is available"):
        _run_gate()
