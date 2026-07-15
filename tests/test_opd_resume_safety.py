from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.opd_retry_contract import (
    OPD_RETRY_CONTRACT_STATUS_KEY,
    OPD_RETRY_CONTRACT_VERSION,
    canonical_opd_optimizer_start_json,
    decode_opd_optimizer_start_json,
    opd_optimizer_start_marker_path,
)

_RUNPOD_FINGERPRINT = "rpk-0123456789ab"


def _remote(*, attempt: int = 0) -> dict:
    return {
        "provider": "runpod",
        "endpoint_id": f"endpoint-{attempt}",
        "endpoint_name": f"endpoint-{attempt}-name",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "job_id": f"job-{attempt}",
        "attempt": attempt,
        "started_ts": float(attempt + 1),
    }


def _opd_spec(run_id: str, *, max_retries: int = 1):
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    return JobSpec(
        run_id=run_id,
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(hf_repo="private/runs", max_examples=1, epochs=1),
        gpu=GpuSpec(type="RTX 4090", max_retries=max_retries),
    )


def _save_status(
    runner,
    spec,
    *,
    state="running",
    next_attempt=0,
    remote=None,
    contracted=True,
):
    kwargs = {"_next_attempt": next_attempt}
    if contracted:
        kwargs["_opd_retry_contract_version"] = OPD_RETRY_CONTRACT_VERSION
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state=state,
            spec=spec.to_dict(),
            remote=remote,
        ),
        **kwargs,
    )


def test_status_initialization_stamps_opd_contract_only_when_explicit(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    contracted = _opd_spec("contract-opd")
    uncontracted = _opd_spec("uncontracted-opd")
    _save_status(runner, contracted)
    _save_status(runner, uncontracted, contracted=False)
    for spec in (
        JobSpec(run_id="contract-sft", algorithm="sft"),
        JobSpec(run_id="contract-grpo", algorithm="grpo"),
    ):
        runner._save_status(
            runner.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
        )

    opd_raw = runner._load_status_json("contract-opd")
    assert opd_raw[OPD_RETRY_CONTRACT_STATUS_KEY] == OPD_RETRY_CONTRACT_VERSION
    assert OPD_RETRY_CONTRACT_STATUS_KEY not in runner.get_status("contract-opd").__dict__
    assert OPD_RETRY_CONTRACT_STATUS_KEY not in runner._load_status_json("uncontracted-opd")
    assert OPD_RETRY_CONTRACT_STATUS_KEY not in runner._load_status_json("contract-sft")
    assert OPD_RETRY_CONTRACT_STATUS_KEY not in runner._load_status_json("contract-grpo")
    with pytest.raises(ValueError, match="cannot be stored for a non-opd run"):
        runner._save_status(
            runner.RunStatus(
                run_id="invalid-contract-sft",
                state="running",
                spec=JobSpec(run_id="invalid-contract-sft", algorithm="sft").to_dict(),
            ),
            _opd_retry_contract_version=OPD_RETRY_CONTRACT_VERSION,
        )
    assert runner._reserve_attempt("contract-sft") == 0
    assert runner._reserve_attempt("contract-grpo") == 0


def test_marker_path_and_canonical_exact_schema():
    raw = canonical_opd_optimizer_start_json(run_id="run-1", attempt=3, seed=42)
    assert raw == (
        b'{"attempt":3,"contract":"flash.opd.optimizer-start","phase":"opd",'
        b'"run_id":"run-1","seed":42,"version":1}'
    )
    assert opd_optimizer_start_marker_path("run-1", 3) == (
        "_opd_retry/run-1/attempts/attempt-3/optimizer-start.v1.json"
    )
    assert decode_opd_optimizer_start_json(
        raw, run_id="run-1", attempt=3, seed=42
    ) == json.loads(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"attempt":3,"contract":"flash.opd.optimizer-start","phase":"opd",'
        b'"run_id":"run-1","seed":42,"version":1,"extra":0}',
        b'{"attempt":true,"contract":"flash.opd.optimizer-start","phase":"opd",'
        b'"run_id":"run-1","seed":42,"version":1}',
        b'{"attempt":3,"contract":"wrong","phase":"opd","run_id":"run-1",'
        b'"seed":42,"version":1}',
        b'{"attempt":3,"attempt":3,"contract":"flash.opd.optimizer-start",'
        b'"phase":"opd","run_id":"run-1","seed":42,"version":1}',
        b'{"attempt": 3, "contract": "flash.opd.optimizer-start", "phase": "opd", '
        b'"run_id": "run-1", "seed": 42, "version": 1}',
        b"\xff",
    ],
)
def test_marker_decode_rejects_malformed_or_noncanonical_evidence(raw):
    with pytest.raises(ValueError, match="marker"):
        decode_opd_optimizer_start_json(raw, run_id="run-1", attempt=3, seed=42)


def test_optimizer_update_orders_marker_before_mutation_exactly_once(monkeypatch):
    from flash.engine.worker import opd

    events = []

    class Grad:
        def mul_(self, value):
            events.append(("rescale", value))

    class Parameter:
        requires_grad = True
        grad = Grad()

    parameter = Parameter()
    model = SimpleNamespace(parameters=lambda: [parameter])
    optimizer = SimpleNamespace(step=lambda: events.append(("step", None)))
    torch = SimpleNamespace(
        nn=SimpleNamespace(
            utils=SimpleNamespace(
                clip_grad_norm_=lambda _parameters, value: events.append(("clip", value))
            )
        )
    )
    monkeypatch.setattr(
        opd,
        "_w",
        SimpleNamespace(
            publish_opd_optimizer_start_marker=lambda: events.append(("marker", None))
        ),
    )

    opt_steps = opd._apply_opd_optimizer_update(
        model=model,
        optimizer=optimizer,
        torch=torch,
        opt_steps=0,
        nseq=1,
        accum_target=2,
    )
    opt_steps = opd._apply_opd_optimizer_update(
        model=model,
        optimizer=optimizer,
        torch=torch,
        opt_steps=opt_steps,
        nseq=1,
        accum_target=1,
    )

    assert opt_steps == 2
    assert events == [
        ("rescale", 2.0),
        ("clip", 1.0),
        ("marker", None),
        ("step", None),
        ("clip", 1.0),
        ("step", None),
    ]


def test_no_signal_does_not_publish_or_mutate(monkeypatch):
    from flash.engine.worker import opd

    events = []
    monkeypatch.setattr(
        opd,
        "_w",
        SimpleNamespace(
            publish_opd_optimizer_start_marker=lambda: events.append("marker")
        ),
    )
    with pytest.raises(ValueError, match="positive sequence counts"):
        opd._apply_opd_optimizer_update(
            model=SimpleNamespace(parameters=lambda: []),
            optimizer=SimpleNamespace(step=lambda: events.append("step")),
            torch=SimpleNamespace(),
            opt_steps=0,
            nseq=0,
            accum_target=1,
        )
    assert events == []


def test_marker_upload_failure_prevents_optimizer_step_and_preserves_retriable(monkeypatch):
    from flash.engine.worker import hf, opd
    from flash.engine.worker.perf import RetriableInfraError

    events = []
    uploads = []
    failure = RetriableInfraError("required upload failed")

    def upload_file(**kwargs):
        uploads.append(kwargs)
        raise failure

    api = SimpleNamespace(upload_file=upload_file)
    monkeypatch.setattr(
        hf,
        "_w",
        SimpleNamespace(
            HF_REPO="private/runs",
            RUN_ID="run-1",
            ATTEMPT=0,
            SEED=42,
            hf_api=lambda: api,
            _remaining_worker_wall_seconds=lambda: None,
        ),
    )
    monkeypatch.setattr(
        opd,
        "_w",
        SimpleNamespace(publish_opd_optimizer_start_marker=hf.publish_opd_optimizer_start_marker),
    )
    torch = SimpleNamespace(
        nn=SimpleNamespace(
            utils=SimpleNamespace(clip_grad_norm_=lambda *_args: events.append("clip"))
        )
    )
    with pytest.raises(RetriableInfraError) as caught:
        opd._apply_opd_optimizer_update(
            model=SimpleNamespace(parameters=lambda: []),
            optimizer=SimpleNamespace(step=lambda: events.append("step")),
            torch=torch,
            opt_steps=0,
            nseq=1,
            accum_target=1,
        )
    assert caught.value.__cause__ is failure
    assert len(uploads) == 1
    assert events == ["clip"]


def test_ambiguous_marker_upload_is_not_retried_or_applied(monkeypatch):
    from flash.engine.worker import hf, opd
    from flash.engine.worker.perf import RetriableInfraError

    events = []

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
    monkeypatch.setattr(
        hf,
        "_w",
        SimpleNamespace(
            HF_REPO="private/runs",
            RUN_ID="run-ambiguous",
            ATTEMPT=2,
            SEED=42,
            hf_api=lambda: api,
            _remaining_worker_wall_seconds=lambda: None,
        ),
    )
    monkeypatch.setattr(
        opd,
        "_w",
        SimpleNamespace(publish_opd_optimizer_start_marker=hf.publish_opd_optimizer_start_marker),
    )
    torch = SimpleNamespace(
        nn=SimpleNamespace(
            utils=SimpleNamespace(clip_grad_norm_=lambda *_args: events.append("clip"))
        )
    )

    with pytest.raises(RetriableInfraError) as caught:
        opd._apply_opd_optimizer_update(
            model=SimpleNamespace(parameters=lambda: []),
            optimizer=SimpleNamespace(step=lambda: events.append("step")),
            torch=torch,
            opt_steps=0,
            nseq=1,
            accum_target=1,
        )

    assert isinstance(caught.value.__cause__, TimeoutError)
    assert "marker-secret" not in str(caught.value)
    assert len(str(caught.value)) <= len(
        "RETRIABLE_INFRA_GPU: required upload of OPD optimizer-start marker failed: "
    ) + 500
    assert len(api.uploads) == 1
    assert api.committed == canonical_opd_optimizer_start_json(
        run_id="run-ambiguous", attempt=2, seed=42
    )
    assert events == ["clip"]


def test_worker_marker_rejects_empty_repo(monkeypatch):
    from flash.engine.worker import hf

    monkeypatch.setattr(
        hf,
        "_w",
        SimpleNamespace(HF_REPO="", RUN_ID="run-1", ATTEMPT=0, SEED=42),
    )
    with pytest.raises(RuntimeError, match="requires a private HF repository"):
        hf.publish_opd_optimizer_start_marker()


def test_worker_marker_starts_no_hf_call_at_deadline(monkeypatch):
    from flash.engine.worker import hf
    from flash.engine.worker.perf import RetriableInfraError

    calls = []
    monkeypatch.setattr(
        hf,
        "_w",
        SimpleNamespace(
            HF_REPO="private/runs",
            RUN_ID="run-1",
            ATTEMPT=0,
            SEED=42,
            hf_api=lambda: calls.append("hf_api"),
            _remaining_worker_wall_seconds=lambda: 0.0,
        ),
    )

    with pytest.raises(RetriableInfraError, match="run wall deadline exceeded"):
        hf.publish_opd_optimizer_start_marker()

    assert calls == []


def test_worker_marker_writes_fsync_and_required_upload(monkeypatch):
    from flash.engine.worker import hf

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
    monkeypatch.setattr(
        hf,
        "_w",
        SimpleNamespace(
            HF_REPO="private/runs",
            RUN_ID="run-1",
            ATTEMPT=4,
            SEED=42,
            hf_api=lambda: SimpleNamespace(upload_file=upload_file),
            _remaining_worker_wall_seconds=lambda: None,
        ),
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
        return [SimpleNamespace(path=path) for path in kwargs["paths"] if path in self.files]

    def upload_file(self, **kwargs):
        path = kwargs["path_in_repo"]
        self.calls.append(("upload_file", kwargs))
        self.files[path] = Path(kwargs["path_or_fileobj"]).read_bytes()
        if self.raise_after_upload:
            raise TimeoutError("commit response was lost")

    def download(self, **kwargs):
        self.calls.append(("download", kwargs))
        path = kwargs["filename"]
        local_path = self.root / path.replace("/", "-")
        local_path.write_bytes(self.files[path])
        return str(local_path)

    def install(self, monkeypatch):
        _install_hf_reader(monkeypatch, self, self.download)


def test_strict_reader_pins_one_sha_and_any_present_marker_blocks(monkeypatch, tmp_path):
    from flash.providers._hf_artifacts import verify_opd_retry_markers_absent

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
        path.write_bytes(
            canonical_opd_optimizer_start_json(run_id="run-1", attempt=1, seed=42)
        )
        return str(path)

    _install_hf_reader(monkeypatch, Api(), download)
    with pytest.raises(RuntimeError, match="may have mutated optimizer state"):
        verify_opd_retry_markers_absent(
            hf_repo="private/runs",
            run_id="run-1",
            seed=42,
            next_attempt=2,
            contract_version=1,
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
    from flash.providers._hf_artifacts import verify_opd_retry_markers_absent

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
    verify_opd_retry_markers_absent(
        hf_repo="private/runs",
        run_id="run-1",
        seed=42,
        next_attempt=2,
        contract_version=1,
    )


@pytest.mark.parametrize("mode", ["malformed", "timeout", "listing"])
def test_strict_reader_malformed_or_outage_blocks(monkeypatch, tmp_path, mode):
    from flash.providers._hf_artifacts import verify_opd_retry_markers_absent

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
        verify_opd_retry_markers_absent(
            hf_repo="private/runs",
            run_id="run-1",
            seed=42,
            next_attempt=1,
            contract_version=1,
        )


@pytest.mark.parametrize(
    ("repo", "version"),
    [("", 1), ("private/runs", 2), ("private/runs", True), ("private/runs", "1")],
)
def test_strict_reader_missing_repo_or_unsupported_contract_blocks(repo, version):
    from flash.providers._hf_artifacts import verify_opd_retry_markers_absent

    with pytest.raises(RuntimeError, match="replacement is blocked"):
        verify_opd_retry_markers_absent(
            hf_repo=repo,
            run_id="run-1",
            seed=42,
            next_attempt=1,
            contract_version=version,
        )


def test_initial_contracted_opd_reservation_skips_empty_attempt_query(monkeypatch, tmp_path):
    import huggingface_hub

    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("initial-opd")
    _save_status(runner, spec, next_attempt=0)
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not query HF")),
    )

    snapshot = runner._verified_opd_next_attempt(spec.run_id)
    assert snapshot == 0
    assert runner._reserve_attempt(spec.run_id, expected_next_attempt=snapshot) == 0


def test_precontract_opd_fails_closed(monkeypatch, tmp_path):
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("precontract-opd")
    _save_status(runner, spec, contracted=False)

    with pytest.raises(RuntimeError, match="contract is missing or invalid"):
        runner._verified_opd_next_attempt(spec.run_id)
    with pytest.raises(RuntimeError, match="contract is missing or invalid"):
        runner._reserve_attempt(spec.run_id, expected_next_attempt=0)


def test_next_attempt_cas_race_blocks_opd_reservation(monkeypatch, tmp_path):
    import flash.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("opd-cas")
    _save_status(runner, spec, next_attempt=0)
    snapshot = runner._verified_opd_next_attempt(spec.run_id)
    status = runner.get_status(spec.run_id)
    runner._save_status(status, _next_attempt=1)

    with pytest.raises(RuntimeError, match="changed after retry verification"):
        runner._reserve_attempt(spec.run_id, expected_next_attempt=snapshot)
    assert runner._load_status_json(spec.run_id)[runner._NEXT_ATTEMPT_KEY] == 1


def test_opd_automatic_retry_after_teardown_requires_all_markers_absent(
    monkeypatch, tmp_path
):
    import flash.providers as providers
    import flash.providers.allocator as allocator
    import flash.runner as runner
    from flash.providers.base import Allocation, Candidate, PollResult
    from flash.runner import lifecycle

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    private_hf = _FakePrivateHf(tmp_path)
    private_hf.install(monkeypatch)
    spec = _opd_spec("automatic-retry-absent")
    _save_status(runner, spec, next_attempt=0)
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=(candidate,),
        ),
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda *_args: None)

    class Provider:
        supports_weight_cache = False

        def __init__(self):
            self.attempts = []
            self.teardown = []
            self.events = []

        def submit_run(self, _spec, _seed, *, attempt, on_handle, **_kwargs):
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

    metrics = lifecycle._submit_seed_supervised(spec, 42, io.StringIO())

    assert metrics == {"train_tokens": 1, "allocated_gpu": "RTX 4090"}
    assert provider.attempts == [0, 1]
    retry_submit = provider.events.index(("submit", 1))
    assert provider.events.index(("cancel", 0)) < retry_submit
    assert provider.events.index(("destroy", 0)) < retry_submit
    assert [name for name, _kwargs in private_hf.calls] == ["repo_info", "get_paths_info"]
    assert private_hf.calls[1][1]["paths"] == [
        opd_optimizer_start_marker_path(spec.run_id, 0)
    ]


def test_failed_attached_opd_worker_decodes_present_marker_after_teardown(
    monkeypatch, tmp_path
):
    import flash.providers as providers
    import flash.runner as runner
    from flash.providers.base import PollResult

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    spec = _opd_spec("attach-opd-block")
    marker_path = opd_optimizer_start_marker_path(spec.run_id, 0)
    private_hf = _FakePrivateHf(tmp_path)
    private_hf.files[marker_path] = canonical_opd_optimizer_start_json(
        run_id=spec.run_id, attempt=0, seed=42
    )
    private_hf.install(monkeypatch)
    _save_status(runner, spec, next_attempt=1, remote=_remote(attempt=0))
    events = []

    class Provider:
        def poll(self, *_args, **_kwargs):
            return PollResult(False, failure="stalled", detail="worker stopped")

        def cancel(self, _handle):
            events.append("cancel")

        def destroy(self, _handle):
            events.append("destroy")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    resumed = []
    monkeypatch.setattr(runner, "_run_training", lambda *_args, **_kwargs: resumed.append(True))

    status = runner.attach_run(spec.run_id, log_stream=io.StringIO())

    assert events == ["cancel", "destroy"]
    assert resumed == []
    assert status.state == "failed"
    assert status.remote is None
    assert "replacement is blocked" in status.error
    assert [name for name, _kwargs in private_hf.calls] == [
        "repo_info",
        "get_paths_info",
        "download",
    ]
    assert private_hf.calls[2][1]["revision"] == "private-pinned-sha"


def test_handleless_opd_recovery_blocks_through_recover_runs(monkeypatch, tmp_path):
    import flash.providers as providers
    import flash.runner as runner
    from flash.server import _runtime

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("handleless-opd-block")
    marker_path = opd_optimizer_start_marker_path(spec.run_id, 0)
    private_hf = _FakePrivateHf(tmp_path)
    private_hf.files[marker_path] = canonical_opd_optimizer_start_json(
        run_id=spec.run_id, attempt=0, seed=42
    )
    private_hf.install(monkeypatch)
    _save_status(runner, spec, state="provisioning", next_attempt=1)
    started = []
    monkeypatch.setattr(runner, "_run_job_background", lambda *_args: started.append(True))
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(_runtime.db, "all_runs", lambda: [{"run_id": spec.run_id}])
    monkeypatch.setattr(providers, "configured_providers", lambda: [])

    _runtime.recover_runs()

    assert started == []
    status = runner.get_status(spec.run_id)
    assert status.state == "failed"
    assert "replacement is blocked" in status.error
    assert [name for name, _kwargs in private_hf.calls] == [
        "repo_info",
        "get_paths_info",
        "download",
    ]


def test_ambiguous_marker_upload_prevents_step_and_blocks_replacement(monkeypatch, tmp_path):
    import flash.providers.allocator as allocator
    import flash.runner as runner
    from flash.engine.worker import hf, opd
    from flash.engine.worker.perf import RetriableInfraError
    from flash.runner import lifecycle

    private_hf = _FakePrivateHf(tmp_path, raise_after_upload=True)
    monkeypatch.setattr(hf, "_sleep_with_hf_deadline", lambda _delay: True)
    monkeypatch.setattr(
        hf,
        "_w",
        SimpleNamespace(
            HF_REPO="private/runs",
            RUN_ID="ambiguous-upload",
            ATTEMPT=0,
            SEED=42,
            hf_api=lambda: private_hf,
            _remaining_worker_wall_seconds=lambda: None,
        ),
    )
    monkeypatch.setattr(
        opd,
        "_w",
        SimpleNamespace(publish_opd_optimizer_start_marker=hf.publish_opd_optimizer_start_marker),
    )
    events = []
    torch = SimpleNamespace(
        nn=SimpleNamespace(
            utils=SimpleNamespace(clip_grad_norm_=lambda *_args: events.append("clip"))
        )
    )

    with pytest.raises(RetriableInfraError, match="required upload"):
        opd._apply_opd_optimizer_update(
            model=SimpleNamespace(parameters=lambda: []),
            optimizer=SimpleNamespace(step=lambda: events.append("step")),
            torch=torch,
            opt_steps=0,
            nseq=1,
            accum_target=1,
        )

    assert events == ["clip"]
    marker_path = opd_optimizer_start_marker_path("ambiguous-upload", 0)
    assert private_hf.files[marker_path] == canonical_opd_optimizer_start_json(
        run_id="ambiguous-upload", attempt=0, seed=42
    )

    private_hf.raise_after_upload = False
    private_hf.install(monkeypatch)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _opd_spec("ambiguous-upload")
    _save_status(runner, spec, next_attempt=1)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replacement must be blocked before allocation")
        ),
    )

    with pytest.raises(RuntimeError, match="may have mutated optimizer state"):
        lifecycle._submit_seed_supervised(spec, 42, io.StringIO())
