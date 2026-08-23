"""Hermetic coverage for the Modal training provider."""

from __future__ import annotations

import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

from flash.providers.base import AllocationConstraints, JobHandle, PollResult, UnsupportedGpuError
from flash.providers.modal import ModalProvider


def test_modal_package_imports_the_third_party_sdk_without_shadowing() -> None:
    from flash.providers.modal import api

    assert api._modal_sdk().__name__ == "modal"


def test_modal_api_creates_a_mocked_sandbox_with_exact_inputs(monkeypatch) -> None:
    from flash.providers.modal import api, auth

    calls = {}

    class _InvalidError(Exception):
        pass

    class _NotFoundError(Exception):
        pass

    class _Client:
        def _close(self):
            calls["closed"] = True

    class _ClientFactory:
        @staticmethod
        def from_credentials(token_id, token_secret):
            calls["credentials"] = (token_id, token_secret)
            return _Client()

    class _App:
        @staticmethod
        def lookup(name, **kwargs):
            calls["app"] = (name, kwargs)
            return "app"

    class _Image:
        @staticmethod
        def from_registry(image):
            calls["image"] = image
            return f"image:{image}"

    class _Sandbox:
        @staticmethod
        def create(*args, **kwargs):
            calls["create"] = (args, kwargs)
            return SimpleNamespace(object_id="sb-1")

    sdk = SimpleNamespace(
        Client=_ClientFactory,
        App=_App,
        Image=_Image,
        Sandbox=_Sandbox,
        exception=SimpleNamespace(InvalidError=_InvalidError, NotFoundError=_NotFoundError),
    )
    monkeypatch.setattr(auth, "load_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr(api, "_modal_sdk", lambda: sdk)

    instance_id = api.create_sandbox(
        "python",
        "-c",
        "bootstrap",
        image="worker:sm90",
        gpu="H100!:8",
        env={"PAYLOAD": "secret"},
        timeout=600,
        name="flash-run-s0-a0",
        tags={api.LABEL_TAG: "flash-run-s0-a0"},
    )

    assert instance_id == "sb-1"
    assert calls["credentials"] == ("id", "secret")
    assert calls["image"] == "worker:sm90"
    assert calls["create"] == (
        ("python", "-c", "bootstrap"),
        {
            "app": "app",
            "name": "flash-run-s0-a0",
            "tags": {api.LABEL_TAG: "flash-run-s0-a0"},
            "image": "image:worker:sm90",
            "gpu": "H100!:8",
            "env": {"PAYLOAD": "secret"},
            "timeout": 600,
            "client": calls["app"][1]["client"],
        },
    )
    assert calls["app"][0] == api.APP_NAME
    assert calls["app"][1]["create_if_missing"] is True
    assert calls["closed"] is True


@pytest.mark.parametrize(
    "error_name",
    [
        "AuthError",
        "PermissionDeniedError",
        "NotFoundError",
        "RequestSizeError",
        "ImageBuildError",
        "InvalidError",
    ],
)
def test_modal_create_clean_rejections_remain_retryable(monkeypatch, error_name) -> None:
    from flash.providers.modal import api, auth

    clean_error = type(error_name, (Exception,), {})

    class _OtherError(Exception):
        pass

    class _Client:
        def _close(self):
            pass

    exception_types = dict.fromkeys(
        (
            "AuthError",
            "PermissionDeniedError",
            "NotFoundError",
            "RequestSizeError",
            "ImageBuildError",
            "ResourceExhaustedError",
            "InvalidError",
            "ConflictError",
        ),
        _OtherError,
    )
    exception_types[error_name] = clean_error
    sdk = SimpleNamespace(
        Client=SimpleNamespace(from_credentials=lambda *_args: _Client()),
        App=SimpleNamespace(lookup=lambda *_args, **_kwargs: "app"),
        Image=SimpleNamespace(from_registry=lambda image: image),
        Sandbox=SimpleNamespace(
            create=lambda *_args, **_kwargs: (_ for _ in ()).throw(clean_error()),
            list=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("unexpected reconciliation")
            ),
        ),
        exception=SimpleNamespace(**exception_types),
    )
    monkeypatch.setattr(auth, "load_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr(api, "_modal_sdk", lambda: sdk)

    with pytest.raises(api.ModalApiError, match=error_name):
        api.create_sandbox(
            "true",
            image="worker:sm90",
            gpu="H100!",
            env={},
            timeout=60,
            name="flash-run-s0-a0",
            tags={api.LABEL_TAG: "flash-run-s0-a0"},
        )


def test_modal_create_ambiguous_empty_reconciliation_stays_terminal(monkeypatch) -> None:
    from flash.providers.base import UnreconciledCreateError
    from flash.providers.modal import api, auth

    class _ConnectionError(Exception):
        pass

    class _OtherError(Exception):
        pass

    class _Client:
        def _close(self):
            pass

    sdk = SimpleNamespace(
        Client=SimpleNamespace(from_credentials=lambda *_args: _Client()),
        App=SimpleNamespace(lookup=lambda *_args, **_kwargs: "app"),
        Image=SimpleNamespace(from_registry=lambda image: image),
        Sandbox=SimpleNamespace(
            create=lambda *_args, **_kwargs: (_ for _ in ()).throw(_ConnectionError()),
            list=lambda **_kwargs: iter(()),
        ),
        exception=SimpleNamespace(
            InvalidError=_OtherError,
            ConflictError=_OtherError,
            NotFoundError=_OtherError,
            ConnectionError=_ConnectionError,
        ),
    )
    monkeypatch.setattr(auth, "load_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr(api, "_modal_sdk", lambda: sdk)

    with pytest.raises(UnreconciledCreateError, match="did not resolve"):
        api.create_sandbox(
            "true",
            image="worker:sm90",
            gpu="H100!",
            env={},
            timeout=60,
            name="flash-run-s0-a0",
            tags={api.LABEL_TAG: "flash-run-s0-a0"},
        )


@pytest.mark.parametrize("error_name", ["ResourceExhaustedError", "ConflictError"])
def test_modal_create_quota_and_conflict_stay_ambiguous(monkeypatch, error_name) -> None:
    """A quota refusal or a name conflict must NOT be treated as a settled rejection.

    Modal raises ``ResourceExhaustedError`` (grpc RESOURCE_EXHAUSTED) for a quota or rate limit,
    which it can do AFTER admitting the work, so a Sandbox may exist with an id this process never
    saw. ``ConflictError`` subclasses ``InvalidError``, so without an explicit exclusion it would
    inherit the clean verdict from its base. Calling either one clean authorizes a second create
    against a resource that may already be billing.
    """
    from flash.providers.base import UnreconciledCreateError
    from flash.providers.modal import api, auth

    ambiguous = type(error_name, (Exception,), {})

    class _OtherError(Exception):
        pass

    class _Client:
        def _close(self):
            pass

    exception_types = dict.fromkeys(
        (
            "AuthError",
            "PermissionDeniedError",
            "NotFoundError",
            "RequestSizeError",
            "ImageBuildError",
            "InvalidError",
            "ConflictError",
            "ResourceExhaustedError",
        ),
        _OtherError,
    )
    exception_types[error_name] = ambiguous
    if error_name == "ConflictError":
        # the real sdk has ConflictError subclass InvalidError, which is what the exclusion guards.
        exception_types["InvalidError"] = ambiguous.__mro__[1]
    sdk = SimpleNamespace(
        Client=SimpleNamespace(from_credentials=lambda *_args: _Client()),
        App=SimpleNamespace(lookup=lambda *_args, **_kwargs: "app"),
        Image=SimpleNamespace(from_registry=lambda image: image),
        Sandbox=SimpleNamespace(
            create=lambda *_args, **_kwargs: (_ for _ in ()).throw(ambiguous()),
            list=lambda **_kwargs: iter(()),
        ),
        exception=SimpleNamespace(**exception_types),
    )
    monkeypatch.setattr(auth, "load_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr(api, "_modal_sdk", lambda: sdk)

    with pytest.raises(UnreconciledCreateError):
        api.create_sandbox(
            "true",
            image="worker:sm90",
            gpu="H100!",
            env={},
            timeout=60,
            name="flash-run-s0-a0",
            tags={api.LABEL_TAG: "flash-run-s0-a0"},
        )


def test_modal_provider_delegates_credentials_pricing_gc_and_orphan_sweep(monkeypatch) -> None:
    from flash.providers.modal import jobs, preflight, pricing

    provider = ModalProvider()
    calls = []
    monkeypatch.setattr(
        preflight,
        "missing_credentials",
        lambda require_hf: calls.append(("credentials", require_hf)) or ["missing"],
    )
    monkeypatch.setattr(
        pricing,
        "hourly_rate",
        lambda gpu: calls.append(("rate", gpu)) or 4.5,
    )
    monkeypatch.setattr(
        jobs,
        "terminate_run_sandboxes",
        lambda run_id: calls.append(("gc", run_id)),
    )
    monkeypatch.setattr(
        jobs,
        "sweep_orphans",
        lambda **kwargs: calls.append(("sweep", kwargs)) or ["sb-1"],
    )

    assert provider._missing_credentials(False) == ["missing"]
    assert provider._hourly_rate("H200") == 4.5
    assert provider._gc("flash-1") is None
    assert provider._sweep_orphans(active_labels={"live"}, known_labels={"live", "done"}) == [
        "sb-1"
    ]
    assert calls == [
        ("credentials", False),
        ("rate", "H200"),
        ("gc", "flash-1"),
        ("sweep", {"active_labels": {"live"}, "known_labels": {"live", "done"}}),
    ]


def test_modal_pricing_uses_only_the_modal_table() -> None:
    from flash.providers.modal.pricing import hourly_rate

    assert hourly_rate("A10") == pytest.approx(1.10)
    assert hourly_rate("A100 SXM 40GB") == pytest.approx(2.10)
    assert hourly_rate("A100 SXM") == pytest.approx(2.50)
    assert hourly_rate("H100") == pytest.approx(3.95)
    assert hourly_rate("H200") == pytest.approx(4.54)
    assert hourly_rate("B200") == pytest.approx(6.25)
    with pytest.raises(UnsupportedGpuError, match="modal does not offer"):
        hourly_rate("A100 PCIe")


def test_modal_pinned_offline_allocation_uses_modal_rate() -> None:
    from flash.providers.modal.pricing import hourly_rate
    from flash.runner.costs import _pinned_offline_allocation

    allocation = _pinned_offline_allocation("modal", "H100", 2)

    assert allocation is not None
    assert (allocation.provider, allocation.gpu_count, allocation.hourly_usd) == (
        "modal",
        2,
        hourly_rate("H100"),
    )


def test_live_candidates_are_static_shapes_without_a_capacity_probe(monkeypatch) -> None:
    provider = ModalProvider()
    monkeypatch.setattr(
        provider,
        "gpu_classes",
        lambda: [
            SimpleNamespace(name="A10", vram_gb=24),
            SimpleNamespace(name="H100", vram_gb=80),
        ],
    )
    monkeypatch.setattr(provider, "hourly_rate", lambda gpu: {"A10": 1.1, "H100": 3.95}[gpu])

    candidates = provider.live_candidates(20, AllocationConstraints(max_gpu_count=8))

    assert provider.live_capacity is False
    assert [(c.gpu, c.gpu_count) for c in candidates] == [
        ("A10", 4),
        ("A10", 2),
        ("A10", 1),
        ("H100", 8),
        ("H100", 4),
        ("H100", 2),
        ("H100", 1),
    ]


def test_live_candidates_honor_exact_type_and_combined_vram(monkeypatch) -> None:
    provider = ModalProvider()
    monkeypatch.setattr(
        provider,
        "gpu_classes",
        lambda: [
            SimpleNamespace(name="A10", vram_gb=24),
            SimpleNamespace(name="H100", vram_gb=80),
        ],
    )
    monkeypatch.setattr(provider, "hourly_rate", lambda _gpu: 1.0)
    constraints = AllocationConstraints(
        gpu_type="A10",
        max_gpu_count=8,
        required_vram_gb=60,
    )

    candidates = provider.live_candidates(20, constraints)

    assert [(c.gpu, c.gpu_count) for c in candidates] == [("A10", 4)]


@pytest.mark.parametrize(
    ("gpu", "count", "expected"),
    [
        ("A10", 1, "A10G!"),
        ("A10", 4, "A10G!:4"),
        ("A100 SXM 40GB", 2, "A100-40GB!:2"),
        ("A100 SXM", 8, "A100-80GB!:8"),
        ("H100", 8, "H100!:8"),
        ("H200", 1, "H200!"),
        ("B200", 2, "B200!:2"),
    ],
)
def test_modal_gpu_request_is_strictly_pinned(gpu, count, expected) -> None:
    from flash.providers.modal.jobs import modal_gpu_request

    assert modal_gpu_request(gpu, count) == expected
    assert "!" in modal_gpu_request(gpu, count)


def test_modal_gpu_request_rejects_unsupported_shapes() -> None:
    from flash.providers.modal.jobs import modal_gpu_request

    with pytest.raises(UnsupportedGpuError):
        modal_gpu_request("A10", 8)
    with pytest.raises(UnsupportedGpuError):
        modal_gpu_request("A100 PCIe", 1)


def test_realized_gpu_attestation_requires_every_requested_board() -> None:
    from flash.providers.modal.jobs import _realized_gpu_class

    assert _realized_gpu_class(["NVIDIA A10G"] * 4, requested="A10", count=4) == "A10"
    assert (
        _realized_gpu_class(
            ["NVIDIA H100 80GB HBM3"] * 8,
            requested="H100",
            count=8,
        )
        == "H100"
    )
    with pytest.raises(RuntimeError, match="realized"):
        _realized_gpu_class(["NVIDIA H200"], requested="H100", count=1)
    with pytest.raises(RuntimeError, match="realized 1 GPU"):
        _realized_gpu_class(["NVIDIA H100 80GB HBM3"], requested="H100", count=2)


def test_bootstrap_environment_carries_verified_capsule_and_payload() -> None:
    from flash.providers.modal import jobs

    payload = {"job_spec_json": "{}", "env": {"HF_TOKEN": "hf"}, "flash_arm": "modal"}
    environment = jobs.bootstrap_environment(payload)

    decoded_payload = json.loads(base64.b64decode(environment[jobs._BOOTSTRAP_PAYLOAD_ENV]))
    capsule = base64.b64decode(environment[jobs._BOOTSTRAP_CAPSULE_ENV])
    assert decoded_payload == payload
    assert hashlib.sha256(capsule).hexdigest() == environment[jobs._BOOTSTRAP_CAPSULE_SHA_ENV]
    assert "capsule failed verification" in jobs._BOOTSTRAP_LAUNCHER


def _generic_handle() -> JobHandle:
    return JobHandle.from_dict(
        {
            "provider": "modal",
            "instance_id": "sb-1",
            "label": "flash-1-s0-a0",
            "gpu_request": "H100!",
            "gpu": "H100",
            "hourly_usd": 3.95,
            "attempt": 0,
            "started_ts": 1.0,
        }
    )


def test_cancel_validates_the_strict_handle_and_terminates_exact_id(monkeypatch) -> None:
    from flash.providers.modal import jobs

    payloads = []
    monkeypatch.setattr(jobs, "cancel", payloads.append)

    ModalProvider().cancel(_generic_handle())

    assert payloads == [_generic_handle().to_dict()]


def test_submit_persists_handle_polls_and_terminates_in_finally(monkeypatch) -> None:
    from flash.providers.modal import api, jobs

    handle = jobs.ModalJobHandle(
        instance_id="sb-1",
        label="flash-1-s0-a0",
        gpu_request="H100!",
        gpu="H100",
        hourly_usd=3.95,
        attempt=0,
        started_ts=1.0,
    )
    spec = SimpleNamespace(
        gpu=SimpleNamespace(type="H100"),
        train=SimpleNamespace(hf_repo="org/repo"),
        run_id="flash-1",
        phase="sft",
    )
    events = []
    monkeypatch.setattr(jobs, "require_deadline_at", lambda deadline: deadline)
    monkeypatch.setattr(jobs, "deploy_and_submit", lambda *args, **kwargs: handle)
    monkeypatch.setattr(jobs, "heartbeat_reader_for", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        jobs,
        "poll_modal_job",
        lambda *args, **kwargs: events.append("poll") or PollResult(True, metrics={}),
    )
    monkeypatch.setattr(
        api, "terminate_sandbox", lambda instance_id: events.append(("terminate", instance_id))
    )
    persisted = []

    result = jobs.submit_run_modal(spec, 0, on_handle=persisted.append, deadline_at=100.0)

    assert result.ok
    assert persisted == [handle.to_dict()]
    assert events == ["poll", ("terminate", "sb-1")]


def test_submit_still_terminates_when_polling_raises(monkeypatch) -> None:
    from flash.providers.modal import api, jobs

    handle = jobs.ModalJobHandle(
        instance_id="sb-1",
        label="flash-1-s0-a0",
        gpu_request="H100!",
        gpu="H100",
        hourly_usd=3.95,
        attempt=0,
        started_ts=1.0,
    )
    spec = SimpleNamespace(
        gpu=SimpleNamespace(type="H100"),
        train=SimpleNamespace(hf_repo="org/repo"),
        run_id="flash-1",
        phase="sft",
    )
    terminated = []
    monkeypatch.setattr(jobs, "require_deadline_at", lambda deadline: deadline)
    monkeypatch.setattr(jobs, "deploy_and_submit", lambda *args, **kwargs: handle)
    monkeypatch.setattr(jobs, "heartbeat_reader_for", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        jobs,
        "poll_modal_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("poll failed")),
    )
    monkeypatch.setattr(api, "terminate_sandbox", terminated.append)

    with pytest.raises(RuntimeError, match="poll failed"):
        jobs.submit_run_modal(spec, 0, deadline_at=100.0)

    assert terminated == ["sb-1"]


def test_run_gc_and_orphan_sweep_use_exact_modal_tags(monkeypatch) -> None:
    from flash.providers.modal import api, jobs

    listings = []
    terminated = []

    def list_sandboxes(*, tags):
        listings.append(tags)
        if api.RUN_TAG in tags:
            return [{"id": "sb-run", "tags": {api.RUN_TAG: "flash-run"}}]
        return [
            {"id": "sb-live", "tags": {api.RUN_TAG: "flash-live"}},
            {"id": "sb-orphan", "tags": {api.RUN_TAG: "flash-done"}},
            {"id": "sb-other", "tags": {api.RUN_TAG: "flash-other"}},
        ]

    monkeypatch.setattr(api, "list_sandboxes", list_sandboxes)
    monkeypatch.setattr(api, "terminate_sandbox", terminated.append)

    assert jobs.terminate_run_sandboxes("run") == ["sb-run"]
    assert jobs.sweep_orphans(active_labels={"live"}, known_labels={"live", "done"}) == [
        "sb-orphan"
    ]
    assert listings == [
        {api.PROVIDER_TAG: "modal", api.RUN_TAG: "flash-run"},
        {api.PROVIDER_TAG: "modal"},
    ]
    assert terminated == ["sb-run", "sb-orphan"]


def test_run_instances_remaining_delegates_exact_run_tag(monkeypatch) -> None:
    from flash.providers.modal import api, jobs

    seen = []
    monkeypatch.setattr(
        api,
        "list_sandboxes",
        lambda *, tags: seen.append(tags) or [{"id": "sb-7", "tags": tags}],
    )

    assert jobs.run_instances_remaining("flash-1") == ["sb-7"]
    assert seen == [{api.PROVIDER_TAG: "modal", api.RUN_TAG: "flash-1"}]
