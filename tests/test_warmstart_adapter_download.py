"""Focused retry tests for pinned warm-start adapter downloads."""

from types import SimpleNamespace

import pytest
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import Timeout

import flash.engine.worker as worker
import flash.engine.worker.adapter as adapter
from flash.engine.worker.hf import RetriableInfraError

_ADAPTER_REF = "Freesolo-Co/flashrun-source:sft/source-run"
_ADAPTER_DIR = "/tmp/evdl/sft/source-run/adapter"
_REVISION = "a" * 40


def _hf_error(status_code: int) -> HfHubHTTPError:
    response = SimpleNamespace(
        status_code=status_code,
        headers={},
        request=SimpleNamespace(),
    )
    return HfHubHTTPError("request failed", response=response)


def _prepare_download(monkeypatch):
    deadline_checks = []
    monkeypatch.setattr(
        worker,
        "JOB_SPEC",
        SimpleNamespace(train=SimpleNamespace(init_from_adapter_revision=_REVISION)),
        raising=False,
    )
    monkeypatch.setattr(
        adapter,
        "_require_hf_deadline_allowance",
        lambda: deadline_checks.append(None),
    )
    monkeypatch.setattr(adapter, "_has_deployable_adapter", lambda path: path == _ADAPTER_DIR)
    return deadline_checks


def test_transient_throttling_retries_then_downloads(monkeypatch):
    import huggingface_hub

    deadline_checks = _prepare_download(monkeypatch)
    calls = []
    sleeps = []
    transient = _hf_error(429)

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise transient

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        adapter,
        "_sleep_with_hf_deadline",
        lambda delay: sleeps.append(delay) or True,
    )

    assert adapter._download_adapter(_ADAPTER_REF) == _ADAPTER_DIR
    assert len(calls) == 3
    assert sleeps == [5.0, 10.0]
    assert len(deadline_checks) == 3


def test_persistent_transient_failure_raises_exact_retriable_error(monkeypatch):
    import huggingface_hub

    deadline_checks = _prepare_download(monkeypatch)
    calls = []
    sleeps = []
    transient = Timeout("timed out")

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        raise transient

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        adapter,
        "_sleep_with_hf_deadline",
        lambda delay: sleeps.append(delay) or True,
    )

    with pytest.raises(RetriableInfraError) as exc_info:
        adapter._download_adapter(_ADAPTER_REF)

    assert type(exc_info.value) is RetriableInfraError
    assert len(calls) == adapter._ADAPTER_DOWNLOAD_RETRIES
    assert sleeps == [5.0, 10.0, 15.0]
    assert len(deadline_checks) == adapter._ADAPTER_DOWNLOAD_RETRIES


def test_terminal_hf_error_fails_without_retry(monkeypatch):
    import huggingface_hub

    deadline_checks = _prepare_download(monkeypatch)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        raise _hf_error(403)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        adapter,
        "_sleep_with_hf_deadline",
        lambda _delay: pytest.fail("terminal errors must not back off"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter._download_adapter(_ADAPTER_REF)

    assert type(exc_info.value) is RuntimeError
    assert str(exc_info.value) == "the prepared warm-start source adapter could not be downloaded"
    assert len(calls) == 1
    assert len(deadline_checks) == 1


def test_deadline_exhausted_during_backoff_raises_retriable_error(monkeypatch):
    import huggingface_hub

    deadline_checks = _prepare_download(monkeypatch)
    calls = []
    sleeps = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        raise Timeout("timed out")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        adapter,
        "_sleep_with_hf_deadline",
        lambda delay: sleeps.append(delay) or False,
    )

    with pytest.raises(RetriableInfraError) as exc_info:
        adapter._download_adapter(_ADAPTER_REF)

    assert type(exc_info.value) is RetriableInfraError
    assert len(calls) == 1
    assert sleeps == [5.0]
    assert len(deadline_checks) == 1


def test_incomplete_snapshot_retries_until_adapter_is_complete(monkeypatch):
    import huggingface_hub

    deadline_checks = _prepare_download(monkeypatch)
    calls = []
    sleeps = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        adapter,
        "_sleep_with_hf_deadline",
        lambda delay: sleeps.append(delay) or True,
    )
    # first returned snapshot is incomplete (missing weights), the retry lands a loadable adapter.
    completeness = iter([False, True])
    monkeypatch.setattr(adapter, "_has_deployable_adapter", lambda _path: next(completeness))

    assert adapter._download_adapter(_ADAPTER_REF) == _ADAPTER_DIR
    assert len(calls) == 2
    assert sleeps == [5.0]
    assert len(deadline_checks) == 2


def test_snapshot_success_but_adapter_never_complete_raises_retriable(monkeypatch):
    import huggingface_hub

    deadline_checks = _prepare_download(monkeypatch)
    calls = []
    sleeps = []

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        adapter,
        "_sleep_with_hf_deadline",
        lambda delay: sleeps.append(delay) or True,
    )
    # snapshot_download keeps returning, but a loadable adapter never materializes -> infra retry.
    monkeypatch.setattr(adapter, "_has_deployable_adapter", lambda _path: False)

    with pytest.raises(RetriableInfraError) as exc_info:
        adapter._download_adapter(_ADAPTER_REF)

    assert type(exc_info.value) is RetriableInfraError
    assert len(calls) == adapter._ADAPTER_DOWNLOAD_RETRIES
    assert sleeps == [5.0, 10.0, 15.0]
    assert len(deadline_checks) == adapter._ADAPTER_DOWNLOAD_RETRIES
