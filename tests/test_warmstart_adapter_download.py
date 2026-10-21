"""Focused retry tests for pinned warm-start adapter downloads."""

from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import Timeout
from safetensors.numpy import save

import flash.engine.worker.model.adapter as adapter
import flash.engine.worker.runtime.state as worker_state
from flash.engine.worker.io.hf import RetriableInfraError

_ADAPTER_REF = "Freesolo-Co/flashrun-source:sft/source-run"
_ADAPTER_DIR = "/tmp/evdl/sft/source-run/adapter"
_REVISION = "a" * 40
_ADAPTER_CONFIG = '{"peft_type":"LORA"}'


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
        worker_state,
        "JOB_SPEC",
        SimpleNamespace(train=SimpleNamespace(init_from_adapter_revision=_REVISION)),
        raising=False,
    )
    monkeypatch.setattr(
        adapter,
        "_require_hf_deadline_allowance",
        lambda: deadline_checks.append(None),
    )
    return deadline_checks


def _write_adapter_files(adir, *, include_tensor_data: bool) -> None:
    adir.mkdir()
    (adir / "adapter_config.json").write_text(_ADAPTER_CONFIG, encoding="utf-8")
    tensor_bytes = save(
        {
            "base_model.model.layers.0.q_proj.lora_A.default.weight": np.ones(
                (1, 1), dtype=np.float32
            )
        }
    )
    if not include_tensor_data:
        tensor_bytes = tensor_bytes[:-4]
    (adir / "adapter_model.safetensors").write_bytes(tensor_bytes)


def test_warmstart_loadability_rejects_truncated_weights(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "adapter_config.json").write_text(_ADAPTER_CONFIG, encoding="utf-8")
    (empty / "adapter_model.safetensors").write_bytes(b"")
    assert adapter._warmstart_adapter_is_loadable(str(empty)) is False

    truncated = tmp_path / "truncated"
    _write_adapter_files(truncated, include_tensor_data=False)
    assert adapter._warmstart_adapter_is_loadable(str(truncated)) is False

    complete = tmp_path / "complete"
    _write_adapter_files(complete, include_tensor_data=True)
    assert adapter._warmstart_adapter_is_loadable(str(complete)) is True


def test_transient_throttling_retries_then_downloads(monkeypatch):
    import huggingface_hub

    deadline_checks = _prepare_download(monkeypatch)
    calls = []
    sleeps = []
    transient = _hf_error(429)
    monkeypatch.setattr(adapter, "_warmstart_adapter_is_loadable", lambda _path: len(calls) >= 3)

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


def test_transient_sidecar_error_accepts_completed_adapter(monkeypatch):
    import huggingface_hub

    deadline_checks = _prepare_download(monkeypatch)
    calls = []
    monkeypatch.setattr(adapter, "_warmstart_adapter_is_loadable", lambda _path: True)

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        raise Timeout("sidecar timed out after adapter weights completed")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        adapter,
        "_sleep_with_hf_deadline",
        lambda _delay: pytest.fail("a complete adapter must not back off"),
    )
    removed = []
    monkeypatch.setattr(
        adapter.shutil,
        "rmtree",
        lambda path, ignore_errors: removed.append((path, ignore_errors)),
    )

    assert adapter._download_adapter(_ADAPTER_REF) == _ADAPTER_DIR
    assert len(calls) == 1
    # only the startup clear runs; a completed adapter must not be discarded after the sidecar error
    assert removed == [(_ADAPTER_DIR, True)]
    assert len(deadline_checks) == 1


def test_persistent_transient_failure_raises_exact_retriable_error(monkeypatch):
    import huggingface_hub

    deadline_checks = _prepare_download(monkeypatch)
    calls = []
    sleeps = []
    transient = Timeout("timed out")
    monkeypatch.setattr(adapter, "_warmstart_adapter_is_loadable", lambda _path: False)

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
    monkeypatch.setattr(adapter, "_warmstart_adapter_is_loadable", lambda _path: False)

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
    monkeypatch.setattr(adapter, "_warmstart_adapter_is_loadable", lambda _path: False)

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
    removed = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        adapter.shutil,
        "rmtree",
        lambda path, ignore_errors: removed.append((path, ignore_errors)),
    )
    monkeypatch.setattr(
        adapter,
        "_sleep_with_hf_deadline",
        lambda delay: sleeps.append(delay) or True,
    )
    # first returned snapshot is incomplete (missing weights), the retry lands a loadable adapter.
    completeness = iter([False, True])
    monkeypatch.setattr(adapter, "_warmstart_adapter_is_loadable", lambda _path: next(completeness))

    assert adapter._download_adapter(_ADAPTER_REF) == _ADAPTER_DIR
    assert len(calls) == 2
    assert sleeps == [5.0]
    # startup clear plus the post-incomplete clear both run
    assert removed == [(_ADAPTER_DIR, True), (_ADAPTER_DIR, True)]
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
    monkeypatch.setattr(adapter, "_warmstart_adapter_is_loadable", lambda _path: False)
    # the directory itself is present on disk, so this test also fails if the completeness check
    # regresses to a directory-only `os.path.isdir` check.
    monkeypatch.setattr(adapter.os.path, "isdir", lambda path: path == _ADAPTER_DIR)

    with pytest.raises(RetriableInfraError) as exc_info:
        adapter._download_adapter(_ADAPTER_REF)

    assert type(exc_info.value) is RetriableInfraError
    assert len(calls) == adapter._ADAPTER_DOWNLOAD_RETRIES
    assert sleeps == [5.0, 10.0, 15.0]
    assert len(deadline_checks) == adapter._ADAPTER_DOWNLOAD_RETRIES


def _warmstart_guard_call(monkeypatch, tmp_path, *, rank, alpha, expected_rank, expected_alpha):
    """Drive the real sft/opd warm-start guard against an on-disk adapter config."""
    import flash.engine.worker.train.entry.sft_train as sft_train

    adapter_dir = tmp_path / "src-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        f'{{"peft_type":"LORA","r":{rank},"lora_alpha":{alpha}}}', encoding="utf-8"
    )
    monkeypatch.setattr(
        worker_state,
        "JOB_SPEC",
        SimpleNamespace(train=SimpleNamespace(init_from_adapter=_ADAPTER_REF)),
    )
    monkeypatch.setattr(adapter, "_download_adapter", lambda _ref: str(adapter_dir))
    monkeypatch.setattr(adapter, "validate_warmstart_adapter", lambda *_a, **_k: None)
    return sft_train._warmstart_adapter_path("m", "", expected_rank, expected_alpha)


def test_warmstart_rejects_an_alpha_change(monkeypatch, tmp_path):
    """A warm start may not silently rescale the adapter it continues.

    rank and alpha together set the LoRA scaling, so continuing at the source rank under a
    different alpha rescales every trained delta. grpo already rereads both from the source; sft
    and opd share this guard, which failed closed on rank and open on alpha.
    """
    with pytest.raises(
        ValueError, match=r"alpha 19 does not match the prepared train\.lora_alpha 77"
    ):
        _warmstart_guard_call(
            monkeypatch, tmp_path, rank=8, alpha=19, expected_rank=8, expected_alpha=77
        )


def test_warmstart_accepts_a_matching_rank_and_alpha(monkeypatch, tmp_path):
    """The guard stays open for the ordinary continue-this-adapter case."""
    resolved = _warmstart_guard_call(
        monkeypatch, tmp_path, rank=8, alpha=19, expected_rank=8, expected_alpha=19
    )
    assert resolved is not None
