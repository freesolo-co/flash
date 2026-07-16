from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest


def _prepared_spec(*, revision: str = "main"):
    from flash.spec import JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        model_revision=revision,
        algorithm="sft",
        train=TrainSpec(epochs=1, max_examples=1),
        run_id="revision-preflight",
    )


def _stub_prepare_dependencies(monkeypatch):
    import flash.catalog as catalog
    import flash.runner as runner

    monkeypatch.setattr(runner, "resolve_model", lambda *args, **kwargs: catalog.MODELS[args[0]])
    monkeypatch.setattr("flash.cost.spec.estimate_for_spec", lambda _spec: SimpleNamespace(total_usd=1.0))
    monkeypatch.setattr(
        "flash.lora_rank.preflight_train_context_within_serving", lambda _spec: None
    )


def _minimal_spec_dict() -> dict:
    return {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "environment": {"id": "freesolo/gsm8k"},
        "train": {"epochs": 1, "max_examples": 1},
    }


@pytest.mark.parametrize("control", ["dry_run", "background"])
def test_schema_rejects_execution_controls_nested_in_spec(control):
    from flash.schema import ConfigError, spec_from_dict

    raw = _minimal_spec_dict()
    raw[control] = True

    with pytest.raises(ConfigError, match=control):
        spec_from_dict(raw)


def test_jobspec_rejects_execution_control_nested_in_spec():
    from flash.spec import JobSpec

    raw = _minimal_spec_dict()
    raw["dry_run"] = True

    with pytest.raises(ValueError, match="dry_run"):
        JobSpec.from_dict(raw)


def test_spec_parsers_accept_valid_spec_without_execution_controls():
    from flash.schema import spec_from_dict
    from flash.spec import JobSpec

    raw = _minimal_spec_dict()

    assert spec_from_dict(raw).model == raw["model"]
    assert JobSpec.from_dict(raw).model == raw["model"]


def test_schema_defers_exact_vram_rejection_for_authored_revision(monkeypatch):
    import flash.providers.allocator as allocator
    from flash.schema import ConfigError, spec_from_dict

    raw = _minimal_spec_dict()
    raw.update(model="Qwen/Qwen3.5-9B", model_revision="refs/pr/123")
    raw["gpu"] = {"exact_type": "RTX 4090"}

    def fail_if_called(*args, **kwargs):
        raise AssertionError("revision-authored parse must defer exact sizing")

    monkeypatch.setattr(allocator, "required_vram_gb", fail_if_called)
    assert spec_from_dict(raw).gpu.exact_type == "RTX 4090"

    raw["model_revision"] = ""
    monkeypatch.setattr(allocator, "required_vram_gb", lambda *args, **kwargs: 80)
    with pytest.raises(ConfigError, match="requires at least"):
        spec_from_dict(raw)


def test_prepare_job_resolves_ref_to_sha_with_operator_token(monkeypatch):
    import huggingface_hub

    import flash.runner as runner

    _stub_prepare_dependencies(monkeypatch)
    seen = {}
    sha = "a" * 40

    class Api:
        def __init__(self, *, token):
            seen["token"] = token

        def model_info(self, model, *, revision):
            seen.update(model=model, revision=revision)
            return SimpleNamespace(sha=sha)

    monkeypatch.setenv("HF_TOKEN", "operator-token")
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)

    prepared = runner.prepare_job(_prepared_spec())

    assert seen == {
        "token": "operator-token",
        "model": "Qwen/Qwen3.5-0.8B",
        "revision": "main",
    }
    assert prepared.public_spec.model_revision == sha
    assert prepared.worker_spec.model_revision == sha


def test_prepare_job_revision_failure_precedes_persistence_and_provider_submit(monkeypatch):
    import huggingface_hub

    import flash.runner as runner

    class Api:
        def __init__(self, *, token):
            pass

        def model_info(self, model, *, revision):
            raise RuntimeError("private provider detail")

    writes = []
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(runner, "_save_status", lambda *args, **kwargs: writes.append(args))

    with pytest.raises(ValueError, match="could not resolve model_revision") as exc_info:
        runner.submit_job(_prepared_spec(), dry_run=True)

    assert "private provider detail" not in str(exc_info.value)
    assert writes == []


def test_prepare_job_moving_ref_persists_first_resolved_commit(monkeypatch):
    import huggingface_hub

    import flash.runner as runner

    _stub_prepare_dependencies(monkeypatch)
    shas = iter(("b" * 40, "c" * 40))

    class Api:
        def __init__(self, *, token):
            pass

        def model_info(self, model, *, revision):
            return SimpleNamespace(sha=next(shas))

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    prepared = runner.prepare_job(_prepared_spec(revision="moving-tag"))

    assert prepared.public_spec.model_revision == "b" * 40
    assert prepared.worker_spec.model_revision == "b" * 40


def test_revision_specific_sizing_uses_hf_geometry_and_rejects_catalog_drift(
    monkeypatch, tmp_path
):
    import flash.engine.vram as vram

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "vocab_size": 248320,
                "hidden_size": 1024,
                "num_hidden_layers": 28,
            }
        )
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda **kwargs: str(config)
    )

    class Api:
        def __init__(self, *, token):
            pass

        def model_info(self, model, **kwargs):
            return SimpleNamespace(safetensors=SimpleNamespace(total=int(0.9e9)))

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    need = vram.model_required_vram_gb(
        "Qwen/Qwen3.5-0.8B", "sft", model_revision="d" * 40
    )
    assert need > 0

    class DriftApi(Api):
        def model_info(self, model, **kwargs):
            return SimpleNamespace(safetensors=SimpleNamespace(total=int(4e9)))

    monkeypatch.setattr("huggingface_hub.HfApi", DriftApi)
    with pytest.raises(ValueError, match="geometry incompatible"):
        vram.model_required_vram_gb(
            "Qwen/Qwen3.5-0.8B", "sft", model_revision="e" * 40
        )


def test_revision_sizing_fails_closed_when_pinned_commit_lacks_param_metadata(
    monkeypatch, tmp_path
):
    # A pinned revision whose Hub metadata exposes no safetensors.total cannot be sized. Revision-aware
    # sizing is authoritative and must fail closed rather than silently reuse the catalog default-revision
    # param count (which would size the exact-GPU preflight on weights the worker never loads).
    import flash.engine.vram as vram

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"vocab_size": 248320, "hidden_size": 1024, "num_hidden_layers": 28})
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(config))

    class NoParamApi:
        def __init__(self, *, token):
            pass

        def model_info(self, model, **kwargs):
            return SimpleNamespace(safetensors=SimpleNamespace(total=None))

    monkeypatch.setattr("huggingface_hub.HfApi", NoParamApi)
    with pytest.raises(ValueError, match="no parameter-count metadata"):
        vram.model_required_vram_gb(
            "Qwen/Qwen3.5-0.8B", "sft", model_revision="f" * 40
        )


def test_check_fit_pinned_metadata_failure_is_unknown_but_sizing_stays_strict(monkeypatch):
    import flash.engine.vram as vram

    failure = RuntimeError("metadata unavailable")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(vram, "fetch_hf_params_b", fail)

    estimate = vram.check_fit(
        "owner/model",
        "sft",
        "RTX 4090",
        model_revision="a" * 40,
    )
    assert estimate.verdict == "unknown"
    assert estimate.params_b is None

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        vram.model_required_vram_gb(
            "owner/model",
            "sft",
            model_revision="a" * 40,
        )


def test_prefetch_error_classification():
    import httpx
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    from requests.exceptions import ConnectionError, Timeout

    from flash.engine.worker.hf import _prefetch_error_is_retriable

    def hf_error(error_type, status):
        response = SimpleNamespace(
            status_code=status,
            headers={},
            request=SimpleNamespace(),
        )
        return error_type("x", response=response)

    assert not _prefetch_error_is_retriable(hf_error(RevisionNotFoundError, 404))
    assert not _prefetch_error_is_retriable(hf_error(RepositoryNotFoundError, 404))
    assert not _prefetch_error_is_retriable(hf_error(GatedRepoError, 403))
    assert not _prefetch_error_is_retriable(EntryNotFoundError("x"))
    assert not _prefetch_error_is_retriable(hf_error(HfHubHTTPError, 401))
    assert not _prefetch_error_is_retriable(hf_error(HfHubHTTPError, 404))
    assert _prefetch_error_is_retriable(ConnectionError("disconnected"))
    assert _prefetch_error_is_retriable(Timeout("timed out"))
    assert _prefetch_error_is_retriable(httpx.ConnectError("disconnected"))
    assert _prefetch_error_is_retriable(httpx.ReadTimeout("timed out"))
    assert _prefetch_error_is_retriable(httpx.RemoteProtocolError("connection reset"))
    assert _prefetch_error_is_retriable(LocalEntryNotFoundError("missing locally"))
    assert _prefetch_error_is_retriable(hf_error(HfHubHTTPError, 503))
    assert _prefetch_error_is_retriable(hf_error(HfHubHTTPError, 429))
    assert _prefetch_error_is_retriable(hf_error(HfHubHTTPError, None))
    assert not _prefetch_error_is_retriable(ValueError("bug"))


def test_prefetch_pinned_revision_does_not_swallow_download_failure(monkeypatch):
    import flash.engine.worker.hf as hf

    monkeypatch.setattr(hf, "_shared_weight_cache_dir", lambda: None)
    monkeypatch.setattr(hf, "_require_hf_deadline_allowance", lambda: None)
    monkeypatch.setattr(hf, "gpu_diagnostics", lambda: {})
    monkeypatch.setattr(hf._w, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("revision not found")),
    )

    with pytest.raises(ValueError, match="revision not found"):
        hf.prefetch_model("owner/model", revision="f" * 40)


def test_prefetch_pinned_revision_wraps_transient_download_failure(monkeypatch):
    from requests.exceptions import Timeout

    import flash.engine.worker.hf as hf
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    transient = Timeout("timed out")
    monkeypatch.setattr(hf, "_shared_weight_cache_dir", lambda: None)
    monkeypatch.setattr(hf, "_require_hf_deadline_allowance", lambda: None)
    monkeypatch.setattr(hf, "gpu_diagnostics", lambda: {})
    monkeypatch.setattr(hf._w, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(transient),
    )

    with pytest.raises(RetriableInfraError, match="pinned model prefetch failed") as exc_info:
        hf.prefetch_model("owner/model", revision="f" * 40)

    assert exc_info.value.__cause__ is transient


def test_adapter_provenance_is_stamped_and_conflicts_fail(monkeypatch):
    import flash.engine.worker.adapter as adapter

    revision = "1" * 40
    config = SimpleNamespace(base_model_name_or_path="owner/model", revision=None)
    model = SimpleNamespace(peft_config={"default": config})

    adapter.stamp_adapter_provenance(model, "owner/model", revision)
    assert config.base_model_name_or_path == "owner/model"
    assert config.revision == revision

    config.revision = "2" * 40
    with pytest.raises(RuntimeError, match="does not match"):
        adapter.stamp_adapter_provenance(model, "owner/model", revision)


def test_opd_model_revision_is_keyword_only():
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    parameter = inspect.signature(OpdVllmRolloutEngine).parameters["model_revision"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_resolve_vocab_size_is_revision_aware_for_open_policy_model(monkeypatch, tmp_path):
    # regression (PR #538 finding 4): SFT batch sizing resolves vocab through resolve_vocab_size. for a
    # pinned revision on an uncataloged open-policy model it must return the commit's real vocab (mirroring
    # resolve_params_b), not the default, so the fused-CE micro-batch cap tracks the served tokenizer.
    from flash.catalog import _DEFAULT_VOCAB_SIZE, resolve_vocab_size

    # cataloged model, no revision -> catalog vocab (unchanged default path).
    assert resolve_vocab_size("Qwen/Qwen3.5-0.8B") == 248320

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"vocab_size": 151936, "hidden_size": 2048, "num_hidden_layers": 24})
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(config))

    class Api:
        def __init__(self, *, token):
            pass

        def model_info(self, model, **kwargs):
            return SimpleNamespace(safetensors=SimpleNamespace(total=int(3e9)))

    monkeypatch.setattr("huggingface_hub.HfApi", Api)

    # uncataloged open-policy model with a pinned revision -> the fetched commit vocab, not the default.
    got = resolve_vocab_size("open-org/Open-Model-3B", revision="d" * 40)
    assert got == 151936
    assert got != _DEFAULT_VOCAB_SIZE
