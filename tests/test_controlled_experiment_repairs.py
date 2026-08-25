from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

import flash.runner.lifecycle.submit as runner_submit


def _prepared_spec(*, revision: str = "a" * 40):
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
    from tests._helpers.profile import attach_sft_profile

    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        model_revision=revision,
        model_revision_auto=True,
        algorithm="sft",
        environment=EnvironmentSpec(id="freesolo/example-project/gsm8k", resolved_sha="e" * 40),
        train=TrainSpec(epochs=1, max_examples=1),
        run_id="revision-preflight",
    )
    return attach_sft_profile(spec)


def _stub_prepare_dependencies(monkeypatch, spec=None):
    import flash.core.catalog as catalog

    monkeypatch.setattr(catalog, "resolve_model", lambda *args, **kwargs: catalog.MODELS[args[0]])
    if spec is not None:
        # sft preparation profiles the packaged dataset itself, which resolves the environment
        # package over the network. these tests are about revision resolution, so the profile is
        # recorded up front and the download never happens. it is keyed on the RESOLVED revision,
        # which is what preparation re-derives before it profiles.
        from tests._helpers.profile import record_sft_profile

        record_sft_profile(spec, monkeypatch)
    monkeypatch.setattr(
        "flash.cost.spec.estimate_for_spec", lambda _spec: SimpleNamespace(total_usd=1.0)
    )
    monkeypatch.setattr(
        "flash.adapters.lora_rank.preflight_train_context_within_serving", lambda _spec: None
    )
    # these tests isolate model revision and structured-output validation. environment image
    # discovery and managed-teacher configuration have their own boundary suites.
    monkeypatch.setattr(
        "flash.content.multimodal.preflight_validate_image_opd",
        lambda _spec, **_kwargs: None,
    )
    monkeypatch.setattr(
        "flash.server.domain.teacher.broker.preflight_validate_managed_teacher",
        lambda _spec: None,
    )


def _minimal_spec_dict() -> dict:
    return {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "sft",
        "environment": {"id": "freesolo/example-project/gsm8k"},
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
    from flash.core.spec import JobSpec

    raw = _minimal_spec_dict()
    raw["dry_run"] = True

    with pytest.raises(ValueError, match="dry_run"):
        JobSpec.from_dict(raw)


def test_spec_parsers_accept_valid_spec_without_execution_controls():
    from flash.core.spec import JobSpec
    from flash.schema import spec_from_dict

    raw = _minimal_spec_dict()

    assert spec_from_dict(raw).model == raw["model"]
    assert JobSpec.from_dict(raw).model == raw["model"]


def test_prepare_job_retains_runner_forced_sft_revision_and_clears_request(monkeypatch):
    import huggingface_hub

    from flash.core.spec import JobSpec

    exact = "d" * 40
    moving_head = "e" * 40
    _stub_prepare_dependencies(monkeypatch, _prepared_spec(revision=exact))
    seen = []

    class Api:
        def __init__(self, *, token):
            pass

        def model_info(self, model, *, revision):
            seen.append(revision)
            return SimpleNamespace(sha=moving_head if revision is None else revision)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    internal = _prepared_spec(revision=exact).to_internal_dict()
    internal.update(model_revision_auto=True, model_revision_force_pin=True)

    prepared = runner_submit.prepare_job(JobSpec.from_dict(internal))

    assert seen == [exact]
    for spec in (prepared.public_spec, prepared.worker_spec):
        assert spec.model_revision == exact
        assert spec.model_revision_auto is True
        assert spec.model_revision_force_pin is False
    assert "model_revision_force_pin" not in prepared.public_spec.to_dict()
    assert prepared.worker_spec.to_internal_dict()["model_revision_force_pin"] is False


def test_revision_specific_sizing_uses_hf_geometry_and_rejects_catalog_drift(monkeypatch, tmp_path):
    import flash.engine.plan.vram as vram
    from flash.core.catalog import MODELS, ModelInfo

    model_id = "test/revision-geometry"
    monkeypatch.setitem(
        MODELS,
        model_id,
        ModelInfo(
            id=model_id,
            display_name="synthetic revision geometry",
            params="0.9B",
            params_b=0.9,
            algos=("sft",),
            min_vram_gb=12,
            vocab_size=248_320,
            hidden_size=1024,
            num_layers=24,
        ),
    )

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "vocab_size": 248320,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
            }
        )
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(config))

    class Api:
        def __init__(self, *, token):
            pass

        def model_info(self, model, **kwargs):
            return SimpleNamespace(safetensors=SimpleNamespace(total=int(0.9e9)))

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    need = vram.model_required_vram_gb(model_id, "sft", model_revision="d" * 40)
    assert need > 0

    captured = {}

    def capture_estimate(*args, **kwargs):
        captured.update(kwargs)
        return 10.0

    with monkeypatch.context() as scoped:
        scoped.setattr(vram, "estimate_vram_gb", capture_estimate)
        vram.model_required_vram_gb(model_id, "sft", model_revision="d" * 40)
    assert captured["model_info"] is None
    assert captured["active_params_b"] == 0.0

    class DriftApi(Api):
        def model_info(self, model, **kwargs):
            return SimpleNamespace(safetensors=SimpleNamespace(total=int(4e9)))

    monkeypatch.setattr("huggingface_hub.HfApi", DriftApi)
    with pytest.raises(ValueError, match="geometry incompatible"):
        vram.model_required_vram_gb(model_id, "sft", model_revision="e" * 40)


def test_revision_sizing_fails_closed_when_pinned_commit_lacks_param_metadata(
    monkeypatch, tmp_path
):
    # A pinned revision whose Hub metadata exposes no safetensors.total cannot be sized. Revision-aware
    # sizing is authoritative and must fail closed rather than silently reuse the catalog default-revision
    # param count (which would size the exact-GPU preflight on weights the worker never loads).
    import flash.engine.plan.vram as vram

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"vocab_size": 248320, "hidden_size": 1024, "num_hidden_layers": 24})
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(config))

    class NoParamApi:
        def __init__(self, *, token):
            pass

        def model_info(self, model, **kwargs):
            return SimpleNamespace(safetensors=SimpleNamespace(total=None))

    monkeypatch.setattr("huggingface_hub.HfApi", NoParamApi)
    with pytest.raises(ValueError, match="no parameter-count metadata"):
        vram.model_required_vram_gb("Qwen/Qwen3.5-9B", "sft", model_revision="f" * 40)


def test_pinned_metadata_failure_keeps_sizing_strict(monkeypatch):
    """A pin whose metadata cannot be read must raise, never size the run on a guess.

    (This also covered `check_fit`, which returned an "unknown" verdict for the same failure. That
    advisory estimator was reachable only from the deleted open-model path and is gone; the strict
    sizing half below is what actually gates a paid allocation.)
    """
    import flash.engine.plan.vram as vram

    # the pinned path reads full geometry, not just the parameter count -- patch what it actually
    # calls, or the "failure" is really an offline network call and the test proves nothing.
    def fail(*args, **kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(vram, "fetch_hf_model_geometry", fail)

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        vram.model_required_vram_gb(
            "Qwen/Qwen3.5-9B",
            "sft",
            model_revision="a" * 40,
        )
    # the same model WITHOUT a pin sizes fine from the catalog, so the raise above is the pinned
    # path failing closed rather than sizing being broken for this model.
    assert vram.model_required_vram_gb("Qwen/Qwen3.5-9B", "sft") > 0


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

    from flash.engine.worker.io.hf import _prefetch_error_is_retriable

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
    assert _prefetch_error_is_retriable(hf_error(HfHubHTTPError, 408))
    assert _prefetch_error_is_retriable(hf_error(HfHubHTTPError, 425))
    assert _prefetch_error_is_retriable(hf_error(HfHubHTTPError, 429))
    assert _prefetch_error_is_retriable(hf_error(HfHubHTTPError, 503))
    assert _prefetch_error_is_retriable(hf_error(HfHubHTTPError, None))
    assert not _prefetch_error_is_retriable(OSError("local file failure"))
    assert not _prefetch_error_is_retriable(ValueError("bug"))


def test_prefetch_pinned_revision_does_not_swallow_download_failure(monkeypatch):
    import flash.engine.worker.io.heartbeat as worker_heartbeat
    import flash.engine.worker.io.hf as worker_hf
    import flash.engine.worker.io.prefetch as worker_prefetch
    import flash.engine.worker.perf as worker_perf

    monkeypatch.setattr(worker_prefetch, "_shared_weight_cache_dir", lambda: None)
    monkeypatch.setattr(worker_hf, "_require_hf_deadline_allowance", lambda: None)
    monkeypatch.setattr(worker_perf, "gpu_diagnostics", dict)
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("revision not found")),
    )

    with pytest.raises(ValueError, match="revision not found"):
        worker_prefetch.prefetch_model("owner/model", revision="f" * 40)


def test_prefetch_pinned_revision_wraps_transient_download_failure(monkeypatch):
    from requests.exceptions import Timeout

    import flash.engine.worker.io.heartbeat as worker_heartbeat
    import flash.engine.worker.io.hf as worker_hf
    import flash.engine.worker.io.prefetch as worker_prefetch
    import flash.engine.worker.perf as worker_perf
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    transient = Timeout("timed out")
    monkeypatch.setattr(worker_prefetch, "_shared_weight_cache_dir", lambda: None)
    monkeypatch.setattr(worker_hf, "_require_hf_deadline_allowance", lambda: None)
    monkeypatch.setattr(worker_perf, "gpu_diagnostics", dict)
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(transient),
    )

    with pytest.raises(RetriableInfraError, match="pinned model prefetch failed") as exc_info:
        worker_prefetch.prefetch_model("owner/model", revision="f" * 40)

    assert exc_info.value.__cause__ is transient


def test_opd_model_revision_is_keyword_only():
    # a pinned revision must never be positional: model_id and model_revision are adjacent strings,
    # so a positional call that transposes them silently loads the wrong commit and breaks the
    # controlled comparison this file exists to protect. the guard moved from trl's rollout engine
    # (OpdVllmRolloutEngine, deleted) to the verl checkpoint watcher, which is what carries the
    # revision through opd now -- it re-exports each checkpoint adapter under that pin.
    from flash.engine.worker.train.entry.sft_train import _VerlCheckpointWatcher
    from flash.engine.worker.train.opd.orchestration.failures import _OpdVerlCheckpointWatcher

    # the opd subclass forwards **kwargs, so the binding it inherits is what has to be keyword-only.
    assert issubclass(_OpdVerlCheckpointWatcher, _VerlCheckpointWatcher)
    parameters = inspect.signature(_VerlCheckpointWatcher.__init__).parameters
    assert parameters["model_revision"].kind is inspect.Parameter.KEYWORD_ONLY
    # model_id sits next to it and is equally transposable.
    assert parameters["model_id"].kind is inspect.Parameter.KEYWORD_ONLY
    # and the subclass must not reintroduce a positional path around the base.
    opd_parameters = inspect.signature(_OpdVerlCheckpointWatcher.__init__).parameters
    assert all(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name, parameter in opd_parameters.items()
        if name != "self"
    )


def test_resolve_vocab_size_is_revision_aware_for_open_policy_model(monkeypatch, tmp_path):
    # regression (PR #538 finding 4): SFT batch sizing resolves vocab through resolve_vocab_size. for a
    # pinned revision on an uncataloged open-policy model it must return the commit's real vocab (mirroring
    # resolve_params_b), not the default, so the fused-CE micro-batch cap tracks the served tokenizer.
    from flash.core.catalog import _DEFAULT_VOCAB_SIZE, resolve_vocab_size

    # cataloged model, no revision -> catalog vocab (unchanged default path).
    assert resolve_vocab_size("Qwen/Qwen3.5-9B") == 248320

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


def _stub_structured_opd_hub(monkeypatch):
    """Pin revision resolution, which runs before the preflight and would otherwise hit the hub."""
    import huggingface_hub

    class Api:
        def __init__(self, *, token=None):
            pass

        def model_info(self, model, **kwargs):
            return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    # the structured path calls this with kwargs the shared stub's lambda does not accept.
    monkeypatch.setattr(
        "flash.adapters.lora_rank.preflight_train_context_within_serving",
        lambda _spec, **_kwargs: None,
    )


def _structured_opd_spec(structured_outputs: str):
    """An opd spec carrying a structured-output constraint, ready for prepare_job."""
    from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3.5-9B",
        model_revision="a" * 40,
        model_revision_auto=True,
        algorithm="opd",
        environment=EnvironmentSpec(id="freesolo/example-project/gsm8k", resolved_sha="e" * 40),
        train=TrainSpec(
            epochs=1,
            max_examples=1,
            teacher_model="parasail-kimi-k3-fast",
            structured_outputs=structured_outputs,
        ),
        run_id="structured-preflight",
    )


def test_structured_opd_guidance_only_feature_is_rejected_before_allocation(monkeypatch):
    """A constraint the worker refuses deterministically must not reach a rented gpu.

    `format`, `multipleOf` and `uniqueItems` need vLLM's guidance backend, and verl OPD pins
    xgrammar for exact forced-token replay -- so the job can NEVER run. That check lived only in the
    worker, past allocation, so the user paid for a card to receive a permanent validation failure
    The generic serving preflight does not catch it: it validates the schema's shape,
    and this schema is perfectly valid.
    """
    _stub_prepare_dependencies(monkeypatch)
    _stub_structured_opd_hub(monkeypatch)
    # allocation-side work must not be reached. estimate_for_spec is the last step of preparation
    # and the gate the run is priced through, so a rejection that happens before it happens before
    # anything is persisted or provisioned.
    monkeypatch.setattr(
        "flash.cost.spec.estimate_for_spec",
        lambda _spec: (_ for _ in ()).throw(AssertionError("preparation must reject first")),
    )
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "format": "email"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    spec = _structured_opd_spec(json.dumps({"json": schema}))

    with pytest.raises(ValueError, match="guidance fallback"):
        runner_submit.prepare_job(spec)


def test_structured_opd_mistral_tokenizer_model_is_rejected_before_allocation(monkeypatch):
    """The other permanently-failing input: a vLLM MistralTokenizer model.

    Decided by the model id alone, so it needs no allocation to judge either.
    """
    _stub_prepare_dependencies(monkeypatch)
    _stub_structured_opd_hub(monkeypatch)
    monkeypatch.setattr(
        "flash.cost.spec.estimate_for_spec",
        lambda _spec: (_ for _ in ()).throw(AssertionError("preparation must reject first")),
    )
    monkeypatch.setattr(
        "flash.engine.worker.train.opd.orchestration.validation._resolve_structured_model_metadata",
        lambda _model, _rev: (151936, ("tekken.json",)),
    )
    spec = _structured_opd_spec(json.dumps({"json": {"type": "object"}}))

    with pytest.raises(ValueError, match="MistralTokenizer"):
        runner_submit.prepare_job(spec)


def test_a_valid_structured_opd_constraint_still_prepares(monkeypatch):
    """The preflight must reject only what the worker rejects.

    A constraint xgrammar can compile has to pass preparation untouched -- otherwise the fix trades
    a paid failure for a submission that refuses valid work, which is worse.
    """
    _stub_prepare_dependencies(monkeypatch)
    _stub_structured_opd_hub(monkeypatch)
    monkeypatch.setattr(
        "flash.engine.worker.train.opd.orchestration.validation._resolve_structured_model_metadata",
        lambda _model, _rev: (151936, ("tokenizer.json",)),
    )
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    spec = _structured_opd_spec(json.dumps({"json": schema}))

    prepared = runner_submit.prepare_job(spec)

    assert prepared.public_spec.algorithm == "opd"
    assert prepared.public_spec.train.structured_outputs
