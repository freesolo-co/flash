"""Tests for the Flash serving wiring (no GPU/network).

Serving is delegated to the freesolo platform's multi-LoRA serving app; flash is a thin
client. These assert the deploy/undeploy/chat HTTP calls (httpx is monkeypatched) and the
dry-run Deployment shaping — there is no flash-owned vLLM endpoint to provision anymore.
"""

from __future__ import annotations

import json
import sys
import types

import pytest


def _stub_adapter_config(
    monkeypatch,
    tmp_path,
    *,
    rank: int = 32,
    config: dict | None = None,
    tensor_files: dict[str, int | None] | None = None,
):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"r": rank} if config is None else config), encoding="utf-8")
    seen: dict = {}

    def fake_hf_hub_download(**kwargs):
        seen.update(kwargs)
        return str(cfg)

    if tensor_files is None:
        tensor_files = {"adapter_model.safetensors": 123}

    class _HfApi:
        def repo_info(self, **kwargs):
            seen["repo_info"] = kwargs
            return types.SimpleNamespace(sha="a" * 40)

        def list_repo_tree(self, **kwargs):
            seen["list_repo_tree"] = kwargs
            prefix = str(kwargs.get("path_in_repo") or "").rstrip("/")
            return [
                types.SimpleNamespace(path=f"{prefix}/{name}", size=size)
                for name, size in tensor_files.items()
            ]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download, HfApi=_HfApi),
    )

    import flash.serve.deploy as deploy

    monkeypatch.setattr(deploy, "_require_serving_capabilities", lambda: None)
    monkeypatch.setattr(deploy, "_wait_revision_ready", lambda *a, **k: {})
    monkeypatch.setattr(
        deploy,
        "_activate_revision",
        lambda run_id, revision, checkpoint, **kwargs: {
            "adapter_id": run_id,
            "target_adapter_revision": revision,
            "checkpoint": checkpoint,
        },
    )
    return seen


def _capture_registration_body(monkeypatch, tmp_path, stub_serving_registry, **deploy_kwargs):
    """Run a non-dry-run deploy_adapter with httpx.post captured; return the POSTed /adapters body."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    _stub_adapter_config(monkeypatch, tmp_path, rank=32)

    seen: dict = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
        seen["json"] = json
        return _Resp()

    monkeypatch.setattr(d.httpx, "post", fake_post)
    run_id = deploy_kwargs.get("run_id", "flash-7-abcd")
    stub_serving_registry(
        {"adapter_id": run_id, "subfolder": f"{deploy_kwargs['adapter_prefix']}/adapter"}
    )
    d.deploy_adapter(**deploy_kwargs)
    return seen["json"]


def test_deploy_dry_run():
    from flash.serve.deploy import deploy_adapter

    dep = deploy_adapter(
        run_id="r1",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/r1/seed0",
        dry_run=True,
    )
    d = dep.to_dict()
    assert d["state"] == "dry_run"
    assert "gpu" not in d
    # The adapter is addressed by its run_id on the freesolo serving app.
    assert d["openai_model"] == "r1"
    assert d["adapter_hf_prefix"] == "sft/r1/seed0/adapter"
    assert "mode" not in d
    assert "est_idle_cost_usd_per_day" not in d


def test_deploy_9b_dry_run_is_not_rejected():
    """The 9B (bf16 LoRA) tier is deployable: freesolo serving folds the bf16 LoRA delta
    into the bf16 base, instead of being rejected up front."""
    from flash.serve.deploy import deploy_adapter

    dep = deploy_adapter(
        run_id="q1",
        model="Qwen/Qwen3.5-9B",
        hf_repo="org/repo",
        adapter_prefix="sft/q1/seed0",
        dry_run=True,
    )
    assert dep.to_dict()["state"] == "dry_run"


def test_deploy_rejects_lora_rank_above_serving_cap():
    from flash.serve.deploy import deploy_adapter

    # Qwen3.5-4B serving cap is now max_lora_rank=64 (doubled from 32); a rank-65 adapter exceeds it.
    with pytest.raises(ValueError, match="max_lora_rank=64"):
        deploy_adapter(
            run_id="r65",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/r65/seed0",
            dry_run=True,
            lora_rank=65,
        )


def test_deploy_rejects_recombined_artifact_rank_above_serving_cap(monkeypatch, tmp_path):
    """Deploy validates the effective artifact rank, not only spec.train.lora_rank."""
    from flash.serve.deploy import deploy_adapter

    # 4B serving cap is now 64; the artifact's effective rank 65 exceeds it even though spec
    # lora_rank (32) fits — deploy must catch the artifact rank, not just the spec rank.
    seen = _stub_adapter_config(monkeypatch, tmp_path, rank=65)

    with pytest.raises(ValueError, match="adapter artifact has rank 65"):
        deploy_adapter(
            run_id="r-recombined",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="grpo/r-recombined/seed0",
            dry_run=False,
            lora_rank=32,
        )
    assert seen["repo_id"] == "org/repo"
    assert seen["filename"] == "grpo/r-recombined/seed0/adapter/adapter_config.json"
    assert seen["repo_type"] == "dataset"


def test_deploy_rejects_adapter_config_without_rank_metadata(monkeypatch, tmp_path):
    from flash.serve.deploy import deploy_adapter

    _stub_adapter_config(monkeypatch, tmp_path, config={})

    with pytest.raises(ValueError, match="no LoRA rank metadata"):
        deploy_adapter(
            run_id="r-missing-rank",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/r-missing-rank/seed0",
            dry_run=False,
            lora_rank=32,
        )


def test_deploy_rejects_falsey_invalid_rank_pattern(monkeypatch, tmp_path):
    from flash.serve.deploy import deploy_adapter

    _stub_adapter_config(monkeypatch, tmp_path, config={"r": 32, "rank_pattern": []})

    with pytest.raises(ValueError, match="invalid rank metadata"):
        deploy_adapter(
            run_id="r-bad-pattern",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/r-bad-pattern/seed0",
            dry_run=False,
            lora_rank=32,
        )


def test_deploy_adapter_rank_download_failure_is_serving_error(monkeypatch):
    import flash.serve.deploy as d

    def fake_hf_hub_download(**_kwargs):
        raise RuntimeError("hub timeout")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    )
    monkeypatch.setattr(d, "resolve_hf_revision", lambda repo: "a" * 40)

    with pytest.raises(
        d.ServingError, match="failed to read org/repo:sft/r-hf-down/seed0/adapter"
    ) as excinfo:
        d.deploy_adapter(
            run_id="r-hf-down",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/r-hf-down/seed0",
            dry_run=False,
            lora_rank=32,
        )
    assert not isinstance(excinfo.value, d.AdapterConfigMissing)


def test_deploy_adapter_missing_config_is_adapter_config_missing(monkeypatch):
    import flash.serve.deploy as d

    class _Response:
        status_code = 404

    class _NotFound(RuntimeError):
        response = _Response()

    def fake_hf_hub_download(**_kwargs):
        raise _NotFound("adapter_config.json not found")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    )
    monkeypatch.setattr(d, "resolve_hf_revision", lambda repo: "a" * 40)

    with pytest.raises(
        d.AdapterConfigMissing, match="failed to read org/repo:sft/r-missing/seed0/adapter"
    ):
        d.deploy_adapter(
            run_id="r-missing",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/r-missing/seed0",
            dry_run=False,
            lora_rank=32,
        )


def test_deploy_adapter_missing_tensor_file_is_adapter_tensor_missing(monkeypatch, tmp_path):
    import flash.serve.deploy as d

    _stub_adapter_config(monkeypatch, tmp_path, tensor_files={"README.md": 123})

    with pytest.raises(d.AdapterTensorMissing, match="no adapter_model tensor file"):
        d.deploy_adapter(
            run_id="r-missing-tensors",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/r-missing-tensors/seed0",
            dry_run=False,
            lora_rank=32,
        )


def test_deploy_adapter_zero_byte_tensor_file_is_adapter_tensor_missing(monkeypatch, tmp_path):
    import flash.serve.deploy as d

    _stub_adapter_config(monkeypatch, tmp_path, tensor_files={"adapter_model.safetensors": 0})

    with pytest.raises(d.AdapterTensorMissing, match="zero-byte adapter tensor"):
        d.deploy_adapter(
            run_id="r-empty-tensors",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/r-empty-tensors/seed0",
            dry_run=False,
            lora_rank=32,
        )


def test_deploy_adapter_rejects_zero_byte_sharded_tensor(monkeypatch, tmp_path):
    import flash.serve.deploy as d

    _stub_adapter_config(
        monkeypatch,
        tmp_path,
        tensor_files={
            "adapter_model-00001-of-00002.safetensors": 456,
            "adapter_model-00002-of-00002.safetensors": 0,
        },
    )

    with pytest.raises(d.AdapterTensorMissing, match=r"adapter_model-00002-of-00002\.safetensors"):
        d.deploy_adapter(
            run_id="r-empty-shard",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/r-empty-shard/seed0",
            dry_run=False,
            lora_rank=32,
        )


def test_deploy_accepts_legacy_bin_adapter_tensor(monkeypatch, tmp_path):
    from flash.serve.deploy import adapter_artifact_lora_rank

    seen = _stub_adapter_config(monkeypatch, tmp_path, tensor_files={"adapter_model.bin": None})

    assert (
        adapter_artifact_lora_rank("org/repo", "sft/r-bin/seed0/adapter", hf_revision="a" * 40)
        == 32
    )
    assert seen["list_repo_tree"]["path_in_repo"] == "sft/r-bin/seed0/adapter"


def test_deploy_adapter_options_are_keyword_only():
    from flash.serve.deploy import deploy_adapter

    with pytest.raises(TypeError):
        deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "org/repo", "sft/r1/seed0", True)


def test_deploy_registers_with_freesolo_serving(monkeypatch, tmp_path, stub_serving_registry):
    """A non-dry-run deploy POSTs the adapter to {FREESOLO_SERVING_URL}/adapters with the
    right body and the internal-key auth header."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    _stub_adapter_config(monkeypatch, tmp_path, rank=32)

    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
        seen["url"] = url
        seen["json"] = json
        seen["headers"] = headers
        seen["follow_redirects"] = follow_redirects
        return _Resp()

    monkeypatch.setattr(d.httpx, "post", fake_post)
    # deploy reads the registry back before reporting ready
    stub_serving_registry(
        {"adapter_id": "flash-7-abcd", "subfolder": "sft/flash-7-abcd/seed0/adapter"}
    )

    dep = d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
    )
    assert seen["url"] == "https://serve.example/adapters"
    revision = "flash-7-abcd@final." + "a" * 40
    assert seen["json"] == {
        "adapter_id": revision,
        "repo_id": "org/repo",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "subfolder": "sft/flash-7-abcd/seed0/adapter",
        # flash always uploads adapters to HF *dataset* repos, so serving must be told to
        # pull from the dataset namespace (else snapshot_download 404s on the model namespace).
        "repo_type": "dataset",
        "checkpoint": "flash-7-abcd",
        "status": "ready",
        "metadata": {
            "record_type": "revision",
            "run_id": "flash-7-abcd",
            "checkpoint_step": None,
            "hf_revision": "a" * 40,
        },
        # per-adapter thinking default carried so serving can apply it as enable_thinking when a
        # raw chat caller omits chat_template_kwargs (deploy_adapter defaults thinking=False).
        "thinking": False,
    }
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    # Modal 303-redirects slow requests to an async-result poll URL, so registration follows them.
    assert seen["follow_redirects"] is True
    assert dep.openai_model == "flash-7-abcd"
    assert dep.endpoint_name == "https://serve.example"
    assert dep.state == "ready"


def test_deploy_registers_structured_outputs_default(monkeypatch, tmp_path, stub_serving_registry):
    """A non-thinking run trained with structured_outputs registers that grammar as the adapter's
    per-adapter guided-decoding DEFAULT, so serving constrains every request the same way training
    did (closes the train/serve exposure-bias gap). The spec is forwarded as its parsed canonical
    StructuredOutputsParams-kwargs dict (serving re-validates it)."""
    schema = {"type": "object", "properties": {"industries": {"type": "array"}}}
    spec = json.dumps({"json": schema})

    body = _capture_registration_body(
        monkeypatch,
        tmp_path,
        stub_serving_registry,
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        structured_outputs=spec,
    )
    assert body["structured_outputs"] == {"json": schema}
    # thinking default is still carried and independent of the constraint.
    assert body["thinking"] is False


def test_deploy_omits_structured_outputs_when_unset(monkeypatch, tmp_path, stub_serving_registry):
    """A run with no structured_outputs registers no serving grammar default (body key absent, not
    an empty/None value that serving would have to interpret)."""
    body = _capture_registration_body(
        monkeypatch,
        tmp_path,
        stub_serving_registry,
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        structured_outputs="",
    )
    assert "structured_outputs" not in body


def test_deploy_skips_structured_outputs_default_for_thinking(
    monkeypatch, tmp_path, stub_serving_registry
):
    """A THINKING run does NOT register a serving grammar default even if a constraint is set:
    serving binds the grammar from token 0 (no reasoning-parser deferral), which would suppress the
    <think> phase. (Training stays compatible via a deferred grammar; serving has no such deferral,
    so the serve-time default is skipped rather than break reasoning.)"""
    spec = json.dumps({"json": {"type": "object"}})
    body = _capture_registration_body(
        monkeypatch,
        tmp_path,
        stub_serving_registry,
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        thinking=True,
        structured_outputs=spec,
    )
    assert "structured_outputs" not in body
    assert body["thinking"] is True


def test_structured_outputs_body_helper():
    """Unit-level contract of the deploy helper: unset/thinking -> None; non-thinking spec ->
    parsed canonical dict; corrupt spec -> ValueError (fail loud, not silently unconstrained)."""
    from flash.serve.deploy import _structured_outputs_body

    assert _structured_outputs_body("r1", "", thinking=False) is None
    assert _structured_outputs_body("r1", "", thinking=True) is None
    spec = json.dumps({"json": {"type": "object"}})
    assert _structured_outputs_body("r1", spec, thinking=True) is None
    assert _structured_outputs_body("r1", spec, thinking=False) == {"json": {"type": "object"}}
    with pytest.raises(ValueError, match="corrupt train"):
        _structured_outputs_body("r1", "{not json", thinking=False)


def test_deploy_includes_org_id_when_provided(monkeypatch, tmp_path, stub_serving_registry):
    """When the deploying org is known, registration carries `org_id` so serving can persist
    hosted_lora_adapters.org_id and later authorize external chat by org. Omitted when unknown."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    _stub_adapter_config(monkeypatch, tmp_path, rank=32)

    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
        seen["json"] = json
        return _Resp()

    monkeypatch.setattr(d.httpx, "post", fake_post)
    # deploy reads the registry back before reporting ready
    stub_serving_registry(
        {"adapter_id": "flash-7-abcd", "subfolder": "sft/flash-7-abcd/seed0/adapter"}
    )

    d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        org_id="org-xyz",
    )
    assert seen["json"]["org_id"] == "org-xyz"

    # No org -> the key is omitted entirely (registration shape unchanged for older callers).
    d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
    )
    assert "org_id" not in seen["json"]


def test_deploy_sends_thinking_default(monkeypatch, tmp_path, stub_serving_registry):
    """Registration carries the run's training `thinking` flag so serving can default
    enable_thinking to it for raw chat callers (those that omit chat_template_kwargs). A
    thinking=true run registers thinking=true; a thinking=false run registers thinking=false."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    _stub_adapter_config(monkeypatch, tmp_path, rank=32)

    seen = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
        seen["json"] = json
        return _Resp()

    monkeypatch.setattr(d.httpx, "post", fake_post)
    # deploy reads the registry back before reporting ready
    stub_serving_registry(
        {"adapter_id": "flash-7-abcd", "subfolder": "sft/flash-7-abcd/seed0/adapter"}
    )

    d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        thinking=True,
    )
    assert seen["json"]["thinking"] is True

    # A non-thinking run registers thinking=false so serving renders enable_thinking=false by
    # default (else Qwen3.5's template default thinking-ON emits a reasoning preamble).
    d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        thinking=False,
    )
    assert seen["json"]["thinking"] is False


def test_deploy_propagates_serving_error(monkeypatch, tmp_path):
    """A non-2xx from the serving app surfaces as a ServingError (the server maps it to a 502)
    instead of swallowing it or letting a raw httpx error escape as an unhandled 500."""
    import flash.serve.deploy as d

    _stub_adapter_config(monkeypatch, tmp_path, rank=32)

    class _Resp:
        status_code = 500

        def raise_for_status(self):
            raise d.httpx.HTTPStatusError("boom", request=None, response=None)

    monkeypatch.setattr(d.httpx, "post", lambda *a, **k: _Resp())
    with pytest.raises(d.ServingError):
        d.deploy_adapter("r1", "Qwen/Qwen3.5-0.8B", "org/repo", "sft/r1/seed0")


def test_undeploy_deletes_on_freesolo_serving(monkeypatch):
    """A terminal /v1 override keeps undeploy calls on the serving control root."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")

    seen = {}

    class _Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "run_id": "flash-7-abcd",
                "disabled_aliases": ["flash-7-abcd"],
                "disabled_revisions": ["flash-7-abcd@final." + "a" * 40],
            }

    def fake_delete(url, headers=None, timeout=None, follow_redirects=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["follow_redirects"] = follow_redirects
        return _Resp(200)

    def fake_get(url, **kwargs):
        seen["verification_url"] = url
        return _Resp(200)

    monkeypatch.setattr(d.httpx, "delete", fake_delete)
    monkeypatch.setattr(d.httpx, "get", fake_get)
    out = d.undeploy_adapter("flash-7-abcd")
    assert out["disabled_aliases"] == ["flash-7-abcd"]
    assert out["serving_deregistered"] is True
    assert seen["url"] == "https://serve.example/adapters/flash-7-abcd"
    assert seen["verification_url"] == "https://serve.example/adapters"
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    # Modal 303-redirects slow requests to an async-result poll URL, so undeploy follows them too.
    assert seen["follow_redirects"] is True

    # A 404 (already gone) returns an empty list, not an error.
    monkeypatch.setattr(d.httpx, "delete", lambda *a, **k: _Resp(404))
    assert d.undeploy_adapter("flash-7-abcd")["serving_deregistered"] is False


def test_undeploy_propagates_serving_error(monkeypatch):
    """A non-404 failure from the serving app surfaces as a ServingError (carrying the upstream
    status, so the server maps it to a 502) — exactly like deploy — instead of letting a raw
    httpx error escape as an unhandled 500. A 404 still no-ops (already-gone is success)."""
    import flash.serve.deploy as d

    class _Resp:
        def __init__(self, code):
            self.status_code = code
            self.text = "kaboom"

        def raise_for_status(self):
            raise d.httpx.HTTPStatusError("boom", request=None, response=self)

    # Non-404 (500) → ServingError carrying the upstream status, not a raw httpx error.
    monkeypatch.setattr(d.httpx, "delete", lambda *a, **k: _Resp(500))
    with pytest.raises(d.ServingError) as ei:
        d.undeploy_adapter("flash-7-abcd")
    assert ei.value.status_code == 500

    # A transport error (never reached the backend) is also translated into a ServingError.
    # httpx.RequestError must carry the originating request (httpx>=0.27); building it with only a
    # message can raise TypeError before undeploy_adapter() can translate it, so mirror the real
    # undeploy call (DELETE {serving}/adapters/{run_id}).
    def _boom_delete(*a, **k):
        raise d.httpx.RequestError(
            "no route to host",
            request=d.httpx.Request("DELETE", "https://serve.example/adapters/flash-7-abcd"),
        )

    monkeypatch.setattr(d.httpx, "delete", _boom_delete)
    with pytest.raises(d.ServingError):
        d.undeploy_adapter("flash-7-abcd")

    # A 404 short-circuits before raise_for_status(), so it stays a no-op success (not a ServingError).
    monkeypatch.setattr(d.httpx, "delete", lambda *a, **k: _Resp(404))
    assert d.undeploy_adapter("flash-7-abcd")["serving_deregistered"] is False


def test_chat_posts_to_freesolo_serving(monkeypatch):
    """A terminal /v1 override produces one OpenAI path for direct chat."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")

    seen = {}
    completion = {
        "object": "chat.completion",
        "model": "flash-7-abcd",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi there"}}],
    }

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return completion

    class _FakeClient:
        # chat() uses an explicit httpx.Client (context manager) so it can follow Modal's 303
        # async-result redirects; the fake records the call and the client kwargs.
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None):
            seen["url"] = url
            seen["json"] = json
            seen["headers"] = headers or {}
            return _Resp()

    monkeypatch.setattr(d.httpx, "Client", _FakeClient)
    out = d.chat(
        run_id="flash-7-abcd",
        messages=[{"role": "user", "content": "2+2?"}],
        temperature=0.0,
        max_tokens=8,
        thinking=True,
    )
    assert seen["url"] == "https://serve.example/v1/chat/completions"
    # Modal 303-redirects slow ASGI requests to an async-result poll URL, so the chat client
    # MUST follow redirects (else httpx raises on the 303 mid cold-start).
    assert seen["client_kwargs"]["follow_redirects"] is True
    assert seen["json"]["model"] == "flash-7-abcd"
    assert seen["json"]["max_tokens"] == 8
    assert seen["json"]["messages"] == [{"role": "user", "content": "2+2?"}]
    # Per-run thinking parity: the thinking flag is forwarded to the chat template so a
    # thinking-trained adapter serves with thinking (not silently dropped).
    assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    # The OpenAI shape is preserved so resp["choices"][0]["message"]["content"] works.
    assert out["choices"][0]["message"]["content"] == "hi there"
    # The control plane is a trusted serving caller, so it presents the internal key — this is
    # what lets `flash chat` keep working when the serving app enforces external chat auth.
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"


def test_chat_stream_yields_openai_sse_content(monkeypatch):
    """A terminal /v1 override produces one OpenAI path for streaming chat."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    seen = {}

    class _StreamResp:
        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
                    'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}',
                    'data: {"choices":[{"delta":{"content":" there"},"finish_reason":null}]}',
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None):
            seen["method"] = method
            seen["url"] = url
            seen["json"] = json
            seen["headers"] = headers or {}
            return _StreamResp()

    monkeypatch.setattr(d.httpx, "Client", _FakeClient)

    chunks = list(
        d.chat_stream(
            run_id="flash-7-abcd",
            messages=[{"role": "user", "content": "2+2?"}],
            temperature=0.0,
            max_tokens=8,
            thinking=True,
        )
    )

    assert chunks == ["hi", " there"]
    assert seen["client_kwargs"]["follow_redirects"] is True
    assert seen["method"] == "POST"
    # Trusted-caller bypass: chat_stream presents the internal key, like the non-streaming chat.
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    assert seen["url"] == "https://serve.example/v1/chat/completions"
    assert seen["json"]["stream"] is True
    assert seen["json"]["model"] == "flash-7-abcd"
    assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_chat_stream_accepts_json_fallback(monkeypatch):
    """A new Flash server can still talk to an older serving app that ignores stream=true.

    Drives a REAL httpx streaming response (MockTransport) so the read-before-.json() contract is
    actually exercised — a stub with a bare .json() would mask the ResponseNotRead bug.
    """
    import httpx

    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")

    def handler(request):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "full reply"}}]}
        )  # httpx sets content-type: application/json

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(d.httpx, "Client", _client)

    assert list(d.chat_stream("flash-7-abcd", [{"role": "user", "content": "hi"}])) == [
        "full reply"
    ]
