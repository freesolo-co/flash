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
    record = {
        "adapter_id": run_id,
        "repo_id": deploy_kwargs["hf_repo"],
        "base_model": deploy_kwargs["model"],
        "repo_type": "dataset",
        "subfolder": f"{deploy_kwargs['adapter_prefix']}/adapter",
        "thinking": bool(deploy_kwargs.get("thinking", False)),
    }
    structured = json.loads(deploy_kwargs.get("structured_outputs") or "null")
    if structured:
        key = (
            "structured_outputs_after_reasoning"
            if deploy_kwargs.get("thinking")
            else "structured_outputs"
        )
        record[key] = structured
    stub_serving_registry(record)
    if deploy_kwargs.get("thinking") and structured:
        registry_get = d.httpx.get

        class _HealthResp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                model = deploy_kwargs["model"]
                return {
                    "reasoning_parser_by_model": {model: "qwen3"},
                    "deferred_structured_outputs_by_model": {
                        model: {"status": "live", "verified": True}
                    },
                }

        monkeypatch.setattr(
            d.httpx,
            "get",
            lambda url, **kwargs: _HealthResp()
            if url.endswith("/healthz")
            else registry_get(url, **kwargs),
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


def test_thinking_structured_dry_run_does_not_probe_or_mutate(monkeypatch):
    import flash.serve.deploy as d

    monkeypatch.setattr(
        d.httpx, "get", lambda *a, **k: pytest.fail("dry-run must not probe serving")
    )
    monkeypatch.setattr(
        d.httpx, "post", lambda *a, **k: pytest.fail("dry-run must not mutate serving")
    )
    dep = d.deploy_adapter(
        run_id="r-dry-structured",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="rl/r-dry-structured/seed0",
        dry_run=True,
        thinking=True,
        structured_outputs=json.dumps({"json_object": True}),
    )
    assert dep.state == "dry_run"
    assert dep.previous_registry_snapshot is None


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

    with pytest.raises(d.ServingError, match="failed to read org/repo:sft/r-hf-down/seed0/adapter") as excinfo:
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

    with pytest.raises(d.AdapterConfigMissing, match="failed to read org/repo:sft/r-missing/seed0/adapter"):
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

    with pytest.raises(
        d.AdapterTensorMissing, match=r"adapter_model-00002-of-00002\.safetensors"
    ):
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

    assert adapter_artifact_lora_rank("org/repo", "sft/r-bin/seed0/adapter") == 32
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
    assert seen["json"] == {
        "adapter_id": "flash-7-abcd",
        "repo_id": "org/repo",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "subfolder": "sft/flash-7-abcd/seed0/adapter",
        # flash always uploads adapters to HF *dataset* repos, so serving must be told to
        # pull from the dataset namespace (else snapshot_download 404s on the model namespace).
        "repo_type": "dataset",
        "status": "ready",
        # Per-adapter thinking default carried so serving can apply it as enable_thinking when a
        # raw chat caller omits chat_template_kwargs (deploy_adapter defaults thinking=False).
        "thinking": False,
    }
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    # Modal 303-redirects slow requests to an async-result poll URL, so registration follows them.
    assert seen["follow_redirects"] is True
    assert dep.openai_model == "flash-7-abcd"
    assert dep.endpoint_name == "https://serve.example"
    assert dep.state == "ready"


def test_deploy_registers_structured_outputs_default(
    monkeypatch, tmp_path, stub_serving_registry
):
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


def test_deploy_registers_deferred_structured_outputs_for_thinking(
    monkeypatch, tmp_path, stub_serving_registry
):
    """A thinking constraint is registered only in the post-reasoning field."""
    spec = json.dumps({"json": {"type": "object"}})
    body = _capture_registration_body(
        monkeypatch,
        tmp_path,
        stub_serving_registry,
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.6-35B-A3B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        thinking=True,
        structured_outputs=spec,
    )
    assert "structured_outputs" not in body
    assert body["structured_outputs_after_reasoning"] == {"json": {"type": "object"}}
    assert body["thinking"] is True


def test_structured_outputs_body_helper():
    """The helper parses canonical constraints and fails loudly on corruption."""
    from flash.serve.deploy import _structured_outputs_body

    assert _structured_outputs_body("") is None
    spec = json.dumps({"json": {"type": "object"}})
    assert _structured_outputs_body(spec) == {"json": {"type": "object"}}
    with pytest.raises(ValueError, match="corrupt train"):
        _structured_outputs_body("{not json")


def test_deploy_includes_org_id_when_provided(monkeypatch, tmp_path, stub_serving_registry):
    """When the deploying org is known, registration carries `org_id` so serving can persist
    hosted_lora_adapters.org_id and later authorize external chat by org. A replacement of an
    existing record inherits the prior owner even when the caller omits org_id, and a truly
    fresh adapter with no org omits the key entirely."""
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
        seen["headers"] = headers
        return _Resp()

    monkeypatch.setattr(d.httpx, "post", fake_post)
    # deploy reads the registry back before reporting ready; one expected record per mutation
    stub_serving_registry(
        {
            "adapter_id": "flash-7-abcd",
            "subfolder": "sft/flash-7-abcd/seed0/adapter",
            "org_id": "org-xyz",
        },
        {
            "adapter_id": "flash-7-abcd",
            "subfolder": "sft/flash-7-abcd/seed0/adapter",
            "org_id": "org-xyz",
        },
    )

    dep = d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        org_id="org-xyz",
    )
    assert seen["json"]["org_id"] == "org-xyz"
    # a fresh adapter has no prior revision to compare against
    assert seen["headers"] is None or "If-Match" not in seen["headers"]
    assert dep.registry_revision == 1

    # Redeploying without an org inherits the recorded owner (ownership never silently drops)
    # and carries If-Match against the previous revision.
    dep = d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
    )
    assert seen["json"]["org_id"] == "org-xyz"
    assert seen["headers"]["If-Match"] == '"1"'
    assert dep.registry_revision == 2


def test_deploy_rejects_owner_change(monkeypatch, tmp_path, stub_serving_registry):
    """Replacing an existing adapter cannot move it to a different org; the deploy fails
    before any registry mutation is attempted."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    _stub_adapter_config(monkeypatch, tmp_path, rank=32)

    def fail_post(*_args, **_kwargs):
        pytest.fail("owner-change rejection must happen before POST /adapters")

    monkeypatch.setattr(d.httpx, "post", fail_post)
    stub_serving_registry(
        {
            "adapter_id": "flash-7-abcd",
            "subfolder": "sft/flash-7-abcd/seed0/adapter",
            "org_id": "org-original",
        }
    )
    # commit the initial record so a previous snapshot exists at revision 1
    d._etag_revision(None)

    with pytest.raises(d.ServingError, match="cannot change owner"):
        d.deploy_adapter(
            run_id="flash-7-abcd",
            model="Qwen/Qwen3.5-0.8B",
            hf_repo="org/repo",
            adapter_prefix="sft/flash-7-abcd/seed0",
            org_id="org-other",
        )


def test_deploy_omits_org_id_for_fresh_unowned_adapter(
    monkeypatch, tmp_path, stub_serving_registry
):
    """A first-time deploy with no org omits the key entirely (registration shape unchanged
    for older callers)."""
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
    stub_serving_registry(
        {"adapter_id": "flash-7-abcd", "subfolder": "sft/flash-7-abcd/seed0/adapter"}
    )

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
        {
            "adapter_id": "flash-7-abcd",
            "subfolder": "sft/flash-7-abcd/seed0/adapter",
            "thinking": True,
        }
    )

    d.deploy_adapter(
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        thinking=True,
    )
    assert seen["json"]["thinking"] is True

    # a non-thinking run registers thinking=false.
    stub_serving_registry(
        {"adapter_id": "flash-7-abcd", "subfolder": "sft/flash-7-abcd/seed0/adapter"}
    )
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
    """undeploy uses If-Match and verifies the disabled tombstone revision."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    seen = {}
    deleted = False

    class _Resp:
        def __init__(self, code, payload=None, headers=None):
            self.status_code = code
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(*_args, **_kwargs):
        record = {
            "adapter_id": "flash-7-abcd",
            "repo_id": "org/repo",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "repo_type": "dataset",
            "thinking": False,
            "status": "disabled" if deleted else "ready",
        }
        return _Resp(
            200,
            {"adapter": record, "org_id": "org-1", "revision": 5 if deleted else 4},
        )

    def fake_delete(url, headers=None, timeout=None, follow_redirects=None):
        nonlocal deleted
        seen.update(
            url=url,
            headers=headers,
            follow_redirects=follow_redirects,
        )
        deleted = True
        return _Resp(200, headers={"ETag": '"5"'})

    monkeypatch.setattr(d.httpx, "get", fake_get)
    monkeypatch.setattr(d.httpx, "delete", fake_delete)
    assert d.undeploy_adapter("flash-7-abcd") == ["flash-7-abcd"]
    assert seen["url"] == "https://serve.example/adapters/flash-7-abcd"
    assert seen["headers"]["If-Match"] == '"4"'
    assert seen["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    assert seen["follow_redirects"] is True
    assert d.undeploy_adapter("flash-7-abcd") == []


def test_undeploy_propagates_serving_error(monkeypatch):
    """A failed ambiguous delete is surfaced with its status and reconciliation state."""
    import httpx

    import flash.serve.deploy as d

    snapshot = {
        "adapter": {
            "adapter_id": "flash-7-abcd",
            "repo_id": "org/repo",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "repo_type": "dataset",
            "thinking": False,
            "status": "ready",
        },
        "org_id": "org-1",
        "revision": 4,
    }

    class _GetResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return snapshot

    monkeypatch.setattr(d.httpx, "get", lambda *a, **k: _GetResp())
    request = httpx.Request("DELETE", "https://serve.example/adapters/flash-7-abcd")
    response = httpx.Response(500, text="kaboom", request=request)
    monkeypatch.setattr(d.httpx, "delete", lambda *a, **k: response)
    with pytest.raises(d.ServingError) as ei:
        d.undeploy_adapter("flash-7-abcd")
    assert ei.value.status_code == 500
    assert ei.value.reconciliation_required is True

    def _boom_delete(*_args, **_kwargs):
        raise httpx.RequestError("no route to host", request=request)

    monkeypatch.setattr(d.httpx, "delete", _boom_delete)
    with pytest.raises(d.ServingError) as ei:
        d.undeploy_adapter("flash-7-abcd")
    assert ei.value.reconciliation_required is True


def test_chat_posts_to_freesolo_serving(monkeypatch):
    """chat POSTs to {FREESOLO_SERVING_URL}/v1/chat/completions addressing the adapter by
    run_id, and returns the parsed OpenAI response dict."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
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
    """chat_stream requests OpenAI streaming and yields assistant content deltas only."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
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


@pytest.mark.parametrize("health_payload", [None, {}, {"capabilities": "bad"}, {"capabilities": []}])
def test_thinking_structured_capability_fails_before_mutation(
    monkeypatch, health_payload
):
    import httpx

    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(d, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    posts = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return health_payload

    def fake_get(url, **kwargs):
        if health_payload is None:
            raise httpx.ConnectError("offline", request=httpx.Request("GET", url))
        return _Resp()

    monkeypatch.setattr(d.httpx, "get", fake_get)
    monkeypatch.setattr(d.httpx, "post", lambda *a, **k: posts.append((a, k)))

    with pytest.raises(d.ServingError, match="No adapter registry mutation was attempted"):
        d.deploy_adapter(
            "r-cap",
            "Qwen/Qwen3.5-0.8B",
            "org/repo",
            "rl/r-cap/seed0",
            thinking=True,
            structured_outputs=json.dumps({"json_object": True}),
        )
    assert posts == []


def test_thinking_structured_capability_precedes_registration(monkeypatch):
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(d, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    events = []
    registry = None

    class _Resp:
        def __init__(self, status_code=200, payload=None, headers=None):
            self.status_code = status_code
            self.payload = payload
            self.headers = headers or {}

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, **_kwargs):
        if url.endswith("/healthz"):
            events.append("healthz")
            return _Resp(
                payload={
                    "reasoning_parser_by_model": {"Qwen/Qwen3.6-35B-A3B": "qwen3"},
                    "deferred_structured_outputs_by_model": {
                        "Qwen/Qwen3.6-35B-A3B": {"status": "live", "verified": True}
                    },
                }
            )
        if registry is None:
            return _Resp(status_code=404)
        return _Resp(payload={"adapter": registry, "org_id": None, "revision": 1})

    def fake_post(url, json=None, **_kwargs):
        nonlocal registry
        events.append("post")
        registry = dict(json)
        return _Resp(headers={"ETag": '"1"'})

    monkeypatch.setattr(d.httpx, "get", fake_get)
    monkeypatch.setattr(d.httpx, "post", fake_post)
    d.deploy_adapter(
        "r-cap",
        "Qwen/Qwen3.6-35B-A3B",
        "org/repo",
        "rl/r-cap/seed0",
        thinking=True,
        structured_outputs=json.dumps({"choice": ["4"]}),
    )
    assert events == ["healthz", "post"]
    assert registry["structured_outputs_after_reasoning"] == {"choice": ["4"]}
    assert "structured_outputs" not in registry


@pytest.mark.parametrize(
    ("thinking", "structured_outputs"),
    [(True, ""), (False, json.dumps({"regex": "4"}))],
)
def test_capability_probe_is_only_for_thinking_constraints(
    monkeypatch, thinking, structured_outputs
):
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(d, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    registry = None
    get_urls = []

    class _Resp:
        def __init__(self, status_code=200, payload=None, headers=None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, **_kwargs):
        get_urls.append(url)
        if registry is None:
            return _Resp(status_code=404)
        return _Resp(payload={"adapter": registry, "org_id": None, "revision": 1})

    def fake_post(url, json=None, **_kwargs):
        nonlocal registry
        registry = dict(json)
        return _Resp(headers={"ETag": '"1"'})

    monkeypatch.setattr(d.httpx, "get", fake_get)
    monkeypatch.setattr(d.httpx, "post", fake_post)
    d.deploy_adapter(
        "r-no-probe",
        "Qwen/Qwen3.5-0.8B",
        "org/repo",
        "rl/r-no-probe/seed0",
        thinking=thinking,
        structured_outputs=structured_outputs,
    )
    assert all(not url.endswith("/healthz") for url in get_urls)


def test_replacement_preserves_org_and_uses_if_match(monkeypatch):
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(d, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    prior = {
        "adapter": {
            "adapter_id": "r-replace",
            "repo_id": "org/old",
            "repo_type": "dataset",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "subfolder": "rl/old/adapter",
            "thinking": False,
            "status": "ready",
        },
        "org_id": "tenant-1",
        "revision": 7,
    }
    current = prior
    seen = {}

    class _Resp:
        def __init__(self, payload=None, headers=None):
            self.status_code = 200
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    monkeypatch.setattr(d.httpx, "get", lambda *a, **k: _Resp(current))

    def fake_post(url, json=None, headers=None, **_kwargs):
        nonlocal current
        seen.update(url=url, json=json, headers=headers)
        current = {"adapter": dict(json), "org_id": "tenant-1", "revision": 8}
        return _Resp(headers={"ETag": '"8"'})

    monkeypatch.setattr(d.httpx, "post", fake_post)
    dep = d.deploy_adapter(
        "r-replace", "Qwen/Qwen3.5-0.8B", "org/new", "rl/new", org_id="tenant-1"
    )
    assert seen["headers"]["If-Match"] == '"7"'
    assert seen["json"]["org_id"] == "tenant-1"
    assert dep.registry_revision == 8
    assert dep.previous_registry_snapshot == prior


def test_replacement_rejects_org_change_before_mutation(monkeypatch):
    import flash.serve.deploy as d

    monkeypatch.setattr(d, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    prior = {
        "adapter": {
            "adapter_id": "r-replace",
            "repo_id": "org/old",
            "repo_type": "dataset",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "subfolder": "rl/old/adapter",
            "thinking": False,
            "status": "ready",
        },
        "org_id": "tenant-1",
        "revision": 7,
    }
    monkeypatch.setattr(d, "snapshot_adapter_record", lambda _run_id: prior)
    monkeypatch.setattr(
        d.httpx, "post", lambda *a, **k: pytest.fail("owner mismatch must not mutate")
    )
    with pytest.raises(d.ServingError, match="cannot change owner"):
        d.deploy_adapter(
            "r-replace",
            "Qwen/Qwen3.5-0.8B",
            "org/new",
            "rl/new",
            org_id="tenant-2",
        )


@pytest.mark.parametrize("committed", [False, True])
def test_ambiguous_post_reconciles_strong_snapshot(monkeypatch, committed):
    import httpx

    import flash.serve.deploy as d

    monkeypatch.setattr(d, "adapter_artifact_lora_rank", lambda *a, **k: 32)
    current = None
    request = httpx.Request("POST", "https://serve.example/adapters")

    class _Resp:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(*_args, **_kwargs):
        if current is None:
            return _Resp(404)
        return _Resp(200, current)

    def fake_post(_url, json=None, **_kwargs):
        nonlocal current
        if committed:
            current = {"adapter": dict(json), "org_id": None, "revision": 1}
        raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setattr(d.httpx, "get", fake_get)
    monkeypatch.setattr(d.httpx, "post", fake_post)
    if committed:
        assert d.deploy_adapter(
            "r-timeout", "Qwen/Qwen3.5-0.8B", "org/new", "rl/new"
        ).registry_revision == 1
    else:
        with pytest.raises(d.ServingError, match="did not commit") as excinfo:
            d.deploy_adapter("r-timeout", "Qwen/Qwen3.5-0.8B", "org/new", "rl/new")
        assert excinfo.value.reconciliation_required is False
