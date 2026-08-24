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

_IMMUTABLE_SERVING_CAPABILITIES = {
    "immutable_adapter_revisions",
    "alias_compare_and_swap",
    "revision_provenance",
}


@pytest.fixture(autouse=True)
def _stub_shared_http_client(monkeypatch):
    import flash.serve.deploy as deploy

    class _Client:
        def request(self, method, url, **kwargs):
            return getattr(deploy.httpx, method.lower())(url, **kwargs)

        def post(self, url, **kwargs):
            timeout = kwargs.pop("timeout", None)
            client = deploy.httpx.Client(
                follow_redirects=True,
                max_redirects=100,
                timeout=timeout,
            )
            return client.post(url, **kwargs)

        def stream(self, method, url, **kwargs):
            timeout = kwargs.pop("timeout", None)
            client = deploy.httpx.Client(
                follow_redirects=True,
                max_redirects=100,
                timeout=timeout,
            )
            return client.stream(method, url, **kwargs)

    client = _Client()
    monkeypatch.setattr(deploy, "_http_client", lambda: client)
    monkeypatch.setattr(deploy, "_chat_http_client", lambda: client)
    monkeypatch.setattr(deploy, "_stream_http_client", lambda: client)


def _stub_adapter_config(
    monkeypatch,
    tmp_path,
    *,
    rank: int = 32,
    config: dict | None = None,
    tensor_files: dict[str, int | None] | None = None,
    stub_capabilities: bool = True,
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

    if stub_capabilities:
        monkeypatch.setattr(
            deploy,
            "_require_serving_capabilities",
            lambda **_kwargs: set(_IMMUTABLE_SERVING_CAPABILITIES),
        )
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


def _capture_registration_body(
    monkeypatch,
    tmp_path,
    stub_serving_registry,
    *,
    events: list[tuple[str, str]] | None = None,
    **deploy_kwargs,
):
    """Run a non-dry-run deploy_adapter and return the posted adapter body."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    _stub_adapter_config(monkeypatch, tmp_path, rank=32, stub_capabilities=False)

    seen: dict = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_post(url, json=None, headers=None, timeout=None, follow_redirects=None):
        if events is not None:
            events.append(("POST", url))
        seen["json"] = json
        return _Resp()

    monkeypatch.setattr(d.httpx, "post", fake_post)
    run_id = deploy_kwargs.get("run_id", "flash-7-abcd")
    stub_serving_registry(
        {"adapter_id": run_id, "subfolder": f"{deploy_kwargs['adapter_prefix']}/adapter"}
    )
    registry_get = d.httpx.get

    class _HealthResp(_Resp):
        def json(self):
            capabilities = set(_IMMUTABLE_SERVING_CAPABILITIES)
            if deploy_kwargs.get("thinking") and deploy_kwargs.get("structured_outputs"):
                capabilities.add(d.THINKING_STRUCTURED_OUTPUTS_CAPABILITY)
            return {"capabilities": sorted(capabilities)}

    def fake_get(url, **kwargs):
        if url == "https://serve.example/healthz":
            if events is not None:
                events.append(("GET", url))
            return _HealthResp()
        return registry_get(url, **kwargs)

    monkeypatch.setattr(d.httpx, "get", fake_get)
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


def test_thinking_structured_dry_run_does_not_probe_capabilities(monkeypatch):
    import flash.serve.deploy as d

    monkeypatch.setattr(
        d,
        "_require_serving_capabilities",
        lambda **_kwargs: pytest.fail("dry run must not probe serving capabilities"),
    )
    dep = d.deploy_adapter(
        run_id="r1",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/r1/seed0",
        dry_run=True,
        thinking=True,
        structured_outputs=json.dumps({"choice": ["4"]}),
    )
    assert dep.state == "dry_run"


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


def test_deploy_27b_dry_run_accepts_rank_at_serving_cap():
    from flash.serve.deploy import deploy_adapter

    dep = deploy_adapter(
        run_id="q27",
        model="Qwen/Qwen3.6-27B",
        hf_repo="org/repo",
        adapter_prefix="sft/q27/seed0",
        dry_run=True,
        lora_rank=64,
    )
    assert dep.to_dict()["state"] == "dry_run"


def test_deploy_27b_rejects_lora_rank_above_serving_cap():
    from flash.serve.deploy import deploy_adapter

    with pytest.raises(ValueError, match="max_lora_rank=64"):
        deploy_adapter(
            run_id="q27-r65",
            model="Qwen/Qwen3.6-27B",
            hf_repo="org/repo",
            adapter_prefix="sft/q27-r65/seed0",
            dry_run=True,
            lora_rank=65,
        )


def test_deploy_rejects_lora_rank_above_serving_cap():
    from flash.core.catalog import serving_lora_rank_cap
    from flash.serve.deploy import deploy_adapter

    # derive the over-cap rank from the catalog rather than hardcoding it: the 4B cap has moved
    # twice (32 -> 64 -> 128), and a literal here silently stops testing the boundary each time.
    cap = serving_lora_rank_cap("Qwen/Qwen3.5-4B")
    assert cap is not None
    with pytest.raises(ValueError, match=f"max_lora_rank={cap}"):
        deploy_adapter(
            run_id="r-over-cap",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/r-over-cap/seed0",
            dry_run=True,
            lora_rank=cap + 1,
        )


def test_deploy_rejects_recombined_artifact_rank_above_serving_cap(monkeypatch, tmp_path):
    """Deploy validates the effective artifact rank, not only spec.train.lora_rank."""
    from flash.core.catalog import serving_lora_rank_cap
    from flash.serve.deploy import deploy_adapter

    # the artifact's effective rank exceeds the cap even though the spec lora_rank (32) fits, so
    # deploy must catch the ARTIFACT rank. derived from the catalog: pinned to a literal, this stops
    # exercising the over-cap branch the moment the cap rises past it (as it just did, 64 -> 128).
    cap = serving_lora_rank_cap("Qwen/Qwen3.5-4B")
    assert cap is not None
    over_cap = cap + 1
    seen = _stub_adapter_config(monkeypatch, tmp_path, rank=over_cap)

    with pytest.raises(ValueError, match=f"adapter artifact has rank {over_cap}"):
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

    with pytest.raises(ValueError, match="invalid rank_pattern"):
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


def test_deploy_rejects_bin_adapter_tensor(monkeypatch, tmp_path):
    from flash.serve.deploy import AdapterTensorMissing, adapter_artifact_metadata

    seen = _stub_adapter_config(monkeypatch, tmp_path, tensor_files={"adapter_model.bin": None})

    with pytest.raises(AdapterTensorMissing, match="has no adapter_model tensor file"):
        adapter_artifact_metadata("org/repo", "sft/r-bin/seed0/adapter", hf_revision="a" * 40)
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
        # no client-sent "status": the serving backend sets it on registration and rejects an
        # incoming "status" as an extra field (HTTP 422).
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

    events: list[tuple[str, str]] = []
    body = _capture_registration_body(
        monkeypatch,
        tmp_path,
        stub_serving_registry,
        events=events,
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        structured_outputs=spec,
    )
    assert events == [
        ("GET", "https://serve.example/healthz"),
        ("POST", "https://serve.example/adapters"),
    ]
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


def test_deploy_registers_structured_outputs_for_thinking_after_capability_probe(
    monkeypatch, tmp_path, stub_serving_registry
):
    spec = json.dumps({"json": {"type": "object"}})
    events: list[tuple[str, str]] = []
    body = _capture_registration_body(
        monkeypatch,
        tmp_path,
        stub_serving_registry,
        events=events,
        run_id="flash-7-abcd",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-abcd/seed0",
        thinking=True,
        structured_outputs=spec,
    )
    assert events == [
        ("GET", "https://serve.example/healthz"),
        ("POST", "https://serve.example/adapters"),
    ]
    assert body["structured_outputs"] == {"json": {"type": "object"}}
    assert body["thinking"] is True


@pytest.mark.parametrize(
    ("health_case", "match"),
    [
        ("missing_immutable_revisions", "immutable_adapter_revisions"),
        ("missing_alias_cas", "alias_compare_and_swap"),
        # NOTE: `missing_provenance` is intentionally NOT here anymore -- revision_provenance is a
        # PREFERRED (warn-only) capability, not a deploy blocker. Its "deploy proceeds when only
        # provenance is missing" behavior is covered by
        # test_missing_provenance_only_still_deploys below.
        ("missing_thinking_structured", "thinking_structured_outputs_deferred_v1"),
        ("malformed", "must return a list field named capabilities"),
        ("invalid_json", "did not return valid JSON"),
        ("http_error", "HTTP 503"),
        ("unreachable", "could not reach the serving backend"),
    ],
)
def test_thinking_structured_capability_failure_never_posts_adapter(
    monkeypatch, tmp_path, health_case, match
):
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    _stub_adapter_config(monkeypatch, tmp_path, rank=32, stub_capabilities=False)
    posts: list[str] = []

    class _HealthResp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            if health_case == "http_error":
                request = d.httpx.Request("GET", "https://serve.example/healthz")
                response = d.httpx.Response(503, request=request, text="unavailable")
                raise d.httpx.HTTPStatusError("unavailable", request=request, response=response)

        def json(self):
            if health_case == "invalid_json":
                raise ValueError("not json")
            if health_case == "malformed":
                return {"capabilities": "thinking_structured_outputs_deferred_v1"}
            capabilities = _IMMUTABLE_SERVING_CAPABILITIES | {
                d.THINKING_STRUCTURED_OUTPUTS_CAPABILITY
            }
            missing = {
                "missing_immutable_revisions": "immutable_adapter_revisions",
                "missing_alias_cas": "alias_compare_and_swap",
                "missing_provenance": "revision_provenance",
                "missing_thinking_structured": d.THINKING_STRUCTURED_OUTPUTS_CAPABILITY,
            }.get(health_case)
            return {"capabilities": sorted(capabilities - ({missing} if missing else set()))}

    def fake_get(url, **_kwargs):
        assert url == "https://serve.example/healthz"
        if health_case == "unreachable":
            request = d.httpx.Request("GET", url)
            raise d.httpx.ConnectError("connection refused", request=request)
        return _HealthResp()

    def fake_post(url, **_kwargs):
        posts.append(url)
        pytest.fail("adapter registration must not be attempted")

    monkeypatch.setattr(d.httpx, "get", fake_get)
    monkeypatch.setattr(d.httpx, "post", fake_post)

    with pytest.raises(d.ServingError, match=match):
        d.deploy_adapter(
            run_id="flash-7-abcd",
            model="Qwen/Qwen3.5-0.8B",
            hf_repo="org/repo",
            adapter_prefix="sft/flash-7-abcd/seed0",
            thinking=True,
            structured_outputs=json.dumps({"choice": ["4"]}),
        )
    assert posts == []


def test_missing_provenance_only_still_deploys(monkeypatch, tmp_path):
    """The production serving backend advertises immutable revisions + alias CAS + the deferred
    thinking/structured-outputs cap, but NOT revision_provenance. That must NOT block the deploy
    (regression guard for the org-wide `serving_contract_unsupported: ... revision_provenance`
    outage): the capability check passes and registration is attempted."""
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    _stub_adapter_config(monkeypatch, tmp_path, rank=32, stub_capabilities=False)
    posts: list[str] = []

    advertised = sorted(
        (_IMMUTABLE_SERVING_CAPABILITIES - {"revision_provenance"})
        | {d.THINKING_STRUCTURED_OUTPUTS_CAPABILITY}
    )

    class _HealthResp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"capabilities": advertised}

    def fake_get(url, **_kwargs):
        assert url == "https://serve.example/healthz"
        return _HealthResp()

    def fake_post(url, **_kwargs):
        posts.append(url)
        raise RuntimeError("REACHED_POST")  # stop after proving the capability gate let us through

    monkeypatch.setattr(d.httpx, "get", fake_get)
    monkeypatch.setattr(d.httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="REACHED_POST"):
        d.deploy_adapter(
            run_id="flash-7-abcd",
            model="Qwen/Qwen3.5-0.8B",
            hf_repo="org/repo",
            adapter_prefix="sft/flash-7-abcd/seed0",
            thinking=True,
            structured_outputs=json.dumps({"choice": ["4"]}),
        )
    assert posts == ["https://serve.example/adapters"]  # registration WAS attempted


def test_deploy_passes_require_provenance_false_when_backend_lacks_it(monkeypatch, tmp_path):
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    _stub_adapter_config(monkeypatch, tmp_path, rank=32, stub_capabilities=False)
    advertised = sorted(
        (_IMMUTABLE_SERVING_CAPABILITIES - {"revision_provenance"})
        | {d.THINKING_STRUCTURED_OUTPUTS_CAPABILITY}
    )
    captured: dict[str, bool] = {}

    class _Resp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"capabilities": advertised}

    def fake_get(url, **_kwargs):
        assert url == "https://serve.example/healthz"
        return _Resp()

    def wait_ready(
        adapter_revision,
        subfolder,
        *,
        expected_identity=None,
        require_provenance=True,
        budget_s=None,
    ):
        captured["require_provenance"] = require_provenance
        captured["budget_s"] = budget_s
        return {}

    monkeypatch.setattr(d.httpx, "get", fake_get)
    monkeypatch.setattr(d.httpx, "post", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(d, "_wait_revision_ready", wait_ready)

    d.deploy_adapter(
        run_id="flash-7-no-provenance",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-no-provenance/seed0",
    )
    assert captured["require_provenance"] is False

    advertised = sorted(_IMMUTABLE_SERVING_CAPABILITIES)
    captured.clear()
    d.deploy_adapter(
        run_id="flash-7-with-provenance",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/flash-7-with-provenance/seed0",
    )
    assert captured["require_provenance"] is True


def test_wait_revision_ready_retries_transient_read_errors(monkeypatch):
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40
    ready = {
        "adapter_id": revision,
        "subfolder": "sft/run-1/seed0/adapter",
        "metadata": {"lifecycle_state": "ready"},
    }
    outcomes = [
        d.ServingError("temporary 503", status_code=503),
        d.ServingError("connection reset"),
        ready,
    ]

    def fake_registered_adapter(adapter_id, *, timeout_s=None):
        assert adapter_id == revision
        assert timeout_s is not None
        assert timeout_s > 0
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, types.SimpleNamespace(headers={})

    sleeps = []
    monkeypatch.setattr(d, "_registered_adapter_response", fake_registered_adapter)
    monkeypatch.setattr(d.time, "sleep", sleeps.append)

    assert d._wait_revision_ready(revision, ready["subfolder"]) == ready
    assert sleeps == [d._readback_delay(0), d._readback_delay(1)]


def test_wait_revision_ready_caps_reads_by_remaining_wall_time(monkeypatch):
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40
    clock = [100.0]
    request_timeouts = []

    def fake_registered_adapter(adapter_id, *, timeout_s=None):
        assert adapter_id == revision
        request_timeouts.append(timeout_s)
        clock[0] += 3.0 if len(request_timeouts) == 1 else float(timeout_s)
        raise d.ServingError("slow transient read", status_code=503)

    def fake_sleep(delay):
        clock[0] += delay

    monkeypatch.setattr(d, "READBACK_DELAY_SECONDS", 1.0)
    monkeypatch.setattr(d.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(d.time, "sleep", fake_sleep)
    monkeypatch.setattr(d, "_registered_adapter_response", fake_registered_adapter)

    with pytest.raises(d.ServingError, match="readiness could not be confirmed"):
        d._wait_revision_ready(revision, "sft/run-1/seed0/adapter", budget_s=5.0)

    assert request_timeouts == [5.0, 1.0]
    assert clock[0] == 105.0


def test_revision_ready_budget_scales_with_base_model_size():
    """A cold engine loads the BASE model, so the readiness budget must track its size.

    The flat 5-minute budget timed out deploys that were still loading a large base and reported it
    as `remained 'registered'`; re-running the identical deploy against the now-warm engine
    succeeded. Bigger base, bigger budget.
    """
    import flash.serve.deploy as d
    from flash.core.catalog import MODELS

    smallest = d.revision_ready_budget_seconds("Qwen/Qwen3.5-0.8B")
    largest = d.revision_ready_budget_seconds("Qwen/Qwen3.6-35B-A3B")
    assert smallest >= d.REVISION_READY_MIN_BUDGET_SECONDS
    assert largest > smallest

    # monotonic in params_b across the whole catalog: a bigger checkpoint never gets less time.
    budgets = [
        (info.params_b, d.revision_ready_budget_seconds(model_id))
        for model_id, info in MODELS.items()
    ]
    for _params_b, budget in budgets:
        assert d.REVISION_READY_MIN_BUDGET_SECONDS <= budget <= d.REVISION_READY_MAX_BUDGET_SECONDS
    ordered = [budget for _params, budget in sorted(budgets)]
    assert ordered == sorted(ordered)
    # and the per-B term is genuinely live below the cap, not a constant the cap flattens: two
    # distinct sub-cap sizes must get distinct budgets, or "scales with size" is untested.
    sub_cap = sorted({b for b in ordered if b < d.REVISION_READY_MAX_BUDGET_SECONDS})
    assert len(sub_cap) >= 2

    # an MoE is sized by its TOTAL params: every expert is resident even though a token routes
    # through few, so active_params_b must not shrink the budget. asserted against the formula
    # directly rather than against the 35B entry, whose scaled value clamps at the cap.
    moe = MODELS["Qwen/Qwen3.6-35B-A3B"]
    assert moe.is_moe
    assert moe.active_params_b < moe.params_b
    by_total = min(
        d.REVISION_READY_MIN_BUDGET_SECONDS + d.REVISION_READY_SECONDS_PER_PARAM_B * moe.params_b,
        d.REVISION_READY_MAX_BUDGET_SECONDS,
    )
    by_active = min(
        d.REVISION_READY_MIN_BUDGET_SECONDS
        + d.REVISION_READY_SECONDS_PER_PARAM_B * moe.active_params_b,
        d.REVISION_READY_MAX_BUDGET_SECONDS,
    )
    assert largest == by_total
    assert largest > by_active


def test_revision_ready_budget_unknown_model_keeps_the_floor():
    """A fork's own catalog entry or a revision-pinned id must not fail the lookup into an error."""
    import flash.serve.deploy as d

    for unknown in ("some-org/not-in-catalog", "", "   "):
        assert d.revision_ready_budget_seconds(unknown) == d.REVISION_READY_MIN_BUDGET_SECONDS


def test_revision_ready_budget_leaves_room_for_the_rest_of_the_deploy():
    """Readiness is one leg of the attempt, and the other legs have no wall-clock bound of their own.

    The same deploy resolves the hub revision, downloads the adapter config to read its rank, checks
    capabilities and registers before this wait, then runs immutable smoke, activates, and verifies
    the alias. all of that has to finish before the control plane declares the attempt abandoned and
    before the CLI's default `--wait` gives up, so the cap must reserve time rather than merely clear
    smoke.
    """
    import flash.serve.deploy as d

    # take the CLI default from the parser rather than restating it, so the two cannot drift apart
    # silently: shrinking bare `--wait` must fail here, not in a deploy.
    from flash.cli import _build_parser
    from flash.server.routes.serving import _DEPLOYMENT_STALE_SECONDS
    from flash.server.routes.serving_smoke import _SMOKE_BUDGET_SECONDS

    cli_default_wait = float(
        _build_parser().parse_args(["models", "deploy", "run-1", "--wait"]).wait
    )

    bounded = d.REVISION_READY_MAX_BUDGET_SECONDS + 2 * _SMOKE_BUDGET_SECONDS
    assert bounded < _DEPLOYMENT_STALE_SECONDS
    # and the CLI must not call a still-progressing deploy failed before the plane reaps it.
    assert bounded < cli_default_wait
    # the unbudgeted hub reads, registration, activation and poll latency get a real reserve.
    reserve = min(_DEPLOYMENT_STALE_SECONDS, cli_default_wait) - bounded
    assert reserve >= 300.0


def test_deploy_funds_the_readiness_wait_from_the_model_budget(monkeypatch, tmp_path):
    """`deploy_adapter` must pass the scaled budget; the default argument alone is the old bug."""
    import flash.serve.deploy as d

    _stub_adapter_config(monkeypatch, tmp_path, rank=32)
    monkeypatch.setattr(d, "resolve_hf_revision", lambda _repo: "a" * 40)
    monkeypatch.setattr(d, "_activate_revision", lambda *a, **k: {"adapter_id": "run-1"})

    class Response:
        status_code = 200

    monkeypatch.setattr(d, "_serving_request", lambda method, url, **kwargs: Response())
    budgets = []

    def wait_ready(revision, subfolder, **kwargs):
        budgets.append(kwargs.get("budget_s"))
        return {}

    monkeypatch.setattr(d, "_wait_revision_ready", wait_ready)

    d.deploy_adapter(
        run_id="run-1",
        model="Qwen/Qwen3.6-35B-A3B",
        hf_repo="org/repo",
        adapter_prefix="sft/run-1/seed0",
    )

    assert budgets == [d.revision_ready_budget_seconds("Qwen/Qwen3.6-35B-A3B")]
    assert budgets[0] > d.REVISION_READY_MIN_BUDGET_SECONDS


def test_revision_ready_timeout_message_is_self_diagnosing(monkeypatch):
    """The timeout must say it IS a timeout, and that retrying is the right response.

    `remained 'registered'` read as a serving/registration fault and sent readers to the wrong
    subsystem. The distinction matters operationally: retrying is correct for this message and
    wrong for an actively rejected adapter.
    """
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40
    clock = [100.0]

    def registered(adapter_id, *, timeout_s=None):
        clock[0] += 1.0
        return (
            {"subfolder": "sft/run-1/seed0/adapter", "metadata": {"lifecycle_state": "queued"}},
            types.SimpleNamespace(headers={}),
        )

    monkeypatch.setattr(d, "_registered_adapter_response", registered)
    monkeypatch.setattr(d.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(d.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))

    with pytest.raises(d.ServingError) as excinfo:
        d._wait_revision_ready(revision, "sft/run-1/seed0/adapter", budget_s=7.0)

    message = str(excinfo.value)
    assert "revision_ready_timeout" in message
    # names the clock and the last observed state, so the reader need not open serve/deploy.py
    assert "7s" in message
    assert "'queued'" in message
    # says which way to act, and distinguishes itself from the rejection case
    assert "retrying" in message
    assert "serving failed to load adapter revision" in message


def test_revision_ready_timeout_reports_the_loader_failure(monkeypatch):
    """A loader complaint that never moved the revision to `failed` must still be reported.

    This is the evidence that says which subsystem is at fault; dropping it is what made the
    original message send readers to serving when the cause was upstream.
    """
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40
    clock = [100.0]

    def registered(adapter_id, *, timeout_s=None):
        clock[0] += 1.0
        return (
            {
                "subfolder": "sft/run-1/seed0/adapter",
                "metadata": {"lifecycle_state": "queued", "failure": "adapter tensors truncated"},
            },
            types.SimpleNamespace(headers={}),
        )

    monkeypatch.setattr(d, "_registered_adapter_response", registered)
    monkeypatch.setattr(d.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(d.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))

    with pytest.raises(d.ServingError) as excinfo:
        d._wait_revision_ready(revision, "sft/run-1/seed0/adapter", budget_s=5.0)

    message = str(excinfo.value)
    assert "adapter tensors truncated" in message
    # and it must NOT prescribe a retry. a truncated artifact survives a warm engine, so the
    # cold-engine advice would send the reader into a futile loop -- the same wrong-direction
    # failure this message exists to fix.
    assert "retrying this deploy is the correct response" not in message
    assert "succeeds against the now-warm engine" not in message
    assert "fix what the loader reported before retrying" in message


def test_revision_ready_timeout_distinguishes_a_never_visible_record(monkeypatch):
    """A revision that 404s for the whole budget is a different fault from a slow load."""
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40
    clock = [100.0]

    def registered(adapter_id, *, timeout_s=None):
        clock[0] += 1.0
        return None, types.SimpleNamespace(headers={})

    monkeypatch.setattr(d, "_registered_adapter_response", registered)
    monkeypatch.setattr(d.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(d.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))

    with pytest.raises(d.ServingError) as excinfo:
        d._wait_revision_ready(revision, "sft/run-1/seed0/adapter", budget_s=5.0)

    message = str(excinfo.value)
    assert "never returned the revision record" in message
    # the remedy must match the diagnostic. no engine state was ever read, so the cold-engine retry
    # advice would contradict the line above it and point at the wrong subsystem.
    assert "registration-visibility problem" in message
    assert "succeeds against the now-warm engine" not in message


def test_a_cleared_loader_failure_is_not_reported_after_the_timeout(monkeypatch):
    """A later record that drops `failure` has withdrawn the complaint.

    Retaining the first one makes the timeout prescribe "fix the artifact" for what the final
    record says is an ordinary cold-engine timeout -- the wrong direction, again.
    """
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40
    clock = [100.0]
    reads = [0]

    def registered(adapter_id, *, timeout_s=None):
        clock[0] += 1.0
        reads[0] += 1
        metadata: dict = {"lifecycle_state": "queued"}
        # only the FIRST read carries a complaint; later authoritative records clear it.
        if reads[0] == 1:
            metadata["failure"] = "adapter tensors truncated"
        return (
            {"subfolder": "sft/run-1/seed0/adapter", "metadata": metadata},
            types.SimpleNamespace(headers={}),
        )

    monkeypatch.setattr(d, "_registered_adapter_response", registered)
    monkeypatch.setattr(d.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(d.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))

    with pytest.raises(d.ServingError) as excinfo:
        d._wait_revision_ready(revision, "sft/run-1/seed0/adapter", budget_s=5.0)

    message = str(excinfo.value)
    assert reads[0] > 1, "the test needs more than one read for the failure to be cleared"
    assert "adapter tensors truncated" not in message
    assert "retrying this deploy is the correct response" in message


def test_a_loader_failure_survives_later_transient_read_errors(monkeypatch):
    """A transient 5xx is not an authoritative record, so it withdraws nothing.

    When the loader named a failure and every later poll 5xxs until the deadline, reporting only
    the read error points at serving -- but serving already said the artifact is wrong, and that
    survives a warm engine. The deterministic complaint is the actionable half and must be kept.
    """
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40
    clock = [100.0]
    reads = [0]

    def registered(adapter_id, *, timeout_s=None):
        clock[0] += 1.0
        reads[0] += 1
        # the FIRST read is authoritative and names the loader's complaint; every later poll is a
        # retryable 5xx that lasts until the budget is spent.
        if reads[0] == 1:
            return (
                {
                    "subfolder": "sft/run-1/seed0/adapter",
                    "metadata": {
                        "lifecycle_state": "queued",
                        "failure": "adapter tensors truncated",
                    },
                },
                types.SimpleNamespace(headers={}),
            )
        raise d.ServingError("serving backend error (HTTP 503)", status_code=503)

    monkeypatch.setattr(d, "_registered_adapter_response", registered)
    monkeypatch.setattr(d.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(d.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))

    with pytest.raises(d.ServingError) as excinfo:
        d._wait_revision_ready(revision, "sft/run-1/seed0/adapter", budget_s=5.0)

    message = str(excinfo.value)
    assert reads[0] > 1, "the test needs the transient errors to follow the authoritative record"
    # the transient error is still reported: it is why readiness could not be confirmed.
    assert "transient" in message
    # but the loader's deterministic complaint must not be dropped on the floor.
    assert "adapter tensors truncated" in message


def test_rejected_adapter_still_fails_distinctly_from_a_timeout(monkeypatch):
    """The rejection path must keep its own message: retrying it is wrong."""
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40
    monkeypatch.setattr(
        d,
        "_registered_adapter_response",
        lambda adapter_id, **_kwargs: (
            {
                "subfolder": "sft/run-1/seed0/adapter",
                "metadata": {"lifecycle_state": "failed", "failure": "rank 64 exceeds engine cap"},
            },
            types.SimpleNamespace(headers={}),
        ),
    )

    with pytest.raises(d.ServingError) as excinfo:
        d._wait_revision_ready(revision, "sft/run-1/seed0/adapter", budget_s=5.0)

    message = str(excinfo.value)
    assert "serving failed to load adapter revision" in message
    assert "rank 64 exceeds engine cap" in message
    assert "revision_ready_timeout" not in message


def test_deploy_ready_read_returned_at_deadline_never_activates(monkeypatch, tmp_path):
    import flash.serve.deploy as d

    real_wait_revision_ready = d._wait_revision_ready
    _stub_adapter_config(monkeypatch, tmp_path, rank=32)
    monkeypatch.setattr(d, "_wait_revision_ready", real_wait_revision_ready)
    monkeypatch.setattr(d, "resolve_hf_revision", lambda _repo: "a" * 40)
    registration_body = {}

    class Response:
        status_code = 200

    def request(method, url, *, json=None, ok_statuses=(), timeout_s=None):
        assert method == "POST"
        registration_body.update(json)
        return Response()

    clock = [100.0]

    def registered(adapter_id, *, timeout_s=None):
        assert adapter_id == registration_body["adapter_id"]
        # the deploy funds this read from the model's own scaled budget, not the bare floor.
        assert timeout_s == d.revision_ready_budget_seconds("Qwen/Qwen3.5-4B")
        clock[0] += timeout_s
        return (
            {
                **registration_body,
                "metadata": {**registration_body["metadata"], "lifecycle_state": "ready"},
            },
            types.SimpleNamespace(headers={}),
        )

    activations = []
    monkeypatch.setattr(d, "_serving_request", request)
    monkeypatch.setattr(d, "_registered_adapter_response", registered)
    monkeypatch.setattr(d.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        d,
        "_activate_revision",
        lambda *args, **kwargs: activations.append((args, kwargs)),
    )

    with pytest.raises(d.ServingError, match="revision_ready_timeout"):
        d.deploy_adapter(
            run_id="run-expiry",
            model="Qwen/Qwen3.5-4B",
            hf_repo="org/repo",
            adapter_prefix="sft/run-expiry/seed0",
        )

    assert activations == []


def test_registered_adapter_caps_request_timeout(monkeypatch):
    import flash.serve.deploy as d

    seen = {}

    class Response:
        status_code = 404

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return Response()

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setattr(d.httpx, "get", fake_get)

    assert d._registered_adapter("run-1", timeout_s=0.75) is None
    assert seen["timeout"] == 0.75


def test_adapter_alias_target_rejects_legacy_record(monkeypatch):
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40
    monkeypatch.setattr(
        d,
        "_registered_adapter",
        lambda run_id: {"adapter_id": run_id, "metadata": {"alias_of": revision}},
    )
    assert d.adapter_alias_target("run-1") == revision

    monkeypatch.setattr(
        d,
        "_registered_adapter",
        lambda run_id: {
            "adapter_id": run_id,
            "status": "disabled",
            "metadata": {"alias_of": revision},
        },
    )
    assert d.adapter_alias_target("run-1") is None

    monkeypatch.setattr(d, "_registered_adapter", lambda run_id: {"adapter_id": run_id})
    with pytest.raises(d.ServingError, match="legacy aliases are unsupported"):
        d.adapter_alias_target("run-1")


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

    monkeypatch.setattr(d.httpx, "delete", fake_delete)
    # no registry-readback get: the structured undeploy response (disabled_aliases /
    # disabled_revisions / serving_deregistered) is authoritative.
    monkeypatch.setattr(
        d.httpx, "get", lambda *a, **k: pytest.fail("undeploy must not read the registry back")
    )
    out = d.undeploy_adapter("flash-7-abcd")
    assert out["disabled_aliases"] == ["flash-7-abcd"]
    assert out["serving_deregistered"] is True
    assert seen["url"] == "https://serve.example/adapters/flash-7-abcd"
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


def test_chat_classifies_retryable_alias_smoke_503_for_the_expected_revision(monkeypatch):
    import flash.serve.deploy as d

    run_id = "run-1"
    revision = f"{run_id}@final." + "a" * 40

    class Response:
        status_code = 503

        def __init__(self):
            self.headers = {"Retry-After": "1.5"}

        def json(self):
            return {
                "error": {
                    "type": "adapter_unavailable",
                    "code": "adapter_loading",
                    "message": "adapter revision is loading",
                    "retryable": True,
                    "requested_model": run_id,
                    "adapter_revision": revision,
                    "retry_after_seconds": 2,
                }
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(d.httpx, "Client", Client)

    with pytest.raises(d.RetryableServingUnavailable) as exc_info:
        d.chat(
            run_id,
            [{"role": "user", "content": "hello"}],
            expected_adapter_revision=revision,
            timeout_s=5.0,
            retry_unavailable=True,
        )

    assert exc_info.value.code == "adapter_loading"
    assert exc_info.value.retry_after_seconds == 1.5


@pytest.mark.parametrize(
    "error",
    [
        {
            "type": "adapter_unavailable",
            "code": "adapter_load_failed",
            "retryable": True,
        },
        {
            "type": "adapter_unavailable",
            "code": "adapter_loading",
            "retryable": False,
        },
    ],
)
def test_chat_fails_closed_for_unrecognized_smoke_503(monkeypatch, error):
    import flash.serve.deploy as d

    revision = "run-1@final." + "a" * 40

    class Response:
        status_code = 503

        def __init__(self):
            self.headers = {"Retry-After": "1"}
            self.request = d.httpx.Request("POST", "https://serve.example/v1/chat/completions")

        def json(self):
            return {
                "error": {
                    **error,
                    "requested_model": revision,
                    "adapter_revision": revision,
                }
            }

        def raise_for_status(self):
            raise d.httpx.HTTPStatusError("unavailable", request=self.request, response=self)

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(d.httpx, "Client", Client)

    with pytest.raises(d.httpx.HTTPStatusError):
        d.chat(
            revision,
            [{"role": "user", "content": "hello"}],
            timeout_s=5.0,
            retry_unavailable=True,
        )


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


def test_chat_preserves_explicit_empty_structured_override_and_omits_none(monkeypatch):
    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")
    requests = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class Client:
        def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return Response()

    monkeypatch.setattr(d, "_chat_http_client", Client)

    messages = [{"role": "user", "content": "hello"}]
    d.chat("run-1", messages, structured_outputs={})
    d.chat("run-1", messages)

    first_url, first = requests[0]
    second_url, second = requests[1]
    assert first_url == second_url == "https://serve.example/v1/chat/completions"
    assert first["json"]["structured_outputs"] == {}
    assert "structured_outputs" not in second["json"]
    assert first["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"
    assert second["headers"]["X-Freesolo-Internal-Key"] == "secret-internal"


def test_chat_sse_preserves_raw_frames_and_forwards_supported_fields(monkeypatch):
    import flash.serve.deploy as d

    seen = {}
    exits = []

    class _StreamResp:
        status_code = 200

        def __init__(self):
            self.headers = {
                "content-type": "text/event-stream",
                "x-freesolo-adapter-revision": "run-1@final." + "a" * 40,
            }

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b'data: {"id":"one","choices":[{"index":3,"delta":{"content":"a"}}]}\n'
            yield b'\ndata: {"id":"one","choices":[{"index":3,"finish_reason":"length"}]}\n\n'
            yield b"data: [DONE]\n\n"

    class _Context:
        def __enter__(self):
            return _StreamResp()

        def __exit__(self, *exc):
            exits.append(exc)
            return False

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            seen.update({"method": method, "url": url, **kwargs})
            return _Context()

    monkeypatch.setattr(d, "_stream_http_client", lambda: _FakeClient())
    monkeypatch.setattr(d, "serving_openai_base_url", lambda: "https://serve.example/v1")
    response = d.chat_sse(
        "run-1",
        [{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=9,
        top_p=0.7,
        stop=["end"],
        chat_template_kwargs={"enable_thinking": False, "custom": 1},
        structured_outputs={"json_object": True},
        stream_options={"include_usage": True},
    )

    frames = list(response.iter_bytes())

    assert frames == [
        b'data: {"id":"one","choices":[{"index":3,"delta":{"content":"a"}}]}\n\n',
        b'data: {"id":"one","choices":[{"index":3,"finish_reason":"length"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert response.status_code == 200
    assert seen["json"] == {
        "model": "run-1",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 9,
        "temperature": 0.2,
        "top_p": 0.7,
        "chat_template_kwargs": {"enable_thinking": False, "custom": 1},
        "stream": True,
        "stop": ["end"],
        "structured_outputs": {"json_object": True},
        "stream_options": {"include_usage": True},
    }
    assert len(exits) == 1


def test_chat_sse_close_before_first_byte_releases_upstream(monkeypatch):
    import flash.serve.deploy as d

    exits = []

    class _StreamResp:
        status_code = 200

        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}

        def iter_bytes(self):
            yield b"data: [DONE]\n\n"

    class _Context:
        def __enter__(self):
            return _StreamResp()

        def __exit__(self, *exc):
            exits.append(exc)
            return False

    class _FakeClient:
        def stream(self, *_args, **_kwargs):
            return _Context()

    monkeypatch.setattr(d, "_stream_http_client", lambda: _FakeClient())
    monkeypatch.setattr(d, "serving_openai_base_url", lambda: "https://serve.example/v1")
    response = d.chat_sse("run-1", [{"role": "user", "content": "hi"}])
    body = response.iter_bytes()

    body.close()

    assert len(exits) == 1


def test_chat_sse_close_without_exhaustion_releases_upstream(monkeypatch):
    import flash.serve.deploy as d

    exits = []

    class _StreamResp:
        status_code = 200

        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}

        def iter_bytes(self):
            yield b'data: {"choices":[]}\n\n'
            yield b"data: [DONE]\n\n"

    class _Context:
        def __enter__(self):
            return _StreamResp()

        def __exit__(self, *exc):
            exits.append(exc)
            return False

    class _FakeClient:
        def stream(self, *_args, **_kwargs):
            return _Context()

    monkeypatch.setattr(d, "_stream_http_client", lambda: _FakeClient())
    monkeypatch.setattr(d, "serving_openai_base_url", lambda: "https://serve.example/v1")
    response = d.chat_sse("run-1", [{"role": "user", "content": "hi"}])
    body = response.iter_bytes()

    assert next(body) == b'data: {"choices":[]}\n\n'
    body.close()
    assert len(exits) == 1


def test_chat_sse_complete_frame_without_done_raises_after_preserving_bytes(monkeypatch):
    import flash.serve.deploy as d
    from flash.client.http import ClientError

    frame = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    class _StreamResp:
        status_code = 200

        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}

        def iter_bytes(self):
            yield frame

    class _Context:
        def __enter__(self):
            return _StreamResp()

        def __exit__(self, *_exc):
            return False

    class _FakeClient:
        def stream(self, *_args, **_kwargs):
            return _Context()

    monkeypatch.setattr(d, "_stream_http_client", lambda: _FakeClient())
    monkeypatch.setattr(d, "serving_openai_base_url", lambda: "https://serve.example/v1")
    stream = d.chat_sse("run-1", [{"role": "user", "content": "hi"}]).iter_bytes()

    assert next(stream) == frame
    with pytest.raises(ClientError, match=r"terminal \[DONE\]"):
        next(stream)


def test_chat_posts_supported_managed_fields(monkeypatch):
    import flash.serve.deploy as d

    seen = {}

    class _Response:
        def __init__(self):
            self.headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Client:
        def post(self, url, **kwargs):
            seen.update({"url": url, **kwargs})
            return _Response()

    monkeypatch.setattr(d, "_chat_http_client", lambda: _Client())
    monkeypatch.setattr(d, "serving_openai_base_url", lambda: "https://serve.example/v1")
    d.chat(
        "run-1",
        [{"role": "user", "content": "hi"}],
        top_p=0.6,
        stop=["end"],
        chat_template_kwargs={"enable_thinking": True, "custom": 2},
        structured_outputs={"regex": "[ab]+"},
    )

    assert seen["json"] == {
        "model": "run-1",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 0.6,
        "chat_template_kwargs": {"enable_thinking": True, "custom": 2},
        "stop": ["end"],
        "structured_outputs": {"regex": "[ab]+"},
    }


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


def test_chat_stream_rejects_sse_eof_without_done(monkeypatch):
    import flash.serve.deploy as d
    from flash.client.http import ClientError

    class _StreamResp:
        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'

    class _Context:
        def __enter__(self):
            return _StreamResp()

        def __exit__(self, *_exc):
            return False

    _erroring_stream_seams(monkeypatch, _Context())
    stream = d.chat_stream("run-1", [{"role": "user", "content": "hi"}])

    assert next(stream) == "partial"
    with pytest.raises(ClientError, match=r"terminal \[DONE\]"):
        next(stream)


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


def _erroring_stream_seams(monkeypatch, resp):
    """Point the chat_stream seams at a fake client whose stream() yields ``resp``."""
    import flash.serve.deploy as d

    class _FakeClient:
        def stream(self, method, url, **kwargs):
            return resp

    monkeypatch.setattr(d, "_stream_http_client", lambda: _FakeClient())
    monkeypatch.setattr(d, "serving_openai_base_url", lambda: "https://serve.example/v1")


def test_chat_stream_upstream_error_raises_before_first_chunk(monkeypatch):
    """An upstream 4xx/5xx raises at chat_stream() call time, not during iteration.

    The serving route wraps only the serve_chat_stream CALL in its try/except; by the time
    the body iterates, the 200 and headers are already flushed. The request and
    raise_for_status therefore must run inside chat_stream itself, and the upstream response
    must be closed on the way out.
    """
    import httpx

    import flash.serve.deploy as d

    exits = []

    class _ErrorResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            exits.append(exc)
            return False

        def raise_for_status(self):
            request = httpx.Request("POST", "https://serve.example/v1/chat/completions")
            response = httpx.Response(502, request=request)
            raise httpx.HTTPStatusError("bad gateway", request=request, response=response)

    _erroring_stream_seams(monkeypatch, _ErrorResp())

    with pytest.raises(httpx.HTTPStatusError):
        d.chat_stream("flash-7-abcd", [{"role": "user", "content": "hi"}])
    assert len(exits) == 1


class _MidstreamFailureResp:
    def __init__(self):
        self.exits = []
        self.headers = {"content-type": "text/event-stream"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exits.append(exc)
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}'
        raise RuntimeError("upstream connection lost")


def test_chat_stream_midstream_failure_raises_and_closes_upstream(monkeypatch):
    """A failure after the first chunk propagates out of the iterator and closes upstream.

    The propagating exception is what makes the serving route abort the chunked body, so the
    client sees a truncated transfer rather than a clean eof."""
    import flash.serve.deploy as d

    resp = _MidstreamFailureResp()
    _erroring_stream_seams(monkeypatch, resp)

    stream = d.chat_stream("flash-7-abcd", [{"role": "user", "content": "hi"}])
    assert next(stream) == "hi"
    with pytest.raises(RuntimeError, match="upstream connection lost"):
        next(stream)
    assert len(resp.exits) == 1


def test_chat_stream_close_without_iterating_closes_upstream(monkeypatch):
    """Closing the returned iterator before reading any chunk still releases the response.

    chat_stream opens the upstream connection eagerly, so the returned generator must already
    be running: close() on a never-started generator skips the finally that exits the httpx
    stream context."""
    import flash.serve.deploy as d

    resp = _MidstreamFailureResp()
    _erroring_stream_seams(monkeypatch, resp)

    stream = d.chat_stream("flash-7-abcd", [{"role": "user", "content": "hi"}])
    stream.close()
    assert len(resp.exits) == 1


def test_internal_key_is_stripped_on_cross_origin_redirect(monkeypatch):
    """A redirect off the serving origin must not carry the plane credential.

    httpx strips only Authorization and Cookie when a redirect changes origin, so the custom
    X-Freesolo-Internal-Key header would otherwise be forwarded verbatim to wherever a 302 from
    the serving host points. The client's request hook has to drop it before the hop is sent.
    """
    import httpx

    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")

    seen: list[tuple[str, str | None]] = []

    def handler(request):
        seen.append((request.url.host, request.headers.get("X-Freesolo-Internal-Key")))
        if request.url.host == "serve.example":
            return httpx.Response(302, headers={"Location": "https://attacker.example/capture"})
        return httpx.Response(200, json={})

    with d._new_serving_client(transport=httpx.MockTransport(handler)) as client:
        resp = client.get("https://serve.example/v1/adapters", headers=d._internal_key_header())

    assert resp.status_code == 200
    # first hop (serving origin) carries the key; the redirected hop must not.
    assert seen == [("serve.example", "secret-internal"), ("attacker.example", None)]


def test_internal_key_survives_same_origin_redirect_polls(monkeypatch):
    """Modal's async-result poll flow (same-origin 303s) must keep working with the key."""
    import httpx

    import flash.serve.deploy as d

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "secret-internal")

    seen: list[tuple[str, str | None]] = []

    def handler(request):
        seen.append((request.url.path, request.headers.get("X-Freesolo-Internal-Key")))
        if request.url.path == "/v1/slow":
            return httpx.Response(303, headers={"Location": "https://serve.example/v1/poll"})
        return httpx.Response(200, json={})

    with d._new_serving_client(transport=httpx.MockTransport(handler)) as client:
        resp = client.get("https://serve.example/v1/slow", headers=d._internal_key_header())

    assert resp.status_code == 200
    assert seen == [("/v1/slow", "secret-internal"), ("/v1/poll", "secret-internal")]


def test_serving_clients_bound_redirect_chains(monkeypatch):
    """The redirect cap covers the 30-minute chat window of same-origin polls, but is finite.

    modal emits a 303 poll hop roughly every 150s, so the cap must allow at least
    30 min / 150 s hops or slow cold starts fail with TooManyRedirects short of the
    configured chat timeout. credential scoping does not rely on this cap: the request
    hook strips the internal key on every off-origin hop.
    """
    import httpx

    import flash.serve.deploy as d

    # sized for the 30-minute chat timeout at one poll hop per ~150s, with margin.
    assert d._MAX_REDIRECTS * 150 >= 30 * 60

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")

    def handler(request):
        return httpx.Response(302, headers={"Location": "https://serve.example/v1/again"})

    with d._new_serving_client(transport=httpx.MockTransport(handler)) as client:
        assert client.max_redirects == d._MAX_REDIRECTS
        with pytest.raises(httpx.TooManyRedirects):
            client.get("https://serve.example/v1/loop")
