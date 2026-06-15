"""Tests for the Flash serving wiring (no GPU/network): dry-run + handler shaping."""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


def test_deploy_dry_run():
    from autoslm.serve.deploy import deploy_adapter

    dep = deploy_adapter(
        run_id="r1",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/r1/seed0",
        gpu_name="RTX 4090",
        dry_run=True,
    )
    d = dep.to_dict()
    assert d["state"] == "dry_run"
    assert d["gpu"] == "RTX 4090"
    assert d["openai_model"] == "autoslm-r1"
    assert d["endpoint_name"] == "autoslm-serve-4090-r1"  # per-run endpoint


def test_deploy_rejects_unsupported_gpu():
    from autoslm.providers.base import UnsupportedGpuError
    from autoslm.serve.deploy import deploy_adapter

    with pytest.raises(UnsupportedGpuError):
        deploy_adapter(
            run_id="r1",
            model="Qwen/Qwen3.5-0.8B",
            hf_repo="org/repo",
            adapter_prefix="sft/r1/seed0",
            gpu_name="TPU v5",  # junk still rejects
            dry_run=True,
        )


def test_deploy_unvalidated_class_falls_back_to_runpod():
    # Serving is RunPod-only: a run trained on a class that isn't RunPod-validated
    # (RTX 3090, 24 GB) serves from the cheapest RunPod-VALIDATED class with >= that
    # VRAM — not directly on the unvalidated class (which can fail on first chat).
    from autoslm.providers.base import GPU_INFO, is_validated
    from autoslm.serve.deploy import deploy_adapter

    dep = deploy_adapter(
        run_id="r1",
        model="Qwen/Qwen3.5-0.8B",
        hf_repo="org/repo",
        adapter_prefix="sft/r1/seed0",
        gpu_name="RTX 3090",
        dry_run=True,
    )
    assert dep.gpu != "RTX 3090"
    assert is_validated(dep.gpu, "runpod"), "must serve on a RunPod-validated class"
    assert GPU_INFO[dep.gpu].vram_gb >= 24  # >= RTX 3090's VRAM


def _install_serve_mocks():
    hf = types.ModuleType("huggingface_hub")
    hf.snapshot_download = lambda **k: None
    sys.modules["huggingface_hub"] = hf

    tfm = types.ModuleType("transformers")

    class _Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **k):
            return msgs[-1]["content"]

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    tfm.AutoTokenizer = _Tok
    sys.modules["transformers"] = tfm

    vllm = types.ModuleType("vllm")

    class _O:
        text = "hi there"
        token_ids = (1, 2, 3)

    class _R:
        outputs = (_O(),)
        prompt_token_ids = (1, 2)

    class LLM:
        def __init__(self, *a, **k):
            pass

        def generate(self, prompts, sp, lora_request=None):
            return [_R() for _ in prompts]

    class SamplingParams:
        def __init__(self, *a, **k):
            pass

    vllm.LLM = LLM
    vllm.SamplingParams = SamplingParams
    lora_mod = types.ModuleType("vllm.lora")
    req_mod = types.ModuleType("vllm.lora.request")

    class LoRARequest:
        def __init__(self, *a, **k):
            pass

    req_mod.LoRARequest = LoRARequest
    sys.modules["vllm"] = vllm
    sys.modules["vllm.lora"] = lora_mod
    sys.modules["vllm.lora.request"] = req_mod


def _openai_completion_bytes():
    import json as _json

    return _json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "adapter",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi there"},
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode()


def test_serve_body_openai_shape():
    """The serve handler boots a local vLLM OpenAI server (subprocess) and proxies to it.
    Here a stdlib HTTP server stands in for vLLM; Popen/snapshot_download are mocked.

    Also covers thinking parity: the run's flag rides the chat_template_kwargs and is
    part of the engine key (a thinking deploy boots vLLM with a bigger --max-model-len)."""
    import json as _json
    import sys
    import threading
    import types
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen_bodies = []
    boot_cmds = []

    class _VllmStub(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen_bodies.append(_json.loads(self.rfile.read(n)))
            body = _openai_completion_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 8199), _VllmStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **k: "/tmp"
    sys.modules["huggingface_hub"] = hub

    import subprocess as _sub

    class _FakeProc:
        def poll(self):
            return None

        def kill(self):
            pass

    orig_popen = _sub.Popen

    def _fake_popen(cmd, *a, **k):
        boot_cmds.append(list(cmd))
        return _FakeProc()

    _sub.Popen = _fake_popen
    payload = {
        "hf_repo": "org/repo",
        "model": "Qwen/Qwen3.5-0.8B",
        "adapter_prefix": "sft/r1/seed0",
        "token": "x",
        "served_model": "autoslm-r1",
        "messages": [{"role": "user", "content": "2+2?"}],
        "max_tokens": 8,
    }
    try:
        import autoslm.serve.deploy as d

        for k in ("_AUTOSLM_KEY", "_AUTOSLM_PROC"):
            d.__dict__.pop(k, None)
        resp = d._serve_body(dict(payload))
        resp_thinking = d._serve_body({**payload, "thinking": True})
    finally:
        _sub.Popen = orig_popen
        server.shutdown()
        sys.modules.pop("huggingface_hub", None)
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "autoslm-r1"
    assert resp["choices"][0]["message"]["content"] == "hi there"
    assert resp["choices"][0]["message"]["role"] == "assistant"
    # thinking disabled by default; the run's flag turns it on
    assert seen_bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen_bodies[1]["chat_template_kwargs"] == {"enable_thinking": True}
    assert resp_thinking["object"] == "chat.completion"
    # the thinking flag is part of the engine key: a second boot with more context
    assert len(boot_cmds) == 2
    assert boot_cmds[0][boot_cmds[0].index("--max-model-len") + 1] == "2048"
    assert boot_cmds[1][boot_cmds[1].index("--max-model-len") + 1] == "4096"


def test_serve_body_drops_template_kwargs_on_old_vllm():
    """Older vLLM 400s on chat_template_kwargs; the handler retries without it."""
    import json as _json
    import sys
    import threading
    import types
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen_bodies = []

    class _OldVllmStub(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = _json.loads(self.rfile.read(n))
            seen_bodies.append(body)
            if "chat_template_kwargs" in body:
                self.send_response(400)
                self.end_headers()
                return
            out = _openai_completion_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 8199), _OldVllmStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **k: "/tmp"
    sys.modules["huggingface_hub"] = hub

    import subprocess as _sub

    class _FakeProc:
        def poll(self):
            return None

        def kill(self):
            pass

    orig_popen = _sub.Popen
    _sub.Popen = lambda *a, **k: _FakeProc()
    try:
        import autoslm.serve.deploy as d

        for k in ("_AUTOSLM_KEY", "_AUTOSLM_PROC"):
            d.__dict__.pop(k, None)
        resp = d._serve_body(
            {
                "hf_repo": "org/repo",
                "model": "Qwen/Qwen3.5-0.8B",
                "adapter_prefix": "sft/r1/seed0",
                "token": "x",
                "served_model": "autoslm-r1",
                "messages": [{"role": "user", "content": "2+2?"}],
                "max_tokens": 8,
            }
        )
    finally:
        _sub.Popen = orig_popen
        server.shutdown()
        sys.modules.pop("huggingface_hub", None)
    assert resp["object"] == "chat.completion"
    assert len(seen_bodies) == 2
    assert "chat_template_kwargs" in seen_bodies[0]
    assert "chat_template_kwargs" not in seen_bodies[1]
