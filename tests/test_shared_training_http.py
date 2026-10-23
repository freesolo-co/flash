"""HTTP authorization and complete client-side recipes against a resident model."""

import os
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from flash.training.shared.client import Client
from flash.training.shared.contracts import Principal, UnavailableError
from flash.training.shared.http import create_app, token_authentication
from flash.training.shared.journal import Journal
from flash.training.shared.recipes import grpo_step, opd_step, sft_step
from flash.training.shared.runtime import Runtime

TOKEN_A = "a" * 40
TOKEN_B = "b" * 40


class HttpClient(Client):
    def __init__(self, http, token):
        self.http, self.token = http, token

    def request(self, method, path, payload=None):
        result = self.http.request(
            method, path, json=payload, headers={"Authorization": f"Bearer {self.token}"}
        )
        assert result.status_code == 200, result.text
        return result.json()


def test_all_three_recipes_share_base_and_preserve_http_ownership(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    from flash.training.shared.backend import PeftBackend

    torch.manual_seed(9)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            eos_token_id=31,
        )
    )
    base = base.to(os.environ.get("FLASH_SHARED_TEST_DEVICE", "cpu"))
    backend = PeftBackend(base, root=tmp_path / "weights", rank=2, alpha=4)
    journal = Journal(tmp_path / "journal.sqlite", vocab_size=32)
    runtime = Runtime(journal, backend, lock_path=tmp_path / "owner.lock")
    credentials = {TOKEN_A: {"tenant": "a", "owner": "a"}, TOKEN_B: {"tenant": "b", "owner": "b"}}
    app = create_app(runtime, token_authentication(credentials))
    with TestClient(app) as http:
        assert http.get("/usage").status_code == 401
        alice, bob = HttpClient(http, TOKEN_A), HttpClient(http, TOKEN_B)
        sft = alice.create("sft", seed=2, learning_rate=0.01)
        grpo = bob.create("grpo", seed=3, learning_rate=0.01)
        opd = alice.create("opd", seed=4, learning_rate=0.01)
        assert sft_step(sft, [([1, 2], [3, 4])])["version"] == 1
        foreign = http.get(sft.path, headers={"Authorization": f"Bearer {TOKEN_B}"})
        assert foreign.status_code == 404
        result = grpo_step(
            grpo, [[1, 2]], lambda prompt, tokens: sum(tokens), group_size=4, max_tokens=3, seed=42
        )
        assert result["version"] == 1

        def teacher(prompt, tokens):
            return list(range(len(tokens))), [-2.0] * len(tokens)

        assert opd_step(opd, [[1, 2]], teacher, max_tokens=3)["version"] == 1
        saved = sft.checkpoint()
        assert saved["version"] == 1
        sft.close()
        restored = alice.create("sft", seed=2, learning_rate=0.01)
        assert restored.restore(sft.id, 1)["version"] == 1
        assert sft_step(restored, [([1, 2], [3, 4])])["version"] == 2
        for session in (restored, grpo, opd):
            session.close()
        assert backend.adapters == {}
        used_a = alice.request("GET", "/usage")
        used_b = bob.request("GET", "/usage")
        assert used_a["execution_ns"] > 0
        assert used_b["execution_ns"] > 0
        assert (
            used_a["execution_ns"] + used_b["execution_ns"]
            == journal.pool_usage()["attributed_execution_ns"]
        )


def test_runtime_failure_fences_all_tenants_and_retains_usage(tmp_path):
    class BrokenBackend:
        def execute(self, command):
            raise RuntimeError("injected device loss")

    journal = Journal(tmp_path / "journal.sqlite")
    runtime = Runtime(journal, BrokenBackend(), lock_path=tmp_path / "owner.lock")
    runtime.start()
    try:
        a = journal.create(
            Principal("a", "a"), "a", {"algorithm": "sft", "seed": 0, "learning_rate": 0.001}
        )
        b = journal.create(
            Principal("b", "b"), "b", {"algorithm": "opd", "seed": 0, "learning_rate": 0.001}
        )
        runtime.wake.set()
        deadline = time.monotonic() + 5
        while (
            journal.status(Principal("b", "b"), b)["state"] != "lost"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert runtime.failed
        assert journal.status(Principal("a", "a"), a)["state"] == "lost"
        assert journal.status(Principal("b", "b"), b)["state"] == "lost"
        assert journal.pool_usage()["attributed_execution_ns"] > 0
        with pytest.raises(UnavailableError):
            runtime.accept("create", Principal("c", "c"), "c", {"algorithm": "sft"})
    finally:
        runtime.stop()


def test_second_runtime_cannot_recover_live_pool(tmp_path):
    class Backend:
        pass

    journal = Journal(tmp_path / "journal.sqlite")
    first = Runtime(journal, Backend(), lock_path=tmp_path / "owner.lock")
    second = Runtime(journal, Backend(), lock_path=tmp_path / "owner.lock")
    first.start()
    try:
        with pytest.raises(RuntimeError, match="another runtime"):
            second.start()
    finally:
        first.stop()


def test_standard_library_client_over_loopback_http(tmp_path):
    import uvicorn

    class Backend:
        def execute(self, command):
            return {"version": 0} if command["operation"] != "close" else {"closed": True}

        def synchronize(self):
            pass

    journal = Journal(tmp_path / "pool.sqlite")
    runtime = Runtime(journal, Backend(), lock_path=tmp_path / "owner.lock")
    app = create_app(runtime, token_authentication({TOKEN_A: {"tenant": "a", "owner": "a"}}))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"http://127.0.0.1:{listener.getsockname()[1]}"
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            client = Client(address, TOKEN_A)
            session = client.create("sft", key="transport-retry")
            assert client.create("sft", key="transport-retry").id == session.id
            assert session.checkpoint()["version"] == 0
            assert session.close() == {"closed": True}
            assert client.request("GET", "/usage")["execution_ns"] > 0
            with pytest.raises(RuntimeError, match="HTTP 401"):
                Client(address, TOKEN_B).request("GET", "/usage")
        finally:
            server.should_exit = True
            thread.join(timeout=5)
        assert not thread.is_alive()
