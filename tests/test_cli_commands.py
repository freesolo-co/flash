"""In-process CLI coverage: every read/manage command against a fake ApiClient.

`flash login`/`flash train` subprocess flows live in test_cli_managed.py; these tests
drive main() directly so the table rendering, exit codes, and client wiring of
the remaining commands are covered without a server.
"""

from __future__ import annotations

import pytest

from flash.cli import main as cli


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def me(self) -> dict:
        return {"key_prefix": "freesolo", "email": "t@example.com"}

    def models(self, include_experimental: bool = False) -> list[dict]:
        rows = [
            {
                "id": "Qwen/Qwen3.5-0.8B",
                "display_name": "Qwen3 0.6B",
                "params": "0.6B dense",
                "algos": ["sft", "grpo"],
                "min_vram_gb": 12,
                "quant": "bf16",
                "recommended_gpu": "RTX 4090",
                "experimental": False,
                "notes": "",
            }
        ]
        if include_experimental:
            rows.append({**rows[0], "id": "test/experimental-model", "experimental": True})
        return rows

    def list_runs(self) -> list[dict]:
        return [
            {
                "run_id": "flash-1",
                "state": "done",
                "cost_usd": 0.25,
                "updated_at": 1700000000.0,
                "spec": {"model": "Qwen/Qwen3.5-0.8B", "algorithm": "sft"},
            }
        ]

    def get_run(self, run_id: str) -> dict:
        return {
            "run_id": run_id,
            "state": "done",
            "cost_usd": 0.25,
            "error": None,
            "spec": {"model": "Qwen/Qwen3.5-0.8B"},
        }

    def get_logs(self, run_id: str, offset: int = 0) -> dict:
        return {
            "run_id": run_id,
            "logs": "hello from the worker\n",
            "offset": 22,
            "state": "done",
        }

    def cancel_run(self, run_id: str) -> dict:
        self.calls.append(("cancel", run_id))
        return {"run_id": run_id, "state": "cancelled"}

    def deploy(self, run_id: str, mode: str = "dev", idle_timeout_s: int = 300, **_) -> dict:
        self.calls.append(("deploy", run_id, mode, idle_timeout_s))
        return {"run_id": run_id, "mode": mode, "openai_model": f"flash-{run_id}"}

    def undeploy(self, run_id: str) -> dict:
        self.calls.append(("undeploy", run_id))
        return {"run_id": run_id, "deleted_endpoints": ["live-x"]}

    def deployments(self) -> list[dict]:
        return [{"run_id": "flash-1", "mode": "dev", "gpu": "RTX 4090"}]

    def chat(self, run_id: str, messages: list[dict], **_) -> dict:
        self.calls.append(("chat", run_id, messages))
        return {"choices": [{"message": {"content": "42"}}]}


@pytest.fixture
def fake_client(monkeypatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(cli.commands, "client_from_config", lambda *a, **k: client)
    return client


def _run(argv: list[str]) -> int:
    return cli.main(argv)


def test_whoami_prints_identity(fake_client, capsys) -> None:
    assert _run(["whoami"]) == 0
    out = capsys.readouterr().out
    assert "freesolo" in out
    assert "t@example.com" in out


def test_models_table(fake_client, capsys) -> None:
    assert _run(["models"]) == 0
    out = capsys.readouterr().out
    # every catalog model is listed (no experimental/hidden tier)
    assert "Qwen/Qwen3.5-0.8B" in out
    assert "Qwen/Qwen3.5-9B" in out


def test_status_ps_cost_and_logs(fake_client, capsys) -> None:
    assert _run(["status", "flash-1"]) == 0
    assert "done" in capsys.readouterr().out

    assert _run(["ps"]) == 0
    out = capsys.readouterr().out
    assert "flash-1" in out
    assert "done" in out

    assert _run(["cost", "flash-1"]) == 0
    assert "0.25" in capsys.readouterr().out

    assert _run(["logs", "flash-1"]) == 0
    assert "hello from the worker" in capsys.readouterr().out


def test_cancel_deploy_undeploy_deployments(fake_client, capsys) -> None:
    assert _run(["cancel", "flash-1"]) == 0
    assert ("cancel", "flash-1") in fake_client.calls

    assert _run(["deploy", "flash-1", "--mode", "dev", "--idle-timeout", "120"]) == 0
    assert ("deploy", "flash-1", "dev", 120) in fake_client.calls

    assert _run(["deployments"]) == 0
    assert "flash-1" in capsys.readouterr().out

    assert _run(["undeploy", "flash-1"]) == 0
    assert ("undeploy", "flash-1") in fake_client.calls


def test_chat_sends_message_and_prints_reply(fake_client, capsys) -> None:
    assert _run(["chat", "flash-1", "-m", "What is 6*7?"]) == 0
    assert "42" in capsys.readouterr().out
    assert fake_client.calls[-1][0] == "chat"


def test_env_setup_scaffolds_grpo_and_sft_configs(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert _run(["env", "setup"]) == 0

    assert (tmp_path / "environments/starter/environment.py").is_file()
    grpo = tmp_path / "configs/grpo.toml"
    sft = tmp_path / "configs/sft.toml"
    assert grpo.is_file()
    assert sft.is_file()
    assert 'algorithm = "grpo"' in grpo.read_text()
    assert "steps = 150" in grpo.read_text()
    assert 'algorithm = "sft"' in sft.read_text()
    assert "epochs = 1" in sft.read_text()
    assert "configs/grpo.toml" in capsys.readouterr().out


def test_unknown_run_errors_surface_as_nonzero_exit(monkeypatch, capsys) -> None:
    from flash.client import ApiError

    class _Erroring(_FakeClient):
        def get_run(self, run_id: str) -> dict:
            raise ApiError(404, "unknown run")

    monkeypatch.setattr(cli.commands, "client_from_config", lambda *a, **k: _Erroring())
    assert _run(["status", "nope"]) != 0
    assert "unknown run" in capsys.readouterr().err


def test_spec_payload_resolves_worker_pip(monkeypatch, tmp_path) -> None:
    from flash.client.specs import spec_payload
    from flash.spec import EnvironmentSpec, JobSpec

    # An unrecorded env resolves to the Freesolo SDK; the GitHub ref is loaded lazily by the
    # worker at environment load time.
    spec = JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        environment=EnvironmentSpec(id="github:owner/repo@main:env/environment.py"),
    )
    assert spec_payload(spec)["environment"]["pip"] == ["freesolo"]

    # ...and an explicit pip list (the documented escape hatch) wins untouched.
    spec = JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        environment=EnvironmentSpec(
            id="github:owner/repo@main:env/environment.py", pip=("custom==1",)
        ),
    )
    assert list(spec_payload(spec)["environment"]["pip"]) == ["custom==1"]


# ---------------------------------------------------------------------------
# MCP bridge: tool dispatch over the same client surface
# ---------------------------------------------------------------------------


def test_mcp_handle_dispatches_tools(monkeypatch, fake_client, capsys) -> None:
    from flash.mcp import server as mcp

    monkeypatch.setattr(mcp, "client_from_config", lambda *a, **k: fake_client, raising=False)

    models = mcp.handle({"tool": "list_models", "args": {}})
    assert any("Qwen" in str(m) for m in str(models).split(","))

    status = mcp.handle({"tool": "get_run_status", "args": {"run_id": "flash-1"}})
    assert status["state"] == "done"

    logs = mcp.handle({"tool": "get_run_logs", "args": {"run_id": "flash-1"}})
    assert "hello from the worker" in str(logs)

    with pytest.raises(ValueError, match="unknown tool"):
        mcp.handle({"tool": "nope"})
