"""In-process CLI coverage: every read/manage command against a fake ApiClient.

`flash login`/`flash train` subprocess flows live in test_cli_managed.py; these tests
drive main() directly so the table rendering, exit codes, and client wiring of
the remaining commands are covered without a server.
"""

from __future__ import annotations

import io

import pytest

import flash.cli as cli


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.log_text = "hello from the worker\n"

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
            "logs": self.log_text,
            "offset": 22,
            "state": "done",
        }

    def get_worker_output(self, run_id: str) -> dict[str, str]:
        return {"console_sft.txt": "worker stdout line\n"}

    def cancel_run(self, run_id: str) -> dict:
        self.calls.append(("cancel", run_id))
        return {"run_id": run_id, "state": "cancelled"}

    def deploy(self, run_id: str, **_) -> dict:
        self.calls.append(("deploy", run_id))
        return {"run_id": run_id, "openai_model": f"flash-{run_id}"}

    def undeploy(self, run_id: str) -> dict:
        self.calls.append(("undeploy", run_id))
        return {"run_id": run_id, "deleted_endpoints": ["live-x"]}

    def deployments(self) -> list[dict]:
        return [{"run_id": "flash-1", "deployment": {"gpu": "RTX 4090"}}]

    def chat(self, run_id: str, messages: list[dict], **_) -> dict:
        self.calls.append(("chat", run_id, messages))
        return {"choices": [{"message": {"content": "42"}}]}

    def chat_stream(self, run_id: str, messages: list[dict], **_):
        self.calls.append(("chat_stream", run_id, messages))
        yield "4"
        yield "2"


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
    # Rendered as a human card, not raw JSON.
    assert "{" not in out


def test_login_shows_who_you_are(monkeypatch, capsys) -> None:
    # Verify + store are stubbed; login should still surface the identity card itself so the
    # user sees who they are without a separate `flash whoami`. The card is built from the
    # just-verified key via ApiClient, so stub that (not client_from_config).
    monkeypatch.setattr(cli.commands, "verify_freesolo_key", lambda *a, **k: None)
    monkeypatch.setattr(cli.commands, "save_credentials", lambda *a, **k: None)
    monkeypatch.setattr(
        cli.commands,
        "ApiClient",
        lambda *a, **k: type(
            "_C", (), {"me": lambda self: {"key_prefix": "freesolo", "email": "t@example.com"}}
        )(),
    )
    assert _run(["login", "--api-key", "fs-secret-key"]) == 0
    out = capsys.readouterr().out
    assert "logged in to flash" in out
    assert "t@example.com" in out
    assert "fs-secret-key" not in out  # never echo the key


def test_login_failure_is_friendly_and_asks_to_retry(monkeypatch, capsys) -> None:
    from flash.client import ClientError

    def _reject(api_key, base_url=None):
        raise ClientError("freesolo rejected this API key")

    monkeypatch.setattr(cli.commands, "verify_freesolo_key", _reject)
    assert _run(["login", "--api-key", "bad-key"]) == 1
    err = capsys.readouterr().err
    assert "login failed" in err
    assert "try again" in err
    assert "founders@freesolo.co" in err
    assert "bad-key" not in err


def test_identity_render_is_ascii_locale_safe(monkeypatch) -> None:
    # Under an ASCII / non-UTF-8 stdout, neither a non-ASCII identity value nor our own
    # punctuation may raise UnicodeEncodeError after a login has already succeeded.
    from flash.cli import render

    class _AsciiStdout:
        encoding = "ascii"

        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(render.sys, "stdout", _AsciiStdout())
    card = render.whoami({"key_prefix": "fs", "email": "tëst@example.com"})
    fallback = render.login_ok(None)
    for text in (card, fallback):
        text.encode("ascii")  # raises if any non-ASCII slipped through
    assert "run `flash whoami`" in fallback


def test_models_table(fake_client, capsys) -> None:
    assert _run(["models"]) == 0
    out = capsys.readouterr().out
    # every catalog model is listed (no experimental/hidden tier)
    assert "Qwen/Qwen3.5-0.8B" in out
    assert "Qwen/Qwen3.5-9B" in out
    assert "Qwen/Qwen3.5-2B" in out
    assert "openbmb/MiniCPM5-1B" in out
    # only bare model ids, none of the extra per-model detail columns
    assert "2.3B" not in out
    assert "dense" not in out
    assert "(text-only fine-tune)" not in out
    assert "algos=" not in out
    assert "bf16" not in out
    assert "thinking=" not in out


def test_gpus_tip_omits_config_knobs(fake_client, capsys) -> None:
    assert _run(["gpus"]) == 0
    out = capsys.readouterr().out
    assert "GPU class selection is fully automatic" in out
    assert "You can still tune" not in out
    assert "[gpu] config table" not in out


def test_status_runs_and_status_logs(fake_client, capsys) -> None:
    assert _run(["status", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert "done" in out
    assert "cost_usd" in out

    assert _run(["runs"]) == 0
    out = capsys.readouterr().out
    assert "ALGO" in out
    assert "flash-1" in out
    assert "done" in out
    assert "SFT" in out

    assert _run(["status", "flash-1", "--logs"]) == 0
    out = capsys.readouterr().out
    assert "hello from the worker" in out
    # --logs always appends the real train-subprocess stdout fetched from the run's HF repo.
    assert "----- console_sft.txt -----" in out
    assert "worker stdout line" in out
    assert "cost_usd" in out

    assert _run(["status", "flash-1", "--follow"]) == 0
    out = capsys.readouterr().out
    assert "hello from the worker" in out
    assert "cost_usd" in out


def test_status_logs_separates_partial_log_line_from_json(fake_client, capsys) -> None:
    fake_client.log_text = "partial log line"
    fake_client.get_worker_output = lambda run_id: {}  # isolate: log->status separation only

    assert _run(["status", "flash-1", "--logs"]) == 0
    out = capsys.readouterr().out
    assert "partial log line\n{" in out


def test_follow_logs_shows_tty_spinner_while_waiting(monkeypatch, capsys) -> None:
    class _TTYBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    class _WaitingClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.pages = iter(
                [
                    {"run_id": "flash-spin", "logs": "", "offset": 0, "state": "queued"},
                    {
                        "run_id": "flash-spin",
                        "logs": "worker ready\n",
                        "offset": 13,
                        "state": "done",
                    },
                ]
            )

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            return next(self.pages)

    stderr = _TTYBuffer()
    monkeypatch.setattr(cli.commands.sys, "stderr", stderr)
    monkeypatch.setattr(cli.commands.time, "sleep", lambda _seconds: None)

    state = cli.commands._poll_logs(_WaitingClient(), "flash-spin", interval=0.2)

    assert state == "done"
    assert capsys.readouterr().out == "worker ready\n"
    err = stderr.getvalue()
    assert "following logs for flash-spin (queued)" in err
    assert "\r" in err
    assert err.endswith("\r")


@pytest.mark.parametrize("removed", ["cost", "attach", "logs"])
def test_legacy_run_commands_removed(fake_client, removed) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run([removed, "flash-1"])
    assert excinfo.value.code == 2


def test_cancel_deploy_undeploy_deployments(fake_client, capsys) -> None:
    assert _run(["cancel", "flash-1"]) == 0
    assert ("cancel", "flash-1") in fake_client.calls

    assert _run(["deploy", "flash-1"]) == 0
    assert ("deploy", "flash-1") in fake_client.calls

    assert _run(["deployments"]) == 0
    assert "flash-1" in capsys.readouterr().out

    assert _run(["undeploy", "flash-1"]) == 0
    assert ("undeploy", "flash-1") in fake_client.calls


def test_chat_sends_message_and_prints_reply(fake_client, capsys) -> None:
    assert _run(["chat", "flash-1", "-m", "What is 6*7?"]) == 0
    assert "42" in capsys.readouterr().out
    assert fake_client.calls[-1][0] == "chat_stream"


def test_env_setup_scaffolds_grpo_and_sft_configs(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert _run(["env", "setup"]) == 0

    assert (tmp_path / "environment.py").is_file()
    dataset = tmp_path / "datasets/train.jsonl"
    assert dataset.is_file()
    assert '"input":"What is 2 + 2?"' in dataset.read_text()
    grpo = tmp_path / "configs/rl.toml"
    sft = tmp_path / "configs/sft.toml"
    assert grpo.is_file()
    assert sft.is_file()
    assert not (tmp_path / "configs/endpoints.toml").exists()
    assert 'algorithm = "grpo"' in grpo.read_text()
    assert "steps = 150" in grpo.read_text()
    assert 'algorithm = "sft"' in sft.read_text()
    assert "epochs = 1" in sft.read_text()
    training = tmp_path / "TRAINING.md"
    assert training.is_file()
    training_text = training.read_text(encoding="utf-8")
    assert "how to actually improve a model with Flash" in training_text
    assert "## Using Flash" in training_text  # end-to-end library usage, not just conventions
    out = capsys.readouterr().out
    assert "datasets/train.jsonl" in out
    assert "configs/rl.toml" in out
    assert "TRAINING.md" in out


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

    # An unrecorded env resolves to the Freesolo SDK; the env is loaded lazily by the worker.
    spec = JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        environment=EnvironmentSpec(id="owner/env"),
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
