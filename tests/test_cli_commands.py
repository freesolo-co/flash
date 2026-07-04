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
        logs = self.log_text[max(0, int(offset)) :]
        return {
            "run_id": run_id,
            "logs": logs,
            "offset": len(self.log_text),
            "state": "done",
        }

    def get_worker_output(self, run_id: str) -> dict[str, str]:
        return {"console_sft.txt": "worker stdout line\n"}

    def cancel_run(self, run_id: str) -> dict:
        self.calls.append(("cancel", run_id))
        return {"run_id": run_id, "state": "cancelled"}

    def deploy(self, run_id: str, **kwargs) -> dict:
        self.calls.append(("deploy", run_id, kwargs))
        return {
            "run_id": run_id,
            "openai_model": f"flash-{run_id}",
            "endpoint_name": "https://serve.example",
            "state": "deploying",
        }

    def undeploy(self, run_id: str) -> dict:
        self.calls.append(("undeploy", run_id))
        return {"run_id": run_id, "deleted_endpoints": ["live-x"]}

    def deployments(self) -> list[dict]:
        return [{"run_id": "flash-1", "deployment": {"state": "ready"}}]

    def chat(self, run_id: str, messages: list[dict], **_) -> dict:
        self.calls.append(("chat", run_id, messages))
        return {"choices": [{"message": {"content": "42"}}]}

    def chat_stream(self, run_id: str, messages: list[dict], **_):
        self.calls.append(("chat_stream", run_id, messages))
        yield "4"
        yield "2"

    def export(
        self,
        run_id: str,
        *,
        repository: str,
        hf_token: str,
        private: bool = True,
    ) -> dict:
        self.calls.append(("export", run_id, repository, hf_token, private))
        return {
            "run_id": run_id,
            "adapter_id": run_id,
            "repository": repository,
            "url": f"https://huggingface.co/{repository}",
            "source": "org/runs:rl/x/seed0/adapter",
        }


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
    assert "cheapest validated managed class" in out
    assert "runpod" not in out.lower()
    assert "lambda" not in out.lower()
    assert "You can still tune" not in out
    assert "[gpu] config table" not in out


def test_status_runs_and_log_command(fake_client, capsys) -> None:
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

    assert _run(["status", "flash-1", "--follow"]) == 0
    out = capsys.readouterr().out
    assert "cost_usd" in out
    assert "hello from the worker" not in out

    assert _run(["log", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert "hello from the worker" in out
    assert "----- console_sft.txt -----" in out
    assert "worker stdout line" in out
    assert "cost_usd" not in out


def test_log_prints_partial_log_line_with_newline(fake_client, capsys) -> None:
    fake_client.log_text = "partial log line"
    fake_client.get_worker_output = lambda run_id: {}

    assert _run(["log", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert out == "partial log line\n"


def test_log_snapshot_reads_one_offset_page_without_status(fake_client, capsys) -> None:
    calls = []
    pages = {
        0: {"run_id": "flash-1", "logs": "first\n", "offset": 6, "state": "running"},
        6: {"run_id": "flash-1", "logs": "second\n", "offset": 13, "state": "done"},
        13: {"run_id": "flash-1", "logs": "", "offset": 13, "state": "done"},
    }

    def get_logs(run_id: str, offset: int = 0) -> dict:
        calls.append(offset)
        return pages[offset]

    fake_client.get_logs = get_logs
    fake_client.get_worker_output = lambda run_id: {}

    assert _run(["log", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert out == "first\n"
    assert calls == [0]


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
            self.statuses = iter(
                [
                    {"run_id": "flash-spin", "state": "queued"},
                    {"run_id": "flash-spin", "state": "done"},
                ]
            )

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            return next(self.pages)

        def get_run(self, run_id: str) -> dict:
            return next(self.statuses)

    stderr = _TTYBuffer()
    monkeypatch.setattr(cli.commands.sys, "stderr", stderr)
    monkeypatch.setattr(cli.commands.time, "sleep", lambda _seconds: None)

    state, printed_any = cli.commands._poll_logs(_WaitingClient(), "flash-spin", interval=0.2)

    assert state == "done"
    assert printed_any is True
    assert capsys.readouterr().out == "worker ready\n"
    err = stderr.getvalue()
    assert "following logs for flash-spin (queued)" in err
    assert "\r" in err
    assert err.endswith("\r")


def test_follow_logs_uses_status_progress_when_log_tail_lags(monkeypatch, capsys) -> None:
    class _TTYBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    class _LaggingLogClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.statuses = iter(
                [
                    {
                        "run_id": "flash-lag",
                        "state": "running",
                        "last_heartbeat": {"stage": "rl_step", "step": 42},
                        "realized_cost_usd": 1.23456,
                    },
                    {
                        "run_id": "flash-lag",
                        "state": "done",
                        "last_heartbeat": {"stage": "rl_train_done"},
                        "realized_cost_usd": 1.5,
                    },
                ]
            )

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            # Stale/lossy log stream: no bytes and a non-terminal page state forever.
            return {"run_id": run_id, "logs": "", "offset": 0, "state": "running"}

        def get_run(self, run_id: str) -> dict:
            return next(self.statuses)

    stderr = _TTYBuffer()
    monkeypatch.setattr(cli.commands.sys, "stderr", stderr)
    monkeypatch.setattr(cli.commands.time, "sleep", lambda _seconds: None)

    state, printed_any = cli.commands._poll_logs(_LaggingLogClient(), "flash-lag", interval=0.2)

    assert state == "done"
    assert printed_any is False
    assert capsys.readouterr().out == ""
    err = stderr.getvalue()
    assert "stage=rl_step" in err
    assert "step=42" in err
    assert "realized_cost=$1.2346" in err


def test_cancel_deploy_undeploy_deployments(fake_client, capsys) -> None:
    assert _run(["cancel", "flash-1"]) == 0
    assert ("cancel", "flash-1") in fake_client.calls

    assert _run(["deploy", "flash-1"]) == 0
    assert ("deploy", "flash-1", {"dry_run": False, "verify": True}) in fake_client.calls

    assert _run(["deploy", "flash-1/step-40"]) == 0
    assert ("deploy", "flash-1/step-40", {"dry_run": False, "verify": True}) in fake_client.calls
    err = capsys.readouterr().err
    assert "flash undeploy flash-1`" in err
    assert "flash undeploy flash-1/step-40`" not in err

    assert _run(["deployments"]) == 0
    assert "flash-1" in capsys.readouterr().out

    assert _run(["undeploy", "flash-1"]) == 0
    assert ("undeploy", "flash-1") in fake_client.calls


def test_chat_sends_message_and_prints_reply(fake_client, capsys) -> None:
    assert _run(["chat", "flash-1", "-m", "What is 6*7?"]) == 0
    assert "42" in capsys.readouterr().out
    assert fake_client.calls[-1][0] == "chat_stream"


def test_chat_checkpoint_ref_uses_base_run_id(fake_client) -> None:
    assert _run(["chat", "flash-1/step-40", "-m", "What is 6*7?"]) == 0
    assert fake_client.calls[-1][0] == "chat_stream"
    assert fake_client.calls[-1][1] == "flash-1"


def test_chat_system_flag_prepends_system_message(fake_client) -> None:
    """--system gives evals training-prompt parity without calling the HTTP API directly."""
    assert _run(["chat", "flash-1", "-m", "What is 6*7?", "--system", "be brief"]) == 0
    _, _, messages = fake_client.calls[-1]
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "What is 6*7?"},
    ]


def test_chat_without_system_flag_sends_user_message_only(fake_client) -> None:
    assert _run(["chat", "flash-1", "-m", "What is 6*7?"]) == 0
    _, _, messages = fake_client.calls[-1]
    assert messages == [{"role": "user", "content": "What is 6*7?"}]


@pytest.mark.parametrize("flag", ["--enable-thinking", "--disable-thinking"])
def test_chat_does_not_expose_thinking_override_flags(fake_client, flag) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(["chat", "flash-1", "-m", "What is 6*7?", flag])
    assert excinfo.value.code == 2


def test_env_setup_scaffolds_grpo_and_sft_configs(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert _run(["env", "setup"]) == 0

    assert (tmp_path / "environment.py").is_file()
    dataset = tmp_path / "dataset/train.jsonl"
    assert dataset.is_file()
    assert not (tmp_path / "datasets").exists()
    assert '"input":"What is 2 + 2?"' in dataset.read_text()
    grpo = tmp_path / "configs/rl.toml"
    sft = tmp_path / "configs/sft.toml"
    assert grpo.is_file()
    assert sft.is_file()
    assert not (tmp_path / "configs/endpoints.toml").exists()
    assert 'algorithm = "grpo"' in grpo.read_text()
    assert "steps = 150" in grpo.read_text()
    assert "cheapest fitting managed class" in grpo.read_text()
    assert "private environment-scoped repo" in grpo.read_text()
    assert 'algorithm = "sft"' in sft.read_text()
    assert "epochs = 1" in sft.read_text()
    assert "max_examples = 2" in sft.read_text()
    assert "cheapest fitting managed class" in sft.read_text()
    assert "private environment-scoped repo" in sft.read_text()
    training = tmp_path / "TRAINING.md"
    assert training.is_file()
    training_text = training.read_text(encoding="utf-8")
    assert "how to actually improve a model with Flash" in training_text
    assert "## Using Flash" in training_text  # end-to-end library usage, not just conventions
    assert "## Common Flash issues and mitigations" in training_text
    assert "Trying to pin managed infrastructure" in training_text
    assert "response_text.thinking" in training_text
    assert "Qwen3.5 thinking multi-turn SFT" in training_text
    assert "longest shared token prefix" in training_text
    assert "flash env pull your-org/my-env" in training_text
    assert "private environment-scoped repo" in training_text
    assert "flash checkpoints <run-id>" in training_text
    assert "flash deployments" in training_text
    assert "flash export --adapter-id <run-id> --repository <you>/<repo>" in training_text
    assert "HF_TOKEN" in training_text
    assert "runpod" not in training_text.lower()
    assert "lambda" not in training_text.lower()
    out = capsys.readouterr().out
    assert "dataset/train.jsonl" in out
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
    assert spec_payload(spec)["environment"]["pip"] == ["freesolo>=0.2.54"]

    # ...and an explicit pip list (the documented escape hatch) wins untouched.
    spec = JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        environment=EnvironmentSpec(
            id="github:owner/repo@main:env/environment.py", pip=("custom==1",)
        ),
    )
    assert list(spec_payload(spec)["environment"]["pip"]) == ["custom==1"]


def test_export_uses_api_key_flag_and_forwards_args(fake_client, capsys, monkeypatch) -> None:
    # The --api-key flag is the destination HF token; checkpoint refs and --public are forwarded.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert (
        _run(
            [
                "export",
                "--adapter-id",
                "flash-1/step-40",
                "--repository",
                "me/adapters",
                "--api-key",
                "hf_flag",
                "--public",
            ]
        )
        == 0
    )
    assert ("export", "flash-1/step-40", "me/adapters", "hf_flag", False) in fake_client.calls
    # The destination repo / url are reported back to the user.
    out = capsys.readouterr().out
    assert "me/adapters" in out


def test_export_reads_hf_token_from_env_and_defaults_private(fake_client, monkeypatch) -> None:
    # No --api-key: the token resolves from HF_TOKEN, and the repo defaults to private.
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert _run(["export", "--adapter-id", "flash-1", "--repository", "me/adapters"]) == 0
    assert ("export", "flash-1", "me/adapters", "hf_env", True) in fake_client.calls


def test_export_without_token_errors_cleanly(fake_client, monkeypatch, capsys, tmp_path) -> None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # A clean cwd so a stray local .env can't supply a token.
    monkeypatch.chdir(tmp_path)
    assert _run(["export", "--adapter-id", "flash-1", "--repository", "me/adapters"]) == 1
    err = capsys.readouterr().err
    assert "HuggingFace token" in err
    # The control plane is never contacted when there's no token to send.
    assert not any(call[0] == "export" for call in fake_client.calls)


def test_deploy_enqueues_server_side_verification(fake_client, capsys) -> None:
    assert _run(["deploy", "flash-1"]) == 0
    assert ("deploy", "flash-1", {"dry_run": False, "verify": True}) in fake_client.calls
    assert not any(c[0] == "chat" for c in fake_client.calls)
    err = capsys.readouterr().err
    assert "flash deployments" in err
    assert "OpenAI-compatible base URL" in err


def test_deploy_checkpoint_enqueues_base_run_deployment(fake_client) -> None:
    assert _run(["deploy", "flash-1/step-40"]) == 0
    assert ("deploy", "flash-1/step-40", {"dry_run": False, "verify": True}) in fake_client.calls
    assert not any(c[0] == "chat" for c in fake_client.calls)


def test_deploy_no_verify_skips_server_smoke(fake_client, capsys) -> None:
    assert _run(["deploy", "flash-1", "--no-verify"]) == 0
    assert ("deploy", "flash-1", {"dry_run": False, "verify": False}) in fake_client.calls
    assert not any(c[0] == "chat" for c in fake_client.calls)
    assert "smoke verification was skipped" in capsys.readouterr().err


def test_deploy_dry_run_skips_active_deployment_note(fake_client, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "dry_run"},
        raising=False,
    )
    assert _run(["deploy", "flash-1", "--dry-run"]) == 0
    assert not any(c[0] == "chat" for c in fake_client.calls)
    assert "flash deployments" not in capsys.readouterr().err


def test_deploy_failed_state_exits_nonzero(fake_client, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {
            "run_id": run_id,
            "state": "failed",
            "error": "smoke generation failed",
        },
        raising=False,
    )

    assert _run(["deploy", "flash-1"]) == 1
    err = capsys.readouterr().err
    assert "deployment failed: smoke generation failed" in err
    assert "once it is ready" not in err
