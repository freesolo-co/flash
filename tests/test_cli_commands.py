"""In-process CLI coverage: every read/manage command against a fake ApiClient.

`flash login`/`flash train` subprocess flows live in test_cli_managed.py; these tests
drive main() directly so the table rendering, exit codes, and client wiring of
the remaining commands are covered without a server.
"""

from __future__ import annotations

import io
import json
import types

import pytest

import flash.cli as cli
from flash.providers._poll import _format_heartbeat


def test_format_heartbeat_appends_named_reward_metrics() -> None:
    heartbeat = {"stage": "rl_step", "step": 4, "reward": 0.65}
    base_line = _format_heartbeat(heartbeat)

    assert base_line == "worker: stage=rl_step step=4 reward=0.650"
    assert _format_heartbeat({**heartbeat, "reward_metrics": {"success": 0.8, "format": 0.5}}) == (
        base_line + " success=0.800 format=0.500"
    )


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.log_text = "hello from the worker\n"

    def me(self) -> dict:
        return {"key_prefix": "freesolo", "email": "t@example.com"}

    def create_run(
        self,
        spec: dict,
        runtime_secrets=None,
        dry_run: bool = False,
        client_train_schema=None,
    ) -> dict:
        self.calls.append(("create_run", spec, runtime_secrets, dry_run, client_train_schema))
        response = {"run_id": "flash-dry", "state": "dry_run", "spec": spec}
        if dry_run:
            response["train_schema_compatibility"] = {"status": "agreement"}
        return response

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
                "spec": {
                    "project": "11111111-1111-4111-8111-111111111111",
                    "model": "Qwen/Qwen3.5-0.8B",
                    "algorithm": "sft",
                },
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

    def checkpoints(self, run_id: str) -> list[dict]:
        self.calls.append(("checkpoints", run_id))
        return [{"step": 20}, {"step": 40}]

    def deploy(self, run_id: str, **kwargs) -> dict:
        self.calls.append(("deploy", run_id, kwargs))
        return {
            "run_id": run_id,
            "openai_model": f"flash-{run_id}",
            "endpoint_name": "https://serve.example",
            "openai_base_url": "https://serve.example/v1",
            "state": "deploying",
        }

    def undeploy(self, run_id: str) -> dict:
        self.calls.append(("undeploy", run_id))
        return {"run_id": run_id, "deleted_endpoints": ["live-x"]}

    def deployments(self, timeout: float | None = None) -> list[dict]:
        self.calls.append(("deployments", timeout))
        return [
            {
                "run_id": "flash-1",
                "deployment": {
                    "state": "ready",
                    "endpoint_name": "https://serve.example",
                    "openai_base_url": "https://serve.example/v1",
                },
            }
        ]

    def deployment_for(self, run_id: str) -> dict | None:
        self.calls.append(("deployment_for", run_id))
        for entry in self.deployments():
            deployment = entry.get("deployment") or {}
            if entry.get("run_id") == run_id.split("/", 1)[0]:
                return {**deployment, "run_id": entry["run_id"]}
        return None

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


@pytest.fixture(autouse=True)
def project_api(monkeypatch):
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("https://flash.test", "fslo-test")
    )
    monkeypatch.setattr(
        cli.commands, "load_credentials", lambda: ("https://flash.test", "fslo-test")
    )
    monkeypatch.setattr(
        "flash.client.get_project", lambda project_id, api_key: {"id": project_id, "name": "Test"}
    )
    monkeypatch.setattr(
        "flash.client.list_projects",
        lambda api_key: [{"id": "11111111-1111-4111-8111-111111111111", "name": "Test project"}],
    )


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


def test_project_create_prints_only_returned_id_in_plain_mode(monkeypatch, capsys) -> None:
    seen = {}
    monkeypatch.setattr(
        cli.commands, "load_credentials", lambda: ("https://flash.test", "fslo-test")
    )

    def create(name, description, api_key):
        seen.update(name=name, description=description, api_key=api_key)
        return {"id": "33333333-3333-4333-8333-333333333333"}

    monkeypatch.setattr("flash.client.create_project", create)
    assert _run(["projects", "create", " My project ", "--description", " desc "]) == 0
    assert capsys.readouterr().out == "33333333-3333-4333-8333-333333333333\n"
    assert seen == {"name": " My project ", "description": " desc ", "api_key": "fslo-test"}


def test_projects_list_prints_ids_and_names(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "flash.client.list_projects",
        lambda api_key: [
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "name": "Test project",
            }
        ],
    )

    assert _run(["projects", "list"]) == 0
    assert capsys.readouterr().out == ("11111111-1111-4111-8111-111111111111\tTest project\n")


def test_projects_create_uses_plural_group(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "flash.client.create_project",
        lambda name, description, api_key: {"id": "33333333-3333-4333-8333-333333333333"},
    )

    assert _run(["projects", "create", "My project"]) == 0
    assert capsys.readouterr().out == "33333333-3333-4333-8333-333333333333\n"


def test_train_cost_requires_explicit_project(tmp_path, capsys) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-4B"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "acme/example"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 1\n",
        encoding="utf-8",
    )

    assert _run(["train", str(config), "--cost"]) == 1
    assert "project" in capsys.readouterr().err.lower()


def test_env_setup_maps_inaccessible_project_to_client_error(monkeypatch) -> None:
    from argparse import Namespace

    from flash.cli import env_setup
    from flash.client import ApiError, ClientError

    monkeypatch.setattr(
        "flash.client.get_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ApiError(403, "forbidden")),
    )

    with pytest.raises(ClientError, match="not accessible") as excinfo:
        env_setup._require_setup_project(Namespace(project="11111111-1111-4111-8111-111111111111"))
    assert type(excinfo.value) is ClientError


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
    assert _run(["models", "list"]) == 0
    out = capsys.readouterr().out
    # every catalog model is listed (no experimental/hidden tier)
    assert "Qwen/Qwen3.5-0.8B" in out
    assert "Qwen/Qwen3.5-9B" in out
    assert "Qwen/Qwen3.6-27B" in out
    assert "Qwen/Qwen3.5-2B" in out
    # only bare model ids, none of the extra per-model detail columns
    assert "2.3B" not in out
    assert "dense" not in out
    assert "(text-only fine-tune)" not in out
    assert "algos=" not in out
    assert "bf16" not in out
    assert "thinking=" not in out


def test_gpus_tip_explains_automatic_default_and_type_pin(fake_client, capsys) -> None:
    assert _run(["gpus"]) == 0
    out = capsys.readouterr().out
    assert "GPU allocation is automatic by default" in out
    assert "cheapest validated class" in out
    assert 'type = "<CLASS>"' in out
    assert "don't pin" not in out
    assert "cannot pin" not in out
    assert "runpod" not in out.lower()
    assert "lambda" not in out.lower()


def _train_config(tmp_path, *, extra_train: str = ""):
    path = tmp_path / "train.toml"
    path.write_text(
        'model = "Qwen/Qwen3.5-4B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "sft"\n'
        '[environment]\nid = "owner/env"\n'
        f"[train]\nepochs = 1\nmax_examples = 2\n{extra_train}"
    )
    return path


@pytest.mark.parametrize(
    ("compatibility", "expected"),
    [
        (
            {
                "status": "agreement",
                "client_only": [],
                "server_only": [],
                "introduced_in_differences": [],
            },
            "schemas agree exactly",
        ),
        (
            {
                "status": "disagreement",
                "client_only": ["future_knob"],
                "server_only": ["server_knob"],
                "introduced_in_differences": [
                    {"key": "epochs", "client": "0.2.1", "server": "0.2.0"}
                ],
            },
            "client-only keys: future_knob",
        ),
        (None, "unverifiable (legacy server)"),
    ],
)
def test_train_dry_run_keeps_compatibility_on_stderr(
    fake_client, tmp_path, capsys, compatibility, expected
) -> None:
    if compatibility is None:
        original_create_run = fake_client.create_run

        def create_run_without_compatibility(*args, **kwargs):
            response = original_create_run(*args, **kwargs)
            response.pop("train_schema_compatibility", None)
            return response

        fake_client.create_run = create_run_without_compatibility
    else:
        original_create_run = fake_client.create_run

        def create_run_with_compatibility(*args, **kwargs):
            response = original_create_run(*args, **kwargs)
            response["train_schema_compatibility"] = compatibility
            return response

        fake_client.create_run = create_run_with_compatibility

    assert _run(["train", str(_train_config(tmp_path)), "--dry-run"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    call = next(call for call in fake_client.calls if call[0] == "create_run")

    assert "train_schema_compatibility" not in payload
    assert expected in captured.err
    assert call[2] is None
    assert call[3] is True
    assert call[4]["authored_keys"] == ["epochs", "max_examples"]
    assert call[1]["train"] == {"epochs": 1, "max_examples": 2}


def test_train_dry_run_sends_declared_runtime_secrets(
    fake_client, tmp_path, capsys, monkeypatch
) -> None:
    config = tmp_path / "train.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-4B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "sft"\n'
        '[environment]\nid = "owner/env"\nsecrets = ["SERPAPI_API_KEY"]\n'
        "[train]\nepochs = 1\nmax_examples = 2\n"
    )
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-secret")

    assert _run(["train", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    call = next(call for call in fake_client.calls if call[0] == "create_run")

    assert call[2] == {"SERPAPI_API_KEY": "serp-secret"}
    assert call[3] is True


def test_train_dry_run_enriches_legacy_unknown_authored_key_rejection(
    fake_client, tmp_path, capsys, monkeypatch
) -> None:
    from flash.client import ApiError

    detail = "[train] unknown key(s): teacher_model (allowed: epochs, hf_repo, max_examples)"

    def reject(*_args, **_kwargs):
        raise ApiError(400, detail)

    monkeypatch.setattr(fake_client, "create_run", reject)
    config = _train_config(tmp_path, extra_train='teacher_model = "glm-5.2"\n')

    assert _run(["train", str(config), "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert detail in captured.err
    assert "teacher_model (minimum released Flash version 0.2.56)" in captured.err
    assert "client/server [train] schemas disagree" in captured.err


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (400, "budget precheck rejected this run"),
        (
            400,
            "[train] unknown key(s): structured_outputs (allowed: epochs, hf_repo, max_examples)",
        ),
        (
            400,
            "[train] unknown key(s): future_knob (allowed: epochs, hf_repo, max_examples)",
        ),
        (
            500,
            "[train] unknown key(s): teacher_model (allowed: epochs, hf_repo, max_examples)",
        ),
    ],
)
def test_train_dry_run_does_not_enrich_unrelated_or_unknown_errors(
    fake_client, tmp_path, capsys, monkeypatch, status, detail
) -> None:
    from flash.client import ApiError

    def reject(*_args, **_kwargs):
        raise ApiError(status, detail)

    monkeypatch.setattr(fake_client, "create_run", reject)
    config = _train_config(tmp_path, extra_train='teacher_model = "glm-5.2"\n')

    assert _run(["train", str(config), "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: {detail}\n"


def test_train_dry_run_authoritative_rejection_keeps_stdout_empty(
    fake_client, tmp_path, capsys, monkeypatch
) -> None:
    from flash.client import ApiError

    def reject(*_args, **_kwargs):
        raise ApiError(
            400,
            "unknown key(s): future_knob. Unsupported authored [train] key(s): "
            "future_knob (minimum released Flash version 0.3.0); "
            "client/server [train] schemas disagree",
        )

    monkeypatch.setattr(fake_client, "create_run", reject)

    assert _run(["train", str(_train_config(tmp_path)), "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "future_knob" in captured.err
    assert "minimum released Flash version 0.3.0" in captured.err
    assert "schemas disagree" in captured.err


def test_train_live_and_dry_run_send_the_same_sparse_spec(fake_client, tmp_path, capsys) -> None:
    config = _train_config(tmp_path)

    assert _run(["train", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    assert _run(["train", str(config), "--background"]) == 0
    capsys.readouterr()
    calls = [call for call in fake_client.calls if call[0] == "create_run"]

    assert calls[0][1] == calls[1][1]
    assert calls[0][1]["train"] == {"epochs": 1, "max_examples": 2}
    assert calls[0][3] is True
    assert calls[0][4] is not None
    assert calls[1][3] is False
    assert calls[1][4] == calls[0][4]


def test_status_runs_and_log_command(fake_client, capsys) -> None:
    assert _run(["runs", "status", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert "done" in out
    assert "cost_usd" in out

    assert _run(["runs", "list"]) == 0
    out = capsys.readouterr().out
    assert "ALGO" in out
    assert "flash-1" in out
    assert "done" in out
    assert "SFT" in out

    assert _run(["runs", "status", "flash-1", "--follow"]) == 0
    out = capsys.readouterr().out
    assert "cost_usd" in out
    assert "hello from the worker" not in out

    assert _run(["runs", "log", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert "hello from the worker" in out
    assert "----- console_sft.txt -----" in out
    assert "worker stdout line" in out
    assert "cost_usd" not in out


def test_log_prints_partial_log_line_with_newline(fake_client, capsys) -> None:
    fake_client.log_text = "partial log line"
    fake_client.get_worker_output = lambda run_id: {}

    assert _run(["runs", "log", "flash-1"]) == 0
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

    assert _run(["runs", "log", "flash-1"]) == 0
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


def test_follow_logs_prints_heartbeat_metrics_once_per_step(monkeypatch, capsys) -> None:
    metric_one = {
        "step": 1,
        "reward": 0.75,
        "reward_std": 0.12,
        "grad_norm": 1.5,
        "kl": 0.03,
        "entropy": 0.82,
        "frac_reward_zero_std": 0.25,
        "mean_completion_tokens": 48.5,
        "truncation_rate": 0.125,
        "max_completion_tokens": 256,
    }
    metric_two = {
        "step": 2,
        "reward": 0.8,
        "reward_std": 0.1,
        "grad_norm": 1.25,
        "kl": None,
        "entropy": 0.79,
        "frac_reward_zero_std": 0.0,
        "mean_completion_tokens": 51.0,
        "truncation_rate": 0.25,
        "max_completion_tokens": 256,
    }

    class _MetricClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.statuses = iter(
                [
                    {
                        "run_id": "flash-metrics",
                        "state": "running",
                        "last_heartbeat": {"stage": "rl_step", "metrics_last": [metric_one]},
                    },
                    {
                        "run_id": "flash-metrics",
                        "state": "running",
                        "last_heartbeat": {
                            "stage": "rl_step",
                            "metrics_last": [metric_one, metric_two],
                        },
                    },
                    {
                        "run_id": "flash-metrics",
                        "state": "done",
                        "last_heartbeat": {"stage": "rl_step", "metrics_last": [metric_two]},
                    },
                ]
            )

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            return {"run_id": run_id, "logs": "", "offset": 0, "state": "running"}

        def get_run(self, run_id: str) -> dict:
            return next(self.statuses)

    monkeypatch.setattr(cli.commands.time, "sleep", lambda _seconds: None)

    state, printed_any = cli.commands._poll_logs(_MetricClient(), "flash-metrics", interval=0.2)

    assert state == "done"
    assert printed_any is False
    metric_lines = [
        line for line in capsys.readouterr().err.splitlines() if line.startswith("step=")
    ]
    assert metric_lines == [
        "step=1 reward=0.75 reward_std=0.12 grad_norm=1.5 kl=0.03 entropy=0.82 "
        "frac_zero_std=0.25 comp_len=48.5 trunc=0.125 max_comp_tokens=256",
        "step=2 reward=0.8 reward_std=0.1 grad_norm=1.25 entropy=0.79 frac_zero_std=0 "
        "comp_len=51 trunc=0.25 max_comp_tokens=256",
    ]


def test_log_follow_metric_dedup_is_attempt_aware() -> None:
    from flash.cli.commands import _log_follow_metric_rows

    seen = set()
    attempt_one = {
        "last_heartbeat": {
            "attempt": 1,
            "metrics_last": [{"step": 7, "reward": 0.5}],
        }
    }
    attempt_two = {
        "last_heartbeat": {
            "attempt": 2,
            "metrics_last": [{"step": 7, "reward": 0.6}],
        }
    }

    assert _log_follow_metric_rows(attempt_one, seen) == ["step=7 reward=0.5"]
    assert _log_follow_metric_rows(attempt_one, seen) == []
    assert _log_follow_metric_rows(attempt_two, seen) == ["step=7 reward=0.6"]


def test_cancel_surfaces_surviving_checkpoints(fake_client, capsys) -> None:
    """`state=cancelled` + adapter_ref=null + cost=0 reads as discardable, yet the per-step
    deployable checkpoints streamed before the cancel survive it — the cancel output must say
    so (on stderr in the plain path, keeping the stdout JSON machine-readable)."""
    import json as _json

    assert _run(["runs", "cancel", "flash-1"]) == 0
    assert ("checkpoints", "flash-1") in fake_client.calls
    out, err = capsys.readouterr()
    assert _json.loads(out)["state"] == "cancelled"  # stdout stays pure JSON in the plain path
    assert "2 deployable checkpoint(s) survive this cancel" in err
    assert "flash runs checkpoint flash-1" in err
    assert "flash models deploy flash-1/step-40" in err  # points at the newest surviving step


def test_cancel_hint_is_best_effort_when_checkpoint_listing_fails(
    fake_client, capsys, monkeypatch
) -> None:
    """The surviving-checkpoints lookup must never break `flash runs cancel` itself."""

    def boom(run_id):
        raise RuntimeError("backend hiccup")

    monkeypatch.setattr(fake_client, "checkpoints", boom)
    assert _run(["runs", "cancel", "flash-1"]) == 0
    out, err = capsys.readouterr()
    assert '"state": "cancelled"' in out
    assert "deployable checkpoint" not in err


def test_cancel_hint_survives_malformed_checkpoint_shape(fake_client, capsys, monkeypatch) -> None:
    """A checkpoint dict missing 'step' (or carrying a non-orderable value) must NOT crash a cancel
    that already succeeded — the max(step) hint is best-effort. A recoverable step still drives the
    deploy example; when none is recoverable the example is simply dropped (no crash, no bogus step)."""
    monkeypatch.setattr(
        fake_client, "checkpoints", lambda run_id: [{"no_step": 1}, {"step": None}, {"step": 7}]
    )
    assert _run(["runs", "cancel", "flash-1"]) == 0  # did not raise on the malformed entries
    out, err = capsys.readouterr()
    assert '"state": "cancelled"' in out
    assert "3 deployable checkpoint(s) survive this cancel" in err
    assert "flash models deploy flash-1/step-7" in err  # max of the RECOVERABLE steps

    monkeypatch.setattr(fake_client, "checkpoints", lambda run_id: [{"no_step": 1}])
    assert _run(["runs", "cancel", "flash-1"]) == 0
    _, err2 = capsys.readouterr()
    assert "1 deployable checkpoint(s) survive this cancel" in err2
    assert "flash deploy" not in err2


def test_cancel_deploy_undeploy_deployments(fake_client, capsys) -> None:
    assert _run(["runs", "cancel", "flash-1"]) == 0
    assert ("cancel", "flash-1") in fake_client.calls

    assert _run(["models", "deploy", "flash-1"]) == 0
    assert ("deploy", "flash-1", {"dry_run": False}) in fake_client.calls

    assert _run(["models", "deploy", "flash-1/step-40"]) == 0
    assert ("deploy", "flash-1/step-40", {"dry_run": False}) in fake_client.calls
    err = capsys.readouterr().err
    assert "flash models undeploy flash-1`" in err
    assert "flash models undeploy flash-1/step-40`" not in err

    assert _run(["models", "deployments"]) == 0
    deployments_out = capsys.readouterr().out
    assert "flash-1" in deployments_out
    assert "REVISION" in deployments_out

    assert _run(["models", "undeploy", "flash-1"]) == 0
    assert ("undeploy", "flash-1") in fake_client.calls


def test_deployments_json_passes_server_rows_through(fake_client, capsys) -> None:
    assert _run(["models", "deployments", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {
            "run_id": "flash-1",
            "deployment": {
                "state": "ready",
                "endpoint_name": "https://serve.example",
                "openai_base_url": "https://serve.example/v1",
            },
        }
    ]


def test_deployments_json_empty_list(fake_client, monkeypatch, capsys) -> None:
    monkeypatch.setattr(fake_client, "deployments", lambda: [])
    assert _run(["models", "deployments", "--json"]) == 0
    assert capsys.readouterr().out.strip() == "[]"


def test_chat_sends_message_and_prints_reply(fake_client, capsys) -> None:
    assert _run(["models", "chat", "flash-1", "-m", "What is 6*7?"]) == 0
    assert "42" in capsys.readouterr().out
    assert fake_client.calls[-1][0] == "chat_stream"


def test_chat_checkpoint_ref_is_forwarded_unchanged(fake_client) -> None:
    target = "flash-1/step-40"

    assert _run(["models", "chat", target, "-m", "What is 6*7?"]) == 0
    assert fake_client.calls[-1][0] == "chat_stream"
    assert fake_client.calls[-1][1] == target


def test_chat_accepts_full_immutable_revision(fake_client) -> None:
    revision = "flash-1@step-40." + "a" * 40
    assert _run(["models", "chat", revision, "-m", "What is 6*7?"]) == 0
    assert fake_client.calls[-1][0] == "chat_stream"
    assert fake_client.calls[-1][1] == revision


def test_chat_system_flag_prepends_system_message(fake_client) -> None:
    """--system gives evals training-prompt parity without calling the HTTP API directly."""
    assert _run(["models", "chat", "flash-1", "-m", "What is 6*7?", "--system", "be brief"]) == 0
    _, _, messages = fake_client.calls[-1]
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "What is 6*7?"},
    ]


def test_chat_without_system_flag_sends_user_message_only(fake_client) -> None:
    assert _run(["models", "chat", "flash-1", "-m", "What is 6*7?"]) == 0
    _, _, messages = fake_client.calls[-1]
    assert messages == [{"role": "user", "content": "What is 6*7?"}]


@pytest.mark.parametrize("flag", ["--enable-thinking", "--disable-thinking"])
def test_chat_does_not_expose_thinking_override_flags(fake_client, flag) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(["models", "chat", "flash-1", "-m", "What is 6*7?", flag])
    assert excinfo.value.code == 2


def test_env_setup_scaffolds_grpo_and_sft_configs(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

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
    assert 'project = "11111111-1111-4111-8111-111111111111"' in grpo.read_text()
    assert 'algorithm = "grpo"' in grpo.read_text()
    assert "epochs = 1" in grpo.read_text()
    assert "max_examples = 2" in grpo.read_text()
    assert "cheapest fitting managed class" in grpo.read_text()
    assert "private environment-scoped repo" in grpo.read_text()
    assert 'project = "11111111-1111-4111-8111-111111111111"' in sft.read_text()
    assert 'algorithm = "sft"' in sft.read_text()
    assert "epochs = 1" in sft.read_text()
    assert "max_examples = 2" in sft.read_text()
    assert "cheapest fitting managed class" in sft.read_text()
    assert "private environment-scoped repo" in sft.read_text()
    opd = tmp_path / "configs/opd.toml"
    assert opd.is_file()
    opd_text = opd.read_text()
    assert 'project = "11111111-1111-4111-8111-111111111111"' in opd_text
    assert 'algorithm = "opd"' in opd_text
    assert "epochs = 1" in opd_text
    assert "max_examples = 2" in opd_text
    # The teacher key is platform-managed: the scaffold neither declares it as a secret nor tells
    # the user to export it, so the generated config must not mention FIREWORKS_API_KEY at all.
    assert "FIREWORKS_API_KEY" not in opd_text
    assert "secrets" not in opd_text
    assert "platform-managed" in opd_text
    # single-turn opd runs fine, so it carries NO multi-turn "fails fast" warning
    assert "fail fast" not in opd_text
    training = tmp_path / "TRAINING.md"
    assert training.is_file()
    training_text = training.read_text(encoding="utf-8")
    assert "how to actually improve a model with Flash" in training_text
    assert "## Using Flash" in training_text  # end-to-end library usage, not just conventions
    assert "## Common Flash issues and mitigations" in training_text
    assert "GPU selection is not what you expected" in training_text
    assert "response_text.thinking" in training_text
    assert "Qwen3.5 thinking multi-turn SFT" in training_text
    assert "longest shared token prefix" in training_text
    assert "flash env pull your-org/my-env" in training_text
    assert "private environment-scoped repo" in training_text
    assert 'project = "11111111-1111-4111-8111-111111111111"' in training_text
    assert "flash runs checkpoint <run-id>" in training_text
    assert "flash models deployments" in training_text
    assert "flash models export --adapter-id <run-id> --repository <you>/<repo>" in training_text
    assert (
        "flash env push --project 11111111-1111-4111-8111-111111111111 --name my-env ."
        in training_text
    )
    assert "HF_TOKEN" in training_text
    assert "runpod" not in training_text.lower()
    assert "lambda" not in training_text.lower()
    out = capsys.readouterr().out
    assert "dataset/train.jsonl" in out
    assert "configs/rl.toml" in out
    assert "configs/opd.toml" in out
    assert "TRAINING.md" in out


def test_env_setup_multi_turn_scaffolds_opd_for_multi_turn(monkeypatch, tmp_path, capsys) -> None:
    """`flash env setup --multi-turn` scaffolds all three configs (sft/rl/opd). opd now supports
    multi-turn (it rolls out each episode and distils every assistant turn), so the multi-turn opd.toml
    and the starter env docstring must NOT warn it is single-turn only / fails fast."""
    monkeypatch.chdir(tmp_path)

    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--multi-turn"])
        == 0
    )

    env_py = (tmp_path / "environment.py").read_text()
    assert "EnvironmentMultiTurn" in env_py  # genuinely a multi-turn scaffold
    # the docstring documents all three algorithms train off the multi-turn env (no opd carve-out)
    assert "distils EVERY assistant turn" in env_py
    assert "single-turn only" not in env_py
    # all three algorithm configs are scaffolded in multi-turn mode too
    for name in ("configs/sft.toml", "configs/rl.toml", "configs/opd.toml"):
        assert (tmp_path / name).is_file(), name
    opd_text = (tmp_path / "configs/opd.toml").read_text()
    assert 'algorithm = "opd"' in opd_text
    # ...and the multi-turn opd.toml notes it distils every assistant turn, with NO fail-fast warning
    assert "distils EVERY assistant turn" in opd_text
    assert "SINGLE-TURN only" not in opd_text
    assert "fail fast" not in opd_text
    assert "configs/opd.toml" in capsys.readouterr().out


def test_env_setup_default_omits_reasoning(monkeypatch, tmp_path) -> None:
    # Non-interactive (pytest stdin is not a tty) with no flags stays on today's scaffold: no
    # reasoning knobs land in either config.
    monkeypatch.chdir(tmp_path)
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0
    rl = (tmp_path / "configs/rl.toml").read_text()
    sft = (tmp_path / "configs/sft.toml").read_text()
    assert "thinking = true" not in rl
    assert "thinking = true" not in sft
    assert "max_completion_tokens" not in rl
    assert "EnvironmentSingleTurn" in (tmp_path / "environment.py").read_text()


def test_env_setup_reasoning_flag_enables_thinking(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--reasoning"])
        == 0
    )
    rl = (tmp_path / "configs/rl.toml").read_text()
    sft = (tmp_path / "configs/sft.toml").read_text()
    assert "thinking = true" in rl
    assert "thinking = true" in sft
    # GRPO raises the generation budget so reasoning does not truncate the answer.
    assert "max_completion_tokens = 2048" in rl
    # SFT can't share a token budget it doesn't generate; it gets the gold think-tag guidance instead.
    assert "warn_missing_think_tags" in sft
    assert "max_completion_tokens" not in sft


def test_env_setup_no_reasoning_flag_is_explicit_off(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        _run(
            ["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--no-reasoning"]
        )
        == 0
    )
    assert "thinking = true" not in (tmp_path / "configs/rl.toml").read_text()


def test_env_setup_multi_turn_flag_scaffolds_multiturn(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--multi-turn"])
        == 0
    )
    assert "EnvironmentMultiTurn" in (tmp_path / "environment.py").read_text()
    assert "secret whole number" in (tmp_path / "dataset/train.jsonl").read_text()


def test_env_setup_interactive_survey_picks_multi_and_reasoning(monkeypatch, tmp_path) -> None:
    # A real terminal: the two survey questions are asked and answered "2" (multi-turn) then
    # "2" (reasoning). FLASH_STYLE forces the themed path; a fake stdin reports a tty; input()
    # is stubbed so no real keypress is read. CI is cleared so the guard doesn't force defaults.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
    answers = iter(["1", "", "2", "2"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert _run(["env", "setup"]) == 0
    assert "EnvironmentMultiTurn" in (tmp_path / "environment.py").read_text()
    rl = (tmp_path / "configs/rl.toml").read_text()
    assert "thinking = true" in rl
    assert "max_completion_tokens = 2048" in rl


def test_env_setup_interactive_enter_takes_defaults(monkeypatch, tmp_path) -> None:
    # Pressing enter (empty answer) at both questions selects the marked defaults: single-turn,
    # no reasoning — identical to the non-interactive scaffold.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
    answers = iter(["1", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert _run(["env", "setup"]) == 0
    env_py = (tmp_path / "environment.py").read_text()
    assert "EnvironmentSingleTurn" in env_py
    assert "EnvironmentMultiTurn" not in env_py
    assert "thinking = true" not in (tmp_path / "configs/rl.toml").read_text()


def test_env_setup_under_ci_never_prompts(monkeypatch, tmp_path) -> None:
    # Even with a tty and themed output, CI=true must fall back to defaults instead of blocking on
    # a prompt. input() raises so a regression (actually prompting) fails loudly.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))

    def _boom(*a, **k):
        raise AssertionError("prompted under CI")

    monkeypatch.setattr("builtins.input", _boom)
    assert _run(["env", "setup"]) == 1
    assert not (tmp_path / "environment.py").exists()
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0
    assert "EnvironmentSingleTurn" in (tmp_path / "environment.py").read_text()
    assert "thinking = true" not in (tmp_path / "configs/rl.toml").read_text()


def test_env_setup_yes_requires_project_before_creating_files(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _run(["env", "setup", "--yes"]) == 1
    assert "--project PROJECT_UUID is required" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_env_setup_rejects_projectless_or_conflicting_existing_configs(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "sft.toml").write_text('model = "Qwen/Qwen3.5-4B"\n')
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 1
    assert "has no valid top-level project UUID" in capsys.readouterr().err
    assert not (tmp_path / "environment.py").exists()

    (configs / "sft.toml").write_text('project = "22222222-2222-4222-8222-222222222222"\n')
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 1
    assert "does not match selected project" in capsys.readouterr().err
    assert not (tmp_path / "environment.py").exists()


def test_env_setup_reasoning_flag_warns_when_configs_exist(monkeypatch, tmp_path, capsys) -> None:
    # First scaffold (no reasoning), then re-run with --reasoning: the existing configs win and the
    # command warns instead of silently ignoring the flag or writing mismatched configs.
    monkeypatch.chdir(tmp_path)
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0
    capsys.readouterr()
    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--reasoning"])
        == 0
    )
    err = capsys.readouterr().err
    assert "existing configs are no reasoning" in err
    assert "ignoring --reasoning" in err
    assert "thinking = true" not in (tmp_path / "configs/rl.toml").read_text()


def test_unknown_run_errors_surface_as_nonzero_exit(monkeypatch, capsys) -> None:
    from flash.client import ApiError

    class _Erroring(_FakeClient):
        def get_run(self, run_id: str) -> dict:
            raise ApiError(404, "unknown run")

    monkeypatch.setattr(cli.commands, "client_from_config", lambda *a, **k: _Erroring())
    assert _run(["runs", "status", "nope"]) != 0
    assert "unknown run" in capsys.readouterr().err


def test_spec_payload_resolves_worker_pip(monkeypatch, tmp_path) -> None:
    from flash.client.specs import spec_payload
    from flash.spec import EnvironmentSpec, JobSpec

    # An unrecorded env resolves to the Freesolo SDK; the env is loaded lazily by the worker.
    spec = JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        project="11111111-1111-4111-8111-111111111111",
        environment=EnvironmentSpec(id="owner/env"),
    )
    assert spec_payload(spec)["environment"]["pip"] == ["freesolo>=0.2.60"]

    # ...and an explicit pip list (the documented escape hatch) wins untouched.
    spec = JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        project="11111111-1111-4111-8111-111111111111",
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
                "models",
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
    assert _run(["models", "export", "--adapter-id", "flash-1", "--repository", "me/adapters"]) == 0
    assert ("export", "flash-1", "me/adapters", "hf_env", True) in fake_client.calls


def test_export_without_token_errors_cleanly(fake_client, monkeypatch, capsys, tmp_path) -> None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # A clean cwd so a stray local .env can't supply a token.
    monkeypatch.chdir(tmp_path)
    assert _run(["models", "export", "--adapter-id", "flash-1", "--repository", "me/adapters"]) == 1
    err = capsys.readouterr().err
    assert "HuggingFace token" in err
    # The control plane is never contacted when there's no token to send.
    assert not any(call[0] == "export" for call in fake_client.calls)


def test_deploy_enqueues_server_side_verification(fake_client, capsys) -> None:
    assert _run(["models", "deploy", "flash-1"]) == 0
    assert ("deploy", "flash-1", {"dry_run": False}) in fake_client.calls
    assert not any(c[0] == "chat" for c in fake_client.calls)
    err = capsys.readouterr().err
    assert "flash models deployments" in err
    assert "OpenAI-compatible base URL" in err


def test_deploy_checkpoint_enqueues_base_run_deployment(fake_client) -> None:
    assert _run(["models", "deploy", "flash-1/step-40"]) == 0
    assert ("deploy", "flash-1/step-40", {"dry_run": False}) in fake_client.calls
    assert not any(c[0] == "chat" for c in fake_client.calls)


def test_deploy_no_verify_flag_is_removed(fake_client) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(["models", "deploy", "flash-1", "--no-verify"])
    assert excinfo.value.code == 2
    assert not any(call[0] == "deploy" for call in fake_client.calls)


def test_deploy_dry_run_skips_active_deployment_note(fake_client, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "dry_run"},
        raising=False,
    )
    assert _run(["models", "deploy", "flash-1", "--dry-run"]) == 0
    assert not any(c[0] == "chat" for c in fake_client.calls)
    assert "flash models deployments" not in capsys.readouterr().err


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

    assert _run(["models", "deploy", "flash-1"]) == 1
    err = capsys.readouterr().err
    assert "deployment failed: smoke generation failed" in err
    assert "once it is ready" not in err


def _queued_deploy(monkeypatch, fake_client) -> None:
    """Make POST deploy return what the control plane really returns: a queued record."""
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "queued"},
        raising=False,
    )
    monkeypatch.setattr(cli.commands.time, "sleep", lambda _s: None)


def test_deploy_without_wait_returns_while_still_queued(fake_client, monkeypatch, capsys) -> None:
    """No --wait keeps the old behaviour: return immediately, do not poll."""
    _queued_deploy(monkeypatch, fake_client)

    assert _run(["models", "deploy", "flash-1"]) == 0
    assert not any(c[0] == "deployment_for" for c in fake_client.calls)
    assert "deployment state is 'queued'" in capsys.readouterr().err


def test_deploy_wait_polls_until_the_revision_is_servable(fake_client, monkeypatch, capsys) -> None:
    """--wait must not return while the requested revision is still queued.

    deploy returns as soon as the record is persisted, which is normally before the new revision
    can serve a token, so a caller that starts evaluating on that return hits the old revision or
    an error. The wait is what makes the printed record mean "ready".
    """
    _queued_deploy(monkeypatch, fake_client)
    states = iter([{"state": "smoke_testing"}, {"state": "reconciling"}, {"state": "ready"}])
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: next(states), raising=False
    )

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 0
    out, err = capsys.readouterr()
    assert "ready" in out
    assert "queued" not in err
    assert "ctrl-c stops waiting, not the deployment" in err


def test_deploy_wait_stops_on_a_failed_revision(fake_client, monkeypatch, capsys) -> None:
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {"state": "failed", "error": "smoke generation failed"},
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 1
    assert "deployment failed: smoke generation failed" in capsys.readouterr().err


def test_deploy_wait_gives_up_at_the_timeout_without_claiming_success(
    fake_client, monkeypatch, capsys
) -> None:
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {"state": "smoke_testing"},
        raising=False,
    )

    # exit 1, not 0: --wait's contract is "the revision is servable when i return", and a timeout
    # is precisely the case where it is not. exiting 0 here is what lets
    # `deploy --wait && evaluate` proceed against a revision that never became servable.
    assert _run(["models", "deploy", "flash-1", "--wait", "0.01"]) == 1
    err = capsys.readouterr().err
    assert "still 'smoke_testing' after 0.01s" in err
    assert "flash models deployments" in err


def test_deploy_wait_ends_when_the_deployment_stops_being_listed(
    fake_client, monkeypatch, capsys
) -> None:
    """A run drops out of the listing once its deployment is gone, so that is terminal."""
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: None, raising=False
    )

    # the last record seen was still queued, so the requested revision never became servable.
    assert _run(["models", "deploy", "flash-1", "--wait"]) == 1
    assert "no longer an active deployment" in capsys.readouterr().err


def test_deploy_wait_reports_a_rollback_to_a_different_checkpoint_step(
    fake_client, monkeypatch, capsys
) -> None:
    """An absent revision is not always a deleted one.

    `deployment_for` matches the checkpoint step, so a failed `deploy RUN/step-40` that the plane
    rolled back to step-20 reads as absent exactly like a deletion does. Reporting it as "no longer
    an active deployment" named the wrong event and threw away `last_deploy_error`, which is the
    only record of why step-40 did not take -- the restored record carries no `failed` state.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "queued", "requested_at": "T1"},
        raising=False,
    )
    # the requested revision is gone from the listing; the predecessor is what is serving.
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: None, raising=False
    )
    monkeypatch.setattr(
        fake_client,
        "deployments",
        lambda timeout=None: [
            {
                "run_id": "flash-1",
                "deployment": {
                    "run_id": "flash-1",
                    "checkpoint_step": 20,
                    "state": "ready",
                    "requested_at": "T0",
                    "last_deploy_error": "smoke test failed",
                },
            }
        ],
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/step-40", "--wait", "5"]) == 1
    err = capsys.readouterr().err
    assert "smoke test failed" in err, err
    assert "previously deployed revision is still serving" in err, err
    # the wrong explanation must be gone, not merely accompanied by the right one.
    assert "no longer an active deployment" not in err, err


def test_deploy_wait_reports_a_rollback_from_the_final_adapter(
    fake_client, monkeypatch, capsys
) -> None:
    """A bare run id is a revision too, and its failed redeploy rolls back like any other.

    `deploy flash-1` asks for the final adapter, which `parse_checkpoint_ref` reports as step
    `None`. A run already serving `step-20` whose final-adapter redeploy fails is restored to
    step-20 by `mark_deployment_failed`, and `deployment_for` rejects the restored record because
    its non-null step does not match the requested final adapter -- so the bare-run form reads as
    absent exactly like the `/step-N` form does. Exempting it from the rollback lookup reported the
    run as vanished and dropped `last_deploy_error`.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "queued", "requested_at": "T1"},
        raising=False,
    )
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: None, raising=False
    )
    monkeypatch.setattr(
        fake_client,
        "deployments",
        lambda timeout=None: [
            {
                "run_id": "flash-1",
                "deployment": {
                    "run_id": "flash-1",
                    "checkpoint_step": 20,
                    "state": "ready",
                    "requested_at": "T0",
                    "last_deploy_error": "adapter merge failed",
                },
            }
        ],
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1", "--wait", "5"]) == 1
    err = capsys.readouterr().err
    assert "adapter merge failed" in err, err
    assert "previously deployed revision is still serving" in err, err
    assert "no longer an active deployment" not in err, err


def test_deploy_wait_rollback_lookup_stays_inside_the_deadline(
    fake_client, monkeypatch
) -> None:
    """The rollback read is one more read inside the wait, not a second full-length one.

    It runs after a poll that has already spent part of the budget, so bounding it by the
    remainder computed BEFORE that poll hands it time the wait no longer has: a `--wait 5` whose
    poll consumed nearly all five seconds could block for close to ten.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(cli.commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        cli.commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )

    def _poll(run_id, timeout=None):
        # a stalled plane answers at its bound, which is what leaves nothing for the next read.
        clock["t"] += timeout if timeout is not None else 0.0

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)
    reads: list[tuple[float, float | None]] = []

    def _listing(timeout=None):
        reads.append((clock["t"], timeout))
        return []

    monkeypatch.setattr(fake_client, "deployments", _listing, raising=False)

    assert _run(["models", "deploy", "flash-1/step-40", "--wait", "5"]) == 1
    assert reads, "the vanished branch issued no rollback lookup"
    for start, bound in reads:
        assert bound is not None, reads
        # the expired case is allowed the zero-wait one-shot bound and nothing wider.
        assert bound <= max(5.0 - start, cli.commands._DEPLOY_ZERO_WAIT_READ_SECONDS) + 0.001, (
            reads
        )


@pytest.mark.parametrize(
    ("rows", "why"),
    [
        ([], "genuinely deleted"),
        (
            [{"run_id": "other", "deployment": {"run_id": "other", "last_deploy_error": "x"}}],
            "another run entirely",
        ),
        (
            [{"run_id": "flash-1", "deployment": {"run_id": "flash-1", "checkpoint_step": 20}}],
            "same run, no recorded error, so nothing ties it to this request",
        ),
    ],
)
def test_deploy_wait_still_reports_a_vanished_deployment_as_vanished(
    fake_client, monkeypatch, capsys, rows, why
) -> None:
    """The rollback lookup must not swallow the deletion case it was added beside.

    Matching on the run id alone is deliberately wider than `deployment_for`, so each of these has
    to stay out: without the `last_deploy_error` requirement this would report an unrelated or
    concurrently-deployed revision as this command's rollback.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: None, raising=False
    )
    monkeypatch.setattr(fake_client, "deployments", lambda timeout=None: rows, raising=False)

    assert _run(["models", "deploy", "flash-1/step-40", "--wait", "5"]) == 1
    assert "no longer an active deployment" in capsys.readouterr().err, why


def test_deploy_wait_survives_a_transient_control_plane_error(
    fake_client, monkeypatch, capsys
) -> None:
    """One failed poll must not fail a deploy that is progressing fine."""
    _queued_deploy(monkeypatch, fake_client)
    results = iter([cli.commands.ClientError("503"), {"state": "ready"}])

    def _next(run_id, timeout=None):
        value = next(results)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(fake_client, "deployment_for", _next, raising=False)

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 0
    assert "ready" in capsys.readouterr().out


def test_deploy_wait_zero_polls_once_instead_of_being_treated_as_no_wait(
    fake_client, monkeypatch, capsys
) -> None:
    """`--wait 0` is an explicit bound, not an absent flag.

    Gating on truthiness makes 0.0 indistinguishable from None, so the one value that means
    "check, but do not block" silently became "do not check at all" and exited 0 on a queued
    record.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {"state": "smoke_testing"},
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1", "--wait", "0"]) == 1
    assert "waiting up to 0s" in capsys.readouterr().err


def test_deploy_wait_rejects_a_restored_previous_revision(fake_client, monkeypatch, capsys) -> None:
    """A failed redeploy leaves a `ready` record for the PREVIOUS revision.

    mark_deployment_failed restores the old deployment verbatim and records the failure only in
    last_deploy_error, so trusting the state word reports success while the requested checkpoint
    is not the one serving.
    """
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {
            "run_id": run_id,
            "state": "queued",
            "requested_at": "2026-07-29T02:00:00Z",
        },
        raising=False,
    )
    monkeypatch.setattr(cli.commands.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {
            "state": "ready",
            "requested_at": "2026-07-29T01:00:00Z",
            "last_deploy_error": "adapter load failed",
        },
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 1
    err = capsys.readouterr().err
    assert "did not become servable" in err
    assert "adapter load failed" in err


def test_deploy_wait_accepts_a_ready_revision_carrying_a_stale_error(
    fake_client, monkeypatch
) -> None:
    """A last_deploy_error from an EARLIER attempt must not fail the attempt that succeeded.

    The stamps match here, so this record is the revision that was just asked for; treating any
    recorded error as failure would make every retry-after-failure report failure forever.
    """
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {
            "run_id": run_id,
            "state": "queued",
            "requested_at": "2026-07-29T02:00:00Z",
        },
        raising=False,
    )
    monkeypatch.setattr(cli.commands.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {
            "state": "ready",
            "requested_at": "2026-07-29T02:00:00Z",
            "last_deploy_error": "a previous attempt failed",
        },
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 0


def test_deploy_wait_bounds_each_poll_by_the_remaining_time(
    fake_client, monkeypatch, capsys
) -> None:
    """An unbounded read inside a short wait overshoots the deadline the user set.

    The client default is 60s, so `--wait 5` could block roughly a minute inside a single stalled
    request while reporting that it waited five seconds.
    """
    _queued_deploy(monkeypatch, fake_client)
    seen: list[float | None] = []

    def _poll(run_id, timeout=None):
        seen.append(timeout)
        return {"state": "ready"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1", "--wait", "5"]) == 0
    assert seen == [pytest.approx(5.0, abs=0.5)]


def test_deploy_wait_zero_actually_reads_the_current_state(fake_client, monkeypatch) -> None:
    """`--wait 0` means "check once, do not block" -- it must issue that one read.

    The deadline was evaluated before the first poll, so a zero budget was already expired on entry
    and deployment_for never ran. Readiness was then judged from the POST body, which is `queued` on
    every normal async deploy, so `--wait 0` could not succeed even against a ready revision.
    """
    _queued_deploy(monkeypatch, fake_client)
    polls: list[str] = []

    def _poll(run_id, timeout=None):
        polls.append(run_id)
        return {"state": "ready"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1", "--wait", "0"]) == 0
    assert polls == ["flash-1"], polls


def test_deploy_wait_does_not_start_a_read_after_the_deadline_expires(
    fake_client, monkeypatch
) -> None:
    """No read may still be running past the deadline the caller set.

    The remaining time was computed once before sleeping, so the post-sleep request still went out
    with the 1.0s floor: `--wait 0.1` against a stalled plane blocked for over a second past the
    bound it advertised.

    Asserted as "every read finishes by the deadline" rather than as a read COUNT. The count was a
    proxy for it under the original behaviour, where the only way to be late was an extra read; it
    stopped tracking the invariant once the final window began funding a read of its own, which is
    a bounded read strictly inside the deadline rather than an overshoot. Keeping the count would
    have made this test forbid the fix to the blind spot it shares a loop with.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(cli.commands.time, "monotonic", lambda: clock["t"])
    # the sleep is what burns the budget, exactly as a real one would.
    monkeypatch.setattr(
        cli.commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )
    reads: list[tuple[float, float | None]] = []

    def _poll(run_id, timeout=None):
        reads.append((clock["t"], timeout))
        return {"state": "queued"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1", "--wait", "0.1"]) == 1
    assert reads, "the wait issued no read at all"
    # a read starting at `start` and bounded by `bound` occupies the plane until start+bound, so
    # that sum is what has to stay within the advertised wait. this is the assertion the 1.0s floor
    # violated: it put a 0.1s wait on the hook for a full second.
    for start, bound in reads:
        assert bound is not None, reads
        assert start + bound <= 0.1 + 0.001, reads


@pytest.mark.parametrize("state", ["revocation_failed", "some_state_a_newer_plane_added"])
def test_deploy_wait_fails_closed_on_a_terminal_state_that_is_not_ready(
    fake_client, monkeypatch, capsys, state
) -> None:
    """Leaving the busy set is not the same as being servable.

    `revocation_failed` is a real persisted state (a concurrent undeploy whose backend cleanup
    failed), and an unknown state arrives on any client/server skew. Both are non-busy, so gating
    success on "not busy" exited 0 with nothing actually serving.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {"state": state},
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 1
    err = capsys.readouterr().err
    assert "not\nservable" in err or "not servable" in err, err
    assert "once it is ready" not in err, err


def test_deploy_wait_rejects_a_superseding_deploy_that_carries_no_error(
    fake_client, monkeypatch, capsys
) -> None:
    """A concurrent deploy for the same run reaches ready on ITS checkpoint, with no error at all.

    Returning early whenever last_deploy_error was absent meant the stamps were never compared on
    exactly the case that needs them, so `deploy --wait && evaluate` reported success and then
    evaluated the other shell's revision.
    """
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {
            "run_id": run_id,
            "state": "queued",
            "requested_at": "2026-07-29T02:00:00Z",
        },
        raising=False,
    )
    monkeypatch.setattr(cli.commands.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        # ready, no error, different attempt: someone else's deploy.
        lambda run_id, timeout=None: {
            "state": "ready",
            "requested_at": "2026-07-29T03:00:00Z",
        },
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 1
    assert "once it is ready" not in capsys.readouterr().err


def test_deploy_wait_observes_readiness_inside_a_short_window(fake_client, monkeypatch) -> None:
    """Sleeping the entire remainder spends the budget without ever looking again.

    With a `--wait 5` and a 5s poll interval, the first read saw `queued`, the sleep consumed all
    five seconds, and the deadline check exited: a revision that became ready one second in was
    still reported as queued and the command exited 1.
    """
    # _queued_deploy stubs sleep to a no-op, so install the clock AFTER it: a frozen monotonic with
    # a non-advancing sleep is an infinite poll, which is a broken test rather than a caught defect.
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(cli.commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        cli.commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )

    def _poll(run_id, timeout=None):
        # ready one second into the five-second window.
        return {"state": "ready" if clock["t"] >= 1.0 else "queued"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1", "--wait", "5"]) == 0


def test_deploy_wait_observes_readiness_inside_the_final_window(fake_client, monkeypatch) -> None:
    """The last second of a wait must still be watched, not slept through.

    The per-sleep reserve was subtracted only when the slice EXCEEDED it, so a remainder at or under
    the reserve was slept whole and the deadline check ended the wait with no further read. `--wait
    1` therefore could not succeed at all against an async deploy -- one read at t=0, then a full
    second of sleep -- and every longer wait was blind through its final second.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(cli.commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        cli.commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )

    def _poll(run_id, timeout=None):
        # ready half a second into the one-second window.
        return {"state": "ready" if clock["t"] >= 0.5 else "queued"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1", "--wait", "1"]) == 0


def test_deploy_wait_final_window_does_not_poll_unboundedly(fake_client, monkeypatch) -> None:
    """Splitting the final window must be the wait's last sleep, not a converging series.

    Reserving a FRACTION of the remainder rather than a fixed slice never drives the remainder to
    zero, so the loop terminates only on the clock's granularity. Against a stalled plane that is an
    unbounded burst of reads inside the last second -- the failure mode the fixed reserve was chosen
    to avoid, reintroduced at the one point the reserve does not apply.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(cli.commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        cli.commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )
    reads = []

    def _poll(run_id, timeout=None):
        reads.append(clock["t"])
        # never settles: the wait has to end on its own budget, not on the plane's answer.
        return {"state": "queued"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1", "--wait", "1"]) == 1
    # the up-front read plus the one the split funds. a fractional reserve makes this grow without
    # bound; asserting the exact count is what keeps the split from silently becoming that.
    assert len(reads) == 2, reads


def test_deploy_wait_zero_does_not_block_past_its_own_bound(fake_client, monkeypatch) -> None:
    """`--wait 0` advertises "check once, do not block", so its one read must be bounded tightly.

    A ten-second fixed budget let a stalled plane hold a zero-second wait for ten seconds, which is
    the same overshoot the per-poll bound exists to prevent, just smaller.
    """
    _queued_deploy(monkeypatch, fake_client)
    seen: list[float | None] = []

    def _poll(run_id, timeout=None):
        seen.append(timeout)
        return {"state": "ready"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1", "--wait", "0"]) == 0
    assert seen == [pytest.approx(1.0, abs=0.001)], seen


def test_deploy_wait_rejects_a_synchronous_failure_that_returns_the_restored_revision(
    fake_client, monkeypatch, capsys
) -> None:
    """A synchronous deploy returns the FINISHED record, never the queued attempt.

    Under FLASH_DEPLOY_SYNC the POST answers after the job ran, so on failure it returns the
    restored previous `ready` revision. requested and final are then the same row, their stamps
    match by construction, and comparing identity accepted a deploy that never happened.
    """
    settled = {
        "run_id": "flash-1",
        "state": "ready",
        "requested_at": "2026-07-29T01:00:00Z",
        "last_deploy_error": "adapter load failed",
    }
    monkeypatch.setattr(fake_client, "deploy", lambda run_id, **_: dict(settled), raising=False)
    # non-busy on arrival, so _await_deployment returns it without polling at all.
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: dict(settled), raising=False
    )

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 1
    err = capsys.readouterr().err
    assert "did not become servable" in err, err
    assert "adapter load failed" in err, err


def test_deploy_wait_accepts_a_synchronous_success(fake_client, monkeypatch) -> None:
    """The synchronous check keys on a recorded error, so a clean sync deploy still succeeds."""
    settled = {"run_id": "flash-1", "state": "ready", "requested_at": "2026-07-29T02:00:00Z"}
    monkeypatch.setattr(fake_client, "deploy", lambda run_id, **_: dict(settled), raising=False)
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: dict(settled), raising=False
    )

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 0


def test_deploy_notes_name_this_channels_executable(fake_client, monkeypatch, capsys) -> None:
    """The dev channel installs `flash-dev`; a hardcoded `flash ...` hint is not runnable there."""
    monkeypatch.setattr(cli.commands, "CLI_NAME", "flash-dev")
    _queued_deploy(monkeypatch, fake_client)

    assert _run(["models", "deploy", "flash-1"]) == 0
    err = capsys.readouterr().err
    assert "flash-dev models deployments" in err, err
    assert "`flash models" not in err, err


def test_deploy_wait_stops_retrying_a_rejected_key(fake_client, monkeypatch, capsys) -> None:
    """401/403 answers the same way every time, so polling through it just burns the timeout.

    The broad ClientError catch treated a permanent rejection as a transient blip and retried to
    the full default 30-minute deadline before reporting "still queued".
    """
    _queued_deploy(monkeypatch, fake_client)
    calls: list[int] = []

    def _denied(run_id, timeout=None):
        calls.append(1)
        raise cli.commands.ApiError(403, "forbidden")

    monkeypatch.setattr(fake_client, "deployment_for", _denied, raising=False)

    assert _run(["models", "deploy", "flash-1", "--wait"]) == 1
    assert len(calls) == 1
    assert "cannot check flash-1" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1"])
def test_deploy_wait_rejects_a_timeout_that_would_never_expire(fake_client, value, capsys) -> None:
    """`float` accepts nan and inf, and a NaN deadline makes every `remaining <= 0` false.

    The loop then polls forever while the user believes they set a bound, which is worse than the
    unbounded default because the printed timeout says otherwise. Written as `--wait=VALUE` so a
    leading-dash value reaches the validator instead of being read as another option.
    """
    with pytest.raises(SystemExit) as excinfo:
        _run(["models", "deploy", "flash-1", f"--wait={value}"])
    assert excinfo.value.code == 2
    assert "--wait" in capsys.readouterr().err


def test_deploy_wait_before_the_run_id_names_the_real_mistake(fake_client, capsys) -> None:
    """`--wait` takes an optional value, so `deploy --wait flash-1` eats the run id.

    argparse cannot hand the token back, so the error has to say which argument was swallowed;
    the bare "invalid float value: 'flash-1'" reads as if the run id itself were malformed.
    """
    with pytest.raises(SystemExit) as excinfo:
        _run(["models", "deploy", "--wait", "flash-1"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "if 'flash-1' is the run id" in err
    assert "deploy flash-1 --wait" in err


def test_deploy_wait_skips_polling_for_a_dry_run(fake_client, monkeypatch, capsys) -> None:
    """A dry run creates no deployment, so there is nothing to wait on."""
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "dry_run"},
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1", "--dry-run", "--wait"]) == 0
    assert not any(c[0] == "deployment_for" for c in fake_client.calls)
    assert "ctrl-c stops waiting" not in capsys.readouterr().err


def test_log_follow_progress_includes_heartbeat_age() -> None:
    """The follow spinner must show a live heartbeat age so a long quiet phase reads as
    "alive, throttled" instead of a frozen line."""
    import time as _time

    from flash.cli.commands import _log_follow_progress

    status = {
        "state": "running",
        "last_heartbeat": {"stage": "sft_initializing", "step": 3, "ts": _time.time() - 41},
    }
    state, progress = _log_follow_progress(status, "unknown")
    assert state == "running"
    assert "stage=sft_initializing" in progress
    assert "step=3" in progress
    assert "hb=<1m" in progress

    status["last_heartbeat"]["ts"] = _time.time() - 500
    _, progress = _log_follow_progress(status, "unknown")
    assert "hb=8m" in progress

    state, progress = _log_follow_progress({"state": "running"}, "unknown")
    assert "hb=" not in progress  # no heartbeat yet -> no fabricated age

    malformed = {"state": "running", "last_heartbeat": {"stage": "sft_step", "ts": "oops"}}
    _, progress = _log_follow_progress(malformed, "unknown")
    assert "hb=" not in progress  # non-numeric ts -> no fabricated age


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_log_follow_progress_explains_rl_warmup(stage: str) -> None:
    import time as _time

    from flash.cli.commands import _log_follow_progress

    status = {"state": "running", "last_heartbeat": {"stage": stage, "ts": _time.time()}}
    _, progress = _log_follow_progress(status, "unknown")

    assert f"warming up (stage={stage})" in progress
    assert "typically several minutes, sometimes 15-20 min" in progress
    assert "setup is not billed" in progress
    assert "do not cancel" in progress

    status["last_heartbeat"]["stage"] = "rl_step"
    _, progress = _log_follow_progress(status, "unknown")
    assert "warming up" not in progress
    assert "not billed" not in progress


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_log_follow_progress_omits_warmup_claim_for_stale_heartbeat(stage: str) -> None:
    import time as _time

    from flash.cli.commands import _log_follow_progress

    status = {
        "state": "running",
        "last_heartbeat": {"stage": stage, "ts": _time.time() - 1201},
    }
    _, progress = _log_follow_progress(status, "unknown")

    assert f"stage={stage}" in progress
    assert "hb=20m" in progress
    assert "warming up" not in progress
    assert "do not cancel" not in progress


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_log_follow_progress_omits_warmup_claim_for_prior_attempt_heartbeat(stage: str) -> None:
    import time as _time

    from flash.cli.commands import _log_follow_progress

    # remote is on attempt 1 while last_heartbeat is the previous attempt's fresh setup ping: the
    # warmup reassurance must not fire against a superseded attempt before the new worker publishes.
    status = {
        "state": "running",
        "remote": {"attempt": 1},
        "last_heartbeat": {"stage": stage, "ts": _time.time(), "attempt": 0},
    }
    _, progress = _log_follow_progress(status, "unknown")

    assert f"stage={stage}" in progress
    assert "warming up" not in progress
    assert "do not cancel" not in progress


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_log_follow_progress_explains_warmup_when_heartbeat_matches_attempt(stage: str) -> None:
    import time as _time

    from flash.cli.commands import _log_follow_progress

    status = {
        "state": "running",
        "remote": {"attempt": 3},
        "last_heartbeat": {"stage": stage, "ts": _time.time(), "attempt": 3},
    }
    _, progress = _log_follow_progress(status, "unknown")

    assert f"warming up (stage={stage})" in progress
    assert "do not cancel" in progress
