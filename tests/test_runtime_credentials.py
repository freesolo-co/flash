import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from flash.core.spec import JobSpec
from flash.runner.lifecycle import runtime_credentials as vault
from flash.runner.lifecycle import state, status


@pytest.fixture
def run(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec(run_id="credential-test", model="Qwen/Qwen3.5-9B", algorithm="sft")
    value = state.RunStatus(
        run_id=spec.run_id, state="queued", spec=spec.to_dict(), requires_runtime_secrets=True
    )
    state._save_status(value)
    return spec, value


def test_credentials_survive_a_new_process_without_leaking_into_status(run, tmp_path):
    spec, value = run
    credentials = {"OPENROUTER_API_KEY": "private-request-value"}
    vault.save(spec.run_id, credentials)
    assert vault.restore(status.get_status(spec.run_id)) == credentials
    script = (
        "import sys; from flash.runner.lifecycle import state,status,runtime_credentials as v; "
        "state.RUNS_DIR=sys.argv[1]; "
        "assert v.restore(status.get_status('credential-test')) == "
        "{'OPENROUTER_API_KEY': 'private-request-value'}"
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)
    stored = (tmp_path / f"{spec.run_id}.json").read_text()
    assert "OPENROUTER_API_KEY" not in stored
    assert "private-request-value" not in stored
    assert "requires_runtime_secrets" not in value.to_dict()
    assert b"private-request-value" not in vault._path(spec.run_id).read_bytes()
    assert vault._directory().stat().st_mode & 0o777 == 0o700
    for path in vault._directory().iterdir():
        assert path.stat().st_mode & 0o777 == 0o600


def test_concurrent_key_creation_keeps_every_run_decryptable(run):
    def write(index):
        name = f"concurrent-{index}"
        vault.save(name, {"KEY": str(index)})
        return name

    with ThreadPoolExecutor(max_workers=8) as pool:
        names = list(pool.map(write, range(16)))
    for index, name in enumerate(names):
        value = state.RunStatus(run_id=name, state="queued", spec={}, requires_runtime_secrets=True)
        assert vault.restore(value) == {"KEY": str(index)}


@pytest.mark.parametrize("corruption", ["missing", "tampered", "wrong-run", "missing-key"])
def test_missing_or_corrupt_store_fails_closed_without_operator_key_fallback(
    run, monkeypatch, corruption
):
    spec, value = run
    vault.save(spec.run_id, {"OPENROUTER_API_KEY": "private-request-value"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "different-operator-key")
    if corruption == "missing":
        vault.discard(spec.run_id)
    elif corruption == "missing-key":
        (vault._directory() / ".key").unlink()
    elif corruption == "wrong-run":
        vault.save("another-run", {"OPENROUTER_API_KEY": "wrong-tenant-key"})
        vault._path(spec.run_id).write_bytes(vault._path("another-run").read_bytes())
    else:
        vault._path(spec.run_id).write_bytes(b"corrupted-private-request-value")
    with pytest.raises(ValueError, match="cannot restore") as error:
        vault.restore(value)
    assert "private-request-value" not in str(error.value)
    assert "different-operator-key" not in str(error.value)


@pytest.mark.parametrize("terminal", ["done", "failed", "cancelled", "dry_run"])
def test_terminal_status_purges_credentials(run, terminal):
    spec, value = run
    vault.save(spec.run_id, {"KEY": "secret"})
    value.state = terminal
    state._save_status(value)
    assert not vault._path(spec.run_id).exists()


def test_recovered_training_forwards_original_credentials_to_the_provider(run, monkeypatch):
    from flash.runner.supervise import lifecycle, recovery

    spec, _ = run
    credentials = {"OPENROUTER_API_KEY": "request-key"}
    vault.save(spec.run_id, credentials)
    captured = {}

    def train(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(lifecycle, "_run_attempts_supervised", train)
    monkeypatch.setattr(recovery, "_adopt_completed_attempt", lambda *args, **kwargs: True)
    lifecycle._run_training(spec, io.StringIO(), prior_cost=0, source_snapshot={})
    assert captured["runtime_secrets"] == credentials


def test_handleless_recovery_preserves_credentials_and_launch_claim(run, monkeypatch):
    from flash.runner.lifecycle import attempts
    from flash.runner.supervise import lifecycle
    from flash.server.platform import runtime

    spec, _ = run
    vault.save(spec.run_id, {"KEY": "request-key"})
    monkeypatch.setattr(status, "source_snapshot_from_status", lambda *args, **kwargs: {})
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    claim = object()
    monkeypatch.setattr(
        attempts,
        "reserve_handleless_recovery_launch",
        lambda *args, **kwargs: SimpleNamespace(active=False, claim=claim),
    )
    calls = []
    monkeypatch.setattr(lifecycle, "_run_job_background", lambda *args: calls.append(args))
    assert runtime._start_handleless_resubmit(spec, "queued")
    assert calls == [(spec, {"KEY": "request-key"}, claim)]


def test_submission_persists_credentials_before_dispatch_and_skips_dry_run(run, monkeypatch):
    from flash.runner.lifecycle import submit
    from tests._helpers.source_snapshot import valid_source_snapshot

    spec, _ = run
    monkeypatch.setattr(submit.preparation, "_mark_warmstart_source", lambda *args: None)
    monkeypatch.setattr(
        submit.provider_worker, "publish_source_snapshot", lambda *args: valid_source_snapshot()
    )
    monkeypatch.setattr(submit.reporting, "_report_status", lambda *args: None)
    captured = []
    credentials = {"KEY": "secret"}

    def dispatch(_work):
        captured.append(vault.restore(status.get_status(spec.run_id)))

    kwargs = {
        "prepared": submit.PreparedJob(spec, spec, 1),
        "dry_run": False,
        "runtime_secrets": credentials,
        "billing_context": None,
        "platform_context": None,
        "dispatch": dispatch,
    }
    submit._submit_prepared_job(**kwargs)
    assert captured == [credentials]
    vault.discard(spec.run_id)
    submit._submit_prepared_job(**{**kwargs, "dry_run": True})
    assert not vault._path(spec.run_id).exists()


def test_failed_dispatch_keeps_durable_credentials_recoverable(run, monkeypatch):
    from flash.server.routes import runs

    spec, _ = run
    vault.save(spec.run_id, {"KEY": "secret"})
    monkeypatch.setattr(runs, "_record_environment_use", lambda *args, **kwargs: None)
    monkeypatch.setattr(runs.db, "delete_run", lambda *args: pytest.fail("must retain ownership"))
    runs._dispose_failed_submission(
        spec.run_id,
        dry_run=False,
        had_runtime_secrets=True,
        environment_slug=None,
        project_id="",
        reporting_key={},
    )
    assert status.get_status(spec.run_id).state == "queued"
    assert vault.restore(status.get_status(spec.run_id)) == {"KEY": "secret"}
