"""The CLI reports ordinary user errors as a clean one-line message, not a traceback."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(args, env=None):
    full_env = os.environ.copy()
    full_env.pop("FREESOLO_API_KEY", None)  # never let the host's login leak into tests
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "flash.cli", *args],
        cwd=ROOT,
        text=True,
        env=full_env,
        capture_output=True,
        timeout=30,
    )


def _logged_out_env(tmp):
    home = os.path.join(tmp, "home")
    os.makedirs(home, exist_ok=True)
    return {"HOME": home}  # no ~/.flash/config.json -> no Flash key


def test_logged_out_status_is_friendly():
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run(["status", "does-not-exist"], env=_logged_out_env(tmp))
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:")
    assert "flash login" in proc.stderr
    # No raw traceback on stderr.
    assert "Traceback (most recent call last)" not in proc.stderr


def test_bad_model_is_friendly():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write(
                'model = "Not/AReal-Model"\nalgorithm = "grpo"\n'
                '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
            )
        proc = _run(["train", cfg, "--dry-run"], env=_logged_out_env(tmp))
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:")
    assert "unsupported model" in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


def test_missing_config_is_friendly():
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run(
            ["train", os.path.join(tmp, "nope.toml"), "--dry-run"], env=_logged_out_env(tmp)
        )
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:")
    assert "config file not found" in proc.stderr
    # a bare [Errno 2] string and a traceback are both the wrong UX for a mistyped path.
    assert "Errno" not in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


def test_config_pointed_at_a_directory_is_friendly():
    with tempfile.TemporaryDirectory() as tmp:
        # `flash train configs/` or `flash train .` — a directory, not a .toml file.
        proc = _run(["train", tmp, "--dry-run"], env=_logged_out_env(tmp))
    assert proc.returncode == 1
    assert proc.stderr.startswith("error:")
    assert "is a directory" in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr


def test_debug_flag_shows_traceback():
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run(["--debug", "status", "does-not-exist"], env=_logged_out_env(tmp))
    assert proc.returncode != 0
    assert "Traceback (most recent call last)" in proc.stderr


def test_train_without_login_fails_fast():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-4B"\nalgorithm = "grpo"\n'
                '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n[train]\nsteps = 1\nhf_repo = "owner/runs"\n'
            )
        proc = _run(["train", cfg], env=_logged_out_env(tmp))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    # It must fail *before* contacting anything, with the fix spelled out.
    assert "not logged in" in proc.stderr
    assert "flash login" in proc.stderr


def test_missing_env_id_rejected_client_side():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write('model = "Qwen/Qwen3.5-4B"\nalgorithm = "grpo"\n[environment]\n[train]\n')
        # A config without [environment] id is rejected before any network call.
        submit = _run(["train", cfg], env=_logged_out_env(tmp))
        assert submit.returncode == 1
        assert "[environment] id" in submit.stderr


def test_dry_run_needs_no_credentials_or_server():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-4B"\nalgorithm = "grpo"\n'
                '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n[train]\nsteps = 1\nhf_repo = "owner/runs"\n'
            )
        proc = _run(["train", cfg, "--dry-run"], env=_logged_out_env(tmp))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '"state": "dry_run"' in proc.stdout
    assert "live GPU pricing unavailable" not in proc.stderr


def test_cost_needs_no_live_pricing():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "run.toml")
        with open(cfg, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-4B"\nalgorithm = "grpo"\n'
                '[environment]\nid = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
                '[train]\nsteps = 1\nhf_repo = "owner/runs"\n'
            )
        proc = _run(["train", cfg, "--cost"], env=_logged_out_env(tmp))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TOTAL" in proc.stdout
    assert "live GPU pricing unavailable" not in proc.stderr
