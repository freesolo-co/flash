"""Deployment-topology configuration: data dir, logging, and server bind address.

These six variables exist for one reason: an operator who is not Freesolo cannot hardcode where
state lives, what the server binds to, or what it logs. Every one of them defaults to exactly the
value that was compiled in before, so an existing deployment that sets none of them is unaffected.
That "unaffected" claim is the first thing tested here, because it is the one that would break a
running control plane rather than merely inconvenience a new one.
"""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys

import pytest

from flash._internal import logging as flash_logging
from flash._internal.paths import DATA_DIR_ENV, data_dir

# Modules that read the data dir at import time. Resolution is centralized, but these bind the
# result to a module constant, so the subprocess probes below re-import rather than reload.
_STATE_MODULES = (
    "flash.client.config",
    "flash.runner.lifecycle.state",
    "flash.server.platform.db",
)


def _resolve_paths_in(env_overrides: dict[str, str]) -> dict[str, str]:
    """Import the state modules in a fresh interpreter and report the paths they chose.

    A subprocess rather than importlib.reload: these constants are read at import, and other
    modules hold references to them, so reloading in-process would leave a mix of old and new
    bindings and prove nothing about a real startup.
    """
    code = (
        "import json\n"
        "import flash.client.config as c, flash.runner.lifecycle.state as r, "
        "flash.server.platform.db as d\n"
        "from flash._internal.paths import data_dir\n"
        "print(json.dumps({\n"
        "  'config': str(c.CONFIG_PATH),\n"
        "  'config_dir': str(c.CONFIG_DIR),\n"
        "  'runs': r.RUNS_DIR,\n"
        "  'results': r.RESULTS_DIR,\n"
        "  'db': d.DB_PATH,\n"
        "  'root': str(data_dir()),\n"
        "}))\n"
    )
    import os

    env = {k: v for k, v in os.environ.items() if k != DATA_DIR_ENV}
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    return json.loads(result.stdout)


class TestDataDir:
    def test_unset_resolves_exactly_where_state_already_lived(self, tmp_path, monkeypatch):
        """The whole change must be invisible to a deployment that sets nothing.

        Every one of these paths was previously spelled out independently as ``~/.flash/...``.
        If consolidating them moved any single one, an upgrade would silently orphan that
        deployment's logins, run records, or database.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        paths = _resolve_paths_in({"HOME": str(tmp_path)})
        assert paths["config"] == f"{tmp_path}/.flash/config.json"
        assert paths["config_dir"] == f"{tmp_path}/.flash"
        assert paths["runs"] == f"{tmp_path}/.flash/runs"
        assert paths["results"] == f"{tmp_path}/.flash/results"
        assert paths["db"] == f"{tmp_path}/.flash/server.db"

    def test_setting_it_moves_every_consumer_together(self, tmp_path):
        """State must not split: one root, or the CLI and the server disagree about a run."""
        root = tmp_path / "srv" / "flash"
        paths = _resolve_paths_in({DATA_DIR_ENV: str(root)})
        assert paths["config"] == f"{root}/config.json"
        assert paths["runs"] == f"{root}/runs"
        assert paths["results"] == f"{root}/results"
        assert paths["db"] == f"{root}/server.db"
        assert paths["root"] == str(root)

    def test_tilde_is_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(DATA_DIR_ENV, "~/elsewhere")
        assert data_dir() == tmp_path / "elsewhere"

    def test_blank_value_falls_back_to_the_default(self, monkeypatch, tmp_path):
        """An empty or whitespace-only value is an unset variable, not a request for CWD."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(DATA_DIR_ENV, "   ")
        assert data_dir() == tmp_path / ".flash"

    def test_resolution_is_not_frozen_at_import(self, monkeypatch, tmp_path):
        """A process that sets the variable after importing flash still gets the new root."""
        monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "first"))
        assert data_dir() == tmp_path / "first"
        monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "second"))
        assert data_dir() == tmp_path / "second"


class TestLogLevel:
    @pytest.fixture(autouse=True)
    def _restore_logger(self):
        logger = logging.getLogger("flash")
        before = (logger.level, list(logger.handlers))
        yield
        logger.setLevel(before[0])
        logger.handlers[:] = before[1]

    def test_env_var_sets_the_level(self, monkeypatch):
        monkeypatch.setenv("FLASH_LOG_LEVEL", "DEBUG")
        flash_logging.configure_logging()
        assert logging.getLogger("flash").level == logging.DEBUG

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("FLASH_LOG_LEVEL", "debug")
        flash_logging.configure_logging()
        assert logging.getLogger("flash").level == logging.DEBUG

    def test_env_var_overrides_an_entrypoint_default(self, monkeypatch):
        """The server passes default_level=INFO; the operator's value must still win.

        Without this the variable would be inert for the one process that most needs it.
        """
        monkeypatch.setenv("FLASH_LOG_LEVEL", "WARNING")
        flash_logging.configure_logging(default_level=logging.INFO)
        assert logging.getLogger("flash").level == logging.WARNING

    def test_explicit_level_beats_the_env_var(self, monkeypatch):
        """A caller naming a level for one invocation is not overridden by the environment."""
        monkeypatch.setenv("FLASH_LOG_LEVEL", "DEBUG")
        flash_logging.configure_logging(level=logging.ERROR)
        assert logging.getLogger("flash").level == logging.ERROR

    def test_verbosity_flags_beat_the_env_var(self, monkeypatch):
        """`flash -v` is a per-command request and outranks a deployment-wide floor."""
        monkeypatch.setenv("FLASH_LOG_LEVEL", "ERROR")
        flash_logging.configure_logging(verbosity=1)
        assert logging.getLogger("flash").level == logging.INFO

    def test_unparseable_value_falls_back_instead_of_crashing(self, monkeypatch):
        """A typo'd log level must not stop a control plane from booting."""
        monkeypatch.setenv("FLASH_LOG_LEVEL", "LOUD")
        flash_logging.configure_logging(default_level=logging.INFO)
        assert logging.getLogger("flash").level == logging.INFO

    def test_unset_keeps_the_historical_warning_default(self, monkeypatch):
        monkeypatch.delenv("FLASH_LOG_LEVEL", raising=False)
        flash_logging.configure_logging()
        assert logging.getLogger("flash").level == logging.WARNING


class TestLogFormat:
    @pytest.fixture(autouse=True)
    def _restore_logger(self):
        logger = logging.getLogger("flash")
        before = (logger.level, list(logger.handlers))
        yield
        logger.setLevel(before[0])
        logger.handlers[:] = before[1]

    @staticmethod
    def _record() -> logging.LogRecord:
        return logging.LogRecord(
            name="flash.provider",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="allocating %s",
            args=("h100",),
            exc_info=None,
        )

    def _formatter(self):
        flash_logging.configure_logging()
        handlers = [
            h for h in logging.getLogger("flash").handlers if getattr(h, "_flash_console", False)
        ]
        assert len(handlers) == 1
        return handlers[0].formatter

    def test_default_is_the_existing_text_line(self, monkeypatch):
        monkeypatch.delenv("FLASH_LOG_FORMAT", raising=False)
        assert self._formatter().format(self._record()) == "WARNING flash.provider: allocating h100"

    def test_json_emits_one_object_per_line(self, monkeypatch):
        monkeypatch.setenv("FLASH_LOG_FORMAT", "json")
        line = self._formatter().format(self._record())
        assert "\n" not in line
        assert json.loads(line) == {
            "level": "WARNING",
            "logger": "flash.provider",
            "message": "allocating h100",
        }

    def test_json_carries_no_more_than_the_text_line(self, monkeypatch):
        """Switching format must not change WHAT is logged, only how it is punctuated.

        A JSON formatter that reached into the record for extra fields could publish something a
        caller only ever intended for a local console.
        """
        monkeypatch.setenv("FLASH_LOG_FORMAT", "json")
        payload = json.loads(self._formatter().format(self._record()))
        assert set(payload) == {"level", "logger", "message"}

    def test_unknown_format_falls_back_to_text(self, monkeypatch):
        monkeypatch.setenv("FLASH_LOG_FORMAT", "yaml")
        assert self._formatter().format(self._record()) == "WARNING flash.provider: allocating h100"


class TestServerBind:
    @staticmethod
    def _main():
        return importlib.import_module("flash.server.asgi.cli")

    def _parsed(self, monkeypatch, argv, env):
        main = self._main()
        for key in ("FLASH_SERVER_HOST", "FLASH_SERVER_PORT"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        captured = {}
        monkeypatch.setattr(
            main, "run_server", lambda host, port: captured.update(host=host, port=port)
        )
        monkeypatch.setattr(main, "configure_logging", lambda **_: None)
        main.main(argv)
        return captured

    def test_defaults_match_the_previous_hardcoded_values(self, monkeypatch):
        assert self._parsed(monkeypatch, [], {}) == {"host": "127.0.0.1", "port": 8080}

    def test_env_vars_set_the_bind_address(self, monkeypatch):
        got = self._parsed(
            monkeypatch, [], {"FLASH_SERVER_HOST": "0.0.0.0", "FLASH_SERVER_PORT": "9000"}
        )
        assert got == {"host": "0.0.0.0", "port": 9000}

    def test_flags_beat_the_env_vars(self, monkeypatch):
        """An operator typing --port meant that port, whatever the platform injected."""
        got = self._parsed(
            monkeypatch,
            ["--host", "10.0.0.5", "--port", "1234"],
            {"FLASH_SERVER_HOST": "0.0.0.0", "FLASH_SERVER_PORT": "9000"},
        )
        assert got == {"host": "10.0.0.5", "port": 1234}

    @pytest.mark.parametrize("value", ["not-a-port", "0", "65536", "-1"])
    def test_unusable_port_refuses_to_start(self, monkeypatch, value):
        """Falling back to 8080 would leave the plane running on a port nothing routes to.

        A control plane that is up but unreachable looks healthy and is not, which is a worse
        failure than refusing to boot with the reason on stderr.
        """
        with pytest.raises(SystemExit) as excinfo:
            self._parsed(monkeypatch, [], {"FLASH_SERVER_PORT": value})
        assert "FLASH_SERVER_PORT" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["not-a-port", "0", "65536", "-1"])
    def test_port_flag_wins_even_when_the_env_var_is_unusable(self, monkeypatch, value):
        """Precedence must hold for MALFORMED values, not just valid ones.

        Resolving the environment while building the argparse defaults evaluates it before
        parsing, so a broken variable outranked the very flag meant to override it -- the
        operator's escape hatch failed exactly when they needed it.
        """
        assert self._parsed(monkeypatch, ["--port", "9999"], {"FLASH_SERVER_PORT": value}) == {
            "host": "127.0.0.1",
            "port": 9999,
        }

    def test_help_works_with_an_unusable_port_env_var(self, monkeypatch):
        """`--help` must survive a broken environment: it is how you learn the flag exists.

        Erroring here sends someone looking for the override to a message about the thing they
        are trying to override.
        """
        main = self._main()
        monkeypatch.setenv("FLASH_SERVER_PORT", "not-a-port")
        monkeypatch.setattr(main, "run_server", lambda host, port: None)
        monkeypatch.setattr(main, "configure_logging", lambda **_: None)
        with pytest.raises(SystemExit) as excinfo:
            main.main(["--help"])
        assert excinfo.value.code == 0

    def test_blank_host_falls_back_to_the_default(self, monkeypatch):
        assert self._parsed(monkeypatch, [], {"FLASH_SERVER_HOST": "  "})["host"] == "127.0.0.1"

    def test_server_configures_logging_at_info(self, monkeypatch):
        """The server used to attach no handler at all, so its own logs went nowhere.

        This is the assertion that the observability gap stays closed: without the call, every
        provider-resolution and degraded-config message is discarded by the NullHandler.
        """
        main = self._main()
        seen = {}
        monkeypatch.setattr(main, "run_server", lambda host, port: None)
        monkeypatch.setattr(main, "configure_logging", lambda **kw: seen.update(kw))
        main.main([])
        assert seen == {"default_level": logging.INFO}


class TestServerPreflightFailure:
    """A plane missing operator configuration must say so, not raise through the ASGI stack.

    check_run_preflight also runs inside the lifespan, where uvicorn renders a PreflightError as
    an unhandled startup exception: the actionable text arrives after ~20 frames of
    starlette/fastapi/contextlib, which is where a self-hoster's first error used to land.
    """

    @staticmethod
    def _main():
        return importlib.import_module("flash.server.asgi.cli")

    def _run(self, monkeypatch, capsys, exc):
        main = self._main()
        monkeypatch.setattr(main, "configure_logging", lambda **_: None)

        def _boom(host, port):
            raise exc

        monkeypatch.setattr(main, "run_server", _boom)
        code = main.main([])
        return code, capsys.readouterr()

    def test_missing_operator_config_exits_3_with_the_message_on_stderr(self, monkeypatch, capsys):
        from flash.providers.core.preflight import PreflightError

        message = (
            "the Flash control plane is missing required operator configuration:\n"
            "  - HF_TOKEN: a token with write access\n\nSee SELF_HOSTING.md."
        )
        code, captured = self._run(monkeypatch, capsys, PreflightError(message))

        # 3 is uvicorn's STARTUP_FAILURE, which is what this path exited with when the error came
        # out of the lifespan. Supervision keying on it must not see a different code now.
        assert code == 3
        assert "HF_TOKEN" in captured.err
        assert "SELF_HOSTING.md" in captured.err
        assert captured.err.startswith("error: ")

    def test_the_failure_is_not_reported_as_a_traceback(self, monkeypatch, capsys):
        from flash.providers.core.preflight import PreflightError

        _, captured = self._run(
            monkeypatch, capsys, PreflightError("missing FREESOLO_INTERNAL_KEY")
        )

        combined = captured.out + captured.err
        assert "Traceback" not in combined
        for frame in ("starlette", "contextlib", "asgi"):
            assert frame not in combined.lower(), (
                f"{frame!r} in the output means the operator is reading an ASGI stack again"
            )

    def test_an_unexpected_error_still_propagates(self, monkeypatch, capsys):
        """Only PreflightError is a configuration problem. A bug must not be swallowed as exit 3."""
        with pytest.raises(RuntimeError):
            self._run(monkeypatch, capsys, RuntimeError("something actually broke"))


class TestSqliteBusyTimeout:
    def test_default_is_the_previous_thirty_seconds(self, monkeypatch):
        from flash.server.platform import db

        monkeypatch.delenv(db.BUSY_TIMEOUT_ENV, raising=False)
        assert db.busy_timeout_s() == 30.0

    def test_env_var_sets_the_timeout(self, monkeypatch):
        from flash.server.platform import db

        monkeypatch.setenv(db.BUSY_TIMEOUT_ENV, "120")
        assert db.busy_timeout_s() == 120.0

    @pytest.mark.parametrize("value", ["abc", "0", "-5"])
    def test_unusable_value_keeps_the_default(self, monkeypatch, value):
        """A zero or negative timeout would turn ordinary contention into failed API calls."""
        from flash.server.platform import db

        monkeypatch.setenv(db.BUSY_TIMEOUT_ENV, value)
        assert db.busy_timeout_s() == 30.0

    def test_both_connection_paths_read_it(self, monkeypatch, tmp_path):
        """Two independent sites used to hardcode 30s; both must honour the variable.

        Missing one leaves a deployment that raised the timeout still failing on whichever path
        was overlooked, which is indistinguishable from the variable not working.
        """
        from flash.server.platform import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
        monkeypatch.setenv(db.BUSY_TIMEOUT_ENV, "7")
        timeouts = []
        real_connect = db.sqlite3.connect

        def _recording(path, *args, **kwargs):
            timeouts.append(kwargs.get("timeout"))
            return real_connect(path, *args, **kwargs)

        monkeypatch.setattr(db.sqlite3, "connect", _recording)
        db._INITIALIZED_DATABASES.clear()
        db._CONNECTIONS.__dict__.clear()
        db._connect()
        assert timeouts, "no connection was opened"
        assert all(t is not None and t <= 7.0 for t in timeouts), timeouts
