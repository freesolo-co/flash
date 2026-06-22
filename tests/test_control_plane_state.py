from __future__ import annotations

import importlib
from pathlib import Path


def test_control_plane_state_dir_env_isolates_runner_and_db(monkeypatch, tmp_path):
    import flash.runner as runner
    import flash.server.db as db_mod

    with monkeypatch.context() as m:
        state_dir = tmp_path / "control-plane"
        m.setenv("FLASH_CONTROL_PLANE_STATE_DIR", str(state_dir))
        m.delenv("FLASH_RUNS_DIR", raising=False)
        m.delenv("FLASH_RESULTS_DIR", raising=False)
        m.delenv("FLASH_DB_PATH", raising=False)

        runner = importlib.reload(runner)
        db_mod = importlib.reload(db_mod)

        assert Path(runner.RUNS_DIR) == state_dir / "runs"
        assert Path(runner.RESULTS_DIR) == state_dir / "results"
        assert Path(db_mod.DB_PATH) == state_dir / "server.db"

    importlib.reload(runner)
    importlib.reload(db_mod)


def test_control_plane_state_paths_have_granular_overrides(monkeypatch, tmp_path):
    import flash.runner as runner
    import flash.server.db as db_mod

    with monkeypatch.context() as m:
        m.setenv("FLASH_CONTROL_PLANE_STATE_DIR", str(tmp_path / "ignored-root"))
        m.setenv("FLASH_RUNS_DIR", str(tmp_path / "custom-runs"))
        m.setenv("FLASH_RESULTS_DIR", str(tmp_path / "custom-results"))
        m.setenv("FLASH_DB_PATH", str(tmp_path / "custom.sqlite"))

        runner = importlib.reload(runner)
        db_mod = importlib.reload(db_mod)

        assert Path(runner.RUNS_DIR) == tmp_path / "custom-runs"
        assert Path(runner.RESULTS_DIR) == tmp_path / "custom-results"
        assert Path(db_mod.DB_PATH) == tmp_path / "custom.sqlite"

    importlib.reload(runner)
    importlib.reload(db_mod)
