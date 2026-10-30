"""Retire new OPD requests while retaining persisted jobs and artifact identities."""

import pytest

from flash.core.catalog import public_model_rows, validate_new_run_algorithm
from flash.core.spec import JobSpec
from flash.schema import (
    ConfigError,
    checkpoint_storage_ref,
    parse_adapter_storage_ref,
    spec_and_train_keys_from_file,
)


@pytest.mark.parametrize("algorithm", ["opd", "OPD", "OpD"])
def test_new_opd_requests_explain_migration(algorithm):
    with pytest.raises(ValueError, match="custom algorithm API"):
        validate_new_run_algorithm(algorithm)


def test_public_catalog_excludes_opd_without_mutating_legacy_capabilities():
    from flash.core.catalog import get_model

    assert "opd" in get_model("Qwen/Qwen3.5-9B").algos
    assert all("opd" not in row["algos"] for row in public_model_rows())
    assert "opd" in get_model("Qwen/Qwen3.5-9B").algos


@pytest.mark.parametrize("flags", [[], ["--dry-run"], ["--cost"]])
def test_cli_rejects_opd_before_constructing_a_client(tmp_path, monkeypatch, capsys, flags):
    from flash.cli.commands.ops import train
    from flash.cli.parsing.main import main

    config = tmp_path / "opd.toml"
    config.write_text('algorithm = "opd"\n')
    monkeypatch.setattr(
        train,
        "client_from_config",
        lambda: pytest.fail("retired OPD must be rejected locally"),
    )
    assert main(["train", str(config), *flags]) == 1
    assert "Built-in OPD has been retired" in capsys.readouterr().err


@pytest.mark.parametrize("source", ["file", "extra_config", "override"])
def test_retirement_applies_after_config_composition(tmp_path, source):
    config = tmp_path / "run.toml"
    config.write_text(f'algorithm = "{"opd" if source == "file" else "grpo"}"\n')
    extra = tmp_path / "extra.toml"
    extra.write_text('algorithm = "opd"\n')
    with pytest.raises(ConfigError, match="Built-in OPD has been retired"):
        spec_and_train_keys_from_file(
            str(config),
            extra_configs=[str(extra)] if source == "extra_config" else None,
            overrides=["algorithm=OPD"] if source == "override" else None,
        )


def test_persisted_opd_spec_and_checkpoint_paths_still_round_trip():
    original = JobSpec(algorithm="opd", run_id="legacy-opd")
    restored = JobSpec.from_dict(original.to_internal_dict())
    assert restored.algorithm == "opd"
    assert restored.phase == "opd"
    assert restored.run_id == "legacy-opd"
    ref = checkpoint_storage_ref("owner/artifacts", restored.phase, restored.run_id, 10)
    assert parse_adapter_storage_ref(ref) == (
        "owner/artifacts",
        "opd/legacy-opd/checkpoints/step-10",
    )
