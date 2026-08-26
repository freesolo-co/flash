"""nonshipping serving-profile qualification harness."""

from __future__ import annotations

import pytest

from flash.serve.deployment.profiles import get_profile
from scripts.qualify_serving_profile import build_qualification_plan, main
from tests.test_cli_serve_deploy import CERTIFIED_IMAGE, _args, _stub_resolution


def test_cell_mismatch_fails_before_bundle_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(_args):
        raise AssertionError("bundle resolution ran for the wrong qualification cell")

    monkeypatch.setattr("scripts.qualify_serving_profile._deployment_bundle", _explode)

    with pytest.raises(ValueError, match="does not match CLI cell"):
        build_qualification_plan(_args(dry_run=False), "Qwen/Qwen3.8-27B:modal")


def test_harness_main_is_provider_free_and_prints_only_the_safe_preview(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_resolution(monkeypatch)

    result = main(
        [
            "--qualification-cell",
            "Qwen/Qwen3.8-27B:modal",
            "serve",
            "deploy",
            "--provider",
            "modal",
            "--model",
            "Qwen/Qwen3.8-27B",
            "--run",
            "run1",
            "--deployment-id",
            "deployment1",
            "--image",
            CERTIFIED_IMAGE,
            "--artifact-repo",
            "Freesolo-Co/artifacts",
            "--artifact-subfolder",
            "rl/run1/seed0/adapter",
            "--lora-rank",
            "32",
            "--modal-workspace",
            "workspace",
            "--modal-environment",
            "dev",
            "--modal-region",
            "us-east",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "ModalCreatePlan" in output
    assert "no provider was contacted" in output
    assert "Qwen/Qwen3.8-27B-FP8" in output


def test_named_cell_builds_the_identical_bundle_and_plan_for_qualified_shipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_resolution(monkeypatch)
    args = _args(model="Qwen/Qwen3.8-27B", provider="modal", image=CERTIFIED_IMAGE, dry_run=False)

    bundle, plan = build_qualification_plan(args, "Qwen/Qwen3.8-27B:modal")

    assert bundle.spec.engine.served_model == "Qwen/Qwen3.8-27B-FP8"
    assert plan.bundle is bundle
    assert plan.gpu_request == "H100!:1"
    assert get_profile(args.model).modal_live_qualified is True
