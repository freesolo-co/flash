"""`upload = false` opt-out plumbing (CPU-only, no network).

A single per-run boolean, parsed in spec_from_dict and carried by JobSpec end to end, that
decides whether the run is mirrored to the freesolo dashboard. Five backend reporters carry a
run id and each sits behind `flash.spec.upload_suppressed`: the run record (which carries any
deployment made from it, as a field), its checkpoint metrics and its export event in
flash.server.run_registry; the deployable-checkpoint batch in flash.server.checkpoints; and the
environment-use post in flash.server.routes.runs.

The tests that matter most here are the ones proving the flag CANNOT be invented: a spec that
never set it, or set it to something unparseable, must still report exactly as it did before.
A bug in that direction hides a user's run from their own dashboard silently.
"""

from __future__ import annotations

import types
from dataclasses import replace
from unittest import mock

import pytest

from flash.client.specs import spec_payload
from flash.schema import ConfigError, spec_from_dict, spec_from_file
from flash.server import checkpoints as server_checkpoints
from flash.server import run_registry
from flash.spec import JobSpec, upload_suppressed
from tests._helpers.specs import raw_spec as _raw

PROJECT = "11111111-1111-4111-8111-111111111111"
DEPLOYMENT = {
    "run_id": "run-1",
    "state": "ready",
    "openai_model": "run-1",
    "adapter_revision": "rev-abc",
}


def _status(spec: dict, deployment: dict | None = None):
    """The RunStatus shape the reporters read, with only the fields they touch."""
    status = types.SimpleNamespace(
        run_id="run-1",
        state="succeeded",
        spec=spec,
        # a provisioned run always carries the internal worker-spec carrier; the deployable
        # checkpoint registry reads run_id + hf_repo from it (see _internal_spec_from_status).
        effective_preparation={"worker_spec": spec},
        platform_context={"org_id": "org-1", "user_id": "u1", "api_key_id": "k1"},
        cost_usd=1.0,
        realized_cost_usd=1.0,
        artifacts_dir="/tmp/artifacts",
        error=None,
        deployment=deployment,
        last_heartbeat=None,
        gpu_status=None,
        created_at=0.0,
        updated_at=1.0,
    )
    status.to_dict = lambda: {"adapter_ref": "ref"}
    return status


def test_upload_defaults_true():
    assert spec_from_dict(_raw()).upload is True


def test_upload_can_be_disabled():
    assert spec_from_dict(_raw(upload=False)).upload is False


def test_upload_must_be_boolean():
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(upload="false"))
    assert "boolean" in str(ei.value)


def test_upload_roundtrips_through_dict():
    # the control plane reconstructs the spec from this dict before the reporters read it, so a
    # dropped field here would report a run that asked to stay private.
    spec = spec_from_dict(_raw(upload=False))
    assert JobSpec.from_dict(spec.to_dict()).upload is False
    assert JobSpec.from_dict(spec.to_internal_dict()).upload is False
    # a dict without the field gets the default (ON)
    assert JobSpec.from_dict({"model": "Qwen/Qwen3.5-0.8B"}).upload is True


@pytest.mark.parametrize("value", [None, True, "true", 1])
def test_from_dict_never_invents_an_opt_out(value):
    # the only failure direction that hides a user's run from their own dashboard. a plain
    # coerce_bool reads None as False, which would suppress a run nobody opted out.
    assert JobSpec.from_dict({"model": "m", "upload": value}).upload is True


@pytest.mark.parametrize("value", [False, "false", "False"])
def test_from_dict_preserves_a_real_opt_out(value):
    # the worker round trip carries the spec through the environment, so False arrives as a string.
    assert JobSpec.from_dict({"model": "m", "upload": value}).upload is False


def test_upload_set_override(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-0.8B"\nalgorithm = "sft"\n'
        f'project = "{PROJECT}"\n\n'
        '[environment]\nid = "github:owner/repo@main:env/environment.py"\n\n'
        "[train]\nepochs = 1\nmax_examples = 8\n"
    )
    assert spec_from_file(str(cfg), overrides=["upload=false"]).upload is False


def test_client_payload_carries_the_flag():
    # the server re-parses this payload, so the wire form has to state the opt-out explicitly.
    assert spec_payload(spec_from_dict(_raw(upload=False)))["upload"] is False
    assert spec_payload(spec_from_dict(_raw()))["upload"] is True


def test_opt_out_does_not_excuse_a_missing_project():
    # visibility and identity are separate axes: a private run is still grouped under its project.
    raw = {k: v for k, v in _raw(upload=False).items() if k != "project"}
    with pytest.raises((ConfigError, ValueError)):
        spec_from_dict(raw, project_required=True)


def test_opted_out_run_posts_nothing_to_the_backend():
    spec = spec_from_dict(_raw(upload=False)).to_internal_dict()
    with mock.patch.object(run_registry, "_post") as post:
        assert run_registry.record_training_run(status=_status(spec)) is False
    assert post.call_count == 0


def test_opted_out_deployment_posts_nothing_to_the_backend():
    # a deployment reaches the dashboard as a FIELD of the run report, not its own endpoint, so
    # suppressing the report is what keeps the deployed model from being displayed.
    spec = spec_from_dict(_raw(upload=False)).to_internal_dict()
    with mock.patch.object(run_registry, "_post") as post:
        assert run_registry.record_training_run(status=_status(spec, DEPLOYMENT)) is False
    assert post.call_count == 0


def test_opted_out_export_event_posts_nothing_to_the_backend():
    spec = spec_from_dict(_raw(upload=False)).to_internal_dict()
    with mock.patch.object(run_registry, "_post") as post:
        assert (
            run_registry.record_model_exported(
                status=_status(spec), repository="owner/repo", url="https://hf/x", step=None
            )
            is False
        )
    assert post.call_count == 0


def test_normal_run_still_reports_with_its_deployment():
    # the control that makes the assertions above meaningful: without it, a gate that suppressed
    # EVERY run would pass all three tests above.
    spec = spec_from_dict(_raw()).to_internal_dict()
    with mock.patch.object(run_registry, "_post", return_value=True) as post:
        assert run_registry.record_training_run(status=_status(spec, DEPLOYMENT)) is True
    assert post.call_count == 1
    assert post.call_args[0][1]["deployment"] == DEPLOYMENT


@pytest.mark.parametrize("value", [None, 0, 0.0, "", {}, [], "maybe"])
def test_a_malformed_value_never_suppresses_end_to_end(value):
    """A value nobody chose must not read as a choice to hide the run.

    Asserted through the REAL path -- from_dict, then the reporter -- rather than against the
    predicate alone. An earlier version of this test called the predicate with a raw "", which
    passed while from_dict was quietly turning that same "" into False upstream. The gate was
    innocent and the run was still hidden; only the round trip catches that.
    """
    spec = JobSpec.from_dict({"model": "m", "project": PROJECT, "upload": value})
    assert spec.upload is True
    assert upload_suppressed(spec.to_internal_dict()) is False
    with mock.patch.object(run_registry, "_post", return_value=True) as post:
        assert run_registry.record_training_run(status=_status(spec.to_internal_dict())) is True
    assert post.call_count == 1


def test_a_replaced_none_does_not_serialize_as_an_opt_out():
    # `replace()` bypasses from_dict, so normalization has to live in __post_init__ too.
    spec = replace(spec_from_dict(_raw()), upload=None)
    assert spec.upload is True
    assert spec_payload(spec)["upload"] is True


def test_spec_without_the_field_still_reports():
    # every run submitted before this field existed carries no `upload` key.
    assert upload_suppressed({"model": "m", "project": PROJECT}) is False


@pytest.mark.parametrize("spec", [None, "", [], {"upload": "false"}])
def test_predicate_only_reads_an_explicit_false_from_a_dict(spec):
    # the reporters hand this whatever the plane persisted, including nothing at all.
    assert upload_suppressed(spec) is False


def test_opted_out_checkpoint_metrics_post_nothing():
    """The per-run metrics reporter, which reads the PERSISTED spec rather than its argument."""
    spec = spec_from_dict(_raw(upload=False))
    persisted = spec.to_internal_dict()
    with (
        mock.patch.object(run_registry, "_post") as post,
        mock.patch("flash.runner.get_status", return_value=_status(persisted)),
        mock.patch("flash.runner.adapter_ref", return_value="ref"),
        mock.patch("flash.engine.accounting.sanitize_worker_metrics", side_effect=lambda m: m),
    ):
        assert (
            run_registry.record_training_checkpoint(
                spec=types.SimpleNamespace(run_id="run-1", phase="sft"),
                metrics={"loss": 0.5},
                artifact_path="/tmp/a",
            )
            is False
        )
    assert post.call_count == 0


def test_normal_run_still_posts_checkpoint_metrics():
    # the control: without it a gate that suppressed every run would pass the test above.
    persisted = spec_from_dict(_raw()).to_internal_dict()
    with (
        mock.patch.object(run_registry, "_post", return_value=True) as post,
        mock.patch("flash.runner.get_status", return_value=_status(persisted)),
        mock.patch("flash.runner.adapter_ref", return_value="ref"),
        mock.patch("flash.engine.accounting.sanitize_worker_metrics", side_effect=lambda m: m),
    ):
        assert (
            run_registry.record_training_checkpoint(
                spec=types.SimpleNamespace(run_id="run-1", phase="sft"),
                metrics={"loss": 0.5},
                artifact_path="/tmp/a",
            )
            is True
        )
    assert post.call_count == 1


def test_opted_out_run_registers_no_deployable_checkpoints():
    """The deployable-checkpoint registry: a fifth run-scoped reporter, in its own module.

    It posts the run id, its HF repo and every deployable step. It does not use
    `post_internal_json` like the run_registry reporters, which is exactly why an enumeration of
    that helper's callers missed it -- it builds its request directly.
    """
    persisted = spec_from_dict(_raw(upload=False)).to_internal_dict()
    with (
        mock.patch.object(server_checkpoints, "internal_key", return_value="int-key"),
        mock.patch.object(server_checkpoints, "list_checkpoints") as listed,
        mock.patch.object(server_checkpoints, "_post_checkpoints") as post,
    ):
        assert server_checkpoints.register_checkpoints_best_effort(_status(persisted)) == 0
    assert post.call_count == 0
    # skipped before the HF listing: an opted-out run does no needless network work either.
    assert listed.call_count == 0


def test_normal_run_still_registers_deployable_checkpoints():
    persisted = spec_from_dict(_raw()).to_internal_dict()
    checkpoints = [{"step": 1, "subfolder": "step-1", "repo_id": "owner/repo"}]
    with (
        mock.patch.object(server_checkpoints, "internal_key", return_value="int-key"),
        mock.patch.object(server_checkpoints, "run_org_id", return_value="org-1"),
        mock.patch.object(server_checkpoints, "list_checkpoints", return_value=checkpoints),
        mock.patch.object(server_checkpoints, "_post_checkpoints", return_value={}) as post,
    ):
        assert server_checkpoints.register_checkpoints_best_effort(_status(persisted)) == 1
    assert post.call_count == 1
    assert post.call_args.kwargs["body"]["runId"] == "run-1"


def _train_config(tmp_path) -> str:
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-0.8B"\nalgorithm = "sft"\n'
        f'project = "{PROJECT}"\n\n'
        '[environment]\nid = "github:owner/repo@main:env/environment.py"\n\n'
        "[train]\nepochs = 1\nmax_examples = 8\n"
    )
    return str(cfg)


@pytest.mark.parametrize(
    ("no_upload", "expected"), [(True, False), (False, True)], ids=["flag", "no-flag"]
)
def test_cli_no_upload_flag_reaches_the_submitted_payload(tmp_path, no_upload, expected):
    # the flag is the only way to opt out without editing the config, so it has to survive all
    # the way into the wire payload the plane re-parses.
    from flash.cli import commands

    args = types.SimpleNamespace(
        config=_train_config(tmp_path),
        overrides=[],
        extra_configs=[],
        cost=False,
        no_upload=no_upload,
    )
    captured: list[dict] = []

    class Submitted(Exception):
        """Stops cmd_train at the payload, so nothing past it has to be stubbed."""

    # bound before the patch: reading it off the module inside `capture` would resolve to the
    # patch itself and recurse.
    real_spec_payload = commands.spec_payload

    def capture(spec, **kwargs):
        captured.append(real_spec_payload(spec, **kwargs))
        raise Submitted

    with (
        mock.patch.object(commands, "client_from_config"),
        mock.patch.object(commands, "spec_payload", capture),
        pytest.raises(Submitted),
    ):
        commands.cmd_train(args)

    assert len(captured) == 1
    assert captured[0]["upload"] is expected


# the `flash env eval --upload` refusal is exercised against the real CLI in
# test_env_evaluations.py, next to the rest of that command's upload contract.
