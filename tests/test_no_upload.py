"""`upload = false` opt-out plumbing (CPU-only, no network).

A single per-run boolean, parsed in spec_from_dict and carried by JobSpec end to end, that
decides whether the run is mirrored to the freesolo dashboard. Every reporter in
flash.server.run_registry funnels through one gate, so the same flag covers the run record,
its checkpoint metrics, any deployment made from it, and its export event.

The tests that matter most here are the ones proving the flag CANNOT be invented: a spec that
never set it, or set it to something unparseable, must still report exactly as it did before.
A bug in that direction hides a user's run from their own dashboard silently.
"""

from __future__ import annotations

import types
from unittest import mock

import pytest

from flash.client.specs import spec_payload
from flash.schema import ConfigError, spec_from_dict, spec_from_file
from flash.server import run_registry
from flash.spec import JobSpec
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


@pytest.mark.parametrize("value", [None, "false", 0, "", "no"])
def test_only_an_explicit_false_suppresses(value):
    # a malformed or absent value must never silently hide a run that asked to be shown.
    assert run_registry._upload_suppressed({"upload": value}) is False


def test_spec_without_the_field_still_reports():
    # every run submitted before this field existed carries no `upload` key.
    assert run_registry._upload_suppressed({"model": "m", "project": PROJECT}) is False
