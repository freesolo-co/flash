"""Shared helper that satisfies the mandatory sft workload-profile gate.

``prepare_job`` fails closed for sft until a completed profile with a matching input digest
exists, and the digest covers the immutable model revision and environment sha. Pinning the
model revision then makes sizing revision-aware too, so a real sft submission performs three
hub/github round-trips before it reaches the behaviour most tests are actually about.

Use this only where the test's subject is something else (disk floors, artifact repos, provider
acceptance). Tests that assert profile behaviour itself build their own records so a helper bug
cannot mask a contract bug.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

_MODEL_REVISION = "a" * 40
_ENV_REVISION = "b" * 40
# provenance, not measurement: fixed so a helper-built profile stays byte-identical across runs.
_PROFILE_CREATED_AT = 1_780_000_000.0


def _pin_model_revision(spec: Any) -> Any:
    return spec if spec.model_revision else replace(spec, model_revision=_MODEL_REVISION)


def _pin_env_revision(spec: Any) -> Any:
    env = spec.environment
    if not env.id or env.resolved_sha:
        return spec
    return replace(spec, environment=replace(env, resolved_sha=_ENV_REVISION))


def sft_profile_for(spec: Any, *, input_digest: str, producer_version: str):
    """A schema-valid profile whose shape follows ``spec``'s own train fields.

    The token counts are synthetic: these tests assert that a *matching* profile unblocks
    preparation, not that the measured workload is right (``tests/test_sft_workload.py`` owns
    that). What must stay faithful is the identity the runner and the cost path re-derive --
    digest, producer version, tokenizer revision -- and the internal consistency the schema
    enforces.
    """
    from flash.workload_profile import SftWorkloadProfile, sft_sample_policy

    train = spec.train
    retained = max(1, int(train.max_examples or 8))
    epochs = max(1, int(train.epochs or 1))
    max_length = max(1, int(train.max_context_tokens or 1024))
    per_update = max(1, int(train.batch_size or 8))
    blocks = -(-retained // per_update)  # ceil
    real = retained * 256
    return SftWorkloadProfile(
        input_digest=input_digest,
        producer_version=producer_version,
        tokenizer_revision=spec.model_revision,
        environment_id=spec.environment.id,
        environment_revision=spec.environment.resolved_sha,
        source_examples=retained,
        selected_examples=retained,
        retained_examples=retained,
        dropped_examples=0,
        epochs=epochs,
        max_length=max_length,
        packing_mode="packed",
        architecture_mode="pure-attention",
        packed_blocks=blocks,
        real_tokens_per_epoch=real,
        supervised_tokens_per_epoch=real // 2,
        padded_compute_tokens_per_epoch=real,
        authoritative_real_tokens=real * epochs,
        authoritative_supervised_tokens=real // 2 * epochs,
        authoritative_compute_tokens=real * epochs,
        realized_max_length=min(256, max_length),
        examples_per_update=per_update,
        derived_steps=blocks * epochs,
        authoritative_steps=int(train.max_steps or blocks * epochs),
        packing_efficiency=1.0,
        sample_policy=sft_sample_policy(train.max_examples),
        created_at=_PROFILE_CREATED_AT,
    )


def stub_revision_geometry(monkeypatch) -> None:
    """Resolve revision-specific sizing from the catalog instead of the hub.

    A pinned model revision makes sizing revision-aware, so every quote -- including the lifecycle's
    post-allocation refresh -- would otherwise reach github. The stub returns the catalog's own
    numbers, which is what a matching pin resolves to anyway; revision geometry has its own
    dedicated coverage.
    """
    from flash.engine import vram

    monkeypatch.setattr(
        vram,
        "_validated_revision_geometry",
        lambda _model_id, _revision, info: (info.params_b, info.vocab_size),
    )


def attach_sft_profile(spec: Any) -> Any:
    """Return ``spec`` in the post-``prepare_job`` shape: revisions pinned, profile attached.

    For tests that call a lifecycle entrypoint directly rather than going through submission.
    ``prepare_job`` is the only producer of an sft spec, and it cannot emit one without a matching
    profile, so an unattached sft spec is a shape the lifecycle never legitimately sees. Pure: no
    monkeypatching and no state store, because the lifecycle re-derives the digest from the spec
    itself and never looks the profile record up again.
    """
    if spec.algorithm != "sft":
        return spec

    from dataclasses import replace as _replace

    from flash import __version__
    from flash.workload_profile import sft_profile_input_digest

    pinned = _pin_env_revision(_pin_model_revision(spec))
    if not pinned.environment.id:
        pinned = _replace(
            pinned,
            environment=_replace(
                pinned.environment,
                id="github:owner/repo@main:env/environment.py",
                resolved_sha=_ENV_REVISION,
            ),
        )
    digest = sft_profile_input_digest(
        pinned,
        tokenizer_revision=pinned.model_revision,
        producer_version=__version__,
    )
    profile = sft_profile_for(pinned, input_digest=digest, producer_version=__version__)
    return _replace(
        pinned,
        workload_profile_input_digest=digest,
        workload_profile_producer_version=__version__,
        workload_profile=profile.to_dict(),
    )


def record_sft_profile(runner, spec: Any) -> str:
    """Store the completed profile record for an ALREADY-pinned ``spec``; return its digest.

    This is the state a real resubmit finds: the profile job ran to completion and its record is in
    the store, so ``prepare_job`` re-derives the digest and reads the profile instead of queueing
    another one. Nothing is stubbed, so a test whose subject IS the pinning can pin for real and
    still get past the gate. Call it after ``RUNS_DIR`` is redirected -- it writes through the
    runner's own state store -- and pass the same shas the runner will resolve, because a digest
    keyed on anything else is exactly the mismatch the gate is built to reject.
    """
    from flash.workload_profile import (
        SFT_PROFILE_KIND,
        sft_profile_input_digest,
        sft_profile_run_id,
    )

    producer_version = runner._profile_producer_version()
    digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=spec.model_revision,
        producer_version=producer_version,
    )
    profile = sft_profile_for(spec, input_digest=digest, producer_version=producer_version)
    runner._save_status(
        runner.RunStatus(
            run_id=sft_profile_run_id(digest),
            state="done",
            spec=spec.to_dict(),
            workload_profile_kind=SFT_PROFILE_KIND,
            workload_profile_input_digest=digest,
            workload_profile_producer_version=producer_version,
            workload_profile=profile.to_dict(),
        )
    )
    return digest


def satisfy_sft_profile(runner, monkeypatch, spec: Any):
    """Pin the immutable revisions ``spec`` needs and record its completed profile.

    Returns the pinned spec (the shape ``prepare_job`` will re-derive), so a caller can assert
    against the same identity the runner sees. Call it *after* ``RUNS_DIR`` is redirected: the
    profile record is written through the runner's own state store. A non-sft spec is left
    completely alone, so a parametrized suite can call this for every algorithm.
    """
    if spec.algorithm != "sft":
        return spec

    # The three hub/github round-trips a pinned sft submission performs, replaced in place. Each
    # has its own dedicated coverage (revision resolution, env-sha pinning, revision geometry);
    # reaching the network here would make every sft submit test an integration test.
    monkeypatch.setattr(runner, "_resolve_model_revision", lambda s, **_kw: _pin_model_revision(s))
    monkeypatch.setattr(runner, "_assign_resolved_env_sha", _pin_env_revision)
    stub_revision_geometry(monkeypatch)

    pinned = _pin_env_revision(_pin_model_revision(spec))
    record_sft_profile(runner, pinned)
    return pinned
