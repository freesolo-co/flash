"""Shared weight-cache isolation, sizing, and cold-retry behavior on Modal."""

from __future__ import annotations

import os
import tempfile

import pytest

import flash.engine.worker.io.heartbeat as worker_heartbeat
import flash.engine.worker.runtime.state as worker_state
import flash.providers._lifecycle.net.worker as provider_worker
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
from flash.core.spec import GpuSpec, JobSpec, TrainSpec
from tests._helpers.profile import satisfy_sft_profile
from tests._helpers.source_snapshot import valid_source_snapshot

_SOURCE_SNAPSHOT = valid_source_snapshot()


def _oversized_model_info():
    """A synthetic catalog entry too large for the shared cache.

    Sized off WEIGHT_CACHE_VOLUME_GB so it stays oversized if the volume grows again — the real
    catalog now fits entirely, so the size gate needs a stand-in to stay covered.
    """
    import flash.runner.accounting.weight_cache as runner
    from flash.core.catalog import ModelInfo

    return ModelInfo(
        id="test/oversized",
        display_name="oversized",
        params="huge",
        algos=("sft",),
        min_vram_gb=80,
        params_b=float(runner.WEIGHT_CACHE_VOLUME_GB),  # peak = 4x params_b GB >> the volume
    )


def _vol_spec(name="flash-weights", gb=100, **gpu):
    return JobSpec(
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(network_volume=name, network_volume_gb=gb, **gpu),
        seed=0,
    )


def test_network_volume_round_trips():
    spec = _vol_spec(gb=123)
    again = JobSpec.from_dict(spec.to_internal_dict())
    assert again.gpu.network_volume == "flash-weights"
    assert again.gpu.network_volume_gb == 123


def test_network_volume_absent_from_public_spec():
    # network_volume / network_volume_gb are platform-managed and must NOT appear in the public spec.
    public = _vol_spec(gb=123).to_dict()
    assert "network_volume" not in public["gpu"]
    assert "network_volume_gb" not in public["gpu"]


def test_default_spec_has_no_volume():
    assert JobSpec(model="m").gpu.network_volume is None


def test_stale_datacenter_key_rejected():
    with pytest.raises(ValueError, match=r"gpu has unknown key\(s\): datacenter"):
        JobSpec.from_dict({"model": "m", "gpu": {"datacenter": "EU-RO-1", "network_volume": "v"}})


def test_network_volume_gb_tolerant_of_bad_values():
    # Platform-managed field: null/empty/"0"/0/negative/non-numeric/missing -> default 100 (never
    # crash int(), never round-trip a nonsensical size). Valid positive sizes pass through.
    for raw in (None, "", 0, "0", -5, "-5", "abc", True, False):
        spec = JobSpec.from_dict(
            {"model": "m", "gpu": {"network_volume": "v", "network_volume_gb": raw}}
        )
        assert spec.gpu.network_volume_gb == 100, f"{raw!r} should default to 100"
    assert (
        JobSpec.from_dict({"model": "m", "gpu": {"network_volume": "v"}}).gpu.network_volume_gb
        == 100
    )
    for raw in (200, "150"):
        spec = JobSpec.from_dict(
            {"model": "m", "gpu": {"network_volume": "v", "network_volume_gb": raw}}
        )
        assert spec.gpu.network_volume_gb == int(raw)


def test_weight_cache_env_is_base_model_scoped():
    from flash.providers._lifecycle.net.worker import weight_cache_env

    env = weight_cache_env("/weight-cache")
    # BASE-MODEL-SCOPED: FLASH_WEIGHT_CACHE_DIR points the base-model prefetch at the mount's HF hub
    # layout. It must NOT set a process-global HF_HOME (that leaked env/reward downloads onto the
    # shared multi-tenant mount — issue #252). The executable kernel-JIT caches are also never on it.
    assert env == {"FLASH_WEIGHT_CACHE_DIR": "/weight-cache/hf-cache/hub"}
    assert "HF_HOME" not in env
    for k in (
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TILELANG_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    ):
        assert k not in env


def test_weight_cache_env_custom_mount():
    from flash.providers._lifecycle.net.worker import weight_cache_env

    assert weight_cache_env("/workspace")["FLASH_WEIGHT_CACHE_DIR"] == "/workspace/hf-cache/hub"


def test_build_worker_env_sets_base_model_cache_with_volume():
    from flash.providers._lifecycle.net.worker import build_worker_env

    env = build_worker_env(_vol_spec(), 0)
    assert env["FLASH_WEIGHT_CACHE_DIR"] == "/weight-cache/hf-cache/hub"
    # The leak fix: no process-global HF_HOME redirect, so env/reward downloads use the ephemeral cache.
    assert "HF_HOME" not in env


def test_build_worker_env_no_cache_without_volume():
    from flash.providers._lifecycle.net.worker import build_worker_env

    env = build_worker_env(JobSpec(model="m", seed=0), 0)
    # Without a volume the base-model cache var must NOT be set (pointing at a missing mount).
    assert "FLASH_WEIGHT_CACHE_DIR" not in env
    assert "HF_HOME" not in env


def _patch_prefetch_io(monkeypatch, ephemeral_hub):
    """Stub the side effects of prefetch_model: a fake snapshot_download that records its call and
    materializes a repo dir under cache_dir, the heartbeat/gpu probes, and the ephemeral hub cache."""
    import huggingface_hub
    import huggingface_hub.constants

    import flash.engine.worker.io.hf as hf

    calls = []

    def _fake_snapshot(repo_id, cache_dir=None, ignore_patterns=None, **kw):
        calls.append(
            {"repo_id": repo_id, "cache_dir": cache_dir, "ignore_patterns": ignore_patterns}
        )
        if (
            cache_dir
        ):  # simulate a real download landing on the (mount) cache: create the repo folder
            folder = "models--" + repo_id.replace("/", "--")
            snap = os.path.join(cache_dir, folder, "snapshots", "deadbeef")
            os.makedirs(snap, exist_ok=True)
            # a real download materializes weight files; prefetch validates they exist
            with open(os.path.join(snap, "model.safetensors"), "w") as f:
                f.write("stub")
            return snap
        return "/ephemeral"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(ephemeral_hub))
    monkeypatch.setattr(hf, "gpu_diagnostics", dict)
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda *a, **k: None)
    return hf, calls


def test_prefetch_model_downloads_to_shared_mount_and_links(tmp_path, monkeypatch):
    """With the shared cache attached, the base model is downloaded ONTO the mount (explicit cache_dir)
    and symlinked into the per-worker ephemeral hub cache so the trainer/vLLM hit it without re-download."""
    import os as _os

    mount = tmp_path / "weight-cache"  # the shared multi-tenant mount (must exist)
    mount.mkdir()
    shared_hub = str(mount / "hf-cache" / "hub")
    ephemeral_hub = tmp_path / "ephemeral" / "hub"
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", shared_hub)
    hf, calls = _patch_prefetch_io(monkeypatch, ephemeral_hub)

    hf.prefetch_model("Qwen/Qwen3.5-9B")

    # downloaded straight onto the shared mount, NOT the ephemeral default
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3.5-9B",
            "cache_dir": shared_hub,
            "ignore_patterns": ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
        }
    ]
    folder = "models--Qwen--Qwen3.5-9B"
    dst = ephemeral_hub / folder
    # the base model is now visible in the ephemeral cache as a SYMLINK pointing back to the mount
    assert _os.path.islink(dst)
    assert _os.path.realpath(dst) == _os.path.realpath(_os.path.join(shared_hub, folder))
    # and ONLY the base model is linked — env/reward repos would resolve in (and be written to) the
    # ephemeral cache, never the shared mount.
    assert [p.name for p in ephemeral_hub.iterdir()] == [folder]


def test_prefetch_model_uses_ephemeral_default_without_shared_cache(tmp_path, monkeypatch):
    """No shared cache attached -> cache_dir=None (the worker's ephemeral default) and no symlink."""
    monkeypatch.delenv("FLASH_WEIGHT_CACHE_DIR", raising=False)
    ephemeral_hub = tmp_path / "ephemeral" / "hub"
    hf, calls = _patch_prefetch_io(monkeypatch, ephemeral_hub)

    hf.prefetch_model("Qwen/Qwen3.5-9B")

    assert calls[0]["cache_dir"] is None  # ephemeral default cache, a correct cold run
    assert not ephemeral_hub.exists()  # nothing linked


def test_prefetch_model_falls_back_to_ephemeral_when_mount_absent(tmp_path, monkeypatch):
    """FLASH_WEIGHT_CACHE_DIR set but the mount isn't present (failed/absent attach) -> ephemeral cache,
    no write under the missing mount. Defense-in-depth re-check on the worker itself."""
    missing_hub = str(tmp_path / "weight-cache" / "hf-cache" / "hub")  # parent mount NOT created
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", missing_hub)
    ephemeral_hub = tmp_path / "ephemeral" / "hub"
    hf, calls = _patch_prefetch_io(monkeypatch, ephemeral_hub)

    hf.prefetch_model("Qwen/Qwen3.5-9B")

    assert (
        calls[0]["cache_dir"] is None
    )  # mount absent -> ephemeral, never the missing /weight-cache path


def test_prefetch_model_starts_no_download_at_deadline(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASH_WEIGHT_CACHE_DIR", raising=False)
    hf, calls = _patch_prefetch_io(monkeypatch, tmp_path / "ephemeral" / "hub")
    monkeypatch.setattr(worker_state, "_remaining_worker_wall_seconds", lambda: 0.0)

    hf.prefetch_model("Qwen/Qwen3.5-9B")

    assert calls == []


def test_shared_weight_cache_dir_requires_the_modal_mount(tmp_path, monkeypatch):
    """Resolve the configured cache only when its persistent mount exists."""
    import flash.engine.worker.io.hf as hf

    monkeypatch.delenv("FLASH_WEIGHT_CACHE_DIR", raising=False)
    assert hf._shared_weight_cache_dir() is None  # unset

    mount = tmp_path / "weight-cache"
    mount.mkdir()
    hub = str(mount / "hf-cache" / "hub")
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", hub)
    assert hf._shared_weight_cache_dir() == hub  # mount present

    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", str(tmp_path / "absent" / "hf-cache" / "hub"))
    assert hf._shared_weight_cache_dir() is None  # mount absent -> ephemeral fallback


def test_assign_weight_cache_attaches_to_a_run():
    import flash.runner.accounting.weight_cache as runner

    out = runner._assign_weight_cache_volume(JobSpec(model="m", run_id="r"))
    assert out.gpu.network_volume == runner.WEIGHT_CACHE_VOLUME_NAME == "flash-weights"
    assert out.gpu.network_volume_gb == runner.WEIGHT_CACHE_VOLUME_GB


def test_assign_weight_cache_keeps_a_custom_volume():
    # A NON-shared (per-org / custom) volume is the intended escape hatch — left intact.
    import flash.runner.accounting.weight_cache as runner

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "run_id": "r",
            "gpu": {
                "type": "A10",
                "network_volume": "org-123-private-cache",
                "network_volume_gb": 100,
            },
        }
    )
    out = runner._assign_weight_cache_volume(spec)
    assert out.gpu.network_volume == "org-123-private-cache"  # not the shared cache -> kept


def test_assign_weight_cache_ignores_removed_kill_switch(monkeypatch):
    # Regression: the FLASH_WEIGHT_CACHE=0 kill switch is GONE — fully managed, always on.
    import flash.runner.accounting.weight_cache as runner

    monkeypatch.setenv("FLASH_WEIGHT_CACHE", "0")
    out = runner._assign_weight_cache_volume(JobSpec(model="m", run_id="r"))
    assert out.gpu.network_volume == "flash-weights"  # env ignored


def test_assign_weight_cache_does_not_override_existing():
    import flash.runner.accounting.weight_cache as runner

    spec = _vol_spec(name="explicit-vol")
    # network_volume is managed -> carried by the internal dict; an already-pinned volume is honored.
    spec = JobSpec.from_dict({**spec.to_internal_dict(), "run_id": "r"})
    out = runner._assign_weight_cache_volume(spec)
    assert out.gpu.network_volume == "explicit-vol"  # an explicit/test value is never clobbered


def test_assign_weight_cache_skips_oversized_catalog_model():
    # SIZE GATE: a model whose peak download footprint exceeds the shared cache must NOT be
    # attached — that would redirect the base-model download onto an undersized mount and overflow
    # mid-download. It is left cache-less and downloads to the container disk instead. Every model
    # in the current catalog fits (see test_every_catalog_model_fits_the_weight_cache), so the gate
    # is exercised with a synthetic oversized entry rather than a real one.
    import flash.runner.accounting.weight_cache as runner

    info = _oversized_model_info()
    out = runner._assign_weight_cache_volume(JobSpec(model=info.id, run_id="r"), info)
    assert out.gpu.network_volume is None  # too big for the shared cache -> cache-less


def test_assign_weight_cache_strips_preset_shared_cache_on_oversized_catalog_model():
    # SIZE GATE re-applies to a pre-set SHARED-cache name: a programmatic/stale catalog spec that
    # already pinned ``flash-weights`` for an oversized model must NOT bypass the gate via the
    # "honor an existing volume" no-op and redirect the download onto the undersized mount. It is
    # stripped cache-less.
    import flash.runner.accounting.weight_cache as runner

    info = _oversized_model_info()
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {
                "type": "B200",
                "network_volume": runner.WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": 100,
            },
        }
    )
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume is None  # oversized -> the pre-set shared cache was stripped


def test_assign_weight_cache_keeps_preset_shared_cache_on_fitting_catalog_model():
    # The re-gate only strips when OVERSIZED: a fitting model that already carries the shared cache
    # keeps it (the pin is correct), exercising the honor-existing path for a shared-name spec.
    import flash.runner.accounting.weight_cache as runner
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {
                "type": "H100",
                "network_volume": runner.WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": 100,
            },
        }
    )
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == "flash-weights"  # fits -> kept


def test_assign_weight_cache_keeps_preset_custom_volume_on_oversized_catalog_model():
    # The re-gate is scoped to the SHARED name only: a custom/per-org volume on an oversized model is
    # left intact (the caller owns its sizing — it may be a 200 GB org cache that DOES fit).
    import flash.runner.accounting.weight_cache as runner
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {
                "type": "B200",
                "network_volume": "org-123-big-cache",
                "network_volume_gb": 400,
            },
        }
    )
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == "org-123-big-cache"  # custom volume honored despite size


def test_assign_weight_cache_attaches_fitting_catalog_model():
    # A model whose download fits the cache (with temp headroom) is still attached when info is passed.
    import flash.runner.accounting.weight_cache as runner
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]  # ~19.4 GB download, peak ~39 GB < 100 GB
    out = runner._assign_weight_cache_volume(JobSpec(model=info.id, run_id="r"), info)
    assert out.gpu.network_volume == "flash-weights"


def test_fits_weight_cache_is_size_based():
    import flash.runner.accounting.weight_cache as runner

    # Oversized models are still excluded: the gate is a size check, not "always true".
    assert not runner._fits_weight_cache(_oversized_model_info())


def test_every_catalog_model_fits_the_weight_cache():
    # The largest models have the slowest cold downloads, so they are exactly the ones the cache
    # must cover. WEIGHT_CACHE_VOLUME_GB must stay >= the peak footprint of the biggest catalog
    # entry; if a larger model is added, grow the volume rather than silently skipping its cache.
    import flash.runner.accounting.weight_cache as runner
    from flash.core.catalog import MODELS

    for mid, info in MODELS.items():
        assert runner._fits_weight_cache(info), (
            f"{mid} ({info.params_b}B) no longer fits the "
            f"{runner.WEIGHT_CACHE_VOLUME_GB} GB weight cache"
        )


def test_submit_job_assigns_weight_cache(monkeypatch):
    # Integration: the assignment is wired into submit_job and visible on the effective worker spec.
    # network_volume is platform-managed -> stripped from the public status.spec, so observe the
    # managed assignment on the effective-preparation worker spec the worker actually runs.
    import flash.runner.accounting.weight_cache as runner

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(tmp, "runs"))
        spec = JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "sft",
                "environment": {"id": "github:o/r@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 8},
                "run_id": "flash-wc-1",
            }
        )
        # sft submission is profile-gated; the cache attachment under test is not, so seed the
        # profile rather than exercising the hub round-trips a real submit performs.
        spec = satisfy_sft_profile(monkeypatch, spec)
        status = runner_submit.submit_job(spec, dry_run=True)
    gpu = status.effective_preparation["worker_spec"]["gpu"]
    assert gpu["network_volume"] == "flash-weights"
    assert gpu["network_volume_gb"] == runner.WEIGHT_CACHE_VOLUME_GB
    # and it must NOT leak into the public spec
    assert "network_volume" not in status.spec["gpu"]


def test_drop_weight_cache_clears_volume():
    from flash.runner.supervise.lifecycle import _drop_weight_cache

    assert _drop_weight_cache(_vol_spec()).gpu.network_volume is None


def test_drop_weight_cache_noop_without_volume():
    from flash.runner.supervise.lifecycle import _drop_weight_cache

    spec = JobSpec(model="m")
    assert _drop_weight_cache(spec) is spec  # no copy when there's nothing to drop


def test_drop_weight_cache_preserves_non_shared_escape_hatch_volume():
    # Review (Copilot): the no-capacity cache-drop must NOT strip a non-shared per-org/custom volume —
    # that is the deliberate escape-hatch isolation the run opted into. Only the SHARED
    # platform cache (WEIGHT_CACHE_VOLUME_NAME) is dropped.
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME
    from flash.runner.supervise.lifecycle import _drop_weight_cache

    custom = _vol_spec(name="org-1234-private")
    assert custom.gpu.network_volume != WEIGHT_CACHE_VOLUME_NAME
    out = _drop_weight_cache(custom)
    assert out is custom  # untouched (no copy) — the escape-hatch volume survives the retry
    assert out.gpu.network_volume == "org-1234-private"
    # the SHARED cache, by contrast, IS dropped
    assert _drop_weight_cache(_vol_spec(name=WEIGHT_CACHE_VOLUME_NAME)).gpu.network_volume is None


def test_effective_spec_persists_managed_cache_removal(monkeypatch):
    # The SHARED platform cache may be dropped on a capacity fallback. network_volume is managed and
    # lives only in the prior preparation snapshot, so the committed shared cache is recorded there;
    # the re-prepared cache-less spec must persist without the removal guard firing.
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        import flash.runner.accounting.weight_cache as runner

        public = JobSpec.from_dict(
            {**_vol_spec().to_internal_dict(), "run_id": "managed-cache-fallback"}
        )
        assert public.gpu.network_volume == runner.WEIGHT_CACHE_VOLUME_NAME
        selected_dict = public.to_internal_dict()
        selected_dict["gpu"]["network_volume"] = None
        selected = JobSpec.from_dict(selected_dict)
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=public.run_id,
                state="provisioning",
                spec=public.to_dict(),
                effective_preparation={
                    "worker_spec": public.to_internal_dict(),  # committed WITH the shared cache
                    "adapter_identity": None,
                    "version": 1,
                    "preparation_digest": "seed",
                },
            )
        )

        assert runner_submit._persist_effective_worker_spec(selected)

        stored = runner_status.get_status(public.run_id)
        assert stored.effective_preparation["worker_spec"]["gpu"]["network_volume"] is None
        assert stored.effective_preparation["adapter_identity"] is None
        assert runner_status.effective_spec_from_status(stored).gpu.network_volume is None


def test_effective_spec_rejects_custom_volume_removal(monkeypatch):
    # A per-org escape-hatch volume a run opted into must never be silently removed.
    # network_volume is managed and no longer travels in the public spec, so the committed custom
    # volume lives only in the prior preparation snapshot; dropping it there must fail closed.
    import pytest

    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        import flash.runner.accounting.weight_cache as runner

        committed = JobSpec.from_dict(
            {
                **_vol_spec(name="org-1234-private").to_internal_dict(),
                "run_id": "custom-cache-fallback",
            }
        )
        assert committed.gpu.network_volume != runner.WEIGHT_CACHE_VOLUME_NAME
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=committed.run_id,
                state="provisioning",
                spec=committed.to_dict(),
                effective_preparation={
                    "worker_spec": committed.to_internal_dict(),  # committed WITH the custom volume
                    "adapter_identity": None,
                    "version": 1,
                    "preparation_digest": "seed",
                },
            )
        )
        selected_dict = committed.to_internal_dict()
        selected_dict["gpu"]["network_volume"] = None
        selected = JobSpec.from_dict(selected_dict)

        with pytest.raises(ValueError, match="effective preparation"):
            runner_submit._persist_effective_worker_spec(selected)


def _supervised_walk(monkeypatch, failures):
    """Run the supervised seed loop, returning per-attempt (gpu.network_volume, gpu.type) tuples.

    ``failures`` maps attempt index -> failure category (absent attempt -> success).
    """
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        import flash.providers.modal.jobs as jobs
        from flash.providers.core.base import PollResult

        seen: list = []

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
            seen.append((spec.gpu.network_volume, spec.gpu.type))
            fail = failures.get(attempt)
            if fail:
                return PollResult(False, failure=fail, detail="x")
            return PollResult(True, metrics={"cost_usd": 0.1})

        monkeypatch.setattr(jobs, "submit_attempt", fake_submit)
        monkeypatch.setattr(
            provider_worker, "publish_source_snapshot", lambda _repo=None: _SOURCE_SNAPSHOT
        )
        monkeypatch.setattr(
            runner_status,
            "validate_terminal_source_metrics",
            lambda _status, metrics, expected_attempt=None: (metrics, expected_attempt),
        )

        spec = JobSpec(
            run_id="wc-walk",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        runner_submit.submit_job(spec, dry_run=False, background=False)
        assert runner_status.get_status("wc-walk").state == "done"
        return seen


def test_no_capacity_drops_weight_cache_on_retry(monkeypatch):
    # attempt 0 lands no_capacity (the cache's DC set was starved) -> the retry runs cache-less on
    # the unrestricted all-DC pool. This is what makes the pin impossible: worst case is a cold run.
    seen = _supervised_walk(monkeypatch, {0: "no_capacity"})
    assert seen[0][0] == "flash-weights"  # first attempt used the cache
    assert seen[1][0] is None  # retry dropped it


def test_poll_error_on_volume_attempt_drops_cache(monkeypatch):
    # A deploy/submit poll_error on a volume-backed attempt (e.g. the SDK failing to create/attach a
    # volume) must degrade to a cold no-volume retry, not loop on the same volume-backed spec.
    seen = _supervised_walk(monkeypatch, {0: "poll_error"})
    assert seen[0][0] == "flash-weights"
    assert seen[1][0] is None


def test_cache_drop_does_not_advance_gpu_walk(monkeypatch):
    # The cache-drop transition must retry the SAME (cheapest) GPU cache-less on the wider pool
    # first — the capacity miss may have been the cache DC set, not the GPU class globally.
    seen = _supervised_walk(monkeypatch, {0: "no_capacity"})
    assert seen[1][1] == seen[0][1]  # same GPU class on the cache-drop retry (walk NOT advanced)


def test_cache_drop_then_walks_on_next_failure(monkeypatch):
    # After the cache is dropped, a SUBSEQUENT failure still walks to the next-cheapest GPU.
    seen = _supervised_walk(monkeypatch, {0: "no_capacity", 1: "stalled"})
    assert seen[0][0] == "flash-weights"
    assert seen[1][0] is None
    assert seen[2][0] is None
    assert seen[1][1] == seen[0][1]  # attempt 1: same GPU, cache dropped (no walk)
    assert seen[2][1] != seen[0][1]  # attempt 2: walked to a different GPU class


def test_non_capacity_failure_keeps_weight_cache(monkeypatch):
    # An ordinary infra flake (stall) on a volume attempt must NOT drop the cache — the warm-weights
    # benefit should survive ordinary retries, and the GPU walk advances as usual.
    seen = _supervised_walk(monkeypatch, {0: "stalled"})
    assert seen[0][0] == "flash-weights"
    assert seen[1][0] == "flash-weights"
    assert seen[1][1] != seen[0][1]  # stall walks to the next GPU (cache retained)


def test_catalog_model_ids_are_the_cache_fitting_catalog():
    from flash.core.catalog import MODELS
    from flash.providers.artifacts import weight_cache as preload
    from flash.runner.accounting.weight_cache import _fits_weight_cache

    ids = set(preload.catalog_model_ids())
    # The default preload set is the catalog RESTRICTED to models that fit the weight cache (warming a
    # non-fitting model only overflows the fixed mount). Mirrors the submit path's _fits_weight_cache.
    assert ids == {mid for mid, info in MODELS.items() if _fits_weight_cache(info)}
    assert ids <= set(MODELS)
    # The whole catalog fits the volume, so the large checkpoints — the ones with the
    # slowest cold downloads — are warmed too.
    assert "Qwen/Qwen3.8-27B" in ids
    assert "Qwen/Qwen3.6-35B-A3B" in ids


def test_volume_holds_whole_catalog_with_largest_model_in_transit():
    """The volume must fit every model resident PLUS the largest one's download scratch.

    Persistent volumes evict nothing, so download order cannot prevent the observed 200 GB
    "Disk quota exceeded" failure.
    """
    from flash.runner.accounting.weight_cache import (
        WEIGHT_CACHE_VOLUME_GB,
        weight_cache_catalog_peak_gb,
    )

    needed = weight_cache_catalog_peak_gb()
    assert needed <= WEIGHT_CACHE_VOLUME_GB, (
        f"catalog needs {needed:.1f} GB at peak but the volume is {WEIGHT_CACHE_VOLUME_GB} GB: "
        "the largest model will fail with Disk quota exceeded on every datacenter"
    )


def test_catalog_peak_counts_others_resident_not_just_the_largest():
    """Pin the shape of the calculation, not just today's numbers.

    The bug was sizing the largest model against an EMPTY volume. Peak must therefore exceed both
    the largest model's own peak and the plain resident total.
    """
    from flash.core.catalog import MODELS
    from flash.runner.accounting.weight_cache import (
        _fits_weight_cache,
        weight_cache_catalog_peak_gb,
    )

    sizes = sorted(
        ((info.params_b or 0.0) * 2.0 for info in MODELS.values() if _fits_weight_cache(info)),
        reverse=True,
    )
    resident_total = sum(sizes)
    largest_alone_peak = 2.0 * sizes[0]
    needed = weight_cache_catalog_peak_gb()

    assert needed > largest_alone_peak, "peak ignores the models already on the volume"
    assert needed > resident_total, "peak ignores the largest model's in-transit scratch"


def test_assign_refreshes_a_stale_shared_cache_size():
    """A spec already pinned to the shared cache must still be re-sized to the MANAGED size.

    A correct volume name can still carry the stale 100 GB size and admit models that need 250 GB.
    """
    import flash.runner.accounting.weight_cache as runner
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {
                "type": "H100",
                "network_volume": runner.WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": 100,
            },
        }
    )
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == runner.WEIGHT_CACHE_VOLUME_NAME
    assert out.gpu.network_volume_gb == runner.WEIGHT_CACHE_VOLUME_GB


def test_assign_leaves_a_custom_volume_size_alone():
    """The size refresh is scoped to the SHARED cache: a caller's own volume keeps its own size.

    A per-org volume is the caller's to size, and rewriting it to the managed number would silently
    change what they are billed for.
    """
    import flash.runner.accounting.weight_cache as runner
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {"type": "H100", "network_volume": "my-org-cache", "network_volume_gb": 100},
        }
    )
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == "my-org-cache"
    assert out.gpu.network_volume_gb == 100
