"""Unit tests for docker/kernel_fingerprint.py (the per-arch rebake SSOT).

The whole auto-rebake feature rests on two invariants this file pins:
  * the cheap/expensive split: a base-only change (e.g. an FA3 wheel) moves fp_base but NOT fp_cache
    (cheap re-layer), while a cache-toolchain change (e.g. the fla sha) moves BOTH (paid GPU re-warm),
  * fail-loud parsing: a stale baked cache is a SILENT cold-JIT, so a parser that silently returns a
    constant would hide staleness. Every input must parse, and the parsed values must match the repo.

docker/ is not a package, so import the module by path (the repo already does this style elsewhere).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_kf():
    spec = importlib.util.spec_from_file_location(
        "kernel_fingerprint", ROOT / "docker" / "kernel_fingerprint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kf = _load_kf()


def test_fingerprints_are_deterministic():
    a = kf.fingerprints(ROOT)
    b = kf.fingerprints(ROOT)
    assert a[:2] == b[:2]  # (fp_cache, fp_base) stable across calls
    assert all(re.fullmatch(r"[0-9a-f]{16}", fp) for fp in a[:2])


def test_split_invariant_cheap_vs_expensive():
    """The contract the feature rests on: base-only edits move fp_base only; cache edits move both."""
    cache = {"fla": "sha-A", "from_image": "torch:1"}
    base = {"fa3": "wheel-A", "causal_conv1d": "cc==1"}

    fc0, fb0, _ = kf.compute_fingerprints(cache, base)

    # a base-only change (new FA3 wheel) -> fp_base moves, fp_cache unchanged -> cheap re-layer
    fc1, fb1, _ = kf.compute_fingerprints(cache, {**base, "fa3": "wheel-B"})
    assert fc1 == fc0
    assert fb1 != fb0

    # a cache-toolchain change (new fla sha) -> BOTH move -> paid GPU re-warm
    fc2, fb2, _ = kf.compute_fingerprints({**cache, "fla": "sha-B"}, base)
    assert fc2 != fc0
    assert fb2 != fb0  # fp_base folds fp_cache in, so it always moves when fp_cache does


def test_collect_inputs_populates_every_key_and_matches_repo():
    cache_inputs, base_partial = kf.collect_inputs(ROOT)
    for key in (
        "from_image",
        "fla",
        "tilelang",
        "tvm_ffi",
        "chalk",
        "kernel_warmup_sha256",
    ):
        assert cache_inputs[key], f"cache input {key} not populated"
    for key in (
        "fa2",
        "fa3",
        "causal_conv1d",
        "pip_base",
        "dockerfile_sha256",
        "endpoints_sha256",
        "make_rp_handler_sha256",
    ):
        assert base_partial[key], f"base input {key} not populated"

    # the parsed fla sha must equal the one actually in Dockerfile.worker (guards a silent parser break)
    dockerfile = (ROOT / "Dockerfile.worker").read_text()
    sha = re.search(r"flash-linear-attention\.git@([0-9a-f]{40})\b", dockerfile).group(1)
    assert sha in cache_inputs["fla"]
    # the cache toolchain must NOT leak into the base pip list (else a cache-toolchain bump would fire a re-layer)
    assert not any("liger-kernel" in s or "tilelang" in s for s in base_partial["pip_base"])


def test_bake_kernel_cache_uses_chalk_default_source_of_truth():
    from flash.providers.runpod.train.deps import LATEST_CHALK_MAIN_SHA

    bake_src = (ROOT / "docker" / "bake_kernel_cache.py").read_text()
    assert "DEFAULT_CHALK_SPEC" in bake_src
    assert LATEST_CHALK_MAIN_SHA not in bake_src


def test_dockerfile_only_change_is_a_free_relayer():
    """A Dockerfile.worker edit that isn't a parsed pin (apt/ENV/CMD/cache-dir) must move fp_base
    (a free re-layer) but NOT fp_cache (no paid GPU re-warm). dockerfile_sha256 lives in fp_base."""
    cache_inputs, base_partial = kf.collect_inputs(ROOT)
    fc0, fb0, _ = kf.compute_fingerprints(cache_inputs, base_partial)
    bumped = {**base_partial, "dockerfile_sha256": base_partial["dockerfile_sha256"] + "x"}
    fc1, fb1, _ = kf.compute_fingerprints(cache_inputs, bumped)
    assert fc1 == fc0, "a Dockerfile-only change must NOT move fp_cache (no GPU re-warm)"
    assert fb1 != fb0, "a Dockerfile-only change MUST move fp_base (free re-layer)"


def test_fa3_default_is_in_lockstep():
    """worker-image.yml's FA3 build-arg default must equal Dockerfile.worker's ARG default (today a
    comment-only, untested 'keep in sync'). collect_inputs reads the yml value, so assert it matches."""
    _cache, base_partial = kf.collect_inputs(ROOT)
    dockerfile = (ROOT / "Dockerfile.worker").read_text()
    arg_default = re.search(r"(?m)^ARG FLASH_ATTN_3_SPEC=(\S+)", dockerfile).group(1)
    assert base_partial["fa3"] == arg_default, (
        "Dockerfile.worker ARG FLASH_ATTN_3_SPEC default must match worker-image.yml's build-arg "
        f"default (dockerfile={arg_default}, yml={base_partial['fa3']})"
    )


def test_bake_sms_default_mirrors_baked_per_sm_arches():
    """The auto-rebake gate iterates BAKED_PER_SM_ARCHES; the bake workflow's default `sms` must
    mirror it (today a comment-only claim). Keep them in lockstep."""
    from flash.providers.runpod.train.deps import BAKED_PER_SM_ARCHES

    yml = (ROOT / ".github" / "workflows" / "bake-kernel-cache.yml").read_text()
    default = re.search(r'sms[\s\S]*?default:\s*"([^"]+)"', yml).group(1)
    sms = {s.strip() for s in default.split(",") if s.strip()}
    assert sms == set(BAKED_PER_SM_ARCHES), (
        f"bake default sms {sorted(sms)} must mirror BAKED_PER_SM_ARCHES {sorted(BAKED_PER_SM_ARCHES)}"
    )


def test_parsing_fails_loud_on_a_broken_stack():
    """A missing pin must raise, never silently hash a constant (which would mask a stale cache)."""
    with pytest.raises(ValueError, match="pip stack block"):
        kf._pip_stack_specs("FROM scratch\n# no pip install block here\n")
    with pytest.raises(ValueError, match="could not parse"):
        kf._search(r"NOPE=(\S+)", "nothing here", "a missing thing")
