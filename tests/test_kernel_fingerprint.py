"""Unit tests for docker/kernel_fingerprint.py (the per-arch rebake SSOT).

The whole auto-rebake feature rests on two invariants this file pins:
  * the cheap/expensive split: a base-only change (e.g. an FA2 wheel) moves fp_base but NOT fp_cache
    (cheap re-layer), while a cache-toolchain change (e.g. the fla sha) moves BOTH (paid GPU re-warm),
  * fail-loud parsing: a stale baked cache is a SILENT cold-JIT, so a parser that silently returns a
    constant would hide staleness. Every input must parse, and the parsed values must match the repo.

docker/ is not a package, so import the module by path (the repo already does this style elsewhere).
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
KERNEL_FINGERPRINT_SCRIPT = ROOT / "docker" / "kernel_fingerprint.py"
AUTO_REBAKE_ARCH_COMMAND = "arches=$(python3 docker/kernel_fingerprint.py --print-baked-arches)"


def _load_kf():
    spec = importlib.util.spec_from_file_location("kernel_fingerprint", KERNEL_FINGERPRINT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kf = _load_kf()


def _workflow_block(lines: list[str], *headers: str) -> list[str]:
    for header in headers:
        starts = [i for i, line in enumerate(lines) if line.rstrip() == header]
        if len(starts) != 1:
            raise ValueError(f"expected exactly one {header!r} block, found {len(starts)}")
        start = starts[0]
        indent = len(header) - len(header.lstrip())
        end = next(
            (
                i
                for i, line in enumerate(lines[start + 1 :], start + 1)
                if line.strip()
                and not line.lstrip().startswith("#")
                and len(line) - len(line.lstrip()) <= indent
            ),
            len(lines),
        )
        lines = lines[start + 1 : end]
    return lines


def test_fingerprints_are_deterministic():
    a = kf.fingerprints(ROOT)
    b = kf.fingerprints(ROOT)
    assert a[:2] == b[:2]  # (fp_cache, fp_base) stable across calls
    assert all(re.fullmatch(r"[0-9a-f]{16}", fp) for fp in a[:2])


def test_split_invariant_cheap_vs_expensive():
    """The contract the feature rests on: base-only edits move fp_base only; cache edits move both."""
    cache = {"fla": "sha-A", "from_image": "torch:1"}
    base = {"fa2": "wheel-A", "causal_conv1d": "cc==1"}

    fc0, fb0, _ = kf.compute_fingerprints(cache, base)

    # a base-only change (new FA2 wheel) -> fp_base moves, fp_cache unchanged -> cheap re-layer
    fc1, fb1, _ = kf.compute_fingerprints(cache, {**base, "fa2": "wheel-B"})
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
        "kernel_warmup_sha256",
    ):
        assert cache_inputs[key], f"cache input {key} not populated"
    for key in (
        "fa2",
        "fa2_sha256",
        "causal_conv1d",
        "pip_base",
        "dockerfile_sha256",
        "endpoints_sha256",
        "make_rp_handler_sha256",
        "archive_sha256",
    ):
        assert base_partial[key], f"base input {key} not populated"

    # the parsed fla sha must equal the one actually in Dockerfile.worker (guards a silent parser break)
    dockerfile = (ROOT / "Dockerfile.worker").read_text()
    sha = re.search(r"flash-linear-attention\.git@([0-9a-f]{40})\b", dockerfile).group(1)
    assert sha in cache_inputs["fla"]
    # the cache toolchain must NOT leak into the base pip list (else a cache-toolchain bump would fire a re-layer)
    assert not any("liger-kernel" in s or "tilelang" in s for s in base_partial["pip_base"])


def test_dockerfile_only_change_is_a_free_relayer():
    """A Dockerfile.worker edit that isn't a parsed pin (apt/ENV/CMD/cache-dir) must move fp_base
    (a free re-layer) but NOT fp_cache (no paid GPU re-warm). dockerfile_sha256 lives in fp_base."""
    cache_inputs, base_partial = kf.collect_inputs(ROOT)
    fc0, fb0, _ = kf.compute_fingerprints(cache_inputs, base_partial)
    bumped = {**base_partial, "dockerfile_sha256": base_partial["dockerfile_sha256"] + "x"}
    fc1, fb1, _ = kf.compute_fingerprints(cache_inputs, bumped)
    assert fc1 == fc0, "a Dockerfile-only change must NOT move fp_cache (no GPU re-warm)"
    assert fb1 != fb0, "a Dockerfile-only change MUST move fp_base (free re-layer)"


def test_fa_sha256_digest_only_change_moves_fp_base_not_fp_cache():
    """The exact remediation this fp_base input exists for: a maintainer dispatches worker-image.yml
    with the SAME wheel url but an UPDATED sha256 (a replaced release asset). A digest rotation must
    move fp_base (so auto-rebake re-layers every stale per-arch tag) while leaving fp_cache untouched
    (a wheel digest plays no part in what the Triton/Inductor mega-cache holds, so no GPU re-warm)."""
    fp_cache0, fp_base0, _, _ = kf.fingerprints(ROOT)
    fp_cache1, fp_base1, _, _ = kf.fingerprints(ROOT, fa2_sha256="0" * 64)
    assert fp_cache1 == fp_cache0, "a wheel digest rotation must not invalidate the kernel cache"
    assert fp_base1 != fp_base0, (
        "a wheel digest rotation must move fp_base so auto-rebake re-layers"
    )


def test_cli_fa_sha256_override_moves_fp_base():
    """worker-image.yml invokes the CLI with a resolved --fa2-sha256 value (mirroring --fa2-spec);
    prove the argparse wiring actually reaches base_inputs_partial and not just the Python-level
    collect_inputs kwarg."""
    baseline = json.loads(
        subprocess.run(
            [sys.executable, str(KERNEL_FINGERPRINT_SCRIPT), "--format", "json"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )
    overridden = json.loads(
        subprocess.run(
            [
                sys.executable,
                str(KERNEL_FINGERPRINT_SCRIPT),
                "--format",
                "json",
                "--fa2-sha256",
                "0" * 64,
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )
    assert overridden["fp_cache"] == baseline["fp_cache"]
    assert overridden["fp_base"] != baseline["fp_base"]


def test_the_flash_attn_wheel_digests_are_in_lockstep():
    """The FA2 wheel is verified before install, so its digest is declared twice.

    Dockerfile.worker carries the ARG defaults it verifies against; worker-image.yml passes the same
    values as build-args. A drifted pair means the build verifies against a digest nobody reviewed
    alongside the url, which leaves the fetch effectively unverified.
    """
    dockerfile = (ROOT / "Dockerfile.worker").read_text()
    yml = (ROOT / ".github" / "workflows" / "worker-image.yml").read_text()

    for arg, env in (("FLASH_ATTN_SHA256", "FA2_SHA256"),):
        docker_sha = re.search(rf"(?m)^ARG {arg}=([0-9a-f]{{64}})$", dockerfile)
        assert docker_sha, f"Dockerfile.worker must pin a 64-hex ARG {arg}"
        yml_sha = re.search(rf"{env}:.*?'([0-9a-f]{{64}})'", yml) or re.search(
            rf"(?m)^\s*{env}:\s*([0-9a-f]{{64}})$", yml
        )
        assert yml_sha, f"worker-image.yml must carry {env}"
        assert docker_sha.group(1) == yml_sha.group(1), (
            f"{arg} drift: Dockerfile.worker={docker_sha.group(1)} vs "
            f"worker-image.yml {env}={yml_sha.group(1)}"
        )


def _pyproject_specs(name: str) -> list[str]:
    """Every requirement for `name` declared anywhere in pyproject's dependency tables."""
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data.get("project", {})
    groups: list[list] = [project.get("dependencies", []) or []]
    groups += [v or [] for v in (project.get("optional-dependencies", {}) or {}).values()]
    groups += [v or [] for v in (data.get("dependency-groups", {}) or {}).values()]
    wanted = name.replace("_", "-").lower()
    return [
        str(s)
        for group in groups
        for s in group
        if isinstance(s, str) and kf._pkg_name(s).replace("_", "-").lower() == wanted
    ]


def test_huggingface_hub_floor_is_in_lockstep():
    """The exact worker pin must satisfy every published huggingface_hub compatibility range.

    Both halves need the built-in 429 retry behavior, but the published package must keep compatible
    ranges while the deployed image is exact and reproducible.
    """
    dockerfile = (ROOT / "Dockerfile.worker").read_text()
    docker_hf = [s for s in kf._pip_stack_specs(dockerfile) if kf._pkg_name(s) == "huggingface_hub"]
    # Exactly one pin per source, else a duplicate/drifted entry could hide behind the first match.
    assert len(docker_hf) == 1, (
        f"expected one huggingface_hub pin in Dockerfile.worker, found {docker_hf}"
    )
    project_hf = _pyproject_specs("huggingface_hub")
    assert project_hf, "expected a huggingface-hub requirement in pyproject.toml"
    # public metadata remains a compatible range while the deployed image is exact. prove the exact
    # worker version satisfies every published range instead of requiring byte-identical specifiers.
    docker_requirement = Requirement(docker_hf[0])
    exact = [
        Version(item.version) for item in docker_requirement.specifier if item.operator == "=="
    ]
    assert len(exact) == 1, f"worker image must exactly pin huggingface_hub, got {docker_hf[0]!r}"
    assert all(exact[0] in Requirement(spec).specifier for spec in project_hf), (
        f"huggingface_hub exact worker pin {exact[0]} falls outside public ranges {project_hf}"
    )


def test_freesolo_floor_is_in_lockstep():
    """The exact image SDK pin must satisfy the compatible env-packaging range.

    The image governs the deployed version, while FREESOLO_WORKER_SPEC remains a range for published
    package compatibility. A worker pin outside that range would make the two paths disagree.
    """
    from flash.envs.loading.base import FREESOLO_WORKER_SPEC

    dockerfile = (ROOT / "Dockerfile.worker").read_text()
    docker_fs = [s for s in kf._pip_stack_specs(dockerfile) if kf._pkg_name(s) == "freesolo"]
    # Exactly one pin per source, else a duplicate/drifted entry could hide behind the first match.
    assert len(docker_fs) == 1, f"expected one freesolo pin in Dockerfile.worker, found {docker_fs}"
    docker_requirement = Requirement(docker_fs[0])
    exact = [
        Version(item.version) for item in docker_requirement.specifier if item.operator == "=="
    ]
    assert len(exact) == 1, f"worker image must exactly pin freesolo, got {docker_fs[0]!r}"
    assert exact[0] in Requirement(FREESOLO_WORKER_SPEC).specifier, (
        f"freesolo exact worker pin {exact[0]} falls outside {FREESOLO_WORKER_SPEC!r}"
    )


def test_parse_baked_per_sm_arches_contract():
    annotated = 'BAKED_PER_SM_ARCHES: frozenset[str] = frozenset({"sm90", "sm80"})\n'
    assert kf.parse_baked_per_sm_arches(annotated) == ["sm90", "sm80"]

    malformed = [
        "OTHER = frozenset({'sm80'})\n",
        "BAKED_PER_SM_ARCHES: frozenset[str]\n",
        "BAKED_PER_SM_ARCHES = frozenset({'sm80', 90})\n",
        "BAKED_PER_SM_ARCHES = {'sm80'}\n",
        "BAKED_PER_SM_ARCHES = set({'sm80'})\n",
        "BAKED_PER_SM_ARCHES = frozenset({})\n",
        "OTHER = BAKED_PER_SM_ARCHES = frozenset({'sm80'})\n",
        "BAKED_PER_SM_ARCHES = frozenset({'sm80', 'sm80'})\n",
        "BAKED_PER_SM_ARCHES = frozenset({'sm80'})\nBAKED_PER_SM_ARCHES = frozenset({'sm90'})\n",
    ]
    for source in malformed:
        with pytest.raises(ValueError, match="kernel_fingerprint"):
            kf.parse_baked_per_sm_arches(source)

    for arch in ("", " sm80", "sm80 ", "sm*", "gpu90", "sm90x"):
        source = f"BAKED_PER_SM_ARCHES = frozenset({{{arch!r}}})\n"
        with pytest.raises(ValueError, match="expected sm followed by digits"):
            kf.parse_baked_per_sm_arches(source)


def test_print_baked_arches_cli_exits_before_fingerprint_work(tmp_path):
    worker_dir = tmp_path / "flash" / "providers" / "_lifecycle" / "net"
    worker_dir.mkdir(parents=True)
    (worker_dir / "worker.py").write_text('BAKED_PER_SM_ARCHES = frozenset({"sm90", "sm80"})\n')
    result = subprocess.run(
        [
            sys.executable,
            str(KERNEL_FINGERPRINT_SCRIPT),
            "--root",
            str(tmp_path),
            "--print-baked-arches",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "sm90 sm80\n"


def test_baked_arch_workflows_match_canonical_source():
    from flash.providers._lifecycle.net.worker import BAKED_PER_SM_ARCHES

    worker_source = (ROOT / "flash" / "providers" / "_lifecycle" / "net" / "worker.py").read_text()
    source_arches = kf.parse_baked_per_sm_arches(worker_source)
    canonical_arches = set(BAKED_PER_SM_ARCHES)

    bake_lines = (ROOT / ".github" / "workflows" / "bake-kernel-cache.yml").read_text().splitlines()
    sms_block = _workflow_block(
        bake_lines, "on:", "  workflow_dispatch:", "    inputs:", "      sms:"
    )
    (default,) = [
        match.group(1)
        for line in sms_block
        if (match := re.fullmatch(r'\s*default:\s*"([^"]+)"\s*', line))
    ]
    default_arches = {arch.strip() for arch in default.split(",") if arch.strip()}
    (fallback,) = [
        match.group(1)
        for line in bake_lines
        if (match := re.fullmatch(r'\s*req="\$\{\{ inputs\.sms \|\| \'([^\']+)\' \}\}"\s*', line))
    ]
    fallback_arches = {arch.strip() for arch in fallback.split(",") if arch.strip()}

    include_block = _workflow_block(
        bake_lines, "jobs:", "  bake:", "    strategy:", "      matrix:", "        include:"
    )
    matrix_arch_list = [
        match.group(1)
        for line in include_block
        if (match := re.fullmatch(r"\s*sm:\s*(sm[0-9]+),?\s*", line))
    ]
    matrix_arches = set(matrix_arch_list)

    assert len(source_arches) == len(canonical_arches)
    assert set(source_arches) == canonical_arches == default_arches == fallback_arches
    assert len(matrix_arch_list) == len(matrix_arches)
    assert canonical_arches == matrix_arches

    auto_lines = (ROOT / ".github" / "workflows" / "auto-rebake.yml").read_text().splitlines()
    run_block = _workflow_block(
        auto_lines,
        "jobs:",
        "  gate:",
        "    steps:",
        "      - name: Classify arches",
        "        run: |",
    )

    def active_assignments(lines: list[str]) -> list[str]:
        return [line.strip() for line in lines if re.fullmatch(r"\s*arches=.*", line)]

    assert active_assignments(run_block) == [AUTO_REBAKE_ARCH_COMMAND]
    assert active_assignments([f"# {AUTO_REBAKE_ARCH_COMMAND}"]) == []
    assert active_assignments([AUTO_REBAKE_ARCH_COMMAND] * 2) != [AUTO_REBAKE_ARCH_COMMAND]


def test_parsing_fails_loud_on_a_broken_stack():
    """A missing pin must raise, never silently hash a constant (which would mask a stale cache)."""
    with pytest.raises(ValueError, match="pip stack block"):
        kf._pip_stack_specs("FROM scratch\n# no pip install block here\n")
    with pytest.raises(ValueError, match="could not parse"):
        kf._search(r"NOPE=(\S+)", "nothing here", "a missing thing")
