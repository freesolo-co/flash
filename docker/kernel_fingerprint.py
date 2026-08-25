"""Compute the two fingerprints that decide whether a per-arch (cu128-smXX) image is stale.

This is the single source of truth for "what makes a baked per-arch image out of date". It is
read by three places:
  * worker-image.yml stamps fp_cache/fp_base as OCI labels on the :cu128 base it builds,
  * bake-kernel-cache.yml re-stamps them on the per-arch image it bakes/re-layers,
  * auto-rebake.yml reads the published labels and compares to decide skip / re-layer / re-warm.

Two fingerprints, because "out of date" has two flavors with very different costs:

  * fp_cache = hash of the inputs whose kernels live in the baked mega-cache
    (torch.compiler.save_cache_artifacts: Triton/Inductor/torch.compile). Changing one of these
    INVALIDATES the cache, so the per-arch image needs a real GPU re-warm. These are: the base
    FROM image (torch+triton), the fla git pin, tilelang, apache-tvm-ffi, and the warmup script
    itself (it decides which kernels get compiled).

  * fp_base = hash of everything else baked into :cu128 that is NOT in the cache (FA2 wheel spec
    AND their sha256 digests, causal-conv1d, the non-kernel pip stack, and the baked rp_handler =
    endpoints.py + make_rp_handler.py). Changing one of these leaves the cache valid but means the
    per-arch tag is sitting on an old base, so it just needs a cheap re-layer (no GPU). fp_base folds
    fp_cache in, so it always moves when fp_cache does.

A stale baked cache is a SILENT cold-JIT (the worker only validates by SM, then torch.compiler
ignores a mismatched blob), never a crash. So a missing or wrong input here is invisible at run
time, which is why every parse below FAILS LOUD rather than hashing a None.

Known limitations (deliberately scoped -- each only ever costs a recoverable cold-JIT, never
correctness, and the alternatives over-fire the paid GPU bake):
  * fp_cache hashes the Dockerfile dep PINS, not pip-resolved versions. Every cache input is
    currently pinned exactly (a 40-char git sha or an == pin), so a resolve cannot drift without a
    text change; a future cache-affecting RANGE would reintroduce that gap. (fp_base DOES hash the
    whole Dockerfile.worker, so arbitrary base edits -- apt/ENV/CMD/cache-dir -- still trigger a
    free re-layer; only a cache-affecting change that isn't a parsed pin slips through.)
  * fp_cache hashes runtime/kernel_warmup.py but not its transitive cache-production deps (perf's Hopper
    fla/tilelang setup, docker/bake_pod_entry.py's pod-side install/invoke). The kernel-DETERMINING
    versions (tilelang/tvm-ffi/fla) ARE captured here; that code is otherwise stable or fetched fresh
    at runtime, so a change there only risks a stale (mostly sm90) cache that cold-JITs. Hashing all
    of perf / bake_pod_entry would re-warm 5 GPUs on every unrelated edit.

stdlib only, no flash/torch import, so it runs under a bare python3 in CI (no uv sync needed).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

# label keys are part of the cross-file contract; define them once here so no workflow hardcodes them.
LABEL_FP_CACHE = "co.freesolo.flash.kernelcache.fp_cache"
LABEL_FP_BASE = "co.freesolo.flash.kernelcache.fp_base"
LABEL_REVISION = "org.opencontainers.image.revision"

# pip specs in Dockerfile.worker's main stack whose kernels land in the mega-cache. everything else
# in that block is a base-layer dep (fp_base). fla is matched separately (it carries the git sha).
_CACHE_PKGS = {"tilelang", "apache-tvm-ffi"}
_BAKED_ARCHES_NAME = "BAKED_PER_SM_ARCHES"
_BAKED_ARCH_PATTERN = re.compile(r"sm[0-9]+")


def _search(pattern: str, text: str, what: str, *, flags: int = 0) -> str:
    """Return the first capture group or raise: a silent miss would mask staleness as cold-JIT."""
    m = re.search(pattern, text, flags)
    if not m:
        raise ValueError(f"kernel_fingerprint: could not parse {what} (pattern {pattern!r})")
    return m.group(1)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_baked_per_sm_arches(text: str, *, source: str = "_lifecycle/worker.py") -> list[str]:
    """Parse the canonical baked-architecture frozenset without importing flash."""
    tree = ast.parse(text, filename=source)
    assignments = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == _BAKED_ARCHES_NAME for t in node.targets)
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == _BAKED_ARCHES_NAME
        )
    ]
    context = f"kernel_fingerprint: {_BAKED_ARCHES_NAME} in {source}"
    if len(assignments) != 1:
        raise ValueError(
            f"{context}: expected exactly one top-level assignment, found {len(assignments)}"
        )

    assignment = assignments[0]
    if isinstance(assignment, ast.Assign) and (
        len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name)
    ):
        raise ValueError(f"{context} must use one named target")
    value = assignment.value
    if value is None:
        raise ValueError(f"{context} must have a value")
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "frozenset"
        and len(value.args) == 1
        and not value.keywords
        and isinstance(value.args[0], ast.Set)
    ):
        raise ValueError(f"{context} must be frozenset({{...}}) of string literals")

    arches: list[str] = []
    for element in value.args[0].elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            raise ValueError(f"{context} must contain strings")
        if _BAKED_ARCH_PATTERN.fullmatch(element.value) is None:
            raise ValueError(
                f"{context} has invalid architecture {element.value!r}; expected sm followed by digits"
            )
        arches.append(element.value)
    if not arches:
        raise ValueError(f"{context} must not be empty")
    if len(arches) != len(set(arches)):
        raise ValueError(f"{context} must not contain duplicates")
    return arches


def _pip_stack_specs(dockerfile: str) -> list[str]:
    """The quoted specs from Dockerfile.worker's first `pip install` block (the pinned worker stack)."""
    specs: list[str] = []
    in_block = False
    for line in dockerfile.splitlines():
        if re.match(r"\s*RUN pip install --no-cache-dir", line):
            in_block = True
            continue
        if in_block:
            m = re.search(r'"([^"]+)"', line)
            if m:
                specs.append(m.group(1))
            if not line.rstrip().endswith("\\"):
                break
    if not specs:
        raise ValueError(
            "kernel_fingerprint: could not parse the Dockerfile.worker pip stack block"
        )
    return specs


def _pkg_name(spec: str) -> str:
    """The bare package name from a pip spec (drops version/extras/git url)."""
    return re.split(r"[=<>@ \[]", spec.strip(), maxsplit=1)[0]


def collect_inputs(
    root: Path,
    *,
    fa2_spec: str | None = None,
    fa2_sha256: str | None = None,
) -> tuple[dict, dict]:
    """Parse the authoritative sources into (cache_inputs, base_inputs_partial).

    Each value comes from where the IMAGE actually gets it:
      * pins / FROM / causal-conv1d from Dockerfile.worker (what the image is built from),
      * FA2 from worker-image.yml's build-args (overridable via fa2_spec/fa2_sha256 for the
        resolved-build-arg case), and file hashes for the warmup/handler sources.
    base_inputs_partial does NOT yet include fp_cache; compute_fingerprints folds it in.
    """
    dockerfile = (root / "Dockerfile.worker").read_text()
    worker_image_yml = (root / ".github" / "workflows" / "worker-image.yml").read_text()

    specs = _pip_stack_specs(dockerfile)
    by_name = {_pkg_name(s): s for s in specs}

    def _need(name: str) -> str:
        if name not in by_name:
            raise ValueError(
                f"kernel_fingerprint: {name!r} missing from the Dockerfile.worker stack"
            )
        return by_name[name]

    from_image = _search(r"(?m)^FROM\s+(\S+)", dockerfile, "Dockerfile.worker FROM")
    fla = next((s for s in specs if "flash-linear-attention" in s), None)
    if not fla or not re.search(r"\.git@[0-9a-f]{40}\b", fla):
        raise ValueError(
            "kernel_fingerprint: fla spec missing or not pinned to a 40-char commit sha"
        )
    cache_inputs = {
        "from_image": from_image,
        "fla": fla,
        "tilelang": _need("tilelang"),
        "tvm_ffi": _need("apache-tvm-ffi"),
        "kernel_warmup_sha256": _sha256_file(
            root / "flash" / "engine" / "worker" / "runtime" / "kernel_warmup.py"
        ),
    }

    # worker-image.yml holds the FA2 wheel in a job-level env block (single source for the build-arg
    # and these fingerprint inputs). CI passes the resolved value via --fa2-spec; for a manual
    # bake / the tests we read those env defaults here.
    if fa2_spec is None:
        fa2_spec = _search(
            r"(?m)^\s*FA2_SPEC:\s*(\S+)", worker_image_yml, "worker-image.yml FA2_SPEC"
        )
    # the wheel digests are fp_base inputs too: a wheel replaced at the SAME url only shows up as a
    # sha256 rotation, and if fp_base did not hash it, a maintainer who bumps the checksum after a
    # release asset was replaced would rebuild a base that auto-rebake still calls up to date.
    if fa2_sha256 is None:
        fa2_sha256 = _search(
            r"(?m)^\s*FA2_SHA256:\s*(\S+)", worker_image_yml, "worker-image.yml FA2_SHA256"
        )

    causal_conv1d = _search(
        r'"(causal-conv1d==[^"]+)"', dockerfile, "Dockerfile.worker causal-conv1d pin"
    )
    pip_base = sorted(
        s for s in specs if _pkg_name(s) not in _CACHE_PKGS and "flash-linear-attention" not in s
    )

    base_inputs_partial = {
        "fa2": fa2_spec,
        # NOT cache_inputs: a wheel digest change leaves the Triton/Inductor mega-cache valid (the
        # wheel plays no part in what it caches), so it only needs the free re-layer, never a paid
        # GPU re-warm.
        "fa2_sha256": fa2_sha256,
        "causal_conv1d": causal_conv1d,
        "pip_base": pip_base,
        # whole-Dockerfile hash catches ANY base change the parsed fields miss (apt/ENV/CMD/cache-dir);
        # it lives in fp_base, so such a change triggers only the FREE re-layer, never a paid re-warm.
        "dockerfile_sha256": _sha256_file(root / "Dockerfile.worker"),
        "endpoints_sha256": _sha256_file(
            root / "flash" / "providers" / "runpod" / "serverless" / "endpoints.py"
        ),
        "make_rp_handler_sha256": _sha256_file(root / "docker" / "make_rp_handler.py"),
        "archive_sha256": _sha256_file(root / "flash" / "snapshot" / "archive.py"),
    }
    return cache_inputs, base_inputs_partial


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def compute_fingerprints(cache_inputs: dict, base_inputs_partial: dict) -> tuple[str, str, dict]:
    """Pure: fp_cache from cache_inputs; fp_base from base_inputs with fp_cache folded in.

    Returns (fp_cache, fp_base, base_inputs) where base_inputs is the dict actually hashed (so an
    --explain dump shows the fp_cache that was folded in).
    """
    fp_cache = _hash(cache_inputs)
    base_inputs = {"fp_cache": fp_cache, **base_inputs_partial}
    fp_base = _hash(base_inputs)
    return fp_cache, fp_base, base_inputs


def fingerprints(
    root: Path,
    *,
    fa2_spec: str | None = None,
    fa2_sha256: str | None = None,
) -> tuple[str, str, dict, dict]:
    cache_inputs, base_partial = collect_inputs(
        root,
        fa2_spec=fa2_spec,
        fa2_sha256=fa2_sha256,
    )
    fp_cache, fp_base, base_inputs = compute_fingerprints(cache_inputs, base_partial)
    return fp_cache, fp_base, cache_inputs, base_inputs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="compute the per-arch kernel-cache fingerprints")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent), help="repo root")
    ap.add_argument(
        "--fa2-spec", default=None, help="resolved FLASH_ATTN_SPEC (else read from yml)"
    )
    ap.add_argument(
        "--fa2-sha256", default=None, help="resolved FLASH_ATTN_SHA256 (else read from yml)"
    )
    ap.add_argument("--format", choices=("github", "json"), default="json")
    ap.add_argument("--explain", action="store_true", help="dump the component inputs to stderr")
    ap.add_argument(
        "--print-label-keys", action="store_true", help="print the OCI label keys + exit"
    )
    ap.add_argument(
        "--print-baked-arches",
        action="store_true",
        help="print BAKED_PER_SM_ARCHES from _lifecycle/worker.py + exit",
    )
    args = ap.parse_args(argv)

    if args.print_label_keys:
        print(f"fp_cache={LABEL_FP_CACHE}")
        print(f"fp_base={LABEL_FP_BASE}")
        print(f"revision={LABEL_REVISION}")
        return 0

    root = Path(args.root)
    if args.print_baked_arches:
        worker_source = root / "flash" / "providers" / "_lifecycle" / "net" / "worker.py"
        try:
            arches = parse_baked_per_sm_arches(
                worker_source.read_text(encoding="utf-8"), source=str(worker_source)
            )
        except (OSError, SyntaxError, ValueError) as exc:
            ap.error(str(exc))
        print(" ".join(arches))
        return 0

    fp_cache, fp_base, cache_inputs, base_inputs = fingerprints(
        root,
        fa2_spec=args.fa2_spec,
        fa2_sha256=args.fa2_sha256,
    )
    if args.explain:
        print(
            json.dumps({"cache_inputs": cache_inputs, "base_inputs": base_inputs}, indent=2),
            file=sys.stderr,
        )
    if args.format == "github":
        print(f"fp_cache={fp_cache}")
        print(f"fp_base={fp_base}")
    else:
        print(json.dumps({"fp_cache": fp_cache, "fp_base": fp_base}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
