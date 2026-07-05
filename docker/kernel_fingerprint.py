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
    FROM image (torch+triton), the fla git pin, tilelang, apache-tvm-ffi, the default chalk
    spec, and the warmup script itself (it decides which kernels get compiled).

  * fp_base = hash of everything else baked into :cu128 that is NOT in the cache (FA2/FA3 wheels,
    causal-conv1d, the non-kernel pip stack, and the baked rp_handler = endpoints.py +
    make_rp_handler.py). Changing one of these leaves the cache valid but means the per-arch tag is
    sitting on an old base, so it just needs a cheap re-layer (no GPU). fp_base folds fp_cache in,
    so it always moves when fp_cache does.

A stale baked cache is a SILENT cold-JIT (the worker only validates by SM, then torch.compiler
ignores a mismatched blob), never a crash. So a missing or wrong input here is invisible at run
time, which is why every parse below FAILS LOUD rather than hashing a None.

Known limitations (deliberately scoped -- each only ever costs a recoverable cold-JIT, never
correctness, and the alternatives over-fire the paid GPU bake):
  * fp_cache hashes the Dockerfile dep PINS, not pip-resolved versions. A cache-affecting range
    (the chalk spec) could resolve a newer build on a later worker-image rebuild with
    no text change, leaving fp_cache unmoved. Pin those exactly for airtight coverage. (fp_base DOES
    hash the whole Dockerfile.worker, so arbitrary base edits -- apt/ENV/CMD/cache-dir -- still
    trigger a free re-layer; only a cache-affecting change that isn't a parsed pin slips through.)
  * fp_cache hashes kernel_warmup.py but not its transitive cache-production deps (perf's Hopper
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


def _search(pattern: str, text: str, what: str, *, flags: int = 0) -> str:
    """Return the first capture group or raise: a silent miss would mask staleness as cold-JIT."""
    m = re.search(pattern, text, flags)
    if not m:
        raise ValueError(f"kernel_fingerprint: could not parse {what} (pattern {pattern!r})")
    return m.group(1)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python_string_constant(text: str, name: str, what: str) -> str:
    """Resolve a simple module-level string constant, including f-strings using prior constants."""
    values: dict[str, str] = {}

    def _eval(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                elif (
                    isinstance(value, ast.FormattedValue)
                    and isinstance(value.value, ast.Name)
                    and value.value.id in values
                ):
                    parts.append(values[value.value.id])
                else:
                    return None
            return "".join(parts)
        return None

    tree = ast.parse(text)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = _eval(node.value)
        if value is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    if name not in values:
        raise ValueError(f"kernel_fingerprint: could not parse {what}")
    return values[name]


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
    root: Path, *, fa2_spec: str | None = None, fa3_spec: str | None = None
) -> tuple[dict, dict]:
    """Parse the authoritative sources into (cache_inputs, base_inputs_partial).

    Each value comes from where the IMAGE actually gets it:
      * pins / FROM / causal-conv1d from Dockerfile.worker (what the image is built from),
      * default chalk spec textually from _worker.py (what the bake warms),
      * FA2/FA3 from worker-image.yml's build-args (overridable via fa2_spec/fa3_spec for the
        resolved-build-arg case), and file hashes for the warmup/handler sources.
    base_inputs_partial does NOT yet include fp_cache; compute_fingerprints folds it in.
    """
    dockerfile = (root / "Dockerfile.worker").read_text()
    worker_pkg = (root / "flash" / "providers" / "_worker.py").read_text()
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
    chalk = _python_string_constant(
        worker_pkg, "DEFAULT_CHALK_SPEC", "_worker.py DEFAULT_CHALK_SPEC"
    )

    cache_inputs = {
        "from_image": from_image,
        "fla": fla,
        "tilelang": _need("tilelang"),
        "tvm_ffi": _need("apache-tvm-ffi"),
        "chalk": chalk,
        "kernel_warmup_sha256": _sha256_file(
            root / "flash" / "engine" / "worker" / "kernel_warmup.py"
        ),
    }

    # worker-image.yml holds the FA wheels in a job-level env block (single source for the build-args
    # and these fingerprint inputs). CI passes the resolved values via --fa{2,3}-spec; for a manual
    # bake / the tests we read those env defaults here.
    if fa2_spec is None:
        fa2_spec = _search(
            r"(?m)^\s*FA2_SPEC:\s*(\S+)", worker_image_yml, "worker-image.yml FA2_SPEC"
        )
    if fa3_spec is None:
        # FA3_SPEC: ${{ github.event.inputs.flash_attn_3_spec || '<default wheel>' }} -> the default.
        fa3_spec = _search(
            r"FA3_SPEC:.*?\|\|\s*'([^']+)'", worker_image_yml, "worker-image.yml FA3_SPEC default"
        )

    causal_conv1d = _search(
        r'"(causal-conv1d==[^"]+)"', dockerfile, "Dockerfile.worker causal-conv1d pin"
    )
    pip_base = sorted(
        s for s in specs if _pkg_name(s) not in _CACHE_PKGS and "flash-linear-attention" not in s
    )

    base_inputs_partial = {
        "fa2": fa2_spec,
        "fa3": fa3_spec,
        "causal_conv1d": causal_conv1d,
        "pip_base": pip_base,
        # whole-Dockerfile hash catches ANY base change the parsed fields miss (apt/ENV/CMD/cache-dir);
        # it lives in fp_base, so such a change triggers only the FREE re-layer, never a paid re-warm.
        "dockerfile_sha256": _sha256_file(root / "Dockerfile.worker"),
        "endpoints_sha256": _sha256_file(
            root / "flash" / "providers" / "runpod" / "train" / "endpoints.py"
        ),
        "make_rp_handler_sha256": _sha256_file(root / "docker" / "make_rp_handler.py"),
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
    root: Path, *, fa2_spec: str | None = None, fa3_spec: str | None = None
) -> tuple[str, str, dict, dict]:
    cache_inputs, base_partial = collect_inputs(root, fa2_spec=fa2_spec, fa3_spec=fa3_spec)
    fp_cache, fp_base, base_inputs = compute_fingerprints(cache_inputs, base_partial)
    return fp_cache, fp_base, cache_inputs, base_inputs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="compute the per-arch kernel-cache fingerprints")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent), help="repo root")
    ap.add_argument(
        "--fa2-spec", default=None, help="resolved FLASH_ATTN_SPEC (else read from yml)"
    )
    ap.add_argument(
        "--fa3-spec", default=None, help="resolved FLASH_ATTN_3_SPEC (else read from yml)"
    )
    ap.add_argument("--format", choices=("github", "json"), default="json")
    ap.add_argument("--explain", action="store_true", help="dump the component inputs to stderr")
    ap.add_argument(
        "--print-label-keys", action="store_true", help="print the OCI label keys + exit"
    )
    args = ap.parse_args(argv)

    if args.print_label_keys:
        print(f"fp_cache={LABEL_FP_CACHE}")
        print(f"fp_base={LABEL_FP_BASE}")
        print(f"revision={LABEL_REVISION}")
        return 0

    fp_cache, fp_base, cache_inputs, base_inputs = fingerprints(
        Path(args.root), fa2_spec=args.fa2_spec, fa3_spec=args.fa3_spec
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
