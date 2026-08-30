"""On-GPU provenance and kernel-path probes recorded with every capacity result.

A capacity number is only interpretable next to the hardware and kernel path that produced it. Two
probes matter enough to gate publication:

* **Card identity.** A result labelled B200 that ran on an H200 is worse than no result.
* **GDN prefill backend.** The served Qwen3.5/3.6 models are GatedDeltaNet hybrids. vLLM 0.23.0's
  ``_resolve_gdn_prefill_backend`` grants FlashInfer GDN prefill to SM90 (H100/H200) unconditionally,
  but on SM10.x (B200) only when ``linear_key_head_dim == 128`` AND CUDA runtime major >= 13 AND
  ``_is_libs_cu13_install_intact()``. That last condition fails whenever the two
  ``nvidia-cutlass-dsl-libs-*`` wheels race during install, and it fails SILENTLY: the resolver falls
  through to Triton after a single ``warning_once``.

  The trap is that the engine still boots, still serves, and still bills the B200 rate while running
  the slower prefill kernel. A boot smoke cannot see it. So the benchmark asks the resolver what it
  actually chose and labels the envelope with the answer, rather than assuming the fast path.

This module reports; it does not repair. Repairing the Cutlass install belongs to the runtime image,
not to a measurement tool that would then be measuring its own side effect.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def probe_gpu() -> dict[str, Any]:
    """Device identity, capability, and memory, read from the DRIVER via NVML.

    Deliberately not ``torch.cuda``. Under vLLM V1 the model runs in a separate EngineCore process,
    and this probe runs in the parent Modal class process. ``torch.cuda.current_device()`` and
    friends initialize a CUDA context *here*, and a prior campaign's extra parent-process context
    stole the post-init headroom EngineCore needs for FlashInfer's first-request decode workspace,
    OOM-killing the 35B engine on its first request (see the post-mortem comment in
    ``flash/serving/src/engine/lora_engine.py``). Since the canary runs this probe before every
    warmup and every sweep, a torch-based probe would either reproduce that outage or measure a
    memory shape production deliberately avoids.

    NVML queries the driver without creating a context, so identity costs nothing measurable.
    """
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - NVML ships with the serving image
        return {"available": False, "reason": f"pynvml unavailable: {exc}"}

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return {"available": False, "reason": "NVML reports no devices"}
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result: dict[str, Any] = {
            "available": True,
            "device_count": count,
            "name": str(name),
            "compute_capability": f"{major}.{minor}",
            "total_memory_bytes": int(memory.total),
            "source": "nvml",
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()

    # Version strings are module attributes; reading them creates no context.
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
    except Exception:  # pragma: no cover - torch is present in the serving image
        pass
    return result


def probe_cutlass_integrity() -> dict[str, Any]:
    """Run vLLM's OWN ``_is_libs_cu13_install_intact`` check, uncached.

    The function is ``lru_cache``d upstream, so an earlier in-process call could return a stale
    answer; ``__wrapped__`` reaches the undecorated function when it is available.
    """
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    except Exception as exc:
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"}

    check = getattr(gdn, "_is_libs_cu13_install_intact", None)
    if check is None:
        return {"checked": False, "reason": "vllm build has no _is_libs_cu13_install_intact"}
    try:
        intact = bool(getattr(check, "__wrapped__", check)())
    except Exception as exc:
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "checked": True,
        "intact": intact,
        # The consequence, stated where a reader of the report will see it.
        "implication": (
            "FlashInfer GDN prefill is available on Blackwell"
            if intact
            else "GDN prefill silently falls back to Triton on Blackwell (slower path)"
        ),
    }


def _served_checkpoint(base_model: str) -> tuple[str, str | None]:
    """The repo and revision the ENGINE loads, which is not always the logical base model.

    The 9B and 27B engines serve separate ``-FP8`` repositories, and the 27B pins a
    ``model_revision``. Reading geometry from the mutable logical name could therefore feed the
    resolver different head dims than the running engine once that repository advances, and the
    published backend provenance would describe a checkpoint nobody served.
    """
    from flash.serving.bench.catalog import bench_engine_overrides_for

    try:
        overrides = bench_engine_overrides_for(base_model)
    except Exception:
        return base_model, None
    repo = overrides.get("serve_model_id") or base_model
    revision = overrides.get("model_revision")
    return str(repo), (str(revision) if revision else None)


def _config_checkpoint(base_model: str) -> tuple[str, str | None]:
    """The repo and the exact revision the GDN config is read at.

    Split out of `_gdn_config_values` so the probe REPORTS the revision it read rather than the
    revision it asked for. Those differ for an unpinned repository, and a `config_source` naming
    `None` while the load resolved a commit would make the artifact unfalsifiable exactly where the
    provenance matters most.
    """
    repo, revision = _served_checkpoint(base_model)
    if revision:
        return repo, revision
    commit, _source = _local_snapshot_commit(repo, None, "model")
    return repo, (commit or None)


def _gdn_config_values(base_model: str) -> dict[str, Any]:
    """GDN geometry from the served checkpoint's own config, for binding the resolver's parameters.

    Read from the checkpoint rather than hardcoded: ``linear_key_head_dim == 128`` is the condition
    the resolver tests on SM10.x, so supplying a guessed 128 would make the probe answer its own
    question and report the fast path for a model that does not qualify.

    Read from the SERVED repo at its pinned revision, not the logical base model, so the geometry
    belongs to the weights the engine actually loaded.

    Pinned to the LOCAL cache. An unpinned model resolves through `refs/main`, and a network-capable
    load would re-resolve that ref and rewrite it to whatever the Hub holds NOW -- so this probe
    could advance the very file `_local_snapshot_commit` reads, and the artifact would attribute a
    measured number to a commit the engine never loaded. The engine has already downloaded this
    config by the time any probe runs, so the local read is the same bytes; if it somehow is not
    cached, raising is correct, because the caller records the failure and the gate refuses to
    publish rather than guessing a revision.

    Pinned to the CAPTURED COMMIT as well, not just to the cache. `local_files_only` stops the Hub
    fetch but not the ref lookup: an unpinned repository still resolves `revision=None` through the
    cache's mutable `refs/main`, which a tokenizer or processor download can advance mid-boot. On
    the 35B model -- the one whose weights and tokenizer share a single unpinned repository -- that
    leaves the newer config describing older running weights, and the resolver would bind head dims
    the engine never loaded, publishing a GDN backend for a checkpoint nobody served. So resolve
    against the commit `probe_resolved_revisions` already captured for the model role, which is the
    snapshot the engine's own weights came from. When no commit was captured the ref is the only
    thing available and the pre-existing behaviour stands; the revision gate refuses to publish such
    a container anyway, so this cannot widen what a paid sweep may claim.
    """
    from transformers import AutoConfig

    repo, revision = _config_checkpoint(base_model)
    config = AutoConfig.from_pretrained(
        repo, revision=revision, trust_remote_code=True, local_files_only=True
    )
    text_config = getattr(config, "text_config", config)
    values: dict[str, Any] = {}
    for name in (
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
    ):
        found = getattr(text_config, name, getattr(config, name, None))
        if found is not None:
            values[name] = found
    return values


def _gdn_resolver_kwargs(resolver: Any, base_model: str) -> dict[str, Any]:
    """Bind only the parameters this build's resolver actually declares.

    Signature-driven rather than hardcoded, because the resolver is vLLM-internal and its parameters
    move between builds; a fixed argument list would break on the next bump exactly as the zero-arg
    call broke on this one. A REQUIRED parameter that the config cannot supply raises, and the caller
    records that as a signature mismatch rather than as an unknown backend.
    """
    signature = inspect.signature(resolver)
    available = _gdn_config_values(base_model)
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in available:
            kwargs[name] = available[name]
        elif parameter.default is parameter.empty:
            raise TypeError(f"resolver requires {name!r}, which this model config does not supply")
    return kwargs


def _gdn_backend_in_process(base_model: str) -> dict[str, Any]:
    """Ask the resolver directly. Runs ONLY in the throwaway child, never in the parent.

    Best-effort by design: the resolver is internal to vLLM and its signature may move between
    builds. A failed probe is recorded as unknown rather than being allowed to assert the fast path
    by omission.

    `probe_gdn_backend` explains why this must not run in the engine's own process.
    """
    result: dict[str, Any] = {"base_model": base_model, "resolved": None}
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result

    resolver = getattr(gdn, "_resolve_gdn_prefill_backend", None)
    if resolver is None:
        result["reason"] = "vllm build has no _resolve_gdn_prefill_backend"
        return result
    result["resolver_present"] = True
    result["cutlass"] = probe_cutlass_integrity()
    # Presence of the resolver says nothing about WHICH backend it picks. The Blackwell failure mode
    # is `warning_once` then a silent Triton fallback -- the boot succeeds, serves, and bills the
    # fast-card rate while running the slow kernel. So call it and record what it actually returned;
    # an unresolved probe stays None and the report may not claim a backend.
    # The resolver is vLLM-internal and takes build-dependent arguments (recent builds require the
    # GDN head dims), so a bare `resolver()` raises TypeError on exactly the builds that HAVE it.
    # That exception was swallowed into `reason` and the probe returned resolved=None -- the same
    # unknown a build with no resolver at all produces, so the canary could not tell "this build
    # cannot be probed" from "this call was made wrong". Bind the declared parameters from the
    # model config and record a signature mismatch distinctly rather than as an unknown backend.
    try:
        kwargs = _gdn_resolver_kwargs(resolver, base_model)
    except Exception as exc:
        result["reason"] = f"resolver signature unsupported: {type(exc).__name__}: {exc}"
        result["resolver_signature_mismatch"] = True
        return result
    result["resolver_kwargs"] = sorted(kwargs)
    probed_repo, probed_revision = _config_checkpoint(base_model)
    result["config_source"] = {"repo": probed_repo, "revision": probed_revision}
    try:
        resolved = resolver(**kwargs)
    except TypeError as exc:
        # A TypeError here is an ARGUMENT problem, not a backend answer. Kept distinct so the report
        # cannot read a mis-called probe as evidence about which kernel ran.
        result["reason"] = f"resolver rejected arguments: {exc}"
        result["resolver_signature_mismatch"] = True
        return result
    except Exception as exc:
        result["reason"] = f"resolver raised: {type(exc).__name__}: {exc}"
        return result
    if isinstance(resolved, tuple):
        result["resolved"] = str(resolved[-1]) if resolved else None
        result["resolved_raw"] = [str(part) for part in resolved]
    else:
        result["resolved"] = str(resolved) if resolved is not None else None
    return result


def probe_gdn_backend(base_model: str, *, timeout: float = 120.0) -> dict[str, Any]:
    """What the GDN prefill resolver would choose, asked from a THROWAWAY child process.

    The resolver is not a passive read. In pinned vLLM 0.23.0 it queries
    ``current_platform.get_device_capability``, whose CUDA implementation calls
    ``torch.cuda.get_device_capability`` -- so asking it initializes a CUDA context in whatever
    process asks. That is exactly the cost ``probe_gpu`` goes out of its way to avoid: this probe
    runs in the long-lived Modal class process, and a prior campaign's extra parent-process context
    stole the post-init headroom EngineCore needs for FlashInfer's first-request decode workspace,
    OOM-killing the 35B engine on its first request. Since the canary probes before every warmup and
    every bucket, an in-process resolver call would let the certification perturb or kill the very
    engine it exists to certify.

    So the resolver runs in a child that exits immediately, taking its CUDA context with it. The
    parent reads back only JSON. The NVML-based capability in ``probe_gpu`` is deliberately NOT used
    to reproduce the resolver's decision instead: re-deriving the branch here would be a second
    implementation of a vLLM-internal rule, free to drift from the one the engine actually runs, and
    the entire point of this probe is to report what vLLM chose rather than what we predict it chose.

    A child that crashes, times out, or returns unparseable output is recorded as unknown, never as
    the fast path -- same contract as every other failure mode here.
    """
    payload = json.dumps(base_model)
    program = (
        "import json,sys;"
        "from flash.serving.bench.probe import _gdn_backend_in_process as p;"
        "sys.stdout.write(json.dumps(p(json.loads(sys.stdin.read()))))"
    )
    try:
        # Fixed argv, no shell: the model name travels as JSON on stdin, not in the command.
        completed = subprocess.run(
            [sys.executable, "-c", program],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "base_model": base_model,
            "resolved": None,
            "reason": f"resolver subprocess exceeded {timeout}s",
            "subprocess_isolated": True,
        }
    except Exception as exc:
        return {
            "base_model": base_model,
            "resolved": None,
            "reason": f"resolver subprocess failed to start: {type(exc).__name__}: {exc}",
            "subprocess_isolated": True,
        }

    if completed.returncode != 0:
        return {
            "base_model": base_model,
            "resolved": None,
            "reason": f"resolver subprocess exited {completed.returncode}: {completed.stderr[-400:]}",
            "subprocess_isolated": True,
        }
    try:
        result = json.loads(completed.stdout)
    except ValueError as exc:
        return {
            "base_model": base_model,
            "resolved": None,
            "reason": f"resolver subprocess returned unparseable output: {exc}",
            "subprocess_isolated": True,
        }
    result["subprocess_isolated"] = True
    return result


def probe_engine_kv_cache(engine: Any) -> dict[str, Any]:
    """KV-cache block counts from the running engine, when its build exposes them.

    Recorded because concurrency at a long context is bounded by the KV pool, so the pool size is
    what makes a near-32k concurrency number explicable rather than merely observed.
    """
    out: dict[str, Any] = {}
    for path in ("cache_config", "vllm_config"):
        config = getattr(getattr(engine, "engine", engine), path, None)
        if config is None:
            continue
        cache = getattr(config, "cache_config", config)
        for field in ("num_gpu_blocks", "num_cpu_blocks", "block_size", "cache_dtype"):
            value = getattr(cache, field, None)
            if value is not None:
                out[field] = value if isinstance(value, (int, str)) else str(value)
    return out


# The files whose presence identifies a snapshot as holding a given role's artifacts. A snapshot
# directory contains exactly the files THAT download fetched, so these markers are what distinguish
# the tree the weights came from from the tree the tokenizer came from when the two differ.
_ROLE_MARKERS: dict[str, tuple[str, ...]] = {
    "model": ("config.json", "model.safetensors", "model.safetensors.index.json"),
    "tokenizer": ("tokenizer.json", "tokenizer_config.json"),
    "processor": ("preprocessor_config.json", "processor_config.json"),
}


def _snapshots_holding(root: Path, role: str) -> list[str]:
    """Commits whose cached snapshot actually contains ``role``'s files, newest mtime last."""
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return []
    markers = _ROLE_MARKERS.get(role, ())
    found = [
        entry
        for entry in snapshots.iterdir()
        if entry.is_dir() and any((entry / marker).exists() for marker in markers)
    ]
    return [entry.name for entry in sorted(found, key=lambda e: e.stat().st_mtime)]


def _local_snapshot_commit(
    repo: str, revision: str | None, role: str = "model"
) -> tuple[str | None, str | None]:
    """The commit of the cached snapshot ``role`` was actually loaded from, and how it was found.

    Three shapes, in order of authority:

    * a pinned revision that is already a 40-character hash IS the commit, and the snapshot
      directory's existence confirms the engine loaded that exact tree;
    * otherwise the snapshot directory that CONTAINS this role's files names the commit that role
      came from;
    * otherwise the cache's `refs/<revision>` file records the commit the last download resolved to.

    The middle shape exists because `refs/<revision>` is per-REPOSITORY while provenance is
    per-ROLE. An unpinned repository that advances mid-boot is downloaded twice -- weights resolve
    to commit A, then the tokenizer fetch re-resolves the moving ref and lands B -- and the ref file
    afterwards holds only B. Reading it for both roles records the weights as B as well: a commit
    the weights never came from, on the one hosted model whose model and tokenizer share a single
    unpinned repository, and it is indistinguishable from correct provenance downstream. A snapshot
    directory holds exactly the files its own download fetched, so asking which snapshot contains
    this role's files answers the per-role question the ref file cannot.

    Ambiguity is reported, never guessed: when several snapshots hold the role and none is the ref,
    the source records that rather than picking one.

    Returns `(None, None)` when nothing is available. A missing snapshot must not fall back to a Hub
    lookup: the whole point is to report what this container loaded, and the Hub cannot answer that.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = revision or "main"
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
        return ref.lower(), "pinned-hash"

    root = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
    ref_file = root / "refs" / ref
    ref_commit = ""
    if ref_file.is_file():
        ref_commit = ref_file.read_text().strip()

    holders = _snapshots_holding(root, role)
    if len(holders) == 1:
        return holders[0], "snapshot-contents"
    if holders:
        # Several snapshots carry this role. If the ref is among them the repository simply has
        # older trees cached and the ref still names the current one; otherwise the ref points
        # somewhere this role was never downloaded to, so report the ambiguity instead of choosing.
        if ref_commit and ref_commit in holders:
            return ref_commit, "local-cache-ref"
        return holders[-1], "snapshot-contents-ambiguous"

    if ref_commit:
        return ref_commit, "local-cache-ref"
    return None, None


def probe_resolved_revisions(base_model: str) -> dict[str, Any]:
    """The commit each served repository RESOLVED to, for repositories production leaves unpinned.

    `immutable_serving_revisions` reports only what the engine explicitly pins, and two of the three
    hosted models pin nothing: their artifacts would carry a mutable repository name and no commit,
    so once the repository advances an old curve could no longer identify the weights or tokenizer
    that produced it. A published capacity number that cannot name its checkpoint is not evidence.

    Read from the DOWNLOADED SNAPSHOT, not from the Hub. Asking the Hub returns whatever the
    repository points at now; for an unpinned repository that advanced between engine init and this
    probe, that is a different commit than the one the curve was produced on, and recording it would
    be false provenance -- worse than none, because it looks authoritative. The local cache is the
    only place the actually-loaded commit exists.

    Fail-soft by design: a missing commit is recorded with its reason and never invented. The gate
    that refuses to publish belongs to the caller, and it cannot make that decision from a
    fabricated hash. `pinned` distinguishes a commit production guaranteed from one this run merely
    observed, because only the former is reproducible.
    """
    from flash.serving.src.engine.model_config import (
        immutable_serving_revisions,
        tokenizer_model_for,
    )

    pins = immutable_serving_revisions(base_model)
    served_repo, served_revision = _served_checkpoint(base_model)
    tokenizer_repo = tokenizer_model_for(base_model)

    targets = {
        "model": (served_repo, served_revision or pins.get("model_revision")),
        "tokenizer": (tokenizer_repo, pins.get("tokenizer_revision")),
        "processor": (tokenizer_repo, pins.get("processor_revision")),
    }

    resolved: dict[str, Any] = {}
    for role, (repo, revision) in targets.items():
        entry: dict[str, Any] = {"repo": repo, "pinned": revision}
        try:
            commit, source = _local_snapshot_commit(repo, revision, role)
            entry["commit"] = commit
            entry["source"] = source
            if commit is None:
                entry["reason"] = "no local snapshot found for the loaded repository"
        # Broad on purpose: a missing cache root, an unreadable ref file and an unexpected cache
        # layout must all degrade to a recorded reason rather than killing a lane that has already
        # paid for its boot.
        except Exception as exc:
            entry["commit"] = None
            entry["source"] = None
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        resolved[role] = entry
    return resolved


def probe_all(base_model: str, engine: Any | None = None) -> dict[str, Any]:
    """The full provenance block stored with each model's results."""
    # Revisions FIRST, before anything that reads a checkpoint. `_local_snapshot_commit` reports an
    # unpinned model's commit from the cache's `refs/main`, which is mutable state a Hub-touching
    # load can rewrite underneath it. `_gdn_config_values` is pinned to the local cache so it cannot
    # do that, but the order is the second, independent guarantee: provenance is captured before any
    # later-added probe gets the chance to refresh a ref and silently re-date the measurement.
    payload: dict[str, Any] = {
        "resolved_revisions": probe_resolved_revisions(base_model),
        "gpu": probe_gpu(),
        "gdn_prefill": probe_gdn_backend(base_model),
    }
    if engine is not None:
        payload["kv_cache"] = probe_engine_kv_cache(engine)
    return payload


def gpu_matches(probe: dict[str, Any], expected: str) -> bool:
    """Whether the probed device is the expected card.

    Token match, NOT substring. Vendors decorate the reported name ("NVIDIA L40S",
    "NVIDIA H200 141GB HBM3e") so bare equality would reject the real card -- but a substring test
    makes ``"L4" in "NVIDIA L40S"`` true, and an L4-expected model would silently pass its identity
    gate while running on an L40S. Since the whole point of this check is to attribute a measured
    number to a specific card, it splits the name and compares whole tokens.

    Modal's tier spelling is normalized first: ``H100!`` pins a non-substitutable card and
    ``A100-80GB`` carries its memory in the tier name, neither of which appears in the device name.
    """
    gpu = probe.get("gpu") or {}
    name = str(gpu.get("name") or "")
    tier = expected.upper().rstrip("!").split("-")[0].strip()
    if not tier:
        return False
    return tier in {token.strip() for token in name.upper().replace("-", " ").split()}


__all__ = [
    "gpu_matches",
    "probe_all",
    "probe_cutlass_integrity",
    "probe_engine_kv_cache",
    "probe_gdn_backend",
    "probe_gpu",
    "probe_resolved_revisions",
]
