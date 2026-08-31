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
import types
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
        # The DRIVER version, which `torch.version.cuda` below does not report. That attribute is
        # the CUDA toolkit torch was COMPILED against -- a build-time constant of the wheel, so it
        # is identical across every host running the same image. The kernels actually dispatched,
        # and the perf of an unchanged image, move with the host driver instead. Two blocks
        # measured months apart on the same pinned image can therefore differ materially while
        # every recorded version string stays byte-identical, which is exactly the unexplainable
        # drift the provenance record exists to prevent. Same bytes-vs-str decode as the device
        # name above: NVML returns either depending on the pynvml build.
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", "replace")
        result: dict[str, Any] = {
            "available": True,
            "device_count": count,
            "name": str(name),
            "compute_capability": f"{major}.{minor}",
            "total_memory_bytes": int(memory.total),
            "driver_version": str(driver),
            "source": "nvml",
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()

    # Version strings are module attributes; reading them creates no context.
    #
    # `str(...)`, not the attribute itself. `torch.__version__` is a `TorchVersion` -- a str SUBCLASS
    # defined in `torch.torch_version` -- so it pickles by reference to that module. This payload is
    # returned across a Modal boundary to a local process that has no torch, where unpickling it
    # raises `ModuleNotFoundError: torch` and the whole probe is lost AFTER the GPU has been paid
    # for. A plain `str` carries the same characters with no module to import.
    try:
        import torch

        result["torch_version"] = str(torch.__version__)
        result["cuda_version"] = str(torch.version.cuda) if torch.version.cuda else None
    except Exception:  # pragma: no cover - torch is present in the serving image
        pass
    return result


# The libraries whose version decides prompt assembly or engine execution, and therefore the
# measured curve. The benchmark image installs from `pyproject.toml` RANGES rather than from a
# lockfile, so two builds of the same commit can resolve different versions of any of these while
# every source-level checksum, checkpoint commit and catalog row stays identical.
_RUNTIME_PACKAGES: tuple[str, ...] = (
    "torch",
    "vllm",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "flashinfer-python",
    "xformers",
    "nvidia-cutlass-dsl-libs-cu13",
)


def probe_runtime_packages() -> dict[str, Any]:
    """The versions the image actually RESOLVED for the libraries that can move the curve.

    Torch and CUDA alone do not identify this runtime. A Transformers or tokenizers bump changes
    chat-template rendering and therefore the prompt token counts every bucket is defined by; a vLLM
    or kernel-library bump changes execution. Because the image installs from dependency ranges,
    those can all differ between two builds that report the same commit, checksum and checkpoints --
    and the artifacts would present the resulting curves as comparable.

    Read from installed DISTRIBUTION metadata rather than module attributes: it covers packages that
    expose no ``__version__`` and needs no import, so probing cannot itself initialize a runtime.
    Fail-soft per package, like the rest of this module -- a missing distribution is recorded as
    absent rather than killing a lane that has already paid for its boot.
    """
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str | None] = {}
    for name in _RUNTIME_PACKAGES:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = None
        # Broad on purpose: a malformed dist-info must degrade to a recorded absence, not an error.
        except Exception:
            resolved[name] = None
    return resolved


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
    move between builds. vLLM 0.23.0 takes one ``vllm_config`` object rather than raw head dimensions,
    so build the smallest duck-typed view of the real engine config that its resolver reads. The head
    dimension still comes from the served checkpoint; supplying a guessed value would make the probe
    answer its own question.
    """
    signature = inspect.signature(resolver)
    available = _gdn_config_values(base_model)
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name == "vllm_config":
            # Never default the head dim. On SM10.x the resolver grants FlashInfer only when
            # `linear_key_head_dim == 128`, so a missing value silently answers "triton" -- the
            # exact Blackwell fallback this probe exists to detect, reported as if it were the
            # engine's own decision. An absent dimension is a failed config read, not a backend.
            if "linear_key_head_dim" not in available:
                raise TypeError(
                    "resolver requires 'vllm_config.model_config.hf_text_config."
                    "linear_key_head_dim', which this model config does not supply"
                )
            hf_text_config = types.SimpleNamespace(
                linear_key_head_dim=available["linear_key_head_dim"]
            )
            kwargs[name] = types.SimpleNamespace(
                additional_config=None,
                model_config=types.SimpleNamespace(hf_text_config=hf_text_config),
            )
        elif name in available:
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
    result: dict[str, Any] = {
        "base_model": base_model,
        "resolved": None,
        "source": "resolver",
    }
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
            "source": "resolver",
            "subprocess_isolated": True,
        }
    except Exception as exc:
        return {
            "base_model": base_model,
            "resolved": None,
            "reason": f"resolver subprocess failed to start: {type(exc).__name__}: {exc}",
            "source": "resolver",
            "subprocess_isolated": True,
        }

    if completed.returncode != 0:
        return {
            "base_model": base_model,
            "resolved": None,
            "reason": f"resolver subprocess exited {completed.returncode}: {completed.stderr[-400:]}",
            "source": "resolver",
            "subprocess_isolated": True,
        }
    try:
        result = json.loads(completed.stdout)
    except ValueError as exc:
        return {
            "base_model": base_model,
            "resolved": None,
            "reason": f"resolver subprocess returned unparseable output: {exc}",
            "source": "resolver",
            "subprocess_isolated": True,
        }
    result.setdefault("source", "resolver")
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
#
# `config.json` is deliberately NOT a model marker. It is the one file every metadata-only fetch
# pulls -- this probe's own `AutoConfig.from_pretrained` creates a snapshot containing nothing else
# -- so accepting it made a weightless directory answer "which commit are the weights from". A real
# cached `Qwen/Qwen3.5-0.8B` snapshot holding only config, tokenizer and vocab files reported
# itself as the model commit. The weights are what a capacity number is attributed to, so the model
# role requires a file that actually carries them.
_ROLE_MARKERS: dict[str, tuple[str, ...]] = {
    "model": (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ),
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


def _tokenizer_vocab(tokenizer: Any) -> dict[str, Any] | None:
    """The loaded tokenizer's own vocabulary, as the serialized snapshot files spell it.

    Read from ``backend_tokenizer.to_str()`` rather than ``get_vocab()``: the serialized form is
    the same structure ``tokenizer.json`` stores, so a loaded tokenizer and an on-disk snapshot are
    directly comparable. Only ``model.vocab`` is used. Neighbouring fields (``merges``,
    ``added_tokens``) are re-normalized by the installed ``tokenizers`` version on load, so they
    differ from the file even when the file is exactly the one that was loaded; the vocabulary does
    not, and it is the field that actually differs between two different tokenizer trees.

    Returns ``None`` when the object exposes no serializable backend, which makes the caller fall
    back to reporting ambiguity rather than guessing.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    try:
        vocab = json.loads(backend.to_str()).get("model", {}).get("vocab")
    # Broad on purpose: a slow tokenizer, an unserializable backend, or a future schema change must
    # degrade to "cannot disambiguate", never kill a probe that has already paid for its boot.
    except Exception:
        return None
    return vocab if isinstance(vocab, dict) else None


def _snapshot_holding_this_tokenizer(root: Path, holders: list[str], tokenizer: Any) -> str | None:
    """The one candidate snapshot whose ``tokenizer.json`` matches the tokenizer THIS process loaded.

    ``refs/<revision>`` is per-repository and records only the last download, so when an unpinned
    repository advances mid-boot it names the newest tree for every role. The parent process
    assembles benchmark prompts with the tokenizer it loaded BEFORE the engine's own download, and
    reporting the ref for that role publishes a commit the prompts never came from -- while looking
    exactly like a resolved one. Comparing content answers which tree was loaded without depending
    on when it was loaded.

    ``None`` when zero or several candidates match, so the caller reports ambiguity instead of
    choosing. Several matching means the trees are byte-equal in the only field that could
    distinguish them, which does not license naming one of their commits.
    """
    loaded = _tokenizer_vocab(tokenizer)
    if not loaded:
        return None
    matches = []
    for commit in holders:
        path = root / "snapshots" / commit / "tokenizer.json"
        try:
            on_disk = json.loads(path.read_text()).get("model", {}).get("vocab")
        # A candidate that cannot be read is simply not a match; an unreadable sibling must not
        # prevent the readable one from resolving.
        except Exception:
            continue
        if on_disk == loaded:
            matches.append(commit)
    return matches[0] if len(matches) == 1 else None


def _local_snapshot_commit(
    repo: str, revision: str | None, role: str = "model", tokenizer: Any = None
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
        # Several snapshots carry this role. When the caller handed us the tokenizer this process
        # actually loaded, ask which candidate it came from: the ref cannot answer that. It is
        # per-repository and holds only the last download, so an unpinned repository that advanced
        # between the parent's tokenizer load and the engine's own weight fetch leaves the ref
        # naming the newer tree -- and because the ref IS among the holders, the branch below would
        # return it with a resolved-looking source that the publication gate accepts. The prompts
        # were built with the older tree, so that is false provenance, not ambiguity.
        if tokenizer is not None:
            matched = _snapshot_holding_this_tokenizer(root, holders, tokenizer)
            if matched:
                return matched, "snapshot-contents-verified"
            return holders[-1], "snapshot-contents-ambiguous"
        # If the ref is among them the repository simply has older trees cached and the ref still
        # names the current one; otherwise the ref points somewhere this role was never downloaded
        # to, so report the ambiguity instead of choosing.
        if ref_commit and ref_commit in holders:
            return ref_commit, "local-cache-ref"
        return holders[-1], "snapshot-contents-ambiguous"

    if ref_commit:
        return ref_commit, "local-cache-ref"
    return None, None


def probe_resolved_revisions(base_model: str, tokenizer: Any = None) -> dict[str, Any]:
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

    `tokenizer` is the object the PARENT process loaded and builds every benchmark prompt with. It
    is passed rather than re-derived because the cache alone cannot say which tree that was: the
    parent loads its tokenizer before `AsyncLLMEngine.from_engine_args` runs, so an unpinned
    repository that advances in between leaves two snapshots holding tokenizer files and the ref
    naming the later one. Matching the loaded object's own vocabulary identifies its tree
    regardless of what downloaded afterwards.
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
            # Only the tokenizer role can be content-matched: it is the one whose loaded object
            # this process holds. The weights live in the EngineCore process and the processor is
            # absent for text-only models.
            commit, source = _local_snapshot_commit(
                repo, revision, role, tokenizer if role == "tokenizer" else None
            )
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
    """The full provenance block stored with each model's results.

    The engine's `tokenizer` attribute is the very object the driver fits prompts with, so passing
    it makes the recorded tokenizer commit describe the tree that produced the measured tokens
    rather than whatever the shared cache ref points at by probe time.
    """
    # Revisions FIRST, before anything that reads a checkpoint. `_local_snapshot_commit` reports an
    # unpinned model's commit from the cache's `refs/main`, which is mutable state a Hub-touching
    # load can rewrite underneath it. `_gdn_config_values` is pinned to the local cache so it cannot
    # do that, but the order is the second, independent guarantee: provenance is captured before any
    # later-added probe gets the chance to refresh a ref and silently re-date the measurement.
    resolved_revisions = probe_resolved_revisions(base_model, getattr(engine, "tokenizer", None))
    gdn_prefill = probe_gdn_backend(base_model)

    payload: dict[str, Any] = {
        "resolved_revisions": resolved_revisions,
        "runtime_packages": probe_runtime_packages(),
        "gpu": probe_gpu(),
        "gdn_prefill": gdn_prefill,
    }
    if engine is not None:
        payload["kv_cache"] = probe_engine_kv_cache(engine)
    return _plain_types(payload)


def _plain_types(value: Any) -> Any:
    """Rebuild `value` out of types that exist without the serving image installed.

    This payload crosses a Modal boundary into a LOCAL process that has no torch, no vllm and no
    transformers. Anything typed by those packages pickles by reference to them, and the local
    unpickle then dies with `ModuleNotFoundError` -- after the GPU has been allocated, booted and
    billed. `torch.__version__` is exactly that: a `str` SUBCLASS from `torch.torch_version`, which
    reads as an ordinary string everywhere it is used here and destroys the return anyway.

    Each probe already coerces its own values, and that is where the real contract lives. This is
    the boundary that makes the coercion unmissable: a field added later, or a config attribute that
    quietly becomes an enum in the next vLLM bump, degrades to its `str` rather than costing a run.
    Subclasses are normalized too -- `isinstance` is true for `TorchVersion`, so an identity pass
    would let precisely the failing case through.
    """
    if isinstance(value, dict):
        return {str(key): _plain_types(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_types(item) for item in value]
    if value is None:
        return None
    if type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


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
