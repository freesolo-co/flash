"""shared out-of-process verl harness reused by the sft and opd verl trainers.

verl pins its own torch/vllm, incompatible with flash's, so flash never imports verl in-process: it
runs a verl trainer entrypoint as a subprocess against a separate interpreter and merges the
fsdp-sharded lora checkpoint back into a flash-servable peft adapter. these are the framework- and
algorithm-neutral pieces (interpreter resolution, checkpoint export, provenance, progress
streaming); the sft/opd/grpo modules layer their own dataset rows, hydra overrides, and
orchestration on top.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import ctypes
import json
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as _thread_module
from http.server import ThreadingHTTPServer
from typing import Self

from flash.diagnostics import sanitize_diagnostic

# verl 0.8.0 exactly, plus the truncation-mask, 3d position-id, and ulysses fused-label fixes.
# it must stay on the 0.8.0 base: the opd plugin patches 0.8.0 internals and imports
# verl.trainer.main_ppo_sync, which verl deleted after 0.8.0, and opd's exact-version gate reads the
# version file this branch pins to the release value.
VERL_REQUIREMENT_NAME = "verl"
VERL_REQUIREMENT_URL = (
    "git+https://github.com/freesolo-co/verl@1bea7d6825bbb9d2164e86e379b3680e7c53bb8a"
)
# the pin, as the venv stamp records it. the provisioning install asks for the [vllm] extra of this
# same commit; the stamp stays extra-free so it identifies the verl a venv holds, not how it was
# installed.
VERL_REQUIREMENT = f"{VERL_REQUIREMENT_NAME} @ {VERL_REQUIREMENT_URL}"

# the FA2 wheel the verl interpreter needs, kept byte-identical to Dockerfile.worker's
# ARG FLASH_ATTN_SPEC default so the fallback venv and /opt/verl-venv resolve the same build.
# prebuilt for cu128/torch2.10/cp312; installed --no-build-isolation, never source-built here.
FLASH_ATTN_SPEC = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
    "v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl"
)
# the wheel above is cp312-ONLY, and flash itself supports 3.11 (pyproject requires-python >=3.11),
# so a bare `uv venv` on a 3.11 host builds an interpreter the wheel cannot install into -- and that
# install is required, so the run dies during provisioning instead of training. name the interpreter
# rather than inheriting the host's.
VERL_VENV_PYTHON = "3.12"

# gated-deltanet kernels for the verl interpreter, kept byte-identical to Dockerfile.worker's
# verl-venv layer. fla resets the DeltaNet recurrence at packed example boundaries (cu_seqlens)
# and causal_conv1d resets the short causal conv (seq_idx); WITHOUT them transformers falls back
# to implementations that accept both arguments and silently discard them, so packed GDN training
# is contaminated across example boundaries while appearing patched. both are required in the
# IMAGE; here causal_conv1d stays best-effort to install because this path has no build environment
# to guarantee, but a venv that misses it is left unstamped rather than reused (see below).
FLA_REQUIREMENT = (
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git"
    "@f0e213dbd8b5fb90c3c7eca869ac1706d5377139"
)
CAUSAL_CONV1D_REQUIREMENT = "causal-conv1d==1.6.2.post1"

# the SAME transformers range the main interpreter and Dockerfile.worker's verl-venv layer use.
# is_flash_linear_attention_available() and is_causal_conv1d_available() are find_spec probes whose
# import path moved in 5.13: an unpinned resolve lands 5.14.x, which answers False for BOTH even
# though the packages are installed, so the child reports no gdn boundary-reset capability and
# grpo/opd fail closed on every catalog model. verl and vllm both depend on transformers, so this
# has to be in the OVERRIDE file as well as the direct list -- a direct pin alone loses to their
# transitive declarations.
TRANSFORMERS_REQUIREMENT = "transformers>=5.6,<5.13"

# the stamp must identify every separately installed package the venv holds. omitting flash-attn,
# fla, causal_conv1d, or the transformers range lets an older partial venv match forever; conv1d
# leaves GRPO/OPD failing ``require_gdn_boundary_resets`` with no rebuild path, and a stale venv
# resolved before the transformers pin holds the 5.14.x that makes both probes answer False.
VERL_VENV_STAMP = (
    f"{VERL_REQUIREMENT}\n{FLASH_ATTN_SPEC}\n{FLA_REQUIREMENT}\n{CAUSAL_CONV1D_REQUIREMENT}\n"
    f"{TRANSFORMERS_REQUIREMENT}"
)


# how many times the prebuilt-wheel install is attempted before the arm is handed back to the plane.
# uv already retries the download 3x internally, so each attempt here is a fresh uv invocation, not a
# fresh request: the point is to outlast an outage that spans one uv lifetime, not to hammer github.
FLASH_ATTN_INSTALL_ATTEMPTS = 3
FLASH_ATTN_INSTALL_BACKOFF_S = 15.0


def _install_flash_attn(py: str) -> None:
    """Install the prebuilt FA2 wheel, raising RetriableInfraError rather than dying on a network fault.

    The install is required (verl's cuda path imports flash_attn.bert_padding unguarded), so this
    cannot fail soft. Measured: an arm died on `error sending request ... operation timed out` for
    this exact url, and the identical url served 200 on retry moments later, so "cannot succeed on
    another worker" was simply false.
    """
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    command = ["uv", "pip", "install", "--python", py, "--no-build-isolation", FLASH_ATTN_SPEC]
    for attempt in range(FLASH_ATTN_INSTALL_ATTEMPTS):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as e:
            if attempt == FLASH_ATTN_INSTALL_ATTEMPTS - 1:
                # hand the arm back to the plane as infra-shaped: it retries as job_preempted on a
                # fresh worker instead of failing the run. bounded by the plane's own retry budget
                # (INFRA_RETRY_FLOOR), so a genuinely broken wheel spec still terminates rather than
                # looping -- it just costs a few provisioning attempts to establish that.
                raise RetriableInfraError(
                    f"prebuilt flash-attn wheel install failed {FLASH_ATTN_INSTALL_ATTEMPTS}x "
                    f"(exit {e.returncode}): {FLASH_ATTN_SPEC}"
                ) from e
            time.sleep(FLASH_ATTN_INSTALL_BACKOFF_S * (attempt + 1))


def _install_causal_conv1d(py: str) -> bool:
    """Install the causal conv kernel that resets GDN conv state at packed example boundaries.

    Best-effort, deliberately: unlike flash-attn this cannot fail the provisioning. Dying here would
    turn a compiler hiccup into a dead paid run, and returning False is not the end of the story --
    ``require_gdn_boundary_resets`` raises for grpo/opd on a gdn model whose child cannot reset, so
    the failure still surfaces, with a message naming the missing kernel. SFT is unaffected either
    way (``_packing_mode`` pins gdn to one example per update, so there is nothing to contaminate).

    This is the NO-IMAGE provisioning path. ``Dockerfile.worker`` requires the same package outright,
    because a prebuilt image has no compiler hiccup to forgive: a build that cannot produce the
    kernel should fail before it ships, not after a run has paid for a gpu.
    """
    completed = subprocess.run(
        ["uv", "pip", "install", "--python", py, "--no-build-isolation", CAUSAL_CONV1D_REQUIREMENT],
        check=False,
        env={**os.environ, "CAUSAL_CONV1D_FORCE_BUILD": "TRUE"},
    )
    # a fake subprocess.run in a test may return None; treat only an explicit nonzero as failure.
    if getattr(completed, "returncode", 0) != 0:
        return False
    # a zero exit is not the kernel landing. this is a compiled cuda extension, and an ABI or
    # symbol mismatch installs cleanly and then fails at import -- Dockerfile.worker gates the
    # same package on the same import for the same reason, and uninstalls on failure.
    # transformers' is_causal_conv1d_available() is only a find_spec probe, so a
    # present-but-broken package passes it, and the run crashes at model load instead of
    # safely training unpacked. importing is the only check that distinguishes the two, and
    # reporting the truth here is what keeps the venv unstamped so a later attempt can rebuild
    # it.
    imported = subprocess.run(
        [py, "-c", "import causal_conv1d"],
        check=False,
        capture_output=True,
    )
    if getattr(imported, "returncode", 0) != 0:
        # leave nothing importable behind: a broken build that stays installed would keep passing
        # find_spec probes on this pod for the life of the workdir.
        subprocess.run(
            ["uv", "pip", "uninstall", "--python", py, "causal-conv1d"],
            check=False,
            capture_output=True,
        )
        return False
    return True


def clamp_engine_len(engine_len: int, max_position_embeddings: int | None) -> int:
    """the engine length verl will accept: the job's context, capped at the model's own limit.

    verl raises ValueError when rollout.max_model_len exceeds the model's max_position_embeddings,
    so a job whose max_context_tokens overshoots the architecture would die at rollout startup
    rather than train on a shorter context. clamp instead. an unknown limit (probe failed) passes
    through untouched, leaving verl's own resolution in charge.
    """
    if max_position_embeddings is None or max_position_embeddings <= 0:
        return int(engine_len)
    return min(int(engine_len), int(max_position_embeddings))


def model_max_position_embeddings(model_id: str, revision: str = "") -> int | None:
    """the model's architectural context limit, or None when it cannot be determined.

    mirrors verl's own lookup (workers/rollout/utils.py:get_max_position_embeddings): the top-level
    attribute, falling back to the nested text_config that multimodal architectures use. every model
    in the flash catalog nests it there. None on any probe failure, so a config that cannot be read
    never blocks a run that works today.
    """
    try:
        from transformers import AutoConfig

        from flash.engine.worker.hf import model_revision_kwargs

        cfg = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, **model_revision_kwargs(revision)
        )
        limit = getattr(cfg, "max_position_embeddings", None)
        if limit is None:
            text_cfg = getattr(cfg, "text_config", None)
            limit = getattr(text_cfg, "max_position_embeddings", None) if text_cfg else None
        return int(limit) if limit else None
    except Exception as e:  # an unreadable config must not fail the run
        print(f"[verl] max_position_embeddings probe failed for {model_id!r}: {e}", flush=True)
        return None


# verl ships two fused linear-CE backends and dispatches on `fused_kernel_options.impl_backend`
# (models/transformers/monkey_patch.py). all three flash trainers used to pin `torch`, which is
# verl's own default (config/model/hf_model.yaml:67) -- a restated default rather than a choice.
#
# MEASURED, 4096x2560 hidden into a 248320 vocab, bf16, fwd+bwd, paired and alternating, 10 reps
# after warmup, verl's own entry points on both arms:
#
#     card               sm    triton vs torch   peak VRAM        triton err vs fp32
#     A100 SXM4 80GB     80    3.28x SLOWER      2.75x lower      lower on all 4 quantities
#     A40                86    1.43x SLOWER      2.75x lower      lower on all 4 quantities
#     H100 80GB HBM3     90    2.25x FASTER      2.72x lower      lower on all 4 quantities
#
# so the choice is card-dependent and a blanket flip would slow every sm80/sm86 run. the split is
# sm90: the triton kernel's TMA path is gated on `get_device_capability()[0] >= 9`
# (verl/utils/kernel/kernels.py:48), and below that it takes a general mainloop that loses to the
# torch chunked loop. the H100 win is stable across shapes (1.9x at 1024 tokens, 2.15x at 2048,
# 2.25x at 4096 and 8192).
#
# ACCURACY is not the deciding axis but is worth recording, because "faster" would not be worth a
# worse gradient: measured against an fp32 log-softmax arbiter, triton is closer on EVERY card and
# every quantity -- log_probs 3e-6 vs 7e-4, grad_hidden 6.5e-3 vs 1.8e-2, grad_weight 2.8e-3 vs
# 1.8e-2. the two backends differ from each other by ~1.9e-2 on gradients, and that gap is mostly
# the torch backend's own error, not triton's.
_TRITON_FUSED_CE_MIN_CAPABILITY = (9, 0)


def fused_ce_backend(caps: dict) -> str:
    """`triton` where it is measurably faster (sm90+), `torch` everywhere else.

    Takes the capability from the out-of-process probe (`verl_device_capability`) rather than
    calling `torch.cuda.get_device_capability` here, for two reasons:

    * this runs in the long-lived PARENT, which does not otherwise touch cuda. initializing a
      context here to answer one question retains it for the process lifetime, on the same devices
      `torchrun` is about to own -- unbudgeted VRAM against a reserve sized without it, which is
      exactly the kind of overhead that OOMs a job sized near a card's limit.
    * the question is about verl's kernels, and verl pins its own torch. the parent's torch is the
      wrong interpreter to ask.

    An unanswerable probe (no cuda, no torch, hung import) yields None and falls back to `torch`,
    which is the current behaviour on every card.
    """
    cc = verl_device_capability(caps)
    if cc is None:
        return "torch"
    return "triton" if cc >= _TRITON_FUSED_CE_MIN_CAPABILITY else "torch"


def agent_loop_workers(rollout_batch: int, *, cap: int = 8) -> int:
    """largest divisor of ``rollout_batch`` that is <= ``cap`` (verl's default worker count).

    verl 0.8.0 hardcodes async rollout (ray_trainer.py:914), so every rollout goes through
    AgentLoopManager, which splits the batch across agent.num_workers with DataProto.chunk and
    asserts the split is exact (agent_loop.py:1111 -> protocol.py:874). the worker count must
    therefore divide the batch. returning a divisor keeps the run alive for any batch size while
    preserving parallelism whenever the batch permits it.
    """
    if rollout_batch <= 0:
        raise ValueError("rollout_batch must be positive")
    workers = next(n for n in range(min(cap, rollout_batch), 0, -1) if rollout_batch % n == 0)
    # a PRIME batch has no divisor in (1, cap], so the pool collapses to a single worker and the
    # rollout runs fully serialized -- a silent cap-fold slowdown, not an error. it needs a prime
    # group_size (11 or 13 within the plausible range) to happen, since prompts_per_step alone is
    # multiplied by group_size, so it is rare rather than impossible. say so instead of leaving the
    # operator to infer it from wall-clock.
    if workers == 1 and rollout_batch > 1:
        print(
            f"[flash-verl][warn] rollout batch {rollout_batch} has no divisor <= {cap}, so the "
            f"agent-loop pool is 1 worker and the rollout is serialized. a batch with a small "
            f"factor (adjust prompts_per_step or group_size) restores up to {cap}-way parallelism."
        )
    return workers


# worker threads each bridge serves requests from. the bridges are pure i/o relays (parse json,
# hand the payload to a callback, write json back), so this bounds concurrency, not throughput:
# requests beyond the pool wait in the listen backlog instead of each spawning an os thread.
_BRIDGE_WORKER_THREADS = 16


class _DaemonBridgeThreadPool(ThreadPoolExecutor):
    """A ``ThreadPoolExecutor`` whose workers cannot hold the interpreter open.

    A hung environment reward or teacher callback must not stop the worker from publishing its
    terminal result and releasing a paid GPU -- and it is also the property the old daemon
    ``ThreadingHTTPServer`` had before requests moved onto a pool. Only the thread creation differs
    from the stdlib; queueing, idle reuse, and shutdown are inherited unchanged.
    """

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread = threading.Thread(
                name=f"{self._thread_name_prefix or self}_{num_threads}",
                target=_thread_module._worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            thread.start()
            self._threads.add(thread)
            # deliberately NOT added to _thread_module._threads_queues: that mapping is exactly
            # what the interpreter-exit hook walks to join workers.


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """an http server that serves requests from a fixed thread pool instead of one thread each.

    unbounded ``ThreadingHTTPServer`` can exhaust the thread table and cause
    ``RemoteDisconnected`` (VERL-139). the fixed pool prevents that, while the larger listen backlog
    must absorb rollout bursts instead of resetting queued connections.
    """

    # daemon threads so a stuck handler can never keep the worker process alive at shutdown.
    # `daemon_threads` is a ThreadingMixIn flag, read only by the `process_request` this class
    # overrides, so it does nothing here on its own -- _DaemonBridgeThreadPool below is what
    # actually delivers the property it names.
    daemon_threads = True

    # measured: at the socketserver default of 5, 13 of 64 simultaneous callers were reset by peer.
    request_queue_size = 128

    def __init__(self, *args, worker_threads: int = _BRIDGE_WORKER_THREADS, **kwargs):
        self._bridge_pool = _DaemonBridgeThreadPool(
            max_workers=worker_threads, thread_name_prefix="flash-bridge"
        )
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address) -> None:
        """hand the request to the pool. socketserver's default starts an unbounded thread here."""
        self._bridge_pool.submit(self._run_request, request, client_address)

    def _run_request(self, request, client_address) -> None:
        # mirrors ThreadingMixIn.process_request_thread: the pool owns the thread, so shutdown of
        # the connection (not the server) is all this needs to do.
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def server_close(self) -> None:
        super().server_close()
        # do not wait: a handler blocked on a hung peer must not stop the worker from exiting.
        self._bridge_pool.shutdown(wait=False, cancel_futures=True)


def _cgroup_cpu_quota() -> int | None:
    """cpus this container is actually allowed, from its cgroup quota, or None if unlimited."""
    try:  # cgroup v2: "<quota> <period>", or "max <period>" when uncapped
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota == "max":
            # an uncapped v2 controller is the authoritative answer: return it rather than falling
            # through to v1, whose files can still exist on a hybrid layout and carry a stale or
            # parent-scoped limit that has nothing to do with this container.
            return None
        if int(period) > 0:
            return max(1, int(int(quota) // int(period)))
    except Exception:
        pass
    try:  # cgroup v1: quota and period in separate files, quota -1 when uncapped
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        if quota > 0 and period > 0:
            return max(1, int(quota // period))
    except Exception:
        pass
    return None


# cpus verl's placement group reserves per gpu bundle: RayResourcePool builds
# bundle = {"CPU": max_colocate_count, "GPU": 1} per gpu (single_controller/ray/base.py:143) and
# ResourcePoolManager defaults max_colocate_count to 3 (base.py:189).
_VERL_CPU_PER_GPU_BUNDLE = 3
# plus the singletons outside the bundles: TaskRunner is ray.remote(num_cpus=1) (main_ppo.py:83),
# and opd adds one storage unit.
_VERL_CPU_OVERHEAD = 2


def verl_cpu_demand(gpu_count: int) -> int:
    """cpus verl's placement group needs for ``gpu_count`` gpus, read off the verl source.

    the bundles are requested STRICT_PACK, so this is a single-node demand: 8 gpus want 26 cpus on
    one box, not spread over several.
    """
    return max(1, gpu_count) * _VERL_CPU_PER_GPU_BUNDLE + _VERL_CPU_OVERHEAD


def ray_num_cpus(gpu_count: int = 1, *, cap: int = 16) -> int:
    """cpus to hand ray, sized to the CONTAINER rather than the host.

    ray otherwise sees host cores and eagerly forks enough idle workers to exhaust container memory
    or process limits (VERL-123, VERL-124). prefer cgroup quota, then affinity, with a cap for an
    unconstrained host.

    ``verl_cpu_demand`` is a hard floor: fewer logical CPUs leaves the placement group waiting forever
    because verl validates GPUs only. this controls ray prefill, not GPU parallelism.
    """
    floor = verl_cpu_demand(gpu_count)
    quota = _cgroup_cpu_quota()
    if quota is not None:
        return max(floor, min(quota, cap))
    try:
        affinity = len(os.sched_getaffinity(0))
    except Exception:
        affinity = os.cpu_count() or 1
    return max(floor, min(affinity, cap))


# ---------------------------------------------------------------------------
# batched child-interpreter capability probe
# answer independent questions in one child to pay one torch/verl import. child probes may import only
# packages installed in that venv; parent-only hub reads are interpolated. each answer fails closed
# independently and flushes before the live CUDA GDN smoke, which must run last.
# ---------------------------------------------------------------------------
_CAPABILITY_PROBE = """
import json


def emit(key, value):
    print("FLASH_VERL_CAPS=" + json.dumps({key: value}), flush=True)


try:
    from verl.workers.config.rollout import RolloutConfig

    emit("rollout_fields", sorted(RolloutConfig.__dataclass_fields__))
except Exception:
    emit("rollout_fields", None)

try:
    import torch

    emit(
        "capability",
        list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    )
except Exception:
    emit("capability", None)

try:
    import flashinfer  # noqa: F401

    emit("flashinfer", True)
except Exception:
    emit("flashinfer", False)

try:
    import wandb  # noqa: F401

    emit("wandb", True)
except Exception:
    emit("wandb", False)

# gdn is asked only when the parent resolved a model_type to ask about (a gdn hybrid). the module
# name is a parent-resolved literal: it comes from the checkpoint config, which needs a hub/cache
# read the child must not repeat.
if %(gdn_module)r:
    try:
        import importlib
        import inspect

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        # the reason the LAST failing check produced, so a clean negative can name it. the two
        # package flags cannot express the source check: it fails with both installed, and an
        # operator reading "fla=True causal_conv1d=True" alongside "install fla + causal_conv1d"
        # is being sent to fix something that is not broken.
        why = ""
        ok = is_flash_linear_attention_available() and is_causal_conv1d_available()
        if not ok:
            why = (
                "fla="
                + str(is_flash_linear_attention_available())
                + " causal_conv1d="
                + str(is_causal_conv1d_available())
                + " (install the missing package into the verl interpreter)"
            )
        if ok:
            import causal_conv1d  # noqa: F401  -- fail a built-but-broken ABI here, not at model load

            mod = importlib.import_module(%(gdn_module)r)
            gdn = next(
                (
                    c
                    for n, c in vars(mod).items()
                    if isinstance(c, type) and n.endswith("GatedDeltaNet")
                ),
                None,
            )
            # read THROUGH decorators. transformers wraps this forward in
            # @force_accelerate_hooks("conv1d"), which returns a plain closure and sets no
            # __wrapped__, so both getsource() and inspect.unwrap() return the hook wrapper --
            # ~1.1k chars of accelerate plumbing that mentions neither name. That reads as "this
            # transformers build cannot reset gdn state", which is a FALSE NEGATIVE: every released
            # 5.x reaches the values as kwargs.get("cu_seq_lens_q") / kwargs.get("seq_idx") in the
            # real body, so no version bump can fix it and the raise sends operators to reinstall
            # packages that are already present. The undecorated function is reachable through the
            # wrapper's closure cells; walk them until the source answers the question.
            src = ""
            if gdn is not None:
                fn = gdn.forward
                seen = set()
                while True:
                    src = inspect.getsource(fn)
                    if ("cu_seq_lens_q" in src) and ("seq_idx" in src):
                        break
                    nxt = None
                    for cell in getattr(fn, "__closure__", None) or ():
                        try:
                            val = cell.cell_contents
                        except ValueError:
                            continue
                        if inspect.isfunction(val) and id(val) not in seen:
                            seen.add(id(val))
                            nxt = val
                            break
                    if nxt is None:
                        break
                    fn = nxt
            ok = ("cu_seq_lens_q" in src) and ("seq_idx" in src)
            if not ok:
                # both packages are present, so this is the transformers version, not the env.
                why = (
                    "fla and causal_conv1d are both installed, but "
                    + (
                        "no *GatedDeltaNet class in " + %(gdn_module)r
                        if gdn is None
                        else (
                            "its forward() takes neither cu_seq_lens_q nor seq_idx"
                            if ("cu_seq_lens_q" not in src) and ("seq_idx" not in src)
                            else "its forward() is missing "
                            + ("cu_seq_lens_q" if "cu_seq_lens_q" not in src else "seq_idx")
                        )
                    )
                    + " -- this transformers build cannot reset gdn state at packed boundaries; "
                    + "installing packages will not fix it"
                )
        if ok:
            import torch

            # a causal_conv1d compiled without this arch imports fine and raises at the first
            # forward, so smoke it now: the answer decides whether we pack, and being wrong means
            # training contaminated.
            if torch.cuda.is_available():
                from causal_conv1d import causal_conv1d_fn

                causal_conv1d_fn(
                    torch.zeros(1, 4, 8, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(4, 3, device="cuda", dtype=torch.bfloat16),
                )
                torch.cuda.synchronize()
        if not ok:
            # say WHY on a clean negative too: which of the three checks failed is the difference
            # between "install fla" and "rebuild causal_conv1d for this arch".
            print(
                "[verl] gdn boundary resets unavailable in the child: " + why,
                flush=True,
            )
        emit("gdn_boundary_resets", bool(ok))
    except Exception as e:
        # a bare False here is what made this invisible. the caller raises on a hybrid that cannot
        # reset, and an operator reading only that has no way back to which check fell over --
        # a missing wheel, a broken ABI, or a conv kernel compiled without this arch all look
        # identical from the parent. printed to STDOUT, which probe_verl_capabilities forwards
        # on the '[verl] ' prefix -- stderr would be dropped, it is only reported when the
        # child answers nothing at all.
        import traceback

        print(
            "[verl] gdn boundary resets unavailable in the child: "
            + " | ".join(traceback.format_exc(limit=3).strip().splitlines()),
            flush=True,
        )
        emit("gdn_boundary_resets", False)
"""

# every answer's fail-closed default, used when the child dies, times out, or prints nothing
# readable. `None` means "could not answer" for the two callers that must distinguish that from a
# negative answer: rollout_fields (unknown verl -> do not claim a fork field is missing) and
# capability (unknown card -> leave the library default alone rather than guessing a workaround).
_CAPABILITIES_UNAVAILABLE: dict = {
    "rollout_fields": None,
    "capability": None,
    "flashinfer": False,
    "wandb": False,
    "gdn_boundary_resets": None,
}

# the batched timeout is the sum of the former 300 + 600 + 120 + 120 + 120 second budgets, so the
# final GDN smoke retains its cold-import allowance. reaching the sum indicates a wedged child.
_CAPABILITY_PROBE_TIMEOUT_S = 300 + 600 + 120 + 120 + 120


def probe_verl_capabilities(python_bin: str, gdn_module: str = "") -> dict:
    """ask the VERL CHILD every independent capability question in ONE subprocess.

    ``gdn_module`` is parent-resolved; ``""`` skips the live CUDA GDN smoke. every key has a fail-closed
    default, and incrementally flushed answers survive a later child timeout.
    """
    probe = _CAPABILITY_PROBE % {"gdn_module": gdn_module}
    caps = dict(_CAPABILITIES_UNAVAILABLE)
    timed_out = False
    try:
        done = subprocess.run(
            [python_bin, "-c", probe],
            capture_output=True,
            text=True,
            timeout=_CAPABILITY_PROBE_TIMEOUT_S,
        )
        stdout, stderr, returncode = done.stdout, done.stderr, done.returncode
    except subprocess.TimeoutExpired as e:
        # the killed child still flushed every answer it reached, and TimeoutExpired carries that
        # output. a slow gdn smoke must not retract the cheap answers that already landed -- an
        # empty rollout_fields aborts grpo, which is a worse outcome than training padded.
        timed_out = True
        stdout = (
            e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        )
        stderr = (
            e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        )
        returncode = None
        print(
            f"[verl] capability probe timed out after {_CAPABILITY_PROBE_TIMEOUT_S}s; "
            "keeping the answers it flushed, failing the rest closed"
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[verl] capability probe failed ({e}); treating every capability as unavailable")
        return caps
    answered_any = False
    for line in (stdout or "").splitlines():
        if not line.startswith("FLASH_VERL_CAPS="):
            continue
        with contextlib.suppress(Exception):
            answered = json.loads(line.split("=", 1)[1])
            if isinstance(answered, dict):
                # merge rather than replace: a child from a future/older flash that answers a key
                # this build does not know must not add it, and one it never reached must keep its
                # fail-closed default rather than vanishing from the dict.
                caps.update({k: v for k, v in answered.items() if k in caps})
                answered_any = True
    # the child's own diagnostics, forwarded BEFORE the early return. the gdn question prints why it
    # answered no, and it is answered LAST -- so every real run answers something earlier, and a
    # bare `return caps` here would discard the one line the gate's raise tells operators to read.
    # matched on the prefix rather than forwarded wholesale: the child's stdout also carries verl's
    # own import chatter, which is not worth a parent log line.
    for line in (stdout or "").splitlines():
        if line.startswith("[verl] "):
            print(line, flush=True)
    if answered_any:
        return caps
    if not timed_out:
        print(
            "[verl] capability probe returned no readable answer "
            f"(exit {returncode}); treating every capability as unavailable"
        )
    if stderr:
        print(f"[verl] capability probe stderr: {stderr.strip()[:800]}")
    return caps


def verl_declares_rollout_field(caps: dict, field: str) -> bool:
    """report whether the probed verl declares `field` on RolloutConfig.

    the fork adds rollout fields stock verl does not have. hydra composes an unknown key happily and
    only fails later in omega_conf_to_dataclass, so callers ask here before emitting a fork-only
    override rather than letting the run abort at dataclass conversion. an unanswerable probe reads
    as "not declared", which is what the subprocess version returned when the child could not be
    reached: the caller raises with the cause instead of composing a key that would abort in
    dataclass conversion on a paid gpu.
    """
    fields = caps.get("rollout_fields")
    return bool(fields) and field in fields


def gdn_probe_module(model_id: str, revision: str = "") -> str:
    """the ``transformers.models.*`` modeling module the gdn question must inspect.

    resolved by the PARENT and handed to the child as a literal, because it comes from the
    checkpoint config -- a hub/cache read the child should not repeat. pair every call with
    ``gdn_reset_arch_from_caps`` below so the shim is rendered for the SAME arch the child cleared.
    """
    from flash.engine.worker.packing import gdn_model_type

    model_type = gdn_model_type(model_id, revision=revision)
    return f"transformers.models.{model_type}.modeling_{model_type}"


def gdn_reset_arch_from_caps(caps: dict, gdn_module: str) -> str | None:
    """the architecture to patch when the VERL CHILD can honor packed GDN boundary resets, else None.

    transformers silently discards ``cu_seqlens`` without fla and ``seq_idx`` without causal_conv1d,
    contaminating packed boundaries. probe the child interpreter because it owns the model stack.

    return the verified ``model_type`` rather than a bool so the shim cannot patch a different module;
    ``qwen3_5`` and ``qwen3_5_moe`` are distinct. child probes must not import parent-only ``flash``.
    """
    if not caps.get("gdn_boundary_resets"):
        return None
    # "transformers.models.<arch>.modeling_<arch>" -> "<arch>", read from the string the CHILD was
    # handed rather than re-resolved here: a second resolve could disagree with what was cleared.
    return gdn_module.split(".")[2]


def require_gdn_boundary_resets(caps: dict, gdn_module: str) -> str | None:
    """The arch to patch for a gdn hybrid, raising if the child cannot honor boundary resets.

    On grpo/opd there is no safe fallback, so this raises rather than launching. verl's padded path
    (``use_remove_padding=False``) is boundary-correct but cannot complete a step: flash sets
    ``use_fused_kernels=True`` unconditionally and verl's fsdp engine has no guard for that pair, so
    the padded+fused branch returns a dense ``[bsz, response_len]`` where siblings re-nest via
    ``cu_seqlens``. That fails at EVERY batch size, measured dead at ``padding.py:144`` on four
    matched real-gpu arms, and the error names neither gdn nor this decision -- so letting it launch
    spends the whole rental on an untraceable shape error. Dropping ``use_fused_kernels`` is NOT the
    alternative: ``engine/vram.py`` sizes assuming no dense ``[b, s, vocab]`` logits tensor exists
    (~130 GB at 32k on a 248k vocab), trading this crash for an OOM. Give the child the kernels.

    sft reaches the same branch but is safe without resets: every gdn model profiles as
    exact-unpacked, pinning ``examples_per_update`` to 1 (``sft_workload.py:410``), so there are no
    packed neighbours to contaminate.
    """
    if not gdn_module:
        return None
    arch = gdn_reset_arch_from_caps(caps, gdn_module)
    if arch is None:
        raise RuntimeError(
            "gdn hybrid without child-side boundary resets: the padded fallback "
            "(use_remove_padding=False) is incompatible with the use_fused_kernels=True this "
            "recipe sets, and cannot complete a training step on verl's fsdp engine. see the "
            "'[verl] gdn boundary resets unavailable' line above for why the child could not "
            "honor resets -- installing fla + causal_conv1d in the verl interpreter is the fix, "
            "not disabling fused kernels."
        )
    return arch


def resolve_verl_python(workdir: str, *, install_wandb: bool = False) -> str:
    """return an interpreter that can import verl.

    prefer ``FLASH_VERL_PYTHON`` unchanged; otherwise provision an isolated venv with the pinned verl
    stack and optional wandb. rebuild owned venvs when their stamp differs. a missing or blank preset
    takes the provisioning path.
    """
    preset = os.environ.get("FLASH_VERL_PYTHON", "").strip()
    if preset:
        return preset
    venv = os.path.join(workdir, "verl-venv")
    py = os.path.join(venv, "bin", "python")
    stamp = os.path.join(venv, "flash-verl-requirement")
    installed = ""
    if os.path.exists(stamp):
        with open(stamp) as f:
            installed = f.read().strip()
    if installed != VERL_VENV_STAMP or not os.path.exists(py):
        # a retry reuses the pod workdir, so this venv can be from an earlier attempt, from an earlier
        # flash release pinning a different verl, or from an install that died partway. remove it and
        # start clean: reusing it would train on the wrong verl, and `uv venv` refuses to write into a
        # directory that already holds a pyvenv.cfg, so a half-built venv wedges every later retry.
        shutil.rmtree(venv, ignore_errors=True)
        # dev-only fallback (production uses FLASH_VERL_PYTHON on a prebuilt verl image): verl brings
        # its own torch/vllm, so use a full install rather than --no-deps to include runtime deps.
        subprocess.run(["uv", "venv", "--python", VERL_VENV_PYTHON, venv], check=True)
        # the SAME four overrides Dockerfile.worker writes, for the same reason: this pin
        # set deliberately violates declared ceilings, and a bare pin cannot break
        # any of them -- a pin is a constraint the resolver must satisfy alongside the
        # declaration, so the pair is simply unsatisfiable and the install fails outright.
        # only --override makes uv IGNORE the declaration. all four lines are required;
        # dropping any one leaves the set unsatisfiable or silently wrong: verl +
        # transferqueue declare numpy<2.0.0 -> vllm 0.19.1 needs numpy>=2 verl[vllm]
        # declares vllm>=0.8.5,<=0.12.0 -> flash needs 0.19.1 for the Qwen3.5 archs vllm
        # declares xgrammar>=0.1.32 -> structured opd gates on EXACTLY 0.1.25 verl and
        # vllm both depend on transformers with no upper bound -> an unpinned resolve
        # lands 5.14.x, whose moved import path makes the fla and causal_conv1d find_spec
        # probes answer False for packages that ARE installed, so the child reports no
        # gdn boundary-reset capability and grpo/opd fail closed after the gpu is rented.
        overrides = os.path.join(workdir, "verl-overrides.txt")
        with open(overrides, "w") as f:
            f.write(f"numpy==2.2.6\nxgrammar==0.1.25\nvllm==0.19.1\n{TRANSFORMERS_REQUIREMENT}\n")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                py,
                "--override",
                overrides,
                # the [vllm] extra, matching Dockerfile.worker's VERL_SPEC. verl's bare
                # install_requires omits vllm, and every entrypoint flash launches needs it:
                # main_ppo/main_ppo_sync set rollout.name=vllm.
                f"{VERL_REQUIREMENT_NAME}[vllm] @ {VERL_REQUIREMENT_URL}",
                "vllm==0.19.1",
                "numpy==2.2.6",
                TRANSFORMERS_REQUIREMENT,
                # NOT transitively guaranteed: verl imports these at MODULE level on the launch path
                # (main_ppo -> ppo.ray_trainer -> rollout.llm_server imports cachetools, and
                # rollout.utils imports uvicorn + fastapi), yet declares none of them. vllm happens
                # to pull cachetools and fastapi today, but that is vllm's dependency choice, not a
                # contract verl states -- name them so a vllm respin cannot silently break launch.
                "cachetools",
                "uvicorn",
                "fastapi",
                # opd's entrypoint calls tq.init()/tq.close(); absent from verl's setup.py.
                "TransferQueue==0.1.7",
                # older raises AttributeError on PyArrow PyExtensionType.
                "datasets>=4.7,<6",
                "liger-kernel",
                "bitsandbytes>=0.49",
                "qwen-vl-utils",
                "torchvision",
                "xgrammar==0.1.25",
                "tqdm",
                "pyarrow",
                # gated-deltanet kernels, in LOCKSTEP with Dockerfile.worker's
                # verl-venv layer (same fla sha, same tilelang pins). the model
                # trains in THIS interpreter, so fla being installed in the
                # worker's own env says nothing: without it here, transformers
                # binds chunk_gated_delta_rule to the pure-torch fallback that
                # takes **kwargs and discards cu_seqlens, and packed gdn runs
                # train across example boundaries while looking patched.
                # apache-tvm-ffi is pinned to 0.1.11 because 0.1.12
                # double-registers TVM-FFI and aborts `import tilelang`.
                FLA_REQUIREMENT,
                "tilelang==0.1.11",
                "apache-tvm-ffi==0.1.11",
            ],
            check=True,
        )
        # a SEPARATE install, exactly as Dockerfile.worker runs it: the wheel is prebuilt against
        # torch 2.10, so it needs --no-build-isolation, and that flag must not apply to the resolve
        # above. required, not best-effort -- remove-padding is the default on all three backends and
        # verl's cuda path imports flash_attn.bert_padding unguarded with no sdpa fallback, so a venv
        # without it dies at the first training batch on a paid gpu rather than degrading.
        _install_flash_attn(py)
        conv_installed = _install_causal_conv1d(py)
        # written only after ALL installs succeed, so a venv that died between them is
        # unstamped and the next attempt rebuilds it rather than reusing a
        # half-provisioned interpreter. the conv kernel is best-effort to INSTALL but not
        # optional to RECORD. grpo and opd pack, so require_gdn_boundary_resets raises for
        # a gdn model whose child lacks it; stamping a venv that missed the build would
        # hand every later attempt on this pod the same broken interpreter, with the stamp
        # asserting it is fully provisioned. leaving it unstamped costs one rebuild and
        # gives the next attempt a real chance at the kernel.
        if conv_installed:
            with open(stamp, "w") as f:
                f.write(VERL_VENV_STAMP)
        else:
            print(
                f"[verl] {CAUSAL_CONV1D_REQUIREMENT} did not install; leaving the venv unstamped so "
                "the next attempt rebuilds it rather than reusing an interpreter that cannot honor "
                "gdn boundary resets",
                flush=True,
            )
    if install_wandb:
        # keep wandb outside the stamped rebuild branch. wandb is absent from the stamp, so reused venvs
        # and transient failures must retry it on later runs; installing an already satisfied package
        # is a cheap no-op.
        subprocess.run(["uv", "pip", "install", "--python", py, "wandb"], check=False)
    return py


def resolve_verl_loggers(caps: dict) -> list[str]:
    """verl's ``trainer.logger`` list, gated on the interpreter that actually logs.

    verl logs from its own interpreter, so wandb must be importable THERE, not in flash's env. The
    prebuilt worker image's /opt/verl-venv does not install wandb, and a preset FLASH_VERL_PYTHON is
    returned by resolve_verl_python untouched (flash does not own that interpreter), so a run with
    WANDB_API_KEY set would otherwise ask verl for a logger it cannot import and die at logger init
    on a paid GPU.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return ["console"]
    if not caps.get("wandb"):
        print(
            "[verl] WANDB_API_KEY set but wandb is unavailable in the verl interpreter; "
            "using console logger only"
        )
        return ["console"]
    return ["console", "wandb"]


def verl_device_capability(caps: dict) -> tuple[int, int] | None:
    """device 0's ``(major, minor)`` cuda capability as the VERL interpreter sees it, or None.

    probed in verl's interpreter rather than flash's because verl owns the rollout engine and pins
    its own torch/vllm stack, and both callers below are deciding what that stack can do. None means
    the probe could not answer -- no cuda, no torch, a hung import -- and every caller must read it
    as "leave the default alone" rather than guessing a workaround onto an unknown card.
    """
    cc = caps.get("capability")
    if not cc:
        return None
    try:
        return (int(cc[0]), int(cc[1]))
    except (TypeError, ValueError, IndexError):
        return None


def resolve_rollout_enforce_eager(cc: tuple[int, int] | None) -> bool:
    """whether this GPU must run the rollout eagerly instead of capturing cuda graphs.

    One knob is enough and cannot fight verl's: vllm 0.19.1 resolves ``enforce_eager`` LAST, forcing
    ``compilation_config.mode=NONE`` and ``cudagraph_mode=NONE`` regardless of what was requested
    (``config/vllm.py:847-853`` and ``:1024-1029``). An unanswerable probe (``cc is None``) leaves
    verl's default in place rather than guessing eager onto an unknown card.
    """
    if cc is None:
        return False
    # sm86 is the only measured graph-capture failure. everything else -- sm80/sm89/sm90 and
    # blackwell -- captures. an arch nobody has measured keeps verl's default rather than being
    # forced eager on the strength of one other Ampere minor version's defect.
    if cc == (8, 6):
        print(
            "[verl] sm86: enforce_eager=True for the rollout (vllm 0.19.1 graph capture degenerates "
            "on this arch: completions repeat to the token cap without emitting EOS)"
        )
        return True
    return False


def rollout_sleep_unsupported(model_id: str) -> bool:
    """whether vLLM's sleep/wake cycle is non-functional for this model.

    a few models HANG on wake/reload rather than erroring (live-confirmed, every attempt), so the
    catalog flags them and the rollout engine must stay RESIDENT instead. shared by grpo and opd:
    verl runs the same ``vllm_async_server`` sleep path for both, so the flag has to mean the same
    thing on both drivers or the one that ignores it wedges on the flagged model.
    """
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    return bool(info is not None and getattr(info, "sleep_unsupported", False))


def rollout_resident_overrides(sleep_unsupported: bool) -> list[str]:
    """verl overrides that pin the rollout engine resident, or ``[]`` to keep verl's default.

    disable both ``free_cache_engine`` and ``enable_sleep_mode`` or flagged models hang on wake
    (``vllm_async_server.py:626,648`` gates sleep on both). VRAM sizing already uses the resident
    peak. prefixes differ: the yaml key takes a bare override, while ``enable_sleep_mode`` is
    dataclass-only (``workers/config/rollout.py:277``) and so requires ``+``.
    """
    if not sleep_unsupported:
        return []
    return [
        "actor_rollout_ref.rollout.free_cache_engine=false",
        "+actor_rollout_ref.rollout.enable_sleep_mode=false",
    ]


def trainer_dtype_overrides() -> list[str]:
    """verl overrides that store the frozen base in bf16 instead of verl's fp32 default.

    fsdp.yaml passes its fp32 default to ``from_pretrained`` (``transformer_impl.py:234``). flash
    freezes the base and computes under bf16 mixed precision, while PEFT keeps trainable adapter
    weights fp32 (``peft/mapping_func.py:35``, not overridden by verl), so fp32 base residency has
    no master-weight benefit. this is specific to LoRA, not full tuning.

    set only ``actor`` because ``ref_in_actor`` aliases the reference worker to it
    (``ray_trainer.py:897``), rather than creating a second resident copy.
    """
    return ["actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"]


def resolve_blackwell_attention_backends(
    caps: dict, cc: tuple[int, int] | None
) -> tuple[str | None, str | None]:
    """the rollout ``(attention_backend, mm_encoder_attn_backend)`` this GPU needs, or ``(None, None)``.

    on Blackwell, use FLASHINFER or TRITON_ATTN for decoder attention; TORCH_SDPA is not registered
    there in vllm 0.19.1 (``vllm/platforms/cuda.py:99-111``). pin ViT attention to TORCH_SDPA
    because no published cutlass version satisfies both CUTE APIs; ``cuda.py:344-373`` picks the
    default ViT backend and honors an explicit override. non-Blackwell or unknown caps keep defaults.
    """
    if cc is None:
        return (None, None)
    major = cc[0]
    if major not in (10, 12):
        return (None, None)
    has_flashinfer = bool(caps.get("flashinfer"))
    decoder = "FLASHINFER" if has_flashinfer else "TRITON_ATTN"
    if not has_flashinfer:
        print(
            f"[verl] sm{major}0 (Blackwell): flashinfer is not importable in the verl interpreter "
            "-> attention_backend=TRITON_ATTN (PTX-independent registered decoder backend)"
        )
    else:
        print(
            f"[verl] sm{major}0 (Blackwell): attention_backend=FLASHINFER "
            "(flash-attn PTX is unreliable on consumer Blackwell -> empty-rollout failures)"
        )
    print(
        f"[verl] sm{major}0 (Blackwell): mm_encoder_attn_backend=TORCH_SDPA "
        "(vllm 0.19.1 ViT CUTE flash-attn is unimportable vs every nvidia-cutlass-dsl)"
    )
    return (decoder, "TORCH_SDPA")


def resolve_checkpoint_actor_dir(step_dir: str) -> str:
    """return the directory inside ``global_step_N`` that holds the saved model + ``huggingface/``.

    verl's two trainers do not agree on this layout, and the difference is not configurable:

    * RL nests per-role, ``global_step_N/actor/`` (``trainer/ppo/ray_trainer.py`` builds
      ``os.path.join(local_global_step_folder, "actor")``).
    * SFT writes the shards straight into ``global_step_N/``
      (``utils/checkpoint/checkpoint_handler.py`` passes ``local_path`` unmodified).
    """
    nested = os.path.join(step_dir, "actor")
    if os.path.isdir(os.path.join(nested, "huggingface")):
        return nested
    if os.path.isdir(os.path.join(step_dir, "huggingface")):
        return step_dir
    # neither marker is present (an interrupted save, or a layout this code has not seen). prefer the
    # nested dir when it exists at all so the error names the RL path the caller most likely wanted.
    return nested if os.path.isdir(nested) else step_dir


def latest_global_step_dir(local_dir: str) -> tuple[str, int]:
    """return (actor_dir, step) for the highest global_step_N checkpoint verl wrote."""
    best_step, best = -1, ""
    if os.path.isdir(local_dir):
        for name in os.listdir(local_dir):
            m = re.fullmatch(r"global_step_(\d+)", name)
            if m and int(m.group(1)) > best_step:
                best_step = int(m.group(1))
                best = resolve_checkpoint_actor_dir(os.path.join(local_dir, name))
    if best_step < 0:
        raise RuntimeError(f"no global_step_N checkpoint found under {local_dir}")
    return best, best_step


def completed_checkpoint_step(local_dir: str) -> int:
    """read verl's completion marker; 0 when it is absent or unreadable.

    verl writes ``global_step_N`` in full and only THEN advances
    ``latest_checkpointed_iteration.txt`` (the file ``stage_verl_resume`` writes above), so gating a
    publish on this marker never reads a half-written checkpoint dir. an unreadable marker reads as
    "nothing completed yet" rather than raising: the watchers poll this on a loop, and a torn read
    mid-write must retry on the next tick, not fail the run.
    """
    tracker = os.path.join(local_dir, "latest_checkpointed_iteration.txt")
    try:
        with open(tracker) as file:
            return int(file.read().strip())
    except (FileNotFoundError, OSError, ValueError):
        return 0


def unprocessed_checkpoint_dirs(
    local_dir: str, completed_step: int, processed_steps: set[int]
) -> list[tuple[int, str]]:
    """``(step, dir)`` for every completed ``global_step_N`` not yet in ``processed_steps``, ascending.

    Bounded by ``completed_step`` so a directory verl is still writing is never handed to a
    publisher, and by ``processed_steps`` so each checkpoint is published exactly once.
    """
    found: list[tuple[int, str]] = []
    try:
        names = os.listdir(local_dir)
    except OSError:
        return found
    for name in names:
        match = re.fullmatch(r"global_step_(\d+)", name)
        if match is None:
            continue
        step = int(match.group(1))
        path = os.path.join(local_dir, name)
        if step <= completed_step and step not in processed_steps and os.path.isdir(path):
            found.append((step, path))
    return sorted(found)


def stage_verl_resume(resume_dir: str, local_dir: str, *, job_label: str) -> int:
    """stage a downloaded ``checkpoint-N`` into local_dir where verl looks; return its step.

    the resume artifact is keyed on the run prefix, not the job type, so the control plane hands
    every trainer the same ``checkpoint-N`` layout. verl finds it via
    latest_checkpointed_iteration.txt under trainer.default_local_dir once resume_mode=auto.
    ``job_label`` only names the job in the error raised for an unparseable path, which is the sole
    thing the trainers ever varied here.
    """
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume_dir))
    if match is None:
        raise RuntimeError(f"invalid {job_label} resume checkpoint path {resume_dir!r}")
    step = int(match.group(1))
    shutil.copytree(resume_dir, os.path.join(local_dir, f"global_step_{step}"), dirs_exist_ok=True)
    with open(os.path.join(local_dir, "latest_checkpointed_iteration.txt"), "w") as file:
        file.write(str(step))
    return step


def export_peft_adapter(
    ckpt_actor_dir: str,
    out_adapter_dir: str,
    *,
    base_model_id: str,
    python_bin: str,
) -> None:
    """turn verl's saved lora checkpoint into a flash-servable peft adapter dir.

    verl saves fsdp-sharded checkpoints (model/optim shards + a ``huggingface/`` config+tokenizer
    subfolder) under ``<local_dir>/global_step_N/actor`` for RL and ``<local_dir>/global_step_N``
    for SFT -- pass the dir that ``resolve_checkpoint_actor_dir`` picked, not a hardcoded one.
    """
    os.makedirs(out_adapter_dir, exist_ok=True)
    merge_out = out_adapter_dir.rstrip("/") + "_merge"
    shutil.rmtree(merge_out, ignore_errors=True)
    merge_env = dict(os.environ)
    # the merger reads only local checkpoint files: strip credentials/tokens from its env (least
    # privilege), and keep it fully offline so it never touches hf's rate-limited api.
    for _k in list(merge_env):
        if any(t in _k.upper() for t in ("TOKEN", "SECRET", "API_KEY", "PASSWORD", "CREDENTIAL")):
            merge_env.pop(_k)
    merge_env["HF_HUB_OFFLINE"] = "1"
    merge_env["TRANSFORMERS_OFFLINE"] = "1"
    merge_env["HF_HUB_DISABLE_XET"] = "1"
    subprocess.run(
        [
            python_bin,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            ckpt_actor_dir,
            "--target_dir",
            merge_out,
        ],
        check=True,
        env=merge_env,
    )
    lora_dir = os.path.join(merge_out, "lora_adapter")
    if not os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        raise RuntimeError(
            f"verl model_merger did not produce a peft adapter at {lora_dir} (no adapter_config.json); "
            "the merger output layout must be adjusted for this verl version."
        )
    for name in os.listdir(lora_dir):
        shutil.copy2(os.path.join(lora_dir, name), os.path.join(out_adapter_dir, name))
    shutil.rmtree(merge_out, ignore_errors=True)


def stamp_adapter_dir_provenance(adapter_dir: str, model_id: str, model_revision: str = "") -> None:
    """stamp the saved adapter's immutable base identity into adapter_config.json.

    dir-based analogue of the in-memory peft-model provenance stamp: same validation + fields,
    applied to the json verl produced. raises if the adapter already names a different base.
    """
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    current_base = str(cfg.get("base_model_name_or_path", "") or "").strip()
    if current_base and current_base != model_id:
        raise RuntimeError(
            f"adapter base model {current_base!r} does not match validated target {model_id!r}"
        )
    current_rev = str(cfg.get("revision", "") or "").strip()
    if current_rev and model_revision and current_rev != model_revision:
        raise RuntimeError("adapter base revision does not match the validated target commit")
    cfg["base_model_name_or_path"] = model_id
    cfg["revision"] = model_revision or None
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


# how many of the child's most recent output lines to retain for stall reporting. the child's last
# words before it wedges are the whole diagnostic, and a stall is usually preceded by a short burst
# (a ray warning, a placement-group notice, a partial traceback), so a small window suffices.
CHILD_TAIL_LINES = 60
# per-line cap when rendering the retained tail. verl prints resolved-config blocks thousands of
# characters wide; an unbounded tail would blow the heartbeat payload it has to travel inside.
_CHILD_TAIL_LINE_CHARS = 300
# how many retained lines ride along on a pre-first-step heartbeat. narrower than what is retained:
# this payload is uploaded every tick, so it stays small enough not to bloat the snapshot.
STALL_TAIL_LINES = 15
_RETRIABLE_VERL_CHILD_SIGNATURES = (
    "cudaErrorDevicesUnavailable",
    "CUDA-capable device(s) is/are busy or unavailable",
)


class ChildOutputTail:
    """bounded ring buffer of a subprocess's most recent output lines.

    child stdout is absent from collected logs; only heartbeat markers survive. retain the tail so
    setup stalls can report the child's last words (ISSUES VERL-061).
    """

    def __init__(self, limit: int = CHILD_TAIL_LINES) -> None:
        self._lines: collections.deque[str] = collections.deque(maxlen=limit)
        self._written = 0
        self._retriable_infra_signature: str | None = None
        self._cuda_oom_evidence: str | None = None

    def record(self, line: str) -> None:
        if self._retriable_infra_signature is None:
            self._retriable_infra_signature = next(
                (signature for signature in _RETRIABLE_VERL_CHILD_SIGNATURES if signature in line),
                None,
            )
        if self._cuda_oom_evidence is None:
            from flash.engine.worker.perf.lifecycle import cuda_oom_message_evidence

            self._cuda_oom_evidence = cuda_oom_message_evidence(line)
        text = line.rstrip("\n")
        if text:
            self._lines.append(text[:_CHILD_TAIL_LINE_CHARS])
            self._written += 1

    @property
    def retriable_infra_signature(self) -> str | None:
        """the first stable retriable-infrastructure signature observed in child output."""
        return self._retriable_infra_signature

    @property
    def cuda_oom_evidence(self) -> str | None:
        """the first authoritative cuda oom message evidence observed in child output."""
        return self._cuda_oom_evidence

    @property
    def written(self) -> int:
        """how many non-empty lines the child has produced, ever.

        monotonic and independent of the retention limit, which is what makes it usable as a
        staleness signal: a child looping on the same line still advances this, and a child that has
        gone silent cannot advance it even though its retained tail stays fully populated.
        """
        return self._written

    def tail(self, limit: int | None = None) -> list[str]:
        """the retained lines, oldest first, optionally narrowed to the most recent ``limit``."""
        lines = list(self._lines)
        if limit is not None and limit >= 0:
            lines = lines[len(lines) - limit :] if limit < len(lines) else lines
        return lines


def raise_for_classified_verl_exit(return_code: int, tail: ChildOutputTail) -> None:
    """raise a classified failure when a nonzero verl child reported authoritative evidence."""
    if return_code == 0:
        return
    oom_evidence = tail.cuda_oom_evidence
    if oom_evidence is not None:
        raise RuntimeError(
            f"verl subprocess exited with status {return_code} after reporting {oom_evidence}"
        )
    signature = tail.retriable_infra_signature
    if signature is None:
        return
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    raise RetriableInfraError(
        f"verl subprocess exited with status {return_code} after reporting {signature}"
    )


class ChildTailStaleness:
    """tracks how long a child has been silent, across the ticks that sample its tail.

    the tail alone cannot answer the question a stall actually poses. a child still loading shards
    and a child wedged forever both present a fully populated tail whose newest line is plausible,
    so the only thing separating them is whether the tail CHANGED between two dumps -- and a
    stateless report throws that comparison away, leaving it to be reconstructed by hand from
    consecutive heartbeats after the money is already spent (ISSUES VERL-067). holding the previous
    line count here turns that into a number the first dump already carries.
    """

    def __init__(self) -> None:
        self._written = -1
        self._since = 0

    def observe(self, written: int) -> int:
        """record this tick's line count; return consecutive ticks with no new output.

        0 means the child spoke since the last observation. n>0 means it has been silent for n
        ticks, which is the signal that separates a slow start from a wedge.
        """
        if written != self._written:
            self._written = written
            self._since = 0
        else:
            self._since += 1
        return self._since


def stall_tail_fields(
    step: int,
    tail: ChildOutputTail,
    limit: int = STALL_TAIL_LINES,
    staleness: ChildTailStaleness | None = None,
) -> dict[str, object]:
    """heartbeat fields carrying the child's last words, but only while it has made no progress.

    before the first step, the tail is the only collected setup-stall evidence. with ``staleness``,
    include silent ticks to distinguish a slow start from a wedge. return empty after progress or
    before any child output.
    """
    if step > 0:
        return {}
    recent = tail.tail(limit=limit)
    if not recent:
        # observed even with nothing to report, so a child that starts talking later is measured
        # from its first line rather than from whenever the payload happened to become non-empty.
        if staleness is not None:
            staleness.observe(tail.written)
        return {}
    fields: dict[str, object] = {"child_tail": recent}
    if staleness is not None:
        fields["child_tail_silent_ticks"] = staleness.observe(tail.written)
    return fields


# the ray logs worth keeping when a raylet dies. the driver's own stdout only ever shows the
# downstream symptom ("Failed to register worker to Raylet: ... End of file"); the reason the raylet
# went away is in these. deliberately a small allowlist -- a ray session dir also holds per-worker
# logs that can run to hundreds of files on a 128-core box.
RAY_FAILURE_LOGS = (
    "raylet.out",
    "raylet.err",
    "gcs_server.out",
    "gcs_server.err",
    "dashboard_agent.log",
    "dashboard.log",
)
# per file. enough to carry a stack and the lines before it, without turning an artifact upload into
# the reason a failing run takes even longer to report. this is the ONLY bound on the result: the
# sanitize pass below is given the same number so it can never truncate a tail we chose to keep.
RAY_LOG_TAIL_BYTES = 64 * 1024


def latest_ray_session_dir(
    root: str = "/tmp/ray", *, started_after: float | None = None
) -> str | None:
    """the most recent ray session directory, or None if THIS run never started one.

    ``started_after`` rejects sessions older than the caller's start. a retry reuses the pod workdir
    and /tmp survives it, so a run that fails BEFORE ray starts -- during dependency provisioning or
    model download -- still finds a previous run's session here. uploading that as the current
    attempt's evidence is worse than uploading nothing: it reads as a raylet failure that never
    happened, and sends the next diagnosis after a cause belonging to a different run.
    """
    try:
        names = os.listdir(root)
    except OSError:
        return None
    # stat once, here: a session directory can vanish between listing and stat, and doing it in a
    # `max(key=...)` would let that raise on an already-failing path.
    dated: list[tuple[float, str]] = []
    for name in names:
        if not name.startswith("session_"):
            continue
        path = os.path.join(root, name)
        try:
            if not os.path.isdir(path):
                continue
            # mtime, not the name's timestamp: the directory keeps being written while ray runs, so
            # a session that STARTED before this run but was still live during it is still ours.
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if started_after is not None and mtime < started_after:
            continue
        dated.append((mtime, path))
    if not dated:
        return None
    return max(dated)[1]


def collect_ray_failure_logs(
    *,
    root: str = "/tmp/ray",
    tail_bytes: int = RAY_LOG_TAIL_BYTES,
    started_after: float | None = None,
) -> str:
    """ray's own logs about why a raylet died, as one credential-safe artifact body ("" if none).

    when a raylet dies the driver prints only its own downstream failure, and ray's session dir --
    which holds the actual cause -- lives on the pod and goes away with it. that makes a raylet
    failure undiagnosable from uploaded artifacts and costs a paid gpu run per guess (VERL-115). one
    string rather than a directory of copies: the caller writes it exactly like the traceback
    artifact beside it, so a dying pod does one upload instead of six against the same bounded hf
    deadline allowance, and there is no staging directory whose only purpose is to be uploaded.
    """
    session = latest_ray_session_dir(root, started_after=started_after)
    if session is None:
        return ""
    logs_dir = os.path.join(session, "logs")
    sections: list[str] = []
    for name in RAY_FAILURE_LOGS:
        src = os.path.join(logs_dir, name)
        try:
            with open(src, "rb") as handle:
                # seek relative to the file's OWN end and cap the read: ray may still be writing
                # while this runs, and a getsize()-then-read() would consume from the old offset
                # through the new EOF -- unbounded, on a dying pod with a bounded upload deadline.
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                # the tail, not the head: a crash reason is at the end of the file.
                handle.seek(max(0, size - tail_bytes))
                payload = handle.read(tail_bytes)
        except OSError:
            continue
        # seeking to a byte offset can land mid-codepoint, and a decode error here would lose the
        # whole file for a cosmetic reason on the one path that exists to preserve evidence.
        text = payload.decode("utf-8", errors="replace")
        if size > tail_bytes:
            # never begin mid-line: a tail cut can split a credential and defeat prefix or full-value
            # redaction. drop the partial line, then sanitize multiline values line by line; both are
            # required for third-party ray logs.
            newline = text.find("\n")
            text = text[newline + 1 :] if newline != -1 else ""
            if not text:
                # a single line longer than the whole tail. dropping it is the only safe option, but
                # say so: an empty section would otherwise read as "ray logged nothing here".
                text = f"<omitted: final {tail_bytes} bytes are one unterminated line>"
        sections.append(
            f"===== {name} (last {tail_bytes} bytes) =====\n"
            f"{sanitize_diagnostic(text, limit=tail_bytes)}"
        )
    return "\n\n".join(sections)


def run_verl_training(
    cmd: list[str],
    *,
    env: dict[str, str],
    on_step: Callable[[int], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    heartbeat: Callable[[], None] | None = None,
    step_pattern: str = r"step:\s*(\d+)",
    heartbeat_interval_s: float = 20.0,
    tail: ChildOutputTail | None = None,
) -> int:
    """run a verl trainer subprocess, streaming stdout and surfacing step progress.

    returns the process exit code. stdout+stderr are merged and scanned line by line: ``on_line``
    receives every line, ``on_step`` receives each parsed training step, and ``heartbeat`` is called
    at most once per ``heartbeat_interval_s``. callback failures terminate the child before they are
    re-raised so a failed required checkpoint upload cannot leave paid training running unattended.
    """
    step_re = re.compile(step_pattern)
    child_tail = tail if tail is not None else ChildOutputTail()
    # before the child exists, so any grandchild it orphans reparents here and can actually be
    # reaped. this process is not pid 1 -- the runpod handler is -- so without it every wait below
    # answers ChildProcessError for a zombie nobody will collect.
    adopt_orphaned_descendants()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    # start_new_session makes the leader pid the stable group id even after that leader is reaped.
    process_group_id = proc.pid
    last_hb = 0.0
    try:
        # the child's exit is watched independently of pipe EOF. a grandchild holding the inherited
        # pipe open after the trainer dies would otherwise keep the loop below running forever, on a
        # paid gpu, having already lost the process whose output it is waiting for.
        with _ChildExitWatchdog(
            proc, process_group_id=process_group_id, grace_s=_ORPHANED_PIPE_GRACE_S
        ) as watchdog:
            assert proc.stdout is not None
            for line in proc.stdout:
                # held across the callbacks, not just the arrival: on_step uploads a checkpoint, and
                # a reader inside one is working, not stuck.
                with watchdog.handling_line():
                    print(line, end="", flush=True)
                    child_tail.record(line)
                    if on_line is not None:
                        on_line(line)
                    m = step_re.search(line)
                    if m and on_step is not None:
                        on_step(int(m.group(1)))
                    if heartbeat is not None:
                        now = time.monotonic()
                        if now - last_hb >= heartbeat_interval_s:
                            heartbeat()
                            last_hb = now
    except BaseException:
        kill_process_group(proc, process_group_id=process_group_id)
        raise
    finally:
        if proc.poll() is None:
            # BOUNDED. EOF is not the child's exit, so a child that outlives its own stdout would
            # park the attempt here. one that has not exited within the grace after its stdout closed
            # is not going to, so tear the group down and let the wait inside that call collect it.
            try:
                proc.wait(timeout=_TEARDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                kill_process_group(proc, process_group_id=process_group_id)
        # every job boundary, not just the failing ones. a straggler an earlier teardown SIGKILLed
        # but could not drain in time exits shortly after, and `kill_process_group` -- the only
        # other caller of this -- runs on exceptions alone: a worker whose later jobs all succeed
        # would hold that zombie for its whole life, as pid 1 with nothing else to reap it.
        reap_stragglers()
    collected = proc.returncode
    if collected is None:
        # the bounded wait above can end with nothing collected, which the unbounded one it replaced
        # could not: a group member wedged in uninterruptible io outlives even the SIGKILL inside
        # teardown. `int(None)` would replace that diagnosis with a TypeError, so name the survivor
        # instead -- callers already handle a RuntimeError from here.
        raise RuntimeError(
            f"verl subprocess {proc.pid} did not exit after teardown; its process group is still "
            "holding the gpu"
        )
    return_code = int(collected)
    if watchdog.tore_down and return_code == 0:
        # the child exited 0 but a descendant held the pipe open past the grace, so the group was
        # killed to release it. reporting success here would upload whatever partial artifacts exist
        # as a completed run: the trainer's own exit status says nothing about the descendant that
        # was still running, and that descendant is why the pipe never closed.
        raise RuntimeError(
            f"verl subprocess {proc.pid} exited 0 but a descendant held its output pipe open for "
            f"{_ORPHANED_PIPE_GRACE_S:.0f}s; the process group was torn down to release the gpu"
        )
    try:
        raise_for_classified_verl_exit(return_code, child_tail)
    except BaseException:
        kill_process_group(proc, process_group_id=process_group_id)
        raise
    if return_code != 0:
        # a nonzero exit that carried no recognized signature RETURNS from the classifier rather
        # than raising, so this is the one failing path that reached neither teardown above. the
        # direct child is gone but its group need not be: a surviving EngineCore keeps its cuda
        # context, and a reusable worker then hands the next attempt an occupied gpu.
        kill_process_group(proc, process_group_id=process_group_id)
    return return_code


_TEARDOWN_GRACE_S = 10.0

# grace for a descendant holding stdout open after the trainer exits. observe child exit separately
# from EOF because EngineCore can retain the pipe; allow final flushing before group teardown.
_ORPHANED_PIPE_GRACE_S = 30.0


class _ChildExitWatchdog:
    """Tears the group down when the direct child exits but a descendant holds the pipe open.

    this prevents an EngineCore-held pipe from blocking teardown forever (PR #730). it arms only after
    child exit, so ordinary trainer silence remains ``ChildTailStaleness``'s responsibility.
    """

    def __init__(self, proc: subprocess.Popen, *, process_group_id: int, grace_s: float) -> None:
        self._proc = proc
        self._process_group_id = process_group_id
        self._grace_s = grace_s
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        # bumped by the reader for every line it takes off the pipe. the exit of the child alone is
        # NOT sufficient evidence of a leak: a child can exit having left a full pipe behind, and a
        # reader working through that backlog -- an on_step callback uploading a checkpoint takes
        # minutes -- would otherwise be killed mid-upload and its successful run reported as failed.
        # a stuck reader cannot advance this; a busy one does, which is exactly the distinction.
        self._lines_read = 0
        # how many lines are being HANDLED right now, not merely taken off the pipe. counting only
        # arrivals makes one long callback look identical to a stuck reader -- the counter cannot
        # advance while an upload runs, because the next line is not read until it returns. so
        # progress is "a line arrived OR one is still in hand", and the two together mean the
        # watchdog only ever fires on a reader that is neither receiving nor working.
        self._lines_in_flight = 0
        # read by the caller after the loop ends, to distinguish "the child closed its own pipe" from
        # "we closed it by killing the group out from under a survivor".
        self.tore_down = False

    @contextlib.contextmanager
    def handling_line(self):
        """Wraps the reader's whole per-line body, not just the moment the line arrives.

        Entered once per line and held until that line's callbacks return, so an `on_step` upload
        that outlasts the grace still reads as progress. Plain int stores, so no lock is needed: the
        watchdog only ever compares them, and either order of the two writes below leaves the reader
        looking busy rather than idle.
        """
        self._lines_read += 1
        self._lines_in_flight += 1
        try:
            yield
        finally:
            self._lines_in_flight -= 1

    def __enter__(self) -> Self:
        self._thread = threading.Thread(
            target=self._watch, name="verl-child-exit-watchdog", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._done.set()
        if self._thread is not None:
            # bounded: the thread only ever sleeps on `_done`, so this is a handoff, not a wait on
            # the child. the thread is a daemon regardless, so it can never hold the worker open.
            self._thread.join(timeout=_TEARDOWN_GRACE_S)

    def _watch(self) -> None:
        # `poll` rather than `wait`: this thread must stay responsive to `_done` instead of blocking
        # on the child. both collect the status, and CPython guards that with `_waitpid_lock`, so
        # whichever of the two threads gets there first is the one that sets `returncode`.
        while not self._done.wait(0.5):
            if self._proc.poll() is None:
                continue
            # the child is gone. that is necessary but not sufficient: require the reader to also be
            # making no progress across the grace, so a backlog being worked through is never killed.
            before = self._lines_read
            if self._done.wait(self._grace_s):
                return
            if self._lines_read != before or self._lines_in_flight:
                # the reader is still draining real output, or is inside a callback for a line it
                # already took. either way it is working, so keep watching rather than tearing down.
                continue
            self.tore_down = True
            kill_process_group(self._proc, process_group_id=self._process_group_id)
            return


def _process_is_zombie(pid: int) -> bool:
    """true when `pid` has exited and is only awaiting a reaper.

    parse state after the last ``)`` because ``/proc/<pid>/stat`` comm may contain parentheses. only
    disappearance proves exit; unreadable status counts as alive so EMFILE cannot suppress SIGKILL and
    strand an EngineCore CUDA context.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
            stat = handle.read()
    except (FileNotFoundError, ProcessLookupError):
        return True  # gone, so there is nothing left to wait for
    except OSError:
        return False  # unreadable, which is not evidence of an exit
    _, _, tail = stat.rpartition(")")
    fields = tail.split()
    return bool(fields) and fields[0] == "Z"


def _process_group_addressable(pgid: int) -> bool:
    """true while the kernel still knows `pgid`, whether or not its members are running."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, this process just may not signal it
    except OSError:
        return False
    return True


# pids this process adopted and killed but could not reap before a teardown deadline expired. each
# is confirmed ours -- `waitpid` answered for it rather than raising -- and was still running at
# that point, so it still owes a status. a zombie holds its pid until it is reaped, so nothing
# recorded here can be recycled behind our back and a later sweep can only ever find the same
# process it recorded.
_UNREAPED_STRAGGLERS: set[int] = set()

# PR_SET_CHILD_SUBREAPER, from linux/prctl.h.
_PR_SET_CHILD_SUBREAPER = 36


def adopt_orphaned_descendants() -> bool:
    """become the reaper for descendants orphaned below this process. true if the kernel agreed.

    `waitpid` then raises `ChildProcessError` here and the zombie is recorded as handled while it
    keeps its pid for the worker's whole life -- and the handler only ever waits on the worker
    itself, so nothing else ever collects it either. marking this process a subreaper makes the
    kernel reparent such orphans HERE instead, which is the condition the rest of this module
    already assumes. set once at teardown-path entry rather than at import, so merely importing this
    module cannot change an unrelated process's semantics.
    """
    global _ADOPTS_ORPHANS
    if _ADOPTS_ORPHANS:
        return True
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            return False
    except (OSError, AttributeError, ValueError):
        # no libc, or a kernel without the option (linux < 3.4). the reaps below degrade to what
        # they did before rather than failing the run: a leaked zombie costs a pid, not the job.
        return False
    _ADOPTS_ORPHANS = True
    return True


# whether this process has already claimed orphaned descendants; the prctl is idempotent but the
# call is not free, and teardown runs once per job on a reused worker.
_ADOPTS_ORPHANS = False


def _reap(pid: int) -> bool:
    """wait on `pid` without blocking. true once nothing is owed, false while it still owes a status.

    `ChildProcessError` means the pid is not ours to wait on, so there is no status left for this
    process to take. that answer is only safe because `adopt_orphaned_descendants` runs first: it
    makes an orphaned grandchild reparent to US, so `ChildProcessError` really does mean someone
    else owns the pid. Without it the same error is returned for a zombie nobody will ever reap.
    """
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True  # not ours, so no status is owed to this process
    except (PermissionError, OSError):
        # cannot wait on it, and retrying later would not change that. dropped rather than tracked
        # so the straggler set cannot grow without bound on a path that can never clear it.
        return True
    return reaped != 0


def _reap_group_zombies(pgid: int, skip: int) -> None:
    """wait on any group member this process has adopted, clearing its process-table entry.

    SIGKILL cannot be refused but it also cannot be delivered while a process sits in
    uninterruptible sleep, so one can outlast the drain deadline and only then become a zombie --
    after the last wait this teardown performs. without a record no future wait is ever scheduled
    for it and the entry is permanent on a pid-1 worker.
    """
    for pid in _process_group_members(pgid) or ():
        if pid == skip:
            continue
        if _reap(pid):
            _UNREAPED_STRAGGLERS.discard(pid)
        else:
            _UNREAPED_STRAGGLERS.add(pid)


def reap_stragglers() -> None:
    """take the statuses still owed by processes an earlier teardown could not drain.

    this is the future wait that the final in-loop reap cannot schedule for itself: a member still
    running when its own teardown gave up is cleared by the next one instead.
    """
    for pid in tuple(_UNREAPED_STRAGGLERS):
        if _reap(pid):
            _UNREAPED_STRAGGLERS.discard(pid)


# how long the last drain waits for a straggler that was still running when its teardown gave up.
# it has already been SIGKILLed, so this is the delivery and exit latency of a process leaving
# uninterruptible sleep, not a grace period.
_EXIT_DRAIN_S = 5.0


def _drain_stragglers_before_exit() -> None:
    """block briefly for the statuses this process still owes, because no later teardown will.

    runpod starts a fresh worker subprocess per phase (endpoints.py `_train_body.run_mode`), so remembered stragglers
    otherwise reparent to a handler that never waits for them. perform one final bounded, non-fatal
    wait at interpreter exit.
    """
    deadline = time.monotonic() + _EXIT_DRAIN_S
    while _UNREAPED_STRAGGLERS:
        reap_stragglers()
        if not _UNREAPED_STRAGGLERS or time.monotonic() >= deadline:
            return
        time.sleep(0.05)


atexit.register(_drain_stragglers_before_exit)


def _process_group_alive(pgid: int) -> bool:
    """true while the group still has a RUNNING member.

    A drained verdict is only returned when TWO consecutive walks agree on it, because a single walk
    cannot distinguish a drained group from one it read too early. /proc can be listed while the
    leader is still present, and the leader can then fork and exit before its status is inspected:
    the snapshot is nonempty and zombie-only, yet the child that inherited the group is alive,
    unlisted, and never received the earlier signal -- so teardown returns without SIGKILL and it
    keeps its cuda context.
    """
    if not _process_group_addressable(pgid):
        return False
    for scan in range(2):
        members = _process_group_members(pgid)
        if members is None:
            # /proc could not be enumerated, so fall back to what signal 0 already established. it
            # over-reports rather than returning early on a live process still holding the gpu.
            return True
        if any(not _process_is_zombie(pid) for pid in members):
            return True
        if scan == 0:
            # this walk says drained. it is only believed if a second one, taken after any fork it
            # could have raced, still says so.
            continue
        if not members:
            # the kernel says this group exists and two walks found nobody in it. that contradiction
            # outlives the fork window, so it is the /proc walk that is wrong rather than late: fall
            # back to what signal 0 established rather than reporting a drain the kernel denies.
            return _process_group_addressable(pgid)
    return False


def _process_group_members(pgid: int) -> list[int] | None:
    """The pids currently in `pgid`, or None when /proc cannot answer.

    An empty list is a real answer -- the group has no members left -- and is distinct from None.
    """
    try:
        entries = os.listdir("/proc")
    except OSError:  # pragma: no cover - /proc missing is not reachable on linux
        return None
    members = []
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) == pgid:
                members.append(pid)
        except (ProcessLookupError, PermissionError, OSError):
            # exited between listdir and here, or not ours to inspect. a process this cannot see is
            # one it also could not have signalled.
            continue
    return members


def kill_process_group(proc: subprocess.Popen, *, process_group_id: int | None = None) -> None:
    """signal the child's whole process group, escalating to SIGKILL if anything survives.

    signalling the group rather than the pid is what reaches vllm's EngineCore grandchild; a
    survivor holds its cuda context and strands the gpu for every later run. the escalation is
    driven off the group, not off the direct child: the usual shape of this failure is the trainer
    dying on the term while the EngineCore ignores it, so waiting only on the child returns before
    the survivor is gone. a captured ``process_group_id`` keeps the group addressable after the
    direct child is reaped.
    """
    # grpo drives its own subprocess and calls this directly, never through `run_verl_training`, so
    # the adoption is claimed here too rather than only at the other entry point. idempotent.
    adopt_orphaned_descendants()
    # collect anything a previous teardown killed but could not drain before its deadline. done on
    # entry rather than on exit because that is when such a process has had the longest to die, and
    # it runs before the early returns below so no path through this function skips it.
    reap_stragglers()
    pgid = process_group_id
    if pgid is None:
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            # already reaped, so there is no group id left to address the survivors by.
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=_TEARDOWN_GRACE_S)
            return

    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGTERM)

    deadline = time.monotonic() + _TEARDOWN_GRACE_S
    # reap the direct child before probing: an unwaited zombie leader is still a group member, so the
    # liveness check below cannot otherwise tell an empty group from one that is merely unreaped.
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_TEARDOWN_GRACE_S)
    while _process_group_alive(pgid) and time.monotonic() < deadline:
        _reap_group_zombies(pgid, skip=proc.pid)
        time.sleep(0.1)
    _reap_group_zombies(pgid, skip=proc.pid)
    if not _process_group_alive(pgid):
        return

    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_TEARDOWN_GRACE_S)
    # sigkill cannot be refused, but delivery and reaping are asynchronous and the caller's next job
    # wants the gpu already free. wait for the group to drain -- bounded, since a zombie awaiting a
    # reaper that is not coming stays addressable and there is nothing stronger left to send.
    drain_deadline = time.monotonic() + _TEARDOWN_GRACE_S
    while _process_group_alive(pgid) and time.monotonic() < drain_deadline:
        _reap_group_zombies(pgid, skip=proc.pid)
        time.sleep(0.1)
    # last pass after the loop: a member killed on the final iteration becomes a zombie only once the
    # kernel has posted its status, which can land after the check that ended the loop. without this
    # the group drains but its process-table entry stays behind for the worker's whole lifetime.
    _reap_group_zombies(pgid, skip=proc.pid)


# --------------------------- tf32 matmul (all three verl backends) ---------------------------
# torch's TF32 flags are per-process, so setting them in the flash parent does not affect the verl
# child. opt in inside each child; bf16 parameters are unaffected.
FLASH_TF32_MARKER = "[flash-verl] tf32 matmul/cuDNN enabled"


def render_tf32_shim() -> str:
    """child-side sitecustomize fragment that enables TF32 matmul/cuDNN in the verl interpreter.

    never raises. tf32 is a throughput optimization, so a torch that cannot take these flags must
    leave training running on the default path rather than abort a paid run.
    """
    return f"""
# --- flash: enable tf32 in the trainer process (see backend_common.render_tf32_shim) ---
try:
    import torch as _flash_tf32_torch

    _flash_tf32_torch.set_float32_matmul_precision("high")
    _flash_tf32_torch.backends.cuda.matmul.allow_tf32 = True
    _flash_tf32_torch.backends.cudnn.allow_tf32 = True
    print({FLASH_TF32_MARKER!r}, flush=True)
except Exception:
    pass
"""


# --------------------------- w&b run link (all three verl backends) ---------------------------
# trl spreads wandb_run_info() into its notes, giving the sdk's link_wandb a clickable
# trl spreads wandb_run_info() into its notes, giving the sdk's link_wandb a clickable
# notes["wandb_url"]. verl calls wandb.init INSIDE the training subprocess, so the flash parent's
# wandb.run is None and that spread would be a silent no-op. the url is not derivable up here
# either: it needs entity/project/runs/<wandb-generated-id>, and flash only knows the project and
# its own run NAME. so the child reports it back over the marker channel the parent already scans.
FLASH_WANDB_LINK_MARKER = "FLASH_WANDB_LINK"

_WANDB_LINK_RE = re.compile(rf"{FLASH_WANDB_LINK_MARKER}\s+(\{{.*\}})\s*$")


def render_wandb_link_shim() -> str:
    """child-side sitecustomize fragment that reports the live w&b run's url and id.

    wraps wandb.init rather than reading wandb.run at import time: sitecustomize runs long before
    verl's tracking module initializes the run, so anything read here would always be None. never
    raises. wandb is optional and a logging link must not be able to abort paid training, so every
    failure path leaves the run unlinked instead of dead.
    """
    return f"""
# --- flash: report the w&b run link to the parent (see backend_common.render_wandb_link_shim) ---
try:
    import json as _flash_wandb_json

    import wandb as _flash_wandb

    _flash_wandb_init = _flash_wandb.init

    def _flash_wandb_init_reporting(*args, **kwargs):
        _run = _flash_wandb_init(*args, **kwargs)
        try:
            _url = getattr(_run, "url", None)
            _id = getattr(_run, "id", None)
            if _url:
                print(
                    "{FLASH_WANDB_LINK_MARKER} "
                    + _flash_wandb_json.dumps({{"wandb_url": _url, "wandb_id": _id}}),
                    flush=True,
                )
        except Exception:
            pass
        return _run

    _flash_wandb.init = _flash_wandb_init_reporting
except Exception:
    pass
"""


FLASH_GDN_VARLEN_MARKER = "[flash-verl] gdn packed-boundary resets active"


def render_gdn_varlen_shim(model_type: str) -> str:
    """child-side sitecustomize fragment that resets GDN state at packed example boundaries.

    patch the checkpoint's exact ``model_type`` because ``qwen3_5`` and ``qwen3_5_moe`` use different
    modules; model names are not authoritative. patch the text-model forward so one derivation reaches
    every GDN layer, and use transformers' own boundary helper to match attention segmentation.

    keep padded input inert, preserve explicit kwargs, and fail closed because silent boundary
    contamination is worse than refusing to start.
    """
    return f'''
# --- flash: reset gdn state at packed example boundaries (backend_common.render_gdn_varlen_shim) ---
import importlib as _flash_gdn_importlib

import torch as _flash_gdn_torch
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence as _flash_gdn_is_packed,
    prepare_fa_kwargs_from_position_ids as _flash_gdn_prepare_fa_kwargs,
)

_flash_gdn_modeling = _flash_gdn_importlib.import_module(
    "transformers.models.{model_type}.modeling_{model_type}"
)
# raises if absent, deliberately: this shim is only rendered once the gate says resets are honored,
# so a missing TextModel means the gate and the model disagree -- refuse rather than train packed
# with an unpatched forward.
_flash_gdn_text_model = next(
    c
    for n, c in vars(_flash_gdn_modeling).items()
    if isinstance(c, type) and n.endswith("TextModel")
)


def _flash_gdn_seq_idx(position_ids, cu_seq_lens):
    """per-token example ordinal, int32 (1, total_nnz) -- what causal_conv1d_fn wants."""
    lengths = cu_seq_lens.diff()
    return (
        _flash_gdn_torch.repeat_interleave(
            _flash_gdn_torch.arange(
                lengths.numel(), device=position_ids.device, dtype=_flash_gdn_torch.int32
            ),
            lengths.to(_flash_gdn_torch.int64),
        )
        .unsqueeze(0)
        .contiguous()
    )


def _flash_patch_gdn_varlen():
    original = _flash_gdn_text_model.forward

    def forward(self, *args, **kwargs):
        # only derive what the caller did not supply, and only for a genuinely packed batch.
        if kwargs.get("cu_seq_lens_q") is None and kwargs.get("seq_idx") is None:
            position_ids = kwargs.get("position_ids")
            # the model reshapes position_ids to 4-way mrope internally; the text row is what the
            # boundaries live on. a 3d (4, batch, seq) tensor indexes to it, 2d is already it.
            text_position_ids = position_ids
            if text_position_ids is not None and text_position_ids.ndim == 3:
                text_position_ids = text_position_ids[0]
            if text_position_ids is not None and text_position_ids.ndim == 2:
                if _flash_gdn_is_packed(text_position_ids, text_position_ids.shape[0]):
                    (cu_seq_lens_q, cu_seq_lens_k), (max_q, max_k) = (
                        _flash_gdn_prepare_fa_kwargs(text_position_ids)
                    )
                    kwargs["cu_seq_lens_q"] = cu_seq_lens_q
                    kwargs["cu_seq_lens_k"] = cu_seq_lens_k
                    kwargs["max_length_q"] = max_q
                    kwargs["max_length_k"] = max_k
                    kwargs["seq_idx"] = _flash_gdn_seq_idx(text_position_ids, cu_seq_lens_q)
        return original(self, *args, **kwargs)

    _flash_gdn_text_model.forward = forward


if not getattr(_flash_gdn_text_model.forward, "_flash_gdn_varlen_patched", False):
    _flash_patch_gdn_varlen()
    _flash_gdn_text_model.forward._flash_gdn_varlen_patched = True
    print({FLASH_GDN_VARLEN_MARKER!r}, "{model_type}", flush=True)
'''


def parse_wandb_link(line: str) -> dict | None:
    """the {{wandb_url, wandb_id}} a marker line carries, or None for every other line.

    returns None rather than raising on a malformed payload: this parses child stdout, and a
    truncated or interleaved line under multi-rank logging must not take down the run.
    """
    match = _WANDB_LINK_RE.search(line)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    url = payload.get("wandb_url")
    # https only: the sdk renders this as a clickable dashboard link, and child stdout also carries
    # rollout text, so the field must not be able to become an arbitrary scheme.
    if not isinstance(url, str) or not url.startswith("https://"):
        return None
    wandb_id = payload.get("wandb_id")
    return {"wandb_url": url, "wandb_id": wandb_id if isinstance(wandb_id, str) else None}


# --------------------------- per-step grpo metrics (verl -> `flash runs log -f`) ---------------------------
# an in-process trainer would feed `metrics_last` from a TrainerCallback, which verl cannot host:
# its trainer runs out of process. verl's LocalLogger prints exactly one line per optimizer update
# -- "step:N - key:value - key:value" over every scalar metric -- so the parent reconstructs the
# same backlog from that line. the payload schema is the CLI's, not verl's: keys below are what
# flash/cli/commands.py:_FOLLOW_METRIC_FIELDS renders.

# the left edge every verl step pattern shares. "not part of a longer word" rather than start-of-
# line, because ray tags worker stdout with a "(TaskRunner pid=123) " prefix -- an anchored match
# would parse nothing at all in production. and not "preceded by whitespace" either (VERL-134):
# verl's LocalLogger shares its stream with tqdm, which ends a bar with "]" and no trailing
# newline, so the metric line arrives glued to it -- "...81.49s/it]step:2 - train/loss:...". a \s
# edge matched step 1 and missed every step after it, which froze the sft heartbeat on step 1's
# metrics and kept the zero-grad guard from ever arming. excluding / as well as word characters
# still refuses "global_step:" and a path's ".../step:".
_VERL_STEP_LEFT_EDGE = r"(?:^|[^\w/])"

# the gate the trainers share, via verl_step_number below: any step-tagged line.
_VERL_STEP_GATE_RE = re.compile(_VERL_STEP_LEFT_EDGE + r"step:(\d+)(?:\s|$)")

# the `metrics_last` row pattern additionally requires the " - " metric separator, so a step-tagged
# line carrying no metrics at all cannot open a row.
_VERL_STEP_RE = re.compile(_VERL_STEP_LEFT_EDGE + r"step:(\d+) - ")

# verl metric key -> flash `metrics_last` field. verl has no counterpart for trl's
# frac_reward_zero_std (an advantage-collapse fraction trl computes per group), so that column is
# simply absent for verl runs rather than faked; the renderer skips fields it does not find.
_VERL_METRIC_FIELDS = (
    ("critic/rewards/mean", "reward"),
    ("actor/grad_norm", "grad_norm"),
    ("actor/kl_loss", "kl"),
    ("actor/entropy", "entropy"),
    ("response_length/mean", "mean_completion_tokens"),
    ("response_length/clip_ratio", "truncation_rate"),
)

# verl reduces most metrics with np.mean and formats them through pprint, so under numpy>=2 a value
# prints as "np.float32(1.25)" rather than "1.25". verl's own requirements pin numpy<2, but the
# production interpreter comes from FLASH_VERL_PYTHON (a prebuilt image flash does not own), so
# accept both spellings -- the numpy-2 form would otherwise drop these columns silently.
_NUMPY_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")


def _metric_value(raw: str) -> float | None:
    """one verl-printed scalar as a finite float, or None when it is not usable."""
    wrapper = _NUMPY_SCALAR_RE.match(raw)
    if wrapper is not None:
        raw = wrapper.group(1)
    try:
        value = float(raw)
    except ValueError:
        return None
    # nan/inf survive float() ("nan", "inf") but would render as a meaningless column and can poison
    # a json payload downstream, so drop them the way the trl callback does.
    return value if math.isfinite(value) else None


def parse_verl_metric(line: str, verl_key: str) -> float | None:
    """one verl metric's finite float value from a step line, or None when absent.

    anchors the key on a separator so a metric whose name merely ends with another's
    (response_length_non_aborted/mean vs response_length/mean) cannot cross-match, and accepts the
    numpy>=2 ``np.float64(...)`` spelling (see _NUMPY_SCALAR_RE).
    """
    hit = re.search(rf"(?:^|[\s-]){re.escape(verl_key)}:(\S+)", line)
    return None if hit is None else _metric_value(hit.group(1))


def verl_step_number(line: str) -> int | None:
    """the optimizer step a verl log line is tagged with, or None when it is not a step line.

    the trainers' stdout callbacks gate every metric read on this: a line with no step carries no
    step-scoped metric worth recording. it lives here rather than in each trainer because all three
    verl paths read one stdout format -- VERL-134 was one format change that had to be fixed at
    three copies at once, and a path whose copy was missed drops steps silently rather than failing.
    """
    match = _VERL_STEP_GATE_RE.search(line)
    return None if match is None else int(match.group(1))


def parse_verl_step_metrics(line: str) -> dict | None:
    """the flash `metrics_last` entry a verl step line carries, or None when it carries none.

    malformed multi-rank output must not fail a paid run. return None for validation-only lines with no
    renderable metric, or they can occupy a deduplicated step and displace its training record.
    """
    match = _VERL_STEP_RE.search(line)
    if match is None:
        return None
    metrics: dict[str, float | int] = {}
    for verl_key, flash_key in _VERL_METRIC_FIELDS:
        value = parse_verl_metric(line, verl_key)
        if value is not None:
            metrics[flash_key] = value
    if not metrics:
        return None
    return {"step": int(match.group(1)), **metrics}


def append_step_metrics(backlog: list[dict], metrics: dict, *, limit: int) -> None:
    """record one step in a bounded, de-duplicated backlog, mirroring the trl callback.

    verl reprints a step on a validation pass, and a resumed run replays its resume step, so a
    repeat must replace rather than append -- otherwise the CLI renders the same step twice. the
    heartbeat thread reads ``backlog`` while the stdout loop writes it, so the new contents are
    published in ONE slice assignment: a filter/append/truncate sequence would let that reader
    observe a torn intermediate state with the step momentarily missing.
    """
    step = metrics.get("step")
    kept = [item for item in backlog if item.get("step") != step]
    kept.append(metrics)
    backlog[:] = kept[-limit:]
