"""shared out-of-process verl harness reused by the sft and opd verl trainers.

verl pins its own torch/vllm, incompatible with flash's, so flash never imports verl in-process: it
runs a verl trainer entrypoint as a subprocess against a separate interpreter and merges the
fsdp-sharded lora checkpoint back into a flash-servable peft adapter. these are the framework- and
algorithm-neutral pieces (interpreter resolution, checkpoint export, provenance, progress
streaming); the sft/opd/grpo modules layer their own dataset rows, hydra overrides, and orchestration
on top.
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
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

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
# verl-venv layer. fla resets the DeltaNet recurrence at packed example boundaries (cu_seqlens) and
# causal_conv1d resets the short causal conv (seq_idx); WITHOUT them transformers falls back to
# implementations that accept both arguments and silently discard them, so packed GDN training is
# contaminated across example boundaries while appearing patched. fla is required; causal_conv1d is
# best-effort (a failed build makes the gdn capability answer False, which turns
# remove-padding off and trains padded rather than wrong).
FLA_REQUIREMENT = (
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git"
    "@f0e213dbd8b5fb90c3c7eca869ac1706d5377139"
)
CAUSAL_CONV1D_REQUIREMENT = "causal-conv1d==1.6.2.post1"

# what a provisioned venv HOLDS, which is the only thing the stamp may identify. flash-attn belongs
# in it because it is installed separately from verl and pinned separately: a workdir provisioned by
# a release that installed verl but not flash-attn records the same VERL_REQUIREMENT, so a
# verl-only stamp would let a retry reuse that venv, skip the install, and return an interpreter
# missing exactly the package this path exists to guarantee. fla is in the stamp for the same
# reason: a venv from a release that predates it holds no fla, and reusing it would train gdn
# models padded (or, worse, look patched) forever without ever reprovisioning.
VERL_VENV_STAMP = f"{VERL_REQUIREMENT}\n{FLASH_ATTN_SPEC}\n{FLA_REQUIREMENT}"


# how many times the prebuilt-wheel install is attempted before the arm is handed back to the plane.
# uv already retries the download 3x internally, so each attempt here is a fresh uv invocation, not a
# fresh request: the point is to outlast an outage that spans one uv lifetime, not to hammer github.
FLASH_ATTN_INSTALL_ATTEMPTS = 3
FLASH_ATTN_INSTALL_BACKOFF_S = 15.0


def _install_flash_attn(py: str) -> None:
    """Install the prebuilt FA2 wheel, raising RetriableInfraError rather than dying on a network fault.

    The install is required (verl's cuda path imports flash_attn.bert_padding unguarded), so this
    cannot fail soft. But failing PERMANENTLY is wrong: uv exits 1 on a download timeout, which
    surfaces as CalledProcessError, and _worker_failure_flags classifies anything that is not
    RetriableInfraError/GitHubRateLimitError as retriable=false -- so the plane routes a transient
    github fetch failure to job_failed and burns the arm at attempt=0 on a paid gpu. Measured: an arm
    died on `error sending request ... operation timed out` for this exact url, and the identical url
    served 200 on retry moments later, so "cannot succeed on another worker" was simply false.
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


def _install_causal_conv1d(py: str) -> None:
    """Install the causal conv kernel that resets GDN conv state at packed example boundaries.

    Best-effort, deliberately: unlike flash-attn this cannot fail the provisioning. Without it
    ``gdn_reset_arch_from_caps`` answers None, the caller turns remove-padding off, and
    the model trains on verl's padded path -- slower, but boundary-correct. Dying here instead would
    turn a compiler hiccup into a dead paid run.

    FORCE_BUILD is passed per-call rather than exported, so provisioning leaves no trace in the
    environment verl inherits.
    """
    subprocess.run(
        ["uv", "pip", "install", "--python", py, "--no-build-isolation", CAUSAL_CONV1D_REQUIREMENT],
        check=False,
        env={**os.environ, "CAUSAL_CONV1D_FORCE_BUILD": "TRUE"},
    )


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
    return next(n for n in range(min(cap, rollout_batch), 0, -1) if rollout_batch % n == 0)


# worker threads each bridge serves requests from. the bridges are pure i/o relays (parse json,
# hand the payload to a callback, write json back), so this bounds concurrency, not throughput:
# requests beyond the pool wait in the listen backlog instead of each spawning an os thread.
_BRIDGE_WORKER_THREADS = 16


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """an http server that serves requests from a fixed thread pool instead of one thread each.

    ``ThreadingHTTPServer`` starts a new os thread per request and never bounds them. under a real
    rollout (agent.num_workers x rollout.n concurrent callers, alongside ray, vllm and libgomp
    pools already holding thousands of threads) the container hits its thread limit, and
    ``socketserver.process_request`` dies with ``RuntimeError: can't start new thread``. that kills
    the connection mid-request, so the caller sees ``RemoteDisconnected`` and the whole run fails
    -- VERL-139, observed on a 4090 where libgomp reported "Thread creation failed" first.

    a fixed pool cannot exhaust the thread table no matter how many callers arrive.

    bounding the pool makes the listen backlog load-bearing: with a thread per request the kernel
    queue drained immediately, but a busy pool leaves arrivals sitting in it. socketserver's default
    of 5 is far below a rollout's burst, and overflowing it resets the connection -- which the
    caller sees as the same ``RemoteDisconnected`` this fix exists to remove. so the default here is
    sized for a burst, and subclasses raise it further to their own known batch size.
    """

    # daemon threads so a stuck handler can never keep the worker process alive at shutdown.
    daemon_threads = True

    # measured: at the socketserver default of 5, 13 of 64 simultaneous callers were reset by peer.
    request_queue_size = 128

    def __init__(self, *args, worker_threads: int = _BRIDGE_WORKER_THREADS, **kwargs):
        self._bridge_pool = ThreadPoolExecutor(
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

    ray autodetects cpus from the host and eagerly prestarts one idle worker per core before the
    first task runs (worker_pool.cc "[Eagerly] Start install runtime environment"). on a rented
    single-gpu pod that detection is wrong in the dangerous direction: the pod is a container on a
    much larger host, so a 1x4090 box with 46GB of ram reports the host's 48 cores and ray forks 48
    interpreters that no workload asked for. two ways that kills a run, both observed on real gpus:

      - the forks exhaust host ram. 42 idle workers held 23.29GB USS -- over half the node -- and
        ray's memory monitor killed the actor that mattered (VERL-123, opd).
      - the forks exhaust process/thread limits. worker_pool.cc:704 "Failed to start worker with
        return value system:11: Resource temporarily unavailable" is a FATAL check, so the raylet
        aborts and every actor dies with "Owner's node has crashed" (VERL-124, grpo).

    neither failure names cpus, which is what made them expensive to diagnose. cgroup quota is the
    honest number when the pod sets one; the affinity mask catches cpuset-pinned pods; the cap keeps
    the pool sane on an unconstrained host, where the detected count reflects a machine flash does
    not own. this bounds ray's PREFILL only -- it is not a limit on training parallelism, which is
    sized by gpus, not by this.

    the floor is not optional and must win over BOTH the cap and the detected count. ray's placement
    group is a logical reservation, so a pool smaller than verl_cpu_demand does not oversubscribe --
    it never schedules, and verl waits on ray.get(pg.ready()) with no timeout. verl's own
    _check_resource_available validates GPUs only (base.py), so a cpu shortfall is not reported: the
    run hangs on a rented, billing gpu having printed nothing. that is strictly worse than the crash
    this function exists to prevent, and it is reachable two ways -- an 8-gpu job needs 26 cpus,
    above the 16 cap, and a 2-cpu cgroup quota is below the demand of even a 1-gpu job.
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
#
# every launch used to ask the verl interpreter the same handful of independent questions in
# SEPARATE `python -c` subprocesses -- rollout field, gdn boundary kernels, device capability,
# flashinfer, wandb -- and each one paid the full torch/verl import before answering. the imports
# dominate: the questions themselves are attribute lookups and an import check.
#
# they are also genuinely independent, so one child can answer all of them and print a single JSON
# blob. that collapses N torch imports into 1 on the critical path of every run, before any
# training starts and while the gpu is already rented.
#
# the child still names only packages it is guaranteed to have (verl, torch, transformers,
# causal_conv1d, flashinfer) and never `flash`, which is not installed in the verl venv -- an
# `import flash` here would raise in every child and silently answer "no" to everything. anything
# needing flash code or a hub read is resolved by the PARENT and interpolated in as a literal (see
# gdn_reset_arch_from_caps's docstring for the full reasoning).
#
# EVERY question is answered independently inside its own try/except and defaults to the same
# fail-closed answer the individual probe returned, so one unavailable capability cannot suppress
# the others. a child that dies outright leaves every answer at its fail-closed default, exactly as
# five dead subprocesses would have.
#
# each answer is FLUSHED THE MOMENT IT IS KNOWN, one line per question, and the parent merges the
# lines it received. five separate subprocesses each had their own timeout, so a slow gdn smoke
# could not cost the cheap answers; one shared child would re-couple them if it printed a single
# blob at the end -- a kill during the last question would take every earlier answer with it, and
# the empty rollout_fields that leaves behind aborts grpo outright. answering incrementally keeps
# the cheap questions safe from the expensive one. ORDER IS THE CONTRACT: cheap and fast first, the
# gdn smoke (transformers import + a live cuda kernel) last.
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

        ok = is_flash_linear_attention_available() and is_causal_conv1d_available()
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
            src = inspect.getsource(gdn.forward) if gdn is not None else ""
            ok = ("cu_seq_lens_q" in src) and ("seq_idx" in src)
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
        emit("gdn_boundary_resets", bool(ok))
    except Exception:
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

# this child does in ONE process what five did serially, so its budget is the UNION of theirs, not
# the largest of them. the old per-probe timeouts were 300 (rollout field) + 600 (gdn) + 120
# (capability) + 120 (flashinfer) + 120 (wandb); each paid its own cold torch/verl import, and the
# gdn probe got its full 600s without one of those imports in front of it. keeping 600 here would
# hand the gdn smoke strictly LESS wall-clock than it used to have, on a cold cache, on the run
# where it matters most. the sum is the honest bound, and it is only ever reached by a child that
# is genuinely wedged: the timeout is a backstop, not the expected duration.
_CAPABILITY_PROBE_TIMEOUT_S = 300 + 600 + 120 + 120 + 120


def probe_verl_capabilities(python_bin: str, gdn_module: str = "") -> dict:
    """ask the VERL CHILD every independent capability question in ONE subprocess.

    each question used to cost its own interpreter, and the torch/verl import -- not the question --
    is what they were paying for. batching them collapses that import onto one child per run.

    ``gdn_module`` is the checkpoint's ``transformers.models.*`` modeling module, resolved by the
    parent because it needs a hub/cache read. pass ``""`` to skip the gdn question entirely, which
    is what a non-hybrid model does: nothing consumes the answer, and the smoke runs a live CUDA
    kernel.

    the returned dict always has every key. an unanswerable question holds its fail-closed default
    (see ``_CAPABILITIES_UNAVAILABLE``) rather than being absent, so callers read a value rather
    than testing for membership. a child killed partway through keeps the answers it already
    flushed: it is the questions it never reached that fail closed, not all of them.
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
    override rather than letting the run abort at dataclass conversion.

    an unanswerable probe reads as "not declared", which is what the subprocess version returned
    when the child could not be reached: the caller raises with the cause instead of composing a key
    that would abort in dataclass conversion on a paid gpu.
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

    the boundary shim derives ``cu_seq_lens_q`` and ``seq_idx`` and hands them to the GDN layer, but
    delivering them is not the same as them being used. without flash-linear-attention,
    ``chunk_gated_delta_rule`` binds to transformers' ``torch_chunk_gated_delta_rule``, which takes
    ``**kwargs`` and never reads ``cu_seqlens``; without causal_conv1d, ``causal_conv1d_fn`` is None
    and the conv falls back to a plain ``F.silu(self.conv1d(...))`` that takes no ``seq_idx``. both
    fallbacks accept the arguments and silently discard them, so a run that looks patched trains on
    contaminated boundaries anyway.

    returns the ``model_type`` rather than a bool so the caller cannot hand the shim a DIFFERENT
    architecture than the one just verified. the two must agree: ``qwen3_5_moe`` (the 35B) is a
    separate transformers module from ``qwen3_5`` (the dense models), and patching the wrong one
    leaves the forward unpatched while the gate still reports resets active -- packed, contaminated,
    labelled fixed. a second independent resolve at the call site would make that agreement a
    convention; returning it makes it structural.

    asked of the CHILD interpreter because that is where the model lives: flash installs fla into the
    parent worker, and the verl venv is built without ``--system-site-packages``, so the parent
    answering "yes" says nothing about the interpreter that will run the forward pass.

    the probe is deliberately self-contained -- it imports only transformers and causal_conv1d, never
    ``flash``. flash is pip-installed into the WORKER interpreter at runtime and is not on the verl
    venv's path, so a probe that imported ``flash.engine.worker.packing`` would raise
    ModuleNotFoundError in every child, report "cannot reset", and silently pin every gdn run to the
    padded fallback forever. that fails closed, so it trains correctly -- but the boundary fix would
    never actually engage, and nothing would say so. ``_CAPABILITY_PROBE`` above holds the
    convention: a child probe names only packages the child is guaranteed to have.

    the checks mirror ``packing.gdn_packing_available``: fla present, causal_conv1d imports (catching
    a built-but-broken ABI that find_spec would pass), the kernel actually runs on the live GPU
    (catching a wheel compiled without this arch), and this transformers version's GDN forward really
    threads both kwarg names.

    the arch is recovered from ``gdn_module`` -- the same literal the child was asked about -- so the
    shim still cannot be rendered for a different architecture than the one just verified. an
    unanswerable or negative probe returns None and the caller trains padded.
    """
    if not caps.get("gdn_boundary_resets"):
        return None
    # "transformers.models.<arch>.modeling_<arch>" -> "<arch>", read from the string the CHILD was
    # handed rather than re-resolved here: a second resolve could disagree with what was cleared.
    return gdn_module.split(".")[2]


def resolve_verl_python(workdir: str, *, install_wandb: bool = False) -> str:
    """return an interpreter that can import verl.

    prefers FLASH_VERL_PYTHON (set by the caller when verl is preinstalled, e.g. on a verl image).
    otherwise provisions an isolated venv on the pod so verl's torch/vllm never touch flash's env.
    optionally installs wandb best-effort for callers that enable verl's wandb logger.

    the preset is returned as-is: flash does not own that interpreter and must not mutate it. it can
    hold a verl without the fork's rollout fields, so callers emitting a fork-only override gate it
    on verl_declares_rollout_field.

    a self-provisioned venv is flash's own, and is rebuilt whenever it does not record the current
    VERL_REQUIREMENT, so an empty/absent FLASH_VERL_PYTHON always yields the pinned verl.

    an EMPTY value is deliberately equivalent to absent, and it is the only route a run has: worker
    images can export FLASH_VERL_PYTHON themselves, and [worker_env] can set a key but never delete
    one, so omitting it from the spec leaves the image's value in place. `""` is what lets a spec
    say "ignore the image's interpreter and provision the pinned one".
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
        # the SAME three overrides Dockerfile.worker writes, for the same reason: this pin set
        # deliberately violates three declared ceilings, and a bare pin cannot break any of them --
        # a pin is a constraint the resolver must satisfy alongside the declaration, so the pair is
        # simply unsatisfiable and the install fails outright. only --override makes uv IGNORE the
        # declaration. all three lines are required; dropping any one leaves the set unsatisfiable:
        #   verl + transferqueue declare numpy<2.0.0  -> vllm 0.19.1 needs numpy>=2
        #   verl[vllm] declares vllm>=0.8.5,<=0.12.0  -> flash needs 0.19.1 for the Qwen3.5 archs
        #   vllm declares xgrammar>=0.1.32            -> structured opd gates on EXACTLY 0.1.25
        overrides = os.path.join(workdir, "verl-overrides.txt")
        with open(overrides, "w") as f:
            f.write("numpy==2.2.6\nxgrammar==0.1.25\nvllm==0.19.1\n")
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
                # gated-deltanet kernels, in LOCKSTEP with Dockerfile.worker's verl-venv layer (same
                # fla sha, same tilelang pins). the model trains in THIS interpreter, so fla being
                # installed in the worker's own env says nothing: without it here, transformers binds
                # chunk_gated_delta_rule to the pure-torch fallback that takes **kwargs and discards
                # cu_seqlens, and packed gdn runs train across example boundaries while looking
                # patched. apache-tvm-ffi is pinned to 0.1.11 because 0.1.12 double-registers TVM-FFI
                # and aborts `import tilelang`.
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
        _install_causal_conv1d(py)
        # written only after BOTH installs succeed, so a venv that died between them is unstamped
        # and the next attempt rebuilds it rather than reusing a half-provisioned interpreter.
        with open(stamp, "w") as f:
            f.write(VERL_VENV_STAMP)
        if install_wandb:
            # verl does not pull wandb; install it best-effort so logger setup can fall back to console.
            subprocess.run(["uv", "pip", "install", "--python", py, "wandb"], check=False)
    return py


def resolve_verl_loggers(caps: dict) -> list[str]:
    """verl's ``trainer.logger`` list, gated on the interpreter that actually logs.

    verl logs from its own interpreter, so wandb must be importable THERE, not in flash's env. The
    prebuilt worker image's /opt/verl-venv does not install wandb, and a preset FLASH_VERL_PYTHON is
    returned by resolve_verl_python untouched (flash does not own that interpreter), so a run with
    WANDB_API_KEY set would otherwise ask verl for a logger it cannot import and die at logger init
    on a paid GPU. Fall back to console instead: losing the W&B mirror beats losing the run.

    Never inits a flash-side run -- flash does not train in-process on this path, so it would stay
    empty.
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

    vllm 0.19.1 is pinned on BOTH the baked verl venv and the fallback interpreter, so its
    graph-capture defects are card properties, not trl-specific ones, and verl drives the same
    engine. The retired trl driver handled that with an ALLOWLIST -- graphs only on ``{(8, 0),
    (9, 0)}`` plus Blackwell, eager everywhere else -- which is the shape this function used to
    inherit. An allowlist answers "has anyone run this?", not "does it break?", so it excluded cards
    whose capture was never tested (see sm89 below). This function now excludes only what has been
    measured to fail.

    Note that verl captures MORE aggressively than the configuration those defects were found in:
    ``rollout.enforce_eager`` defaults False (``workers/config/rollout.py:195``) and its async server
    hardcodes ``cudagraph_mode=FULL_AND_PIECEWISE`` (``vllm_async_server.py:237-241``) where trl
    asked only for ``FULL_DECODE_ONLY``. That is why an arch that fails must be excluded outright
    rather than tuned down.

    One knob is enough and cannot fight verl's: vllm 0.19.1 resolves ``enforce_eager`` LAST, forcing
    ``compilation_config.mode=NONE`` and ``cudagraph_mode=NONE`` regardless of what was requested
    (``config/vllm.py:847-853`` and ``:1024-1029``). Overriding ``cudagraph_mode`` instead would
    leave torch.compile on, which is the other half of what sm86 dies in.

    **sm89 (Ada, RTX 4090 / L40S) captures graphs.** It was excluded as "unvalidated" -- inherited
    from the trl driver's allowlist rather than measured -- and it is the catalog's
    ``recommended_gpu`` for Qwen3.5-0.8B and 2B, so the default small-model GRPO and OPD route was
    forfeiting both cuda graphs and torch.compile. Both axes of the exclusion were then measured on
    real hardware:

    - *output*: judged against sm90-with-graphs, the configuration flash already ships, in both raw
      and chat-templated prompt formats. sm89's graph arm introduced no non-termination, no
      repetition loop, and lost no decidable answer its own eager arm got right -- strictly cleaner
      than the shipping reference, which drops one arithmetic token. bit-identity is NOT the bar:
      vllm makes no determinism promise across execution modes, and sm80/sm90 would fail it too.
    - *host memory*: this is what the exclusion actually rested on. an OPD run captured 102 graphs
      and was OOM-killed with the node at 41.51/42.84 GB, and ~29 GB was attributed to those graphs.
      that number was a RESIDUAL (node total minus ray's accounting), not a measurement, and ray
      cannot see vllm's ``EngineCore`` child where the graphs live. measured across the whole process
      tree, capture costs **453 MB** on sm89 0.8B and **426 MB** on 2B -- *less* than sm90's 551 MB.
      the box was already at 97% on ~9.9 GB of baseline engine footprint that eager pays too; graphs
      were the last straw on an undersized host, not the load.

    **sm86 (Ampere, A10/A10G) stays eager.** Not inherited -- reproduced. Under graphs the A10G
    turned "List three colors." into that sentence repeated to the token cap, never emitting EOS,
    where its own eager arm answered in 19 tokens; a second run reproduced the same collapse on a
    different prompt ("Count from 1 to 10." verbatim, repetition 0.96) while sm90 and sm86's own
    eager arm both produced a normal escalating continuation on the identical prompt. That is
    degeneration, not the wording drift graph capture is entitled to.

    B200 (sm100) is deliberately NOT eager -- the trl path returned early there and kept vllm's own
    default, and the b200 rollout work depends on graphs.

    An unanswerable probe (``cc is None``) leaves verl's default in place rather than guessing eager
    onto an unknown card.
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

    verl defaults ``free_cache_engine`` and ``enable_sleep_mode`` BOTH True and offloads the engine
    between steps, so on a flagged model a run wedges at a sleep boundary instead of failing fast.
    both knobs go: ``free_cache_engine`` is what actually gates the sleep()/wake_up() rpcs
    (``vllm_async_server.py:626,648``), and ``enable_sleep_mode`` is what builds the sleep-capable
    engine in the first place (``:265``).

    safe because vram sizing never credits the offload for a flagged job -- grpo sizes it on the
    RESIDENT peak and rejects a config that cannot fit, and opd's estimator ignores ``sleep_offload``
    outright (it is resident-only by construction). staying resident therefore cannot admit an OOM
    that the sleep path would have avoided.

    the two knobs need DIFFERENT hydra prefixes, and getting it wrong kills the run at config parse
    before a single byte is allocated. ``free_cache_engine`` is in ``trainer/config/rollout/
    rollout.yaml`` (:56) so a bare override resolves. ``enable_sleep_mode`` is NOT: it exists only as
    a ``RolloutConfig`` dataclass field (``workers/config/rollout.py:277``), and the yaml that hydra
    builds the struct from omits it -- so a bare override raises ``Key 'enable_sleep_mode' is not in
    struct`` and ``main_ppo`` exits 1. ``+`` appends the key instead of overriding it, which is what
    verl's own error text prescribes. dataclass-default-only keys are the norm here, not the
    exception, so check the yaml before adding a bare override.
    """
    if not sleep_unsupported:
        return []
    return [
        "actor_rollout_ref.rollout.free_cache_engine=false",
        "+actor_rollout_ref.rollout.enable_sleep_mode=false",
    ]


def trainer_dtype_overrides() -> list[str]:
    """verl overrides that store the frozen base in bf16 instead of verl's fp32 default.

    verl's ``trainer/config/engine/fsdp.yaml`` (:33) declares ``model_dtype: fp32``, and that string
    is handed straight to ``from_pretrained`` as ``torch_dtype``
    (``workers/engine/fsdp/transformer_impl.py:234``). the ``if torch_dtype is None`` fallback just
    below it never fires, because the yaml default is not None -- so an un-overridden RL or OPD run
    loads the base at 4 bytes/param. measured at 27B: ``PeftModelForCausalLM contains 27.61B
    parameters`` -> ``After FSDP, memory allocated (GB): 103.02``, which is 27.61e9 * 4 to within
    0.1 GB. that is one fp32 copy, not optimizer state; a rank-32 adapter's moments are noise.

    the fp32 copy buys nothing HERE, though the same key on a full fine-tune would be the master
    weight copy and dropping it would be a real numerics regression. two things make it dead weight
    on flash: FSDP is already wrapped ``MixedPrecision(param_dtype=bf16)`` (:346-356) so every
    parameter is cast to bf16 for compute regardless, and the base is FROZEN -- verl's
    ``ref_in_actor`` (``lora_rank > 0 or lora_adapter_path is not None``) is always true here, see
    ``rl_train.render_kl_ref_adapter_shim``. storing never-updated weights at twice the width to
    read them at half of it is pure residency.

    what is actually optimized stays fp32: peft's ``autocast_adapter_dtype`` defaults True
    (``peft/mapping_func.py:35``, not overridden by verl) and casts adapter weights UP to fp32 for
    training stability, so a bf16 base still yields fp32 ``lora_*`` params.

    only ``actor`` is set. ``ref`` looks like a second resident copy worth shrinking, and is not:
    ``ray_trainer.py:897`` assigns ``self.ref_policy_wg = self.actor_rollout_wg`` -- literally the
    same object -- whenever ``ref_in_actor`` holds, and the separate ref worker at :884 is built only
    ``if not self.ref_in_actor``. the reference policy here is the actor with its adapter disabled,
    so a ``ref`` dtype override would free nothing on any run flash submits.

    the key is declared in the yaml, so it takes a BARE override and ``+`` is REJECTED (``Could not
    append to config. An item is already at ...``). that is the exact inverse of
    ``rollout_resident_overrides``' ``enable_sleep_mode``, which is dataclass-only and requires
    ``+``. neighbouring keys, opposite prefixes: check the yaml per key, never pattern-match off a
    nearby override.
    """
    return ["actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"]


def resolve_blackwell_attention_backends(
    caps: dict, cc: tuple[int, int] | None
) -> tuple[str | None, str | None]:
    """the rollout ``(attention_backend, mm_encoder_attn_backend)`` this GPU needs, or ``(None, None)``.

    vLLM 0.19.1 picks both by capability, and on Blackwell both defaults are wrong:

    * **decoder.** ``_get_backend_priorities`` (vllm/platforms/cuda.py:99-111) heads its non-MLA list
      with FLASH_ATTN for every arch except cc-major 10, and FlashAttention validates at
      ``>= (8, 0)`` -- so sm120 (RTX 5090 / RTX Pro 6000) selects it, and its prebuilt PTX is
      unreliable on consumer Blackwell hosts, which surfaces as SILENT EMPTY ROLLOUTS rather than a
      crash. Pin FLASHINFER, which the same table already prefers on B200. flashinfer can install yet
      be ABI-broken against this torch, so gate on the import and fall back to TRITON_ATTN: a
      registered, PTX-independent decoder backend that trains on the 5090. NOT TORCH_SDPA -- that tag
      is ``""`` in the 0.19.1 registry (ViT-only) and raises at decoder backend validation.
    * **ViT.** ``get_supported_vit_attn_backends`` (cuda.py:344-351) heads its list with FLASH_ATTN on
      every cc>=8.0 card, and on Blackwell that routes to vLLM's CUTE flash-attn, which is
      unimportable against every published ``nvidia-cutlass-dsl``: the vendored cute needs
      ``cutlass.cute.core.ThrMma`` (<=4.5.x) while ``vit_attn_wrappers`` needs
      ``cutlass._mlir_helpers`` (>=4.6.0) -- the two never coexist, so the first ViT attention aborts
      and takes the rollout with it (a version pin cannot fix it; measured 2026-07-07). A VL model
      builds its vision tower even for a text-only rollout, so this reaches text-only GRPO too.
      ``get_vit_attn_backend`` honors an explicit backend unconditionally (cuda.py:367-373), and
      TORCH_SDPA is a supported ViT backend on cc>=8.0, so pinning it sidesteps the CUTE import. The
      decoder attention is unaffected -- that is chosen separately, above.

    The flashinfer answer comes from the VERL interpreter, not flash's: verl owns the rollout engine
    and pins its own vllm stack, so flash's ``import flashinfer`` would answer for the wrong
    environment. A failed flashinfer import degrades to TRITON_ATTN rather than leaving the fragile
    default in place; an unanswerable capability probe (``cc is None``) leaves vllm's defaults alone.
    Both answers ride the one batched child probe, so this asks the interpreter nothing itself.

    Returns ``(None, None)`` off Blackwell, where vLLM's own defaults are correct.
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

    - RL nests per-role, ``global_step_N/actor/`` (``trainer/ppo/ray_trainer.py`` builds
      ``os.path.join(local_global_step_folder, "actor")``).
    - SFT writes the shards straight into ``global_step_N/`` (``utils/checkpoint/
      checkpoint_handler.py`` passes ``local_path=local_global_step_folder`` unmodified).

    So detect the layout from what verl actually wrote rather than hardcoding either convention.
    ``huggingface/`` is the right marker because it is the subfolder ``verl.model_merger`` resolves
    ``hf_model_config_path`` against, which is precisely what a wrong answer here breaks.

    Getting this wrong does not surface as a missing path: ``AutoConfig.from_pretrained`` treats a
    nonexistent local directory as a *hub repo id*, so the failure arrives as
    ``HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`` and reads
    like a credentials or network problem instead of a path bug.
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
    ``verl.model_merger merge`` writes a standard peft adapter (adapter_config.json +
    adapter_model.safetensors) to a ``lora_adapter/`` subfolder of its target; we copy just that
    adapter into flash's adapter dir (the co-produced merged full model is discarded -- flash
    serves the lora on the immutable base).

    verified against verl 0.8 on an h100: merger emits ``<target>/lora_adapter/{adapter_config
    .json,adapter_model.safetensors}`` with ``base_model_name_or_path: null``.
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

    exists because the verl child's stdout reaches **no collected log stream**. ``run_verl_training``
    re-prints every child line to the parent worker's stdout, and the parent's stdout is not on the
    path the plane scrapes -- only the ``HEARTBEAT <json>`` marker line is (``heartbeat.py``). so a
    child that wedges produces zero retrievable bytes: an OPD arm stalled 3/3 attempts on 3 separate
    endpoints and every attempt cost ~50 minutes of paid H100 time to learn nothing (ISSUES VERL-061).

    retaining the tail in memory lets a stall report the child's own last words through the marker
    channel that provably survives, instead of inferring the cause from the outside.
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
        staleness signal: a child looping on the same line still advances this, and a child that
        has gone silent cannot advance it even though its retained tail stays fully populated.
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
    consecutive heartbeats after the money is already spent (ISSUES VERL-067).

    holding the previous line count here turns that into a number the first dump already carries.
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

    a run that is training is diagnosable from its step/loss stream, so attaching the tail then would
    add an uploaded payload every tick for no information. before the first step there is no such
    stream, and that is exactly the window a setup stall lands in -- the child prints its complaint,
    nobody collects the parent's stdout, and the run dies to a watchdog with zero evidence.

    with ``staleness`` supplied, the payload also carries ``child_tail_silent_ticks``: how many
    consecutive samples produced no new child output. the tail says what the child last said; this
    says whether it is still saying anything, which is the difference between a slow start and a
    wedge. without it the two are indistinguishable in a single heartbeat and have to be told apart
    by hand-diffing consecutive dumps.

    returns an empty dict once ``step`` advances, or when the child has said nothing yet.
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

    ray names these ``session_<timestamp>_<pid>`` and also maintains a ``session_latest`` symlink,
    but the symlink is removed on a clean shutdown, so pick the newest real directory instead: the
    case this exists for is the one where ray did NOT shut down cleanly.

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
    failure undiagnosable from uploaded artifacts and costs a paid gpu run per guess (VERL-115).

    one string rather than a directory of copies: the caller writes it exactly like the traceback
    artifact beside it, so a dying pod does one upload instead of six against the same bounded hf
    deadline allowance, and there is no staging directory whose only purpose is to be uploaded.

    ``started_after`` is passed through so a run that failed before ray ever started reports nothing
    rather than a previous run's session.

    best-effort by construction: this runs on a path that is ALREADY failing, so it must not be able
    to replace the real error with its own. every read is guarded and a file that cannot be read is
    simply absent from the result.
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
            # never begin mid-line. ray logs a raylet's argv and flash passes credentials to the
            # worker through the environment, so this is third-party text carrying live tokens.
            # sanitize_diagnostic matches a secret by its `key=` prefix or by its full value, and a
            # tail boundary landing inside either leaves the REST of a real credential unmatched and
            # uploaded. dropping the partial first line removes every single-line secret the
            # boundary could have split, at the cost of one line we could not have redacted anyway.
            # a MULTILINE secret survives this -- its later lines are whole and land after the cut --
            # so sanitize_diagnostic redacts multiline values line-by-line as well; both halves are
            # required, neither is sufficient.
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

    ``tail``, when supplied, retains the child's most recent lines so a caller that observes a stall
    can report what the child last said. the parent's own stdout reaches no collected stream, so this
    buffer is the only way the child's words escape the container (see ``ChildOutputTail``).

    the child gets its own session so teardown can signal the whole process group. verl spawns vllm's
    EngineCore as a grandchild, and terminating only the direct child reparents that grandchild to
    init with its cuda context intact, so the gpu stays allocated and the next run waits forever on
    device memory that nothing will release.
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
        assert proc.stdout is not None
        for line in proc.stdout:
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
            proc.wait()
        # every job boundary, not just the failing ones. a straggler an earlier teardown SIGKILLed
        # but could not drain in time exits shortly after, and `kill_process_group` -- the only other
        # caller of this -- runs on exceptions alone: a worker whose later jobs all succeed would
        # hold that zombie for its whole life, as pid 1 with nothing else to reap it (codex[bot]).
        reap_stragglers()
    return_code = int(proc.returncode)
    try:
        raise_for_classified_verl_exit(return_code, child_tail)
    except BaseException:
        kill_process_group(proc, process_group_id=process_group_id)
        raise
    return return_code


_TEARDOWN_GRACE_S = 10.0


def _process_is_zombie(pid: int) -> bool:
    """true when `pid` has exited and is only awaiting a reaper.

    field 3 of /proc/<pid>/stat is the state character, and it follows a comm field that may itself
    contain spaces and parentheses -- so it is read after the LAST `)`, not by splitting the line.

    only DISAPPEARANCE proves an exit. every other error says this process could not look, not that
    there is nothing there, and the two must not collapse: teardown can run with the worker out of
    file descriptors, where opening /proc/<pid>/stat raises EMFILE for every live member at once.
    reading that as a zombie made the whole group report drained, so SIGKILL was never sent and an
    EngineCore kept its cuda context precisely on a resource-failure path (codex[bot]). so an
    unreadable status counts as alive -- over-reporting costs a signal to a group that ignores it,
    under-reporting strands the gpu.
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

    without this every reap below is a no-op, because this process is NOT pid 1. `Dockerfile.worker`
    runs `/rp_handler.py` as pid 1 and that handler spawns the flash worker as a subprocess
    (`flash/providers/runpod/train/endpoints.py:539`), so an EngineCore orphaned when the trainer
    exits reparents past us to the HANDLER. `waitpid` then raises `ChildProcessError` here and the
    zombie is recorded as handled while it keeps its pid for the worker's whole life -- and the
    handler only ever waits on the worker itself, so nothing else ever collects it either.

    marking this process a subreaper makes the kernel reparent such orphans HERE instead, which is
    the condition the rest of this module already assumes. set once at teardown-path entry rather
    than at import, so merely importing this module cannot change an unrelated process's semantics.
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

    the return value is what distinguishes "reaped" from "ours and still running": `waitpid` answers
    `(0, 0)` for a live child of ours, which is the one case worth remembering. `ChildProcessError`
    means the pid is not ours to wait on, so there is no status left for this process to take.

    that answer is only safe because `adopt_orphaned_descendants` runs first: it makes an orphaned
    grandchild reparent to US, so `ChildProcessError` really does mean someone else owns the pid.
    Without it the same error is returned for a zombie nobody will ever reap (codex[bot]).
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

    an EngineCore orphaned when the trainer exits is reparented HERE, because
    `adopt_orphaned_descendants` claimed it. once SIGKILL turns it into a zombie no signal can clear
    it -- only a wait can -- and leaving it costs the reusable worker one permanent pid per failed or
    cancelled run, walking it toward the pid limit (codex[bot]).

    `skip` is the direct child, which `subprocess.Popen` reaps itself. waiting on it here would take
    the exit status Popen is owed, leaving `returncode` unset for a caller that asks which signal
    reaped it. everything else in the group is a grandchild this process never spawned, so nothing
    else has a reaper to conflict with.

    scoped to the group so it can only ever touch descendants of this teardown, and non-blocking so
    a member that is merely still running costs nothing.

    a member that is ours and STILL RUNNING here is remembered rather than forgotten. SIGKILL cannot
    be refused but it also cannot be delivered while a process sits in uninterruptible sleep, so one
    can outlast the drain deadline and only then become a zombie -- after the last wait this
    teardown performs. without a record no future wait is ever scheduled for it and the entry is
    permanent on a pid-1 worker (codex[bot]).
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

    a straggler is remembered so a FUTURE wait can collect it, and on a long-lived process the next
    teardown is that wait. this process is not long-lived: `flash/providers/runpod/train/
    endpoints.py:538-552` spawns a fresh `flash.engine.worker_entrypoint` per phase and waits for it
    to exit, so the set dies with the phase. the straggler then reparents to the persistent runpod
    handler, which waits only on the worker it spawned, and becomes a zombie for the container's
    whole life (codex[bot]).

    so the last wait happens here rather than at a job boundary that never comes. registered at
    import because the leak is not specific to one entry point: whichever of them ran, the pids are
    recorded in this process and only this process is their reaper.

    bounded and non-fatal. these are pids already SIGKILLed by an earlier teardown, so the wait is
    for a process finishing its exit, not for work; and an interpreter already on its way out must
    not be held up or failed by a reap that costs a pid at worst.
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

    signal 0 checks addressability and delivers nothing, so it succeeds for a group whose only
    remaining member is a zombie. that is not a hypothetical here: the container runs python
    directly with no init, so an orphaned EngineCore is reaped only by this module's own waits and
    can otherwise sit unreaped indefinitely. driving the escalation off addressability alone
    therefore burned the full drain
    deadline on every teardown -- delaying the reusable worker even though the cuda context was
    already released -- and nothing stronger than SIGKILL exists to clear it (codex[bot]).

    So the group is alive only while some member is not a zombie. Membership is read from /proc
    rather than tracked, since the survivors are grandchildren this process never spawned.

    A drained verdict is only returned when TWO consecutive walks agree on it, because a single walk
    cannot distinguish a drained group from one it read too early. /proc can be listed while the
    leader is still present, and the leader can then fork and exit before its status is inspected:
    the snapshot is nonempty and zombie-only, yet the child that inherited the group is alive,
    unlisted, and never received the earlier signal -- so teardown returns without SIGKILL and it
    keeps its cuda context (codex[bot]). Rechecking addressability cannot settle it either, since a
    zombie holds the group id: that answers True for a group that really has drained, which is the
    burn-both-deadlines failure this function exists to avoid. A second walk can, because the fork
    it missed is published by then.
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

    signalling the group rather than the pid is what reaches vllm's EngineCore grandchild; a survivor
    holds its cuda context and strands the gpu for every later run. the escalation is driven off the
    group, not off the direct child: the usual shape of this failure is the trainer dying on the term
    while the EngineCore ignores it, so waiting only on the child returns before the survivor is gone.
    a captured ``process_group_id`` keeps the group addressable after the direct child is reaped.
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
# torch defaults ``cuda.matmul.allow_tf32`` to False, so an fp32 matmul runs at full fp32 rate
# unless something opts in. the model runs in the verl CHILD, and these flags are per-process
# torch state that no child inherits -- setting them in the flash parent (which trains nothing)
# leaves the trainer on the slow path. sft already opts in from inside its own sitecustomize;
# this is that same fragment, shared so grpo and opd get it from the one place.
#
# bf16 params are unaffected: this only changes the reduction precision of matmuls that are
# genuinely fp32 (the fsdp optimizer/grad-norm math and any fp32-autocast region), where tf32
# keeps fp32 range and trades mantissa bits for tensor-core throughput.
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
# notes["wandb_url"]. verl calls wandb.init INSIDE the training subprocess, so the flash parent's
# wandb.run is None and that spread would be a silent no-op. the url is not derivable up here
# either: it needs entity/project/runs/<wandb-generated-id>, and flash only knows the project and
# its own run NAME. so the child reports it back over the marker channel the parent already scans.
FLASH_WANDB_LINK_MARKER = "FLASH_WANDB_LINK"

_WANDB_LINK_RE = re.compile(rf"{FLASH_WANDB_LINK_MARKER}\s+(\{{.*\}})\s*$")


def render_wandb_link_shim() -> str:
    """child-side sitecustomize fragment that reports the live w&b run's url and id.

    wraps wandb.init rather than reading wandb.run at import time: sitecustomize runs long before
    verl's tracking module initializes the run, so anything read here would always be None.

    never raises. wandb is optional and a logging link must not be able to abort paid training, so
    every failure path leaves the run unlinked instead of dead.
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

    verl trains with ``pad_mode: no_padding``, so a micro-batch reaches the model as ONE row of
    ``(1, total_nnz)`` with ``attention_mask=None`` and per-example ``position_ids`` that restart at
    0 (``workers/engine/fsdp/transformer_impl.py`` remove-padding branch). that is real packing, and
    the softmax-attention layers handle it correctly on their own: transformers derives their varlen
    boundaries from exactly those restarting position ids.

    the GatedDeltaNet layers do not. they read their two reset structures out of ``**kwargs``:
    ``seq_idx`` for the causal conv (``modeling_qwen3_5.py`` ``causal_conv1d_fn(..., seq_idx=...)``)
    and ``cu_seq_lens_q`` for the DeltaNet recurrence (``chunk_gated_delta_rule(..., cu_seqlens=)``).
    nothing upstream derives either from ``position_ids``, and verl passes only
    ``{input_ids, attention_mask, position_ids}``, so both arrive ``None``. the conv window and the
    recurrent state then carry from the end of one packed example into the start of the next: every
    example after the first in a micro-batch trains on state that leaked from its neighbour. no
    error, no warning, no metric moves. all six catalog models are GDN hybrids, so this is every run.

    ``model_type`` names the architecture to patch, and MUST be the checkpoint's own -- the dense
    catalog models are ``qwen3_5`` but the 35B MoE (and the 397B teacher) are ``qwen3_5_moe``, a
    separate module whose ``Qwen3_5MoeGatedDeltaNet`` reads the same two kwargs from a
    ``Qwen3_5MoeTextModel`` that the dense patch never touches. hardcoding ``qwen3_5`` would patch a
    class the MoE never calls while the gate still reported resets active: packed, contaminated, and
    labelled fixed. the class is then found by ``TextModel`` suffix rather than by name, for the same
    reason the probe finds ``GatedDeltaNet`` that way. deliberately has NO default: the only sensible
    one would be ``qwen3_5``, which is exactly the wrong answer for the MoE, and a caller that
    forgot the argument would silently get it. pass ``gdn_reset_arch_from_caps``'s return value.

    note that the model NAME does not predict the module: the 27B is Qwen3.6 by name but reports
    ``model_type: qwen3_5``, while the 35B reports ``qwen3_5_moe``. only the config is authoritative.

    patched at the TEXT MODEL forward rather than on the GDN layer, for three reasons: it already
    spreads ``**kwargs`` into every decoder layer, so one patch covers all of them; it sees the
    ``position_ids`` that actually arrived, so the derivation cannot desync from the real batch; and
    it is where an upstream fix would go.

    derivation is delegated to transformers' own ``prepare_fa_kwargs_from_position_ids`` so the
    injected ``cu_seq_lens_q`` is byte-identical to the one the attention layers compute for
    themselves off the same tensor -- a hand-rolled cumsum that disagreed by one would put the two
    halves of the model on different segmentations.

    the ``_is_packed_sequence`` guard keeps this inert on the padded path (``use_remove_padding
    false``), where the batch is rectangular and the real ``attention_mask`` already drives
    ``_update_linear_attn_mask``. an explicitly supplied kwarg always wins.

    this shim raises on failure instead of passing. every other flash shim degrades quietly because
    what it adds is optional; this one is a correctness floor, and a run that silently trains on
    contaminated boundaries is worse than one that refuses to start.
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
# its trainer runs out of process. verl's LocalLogger prints exactly one line per
# optimizer update -- "step:N - key:value - key:value" over every scalar metric -- so the parent
# reconstructs the same backlog from that line. the payload schema is the CLI's, not verl's: keys
# below are what flash/cli/commands.py:_FOLLOW_METRIC_FIELDS renders.

# the left edge every verl step pattern shares. "not part of a longer word" rather than start-of-
# line, because ray tags worker stdout with a "(TaskRunner pid=123) " prefix -- an anchored match
# would parse nothing at all in production.
#
# and not "preceded by whitespace" either (VERL-134): verl's LocalLogger shares its stream with
# tqdm, which ends a bar with "]" and no trailing newline, so the metric line arrives glued to it --
# "...81.49s/it]step:2 - train/loss:...". a \s edge matched step 1 and missed every step after it,
# which froze the sft heartbeat on step 1's metrics and kept the zero-grad guard from ever arming.
# excluding / as well as word characters still refuses "global_step:" and a path's ".../step:".
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

    returns None rather than raising: this parses child stdout under multi-rank logging, where a
    truncated or interleaved line must not take down a paid run.

    a step line with no renderable metric yields None rather than a bare ``{"step": n}``: verl logs
    its pre-training validation pass as its own record at the *current* step counter
    (ray_trainer.py `logger.log(data=val_metrics, step=self.global_steps)`), and every key on that
    line is namespaced val-core/ or val-aux/. keeping it would render a row with a step number and
    no metrics, and -- because the backlog is deduplicated by step -- a resumed run's validation
    pass would land on the resume step and displace a real training row.
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
    repeat must replace rather than append -- otherwise the CLI renders the same step twice.

    the heartbeat thread reads ``backlog`` while the stdout loop writes it, so the new contents are
    published in ONE slice assignment: a filter/append/truncate sequence would let that reader
    observe a torn intermediate state with the step momentarily missing.
    """
    step = metrics.get("step")
    kept = [item for item in backlog if item.get("step") != step]
    kept.append(metrics)
    backlog[:] = kept[-limit:]
