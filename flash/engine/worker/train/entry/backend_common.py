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
import contextlib
import ctypes
import os
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

# verl 0.8.0 exactly, plus the truncation-mask, 3d position-id, and ulysses fused-label fixes, the
# whole-adapter lora bucket sync, and the fused-linear grad-flag preservation.
# it must stay on the 0.8.0 base: the opd plugin patches 0.8.0 internals and imports
# verl.trainer.main_ppo_sync, which verl deleted after 0.8.0, and opd's exact-version gate reads the
# version file this branch pins to the release value.
VERL_REQUIREMENT_NAME = "verl"
VERL_REQUIREMENT_URL = (
    "git+https://github.com/freesolo-co/verl@f71a02ddb32a9c6a6915f7519bda6dede92e9dd0"
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
# sha256 of the wheel above, kept in lockstep with Dockerfile.worker's ARG FLASH_ATTN_SHA256.
# the wheel is a community rebuild published by an individual, not Dao-AILab, and a github release
# asset can be deleted and re-uploaded at the same url by whoever holds the account, so the url
# alone pins nothing, and this interpreter is the one that TRAINS. bump it together with the url.
FLASH_ATTN_SHA256 = "a58c95a080363606e691c342d47dd173b510f4c013a8c5fcb8744def12e36a0f"
# what the install actually asks for. uv hashes the downloaded artifact and refuses to install it
# when the fragment does not match (verified against uv 0.11), so the check happens after the
# download and before anything is unpacked into the venv, the same order as the sha256sum -c on
# the infisical .deb in Dockerfile. the spec above stays bare so it keeps matching Dockerfile.worker's
# ARG default and worker-image.yml's FA2_SPEC, which fetch and verify in two separate steps.
FLASH_ATTN_INSTALL_SPEC = f"{FLASH_ATTN_SPEC}#sha256={FLASH_ATTN_SHA256}"
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
    "@9c8e42e762fce087c27b673af4922795d9edb85e"
)
# the flashqla GDN backend lives in a separate wheel that fla 0.5.2 dispatches to; the child's
# shim binds it on sm90 only (child_io.render_flash_qla_shim).
FLASH_QLA_REQUIREMENT = "flash-qla==0.1.2"
CAUSAL_CONV1D_REQUIREMENT = "causal-conv1d==1.6.2.post1"

# the SAME transformers range the main interpreter and Dockerfile.worker's verl-venv layer use.
# this venv is the interpreter that TRAINS, and transformers owns the gdn modelling code the
# boundary-reset shim patches (chunk_gated_delta_rule, the causal conv, and the kwargs they read),
# so an unbounded resolve here means the training path silently rides a transformers line nothing
# validated. bounding it to the range flash tests against is the whole reason. verl and vllm both
# depend on transformers, so this has to be in the OVERRIDE file as well as the direct list -- a
# direct pin alone loses to their transitive declarations.
#
# NOT the reason: this pin does not affect is_flash_linear_attention_available() or
# is_causal_conv1d_available(). Those are byte-identical in 5.12.1 and 5.14.1 and both open with
# is_torch_cuda_available(), so they answer False on any machine without a gpu regardless of the
# installed version. An earlier revision of this comment claimed a 5.13 import-path move made them
# answer False; that mechanism does not exist. See Dockerfile.worker's sanity block.
TRANSFORMERS_REQUIREMENT = "transformers>=5.8,<5.13"
# exact deployed install and override pin; the range above remains public package metadata.
TRANSFORMERS_INSTALL_REQUIREMENT = "transformers==5.12.1"

# the stamp must identify every separately installed package the venv holds. omitting flash-attn,
# fla, causal_conv1d, or the exact transformers pin lets an older partial venv match forever; conv1d
# leaves GRPO/OPD failing ``require_gdn_boundary_resets`` with no rebuild path, and a stale venv
# resolved before the transformers pin keeps training on a different transformers indefinitely.
# bump when a repair applied at build time changes what a fully provisioned venv looks like. the
# libcudart repair runs only on the rebuild path, so a venv stamped by a release that predates it
# matches, is reused, and keeps tilelang's original stub -- vLLM then aborts its import in the child
# after the gpu is already rented. carrying it in the stamp identity forces exactly one rebuild of
# those venvs instead of re-running the repair on every reuse.
VERL_VENV_BUILD_REPAIRS = "libcudart-stub-neutralized-v1"

# carries FLASH_ATTN_INSTALL_SPEC (the sha256-fragmented spec), not the bare FLASH_ATTN_SPEC: the
# digest is part of the venv's identity, not just how it got installed, so bumping or rotating the
# checksum while the mutable release url stays constant must invalidate a venv stamped under the old
# digest instead of letting it match forever and reuse an install that was never re-verified.
VERL_VENV_STAMP = (
    f"{VERL_REQUIREMENT}\n{FLASH_ATTN_INSTALL_SPEC}\n{FLA_REQUIREMENT}\n{FLASH_QLA_REQUIREMENT}\n"
    f"{CAUSAL_CONV1D_REQUIREMENT}\n{TRANSFORMERS_INSTALL_REQUIREMENT}\n"
    f"{VERL_VENV_BUILD_REPAIRS}"
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

    The spec carries FLASH_ATTN_SHA256, so a wheel that is not the pinned one is rejected between
    download and install rather than baked into the training interpreter. A mismatch exits nonzero
    like any other install failure, so it costs the retry budget and then terminates: noisy, but
    never silent.
    """
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    command = [
        "uv",
        "pip",
        "install",
        "--python",
        py,
        "--no-build-isolation",
        FLASH_ATTN_INSTALL_SPEC,
    ]
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

        from flash.engine.worker.io.hf import model_revision_kwargs

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


def _run_streaming_verl_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
    on_line: Callable[[str], None],
    errors: str | None = None,
    silence_watchdog: VerlChildSilenceWatchdog | None = None,
) -> int:
    """stream a verl subprocess under the shared process-group lifecycle supervisor."""
    adopt_orphaned_descendants()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors=errors,
        bufsize=1,
        start_new_session=True,
    )
    process_group_id = proc.pid
    if silence_watchdog is not None:
        silence_watchdog.bind(
            child_alive=lambda: proc.poll() is None,
            teardown=lambda: kill_process_group(proc, process_group_id=process_group_id),
        )
        silence_watchdog.start()
    try:
        with _ChildExitWatchdog(
            proc, process_group_id=process_group_id, grace_s=_ORPHANED_PIPE_GRACE_S
        ) as watchdog:
            assert proc.stdout is not None
            for line in proc.stdout:
                with watchdog.handling_line():
                    on_line(line)
    except BaseException:
        kill_process_group(proc, process_group_id=process_group_id)
        raise
    finally:
        if silence_watchdog is not None:
            silence_watchdog.stop()
        if proc.poll() is None:
            try:
                proc.wait(timeout=_TEARDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                kill_process_group(proc, process_group_id=process_group_id)
        reap_stragglers()
    if proc.returncode is None:
        raise RuntimeError(
            f"verl subprocess {proc.pid} did not exit after teardown; its process group is still "
            "holding the gpu"
        )
    return_code = int(proc.returncode)
    if watchdog.tore_down and return_code == 0:
        raise RuntimeError(
            f"verl subprocess {proc.pid} exited 0 but a descendant held its output pipe open for "
            f"{_ORPHANED_PIPE_GRACE_S:.0f}s; the process group was torn down to release the gpu"
        )
    if return_code != 0:
        kill_process_group(proc, process_group_id=process_group_id)
    return return_code


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
    silence_watchdog: VerlChildSilenceWatchdog | None = None,
) -> int:
    """run a verl trainer subprocess, streaming stdout and surfacing step progress.

    returns the process exit code. stdout+stderr are merged and scanned line by line: ``on_line``
    receives every line, ``on_step`` receives each parsed training step, and ``heartbeat`` is called
    at most once per ``heartbeat_interval_s``. callback failures terminate the process group before
    they are re-raised. ``silence_watchdog`` tears the group down when a child that reached the fit
    loop stops producing distinct output while the parent is idle.
    """
    child_tail = tail if tail is not None else ChildOutputTail()
    handle_line = build_verl_line_handler(
        child_tail,
        on_step=on_step,
        on_line=on_line,
        heartbeat=heartbeat,
        step_pattern=step_pattern,
        heartbeat_interval_s=heartbeat_interval_s,
        silence_watchdog=silence_watchdog,
    )
    return_code = _run_streaming_verl_subprocess(
        cmd,
        env=env,
        on_line=handle_line,
        silence_watchdog=silence_watchdog,
    )
    raise_for_classified_verl_exit(return_code, child_tail)
    if silence_watchdog is not None:
        silence_watchdog.raise_if_failed()
    return return_code


_TEARDOWN_GRACE_S = 10.0

# grace for a descendant holding stdout open after the trainer exits. observe child exit separately
# from EOF because EngineCore can retain the pipe; allow final flushing before group teardown.
_ORPHANED_PIPE_GRACE_S = 30.0


class _ChildExitWatchdog:
    """Tears the group down when the direct child exits but a descendant holds the pipe open.

    this prevents an EngineCore-held pipe from blocking teardown forever (PR #730). it arms only after
    child exit, so ordinary trainer silence remains ``VerlChildSilenceWatchdog``'s responsibility.
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
    except OSError:
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
        except OSError:
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


# re-exported so the capability probe and the checkpoint exporter stay reachable as
# `backend_common.<name>`: the trainers and the tests both import them from here.
from flash.engine.worker.verl.capabilities import (  # noqa: E402,F401
    _CAPABILITIES_UNAVAILABLE,
    _CAPABILITY_PROBE,
    _CAPABILITY_PROBE_TIMEOUT_S,
    actor_fsdp_strategy_overrides,
    gdn_probe_module,
    gdn_reset_arch_from_caps,
    probe_verl_capabilities,
    require_gdn_boundary_resets,
    resolve_blackwell_attention_backends,
    resolve_rollout_enforce_eager,
    resolve_verl_loggers,
    resolve_verl_python,
    rollout_fp8_kv,
    rollout_layered_summon_overrides,
    rollout_mm_processor_cache_overrides,
    rollout_resident_overrides,
    rollout_sleep_unsupported,
    strict_gdn_probe_module,
    trainer_dtype_overrides,
    verl_declares_rollout_field,
    verl_device_capability,
)
from flash.engine.worker.verl.checkpoints import (  # noqa: E402,F401
    completed_checkpoint_step,
    export_peft_adapter,
    latest_global_step_dir,
    resolve_checkpoint_actor_dir,
    stage_verl_resume,
    stamp_adapter_dir_provenance,
    undiscovered_checkpoint_dirs,
)
from flash.engine.worker.verl.child_io import (  # noqa: E402,F401
    _VERL_METRIC_FIELDS,
    FLASH_CUDART_STUB_MARKER,
    FLASH_GDN_VARLEN_MARKER,
    FLASH_TF32_MARKER,
    FLASH_WANDB_LINK_MARKER,
    SHIM_FRAGMENT_FAILED_EXIT_CODE,
    append_step_metrics,
    parse_verl_metric,
    parse_verl_step_metrics,
    parse_wandb_link,
    read_applied_shim_markers,
    render_sitecustomize_bootstrap,
    render_tf32_shim,
    render_tilelang_cudart_shim,
    shim_marker_file,
    verify_applied_shim_markers,
    verl_step_number,
)

# re-exported so the child-tail and ray-log diagnostics stay reachable as
# `backend_common.<name>`: the trainers and the tests both read them from here.
from flash.engine.worker.verl.diagnostics import (  # noqa: E402,F401
    _CHILD_TAIL_LINE_CHARS,
    CHILD_TAIL_LINES,
    RAY_FAILURE_LOGS,
    RAY_LOG_TAIL_BYTES,
    STALL_TAIL_LINES,
    ChildOutputTail,
    ChildTailStaleness,
    VerlChildSilenceWatchdog,
    build_verl_line_handler,
    collect_ray_failure_logs,
    latest_ray_session_dir,
    raise_for_classified_verl_exit,
    stall_tail_fields,
)
