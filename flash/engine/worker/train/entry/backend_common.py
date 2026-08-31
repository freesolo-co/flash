"""shared out-of-process verl harness reused by the sft and opd verl trainers.

verl pins its own torch/vllm, incompatible with flash's, so flash never imports verl in-process: it
runs a verl trainer entrypoint as a subprocess against a separate interpreter and merges the
fsdp-sharded lora checkpoint back into a flash-servable peft adapter. these are the framework- and
algorithm-neutral pieces (interpreter resolution, checkpoint export, provenance, progress
streaming); the sft/opd/grpo modules layer their own dataset rows, hydra overrides, and
orchestration on top.
"""

from __future__ import annotations

import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as _thread_module
from http.server import ThreadingHTTPServer

# active callers still import these package-install contracts from backend_common. the owning
# implementation is the neutral verl leaf so capability imports cannot cycle through this facade.
from flash.engine.worker.verl.install import (  # noqa: F401
    CAUSAL_CONV1D_REQUIREMENT,
    FLA_REQUIREMENT,
    FLASH_ATTN_INSTALL_ATTEMPTS,
    FLASH_ATTN_INSTALL_BACKOFF_S,
    FLASH_ATTN_INSTALL_SPEC,
    FLASH_ATTN_SHA256,
    FLASH_ATTN_SPEC,
    FLASH_QLA_REQUIREMENT,
    TRANSFORMERS_INSTALL_REQUIREMENT,
    TRANSFORMERS_REQUIREMENT,
    VERL_REQUIREMENT,
    VERL_REQUIREMENT_NAME,
    VERL_REQUIREMENT_URL,
    VERL_VENV_BUILD_REPAIRS,
    VERL_VENV_PYTHON,
    VERL_VENV_STAMP,
    _install_causal_conv1d,
    _install_flash_attn,
    resolve_verl_python,
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


# direct aliases keep the historical backend_common surface while the neutral leaf owns execution.
# re-exported so the capability probe and the checkpoint exporter stay reachable as
# `backend_common.<name>`: the trainers and the tests both import them from here.
from flash.engine.worker.verl.capabilities import (  # noqa: E402,F401
    _CAPABILITIES_UNAVAILABLE,
    _CAPABILITY_PROBE,
    _CAPABILITY_PROBE_TIMEOUT_S,
    actor_fsdp_strategy_overrides,
    fused_ce_backend,
    gdn_probe_module,
    gdn_reset_arch_from_caps,
    probe_verl_capabilities,
    require_gdn_boundary_resets,
    resolve_blackwell_attention_backends,
    resolve_rollout_enforce_eager,
    resolve_verl_loggers,
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
from flash.engine.worker.verl.process import (  # noqa: E402,F401
    _ADOPTS_ORPHANS,
    _EXIT_DRAIN_S,
    _ORPHANED_PIPE_GRACE_S,
    _TEARDOWN_GRACE_S,
    _UNREAPED_STRAGGLERS,
    _ChildExitWatchdog,
    _drain_stragglers_before_exit,
    _process_group_addressable,
    _process_group_alive,
    _process_group_members,
    _process_is_zombie,
    _reap,
    _reap_group_zombies,
    _run_streaming_verl_subprocess,
    adopt_orphaned_descendants,
    kill_process_group,
    reap_stragglers,
    run_verl_training,
)
