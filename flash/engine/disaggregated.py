"""Disaggregated (multi-GPU async) GRPO rollout: launch a ``trl vllm-serve`` rollout server on
the inference GPUs and point the GRPO trainer at it (``vllm_mode="server"``), so generation for
the next batch overlaps the current optimizer step instead of time-sharing one GPU.

verl ref (3D-HybridEngine / flexible device mapping + async rollout):
https://github.com/verl-project/verl

The command/env builders here are PURE (no torch, no subprocess) so the launch contract is
unit-testable on CPU; only :func:`launch_vllm_server` / :func:`wait_for_server_health` /
:func:`detect_total_gpus` touch the system. The device split itself lives in
:mod:`flash.engine.rollout_bench` (``select_rollout_split``).

TRL 1.6 server-mode contract (verified against trl/scripts/vllm_serve.py + trl/generation/
vllm_client.py): the trainer's ``VLLMClient`` waits for the server (``vllm_server_timeout``),
opens a NCCL weight-sync group on ``vllm_group_port`` (default 51216), and on every generation
batch ``sync_weights()`` POSTs ``/update_named_param/`` then broadcasts the (PEFT-merged) weights
over NCCL — so weight sync is automatic; we only have to bring the server up first.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
import urllib.error
import urllib.request

from flash.engine.rollout_bench import RolloutSplit

# Loopback HTTP port the rollout server binds (trainer connects via vllm_server_base_url) and the
# NCCL weight-sync group port. Single-node (all GPUs on one rented instance) -> loopback, so no
# firewall concerns. Both overridable for the rare port clash.
DEFAULT_SERVER_PORT = 8000
DEFAULT_GROUP_PORT = 51216
# vLLM rollout server gets the whole inference card (no trainer sharing it, unlike colocate's ~0.45).
DEFAULT_SERVER_GPU_UTIL = 0.90


def _port_env(name: str, default: int) -> int:
    """Operator port override, tolerant of the ``[worker_env]`` stringification quirk.

    A blank / whitespace / non-numeric value (common when env is populated via TOML
    ``[worker_env]``) falls back to ``default`` with a warning rather than raising a
    ``ValueError`` that would crash the worker at setup time.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        print(f"[rl][disagg] WARNING: {name}={raw!r} is not an int — using default {default}")
        return default


def server_port() -> int:
    return _port_env("FLASH_VLLM_SERVER_PORT", DEFAULT_SERVER_PORT)


def group_port() -> int:
    return _port_env("FLASH_VLLM_GROUP_PORT", DEFAULT_GROUP_PORT)


def server_base_url(port: int | None = None) -> str:
    return f"http://127.0.0.1:{port or server_port()}"


def detect_total_gpus(env: dict | None = None) -> int:
    """Number of GPUs physically on this node, WITHOUT initializing a torch CUDA context.

    ``torch.cuda.device_count()`` is avoided on the hot path because the disaggregated split must
    set ``CUDA_VISIBLE_DEVICES`` on the trainer process *before* any CUDA context is created;
    querying torch here could bind the context to all visible devices first. Prefer the explicit
    ``FLASH_GPU_COUNT`` the provisioner sets (= ``[gpu] count``), else count ``nvidia-smi -L``.
    """
    env = env if env is not None else os.environ
    # GROUND TRUTH = the GPUs actually visible to THIS container (`nvidia-smi -L`, CUDA-free, honors
    # the container's NVIDIA_VISIBLE_DEVICES). The provisioner's FLASH_GPU_COUNT (= [gpu] count)
    # is only the REQUESTED count — a provider can rent a 2-GPU offer yet expose 1 GPU to the
    # container (observed on Vast), so trusting it would make the split assign a nonexistent device.
    # We take the real count and only fall back to the hint when nvidia-smi is unavailable.
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        # Raw container GPU/visibility diagnostic — definitively shows what the container sees and
        # whether an env var (CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES) is hiding GPUs vs the
        # provider genuinely exposing fewer. Printed once at split time.
        print(
            f"[rl][disagg][diag] nvidia-smi -L:\n{out.stdout.strip()}\n"
            f"[rl][disagg][diag] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')!r} "
            f"NVIDIA_VISIBLE_DEVICES={env.get('NVIDIA_VISIBLE_DEVICES')!r} "
            f"FLASH_GPU_COUNT={env.get('FLASH_GPU_COUNT')!r}"
        )
        lines = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("GPU ")]
        if lines:
            real = len(lines)
            hint = env.get("FLASH_GPU_COUNT")
            if hint and hint.isdigit() and int(hint) != real:
                hint_n = int(hint)
                if real < hint_n:
                    # UNDER-provisioned: the provider exposed fewer GPUs than rented (observed on
                    # Vast). Use the real count — the missing devices don't exist, so the requested
                    # split is impossible; run_rl's select_rollout_split then raises if the smaller
                    # node can't satisfy inference_gpus, with a clear message.
                    print(
                        f"[rl][disagg] WARNING: [gpu] count requested {hint} but the container "
                        f"exposes {real} GPU(s) (nvidia-smi) — using {real}. The provider "
                        f"under-provisioned the node."
                    )
                    return real
                # OVER-exposed: the host shows MORE GPUs than the allocator sized/billed (e.g. a
                # shared 8-GPU box rented as 2). The topology — world size, effective batch, and the
                # split — was fixed at submit time from [gpu].count, so HONOR the requested count and
                # ignore the extra devices rather than silently expanding into GPUs the experiment
                # was not sized for (a 7:1 DDP run instead of the intended 1:1). trainer_devices /
                # infer_devices index the FIRST GPUs, so capping is safe.
                print(
                    f"[rl][disagg] WARNING: the container exposes {real} GPU(s) (nvidia-smi) but "
                    f"[gpu] count requested {hint} — using {hint_n} (the sized/billed topology); "
                    f"ignoring the extra device(s)."
                )
                return hint_n
            return real
    except Exception:
        pass
    explicit = env.get("FLASH_GPU_COUNT")
    if explicit and explicit.isdigit() and int(explicit) >= 1:
        return int(explicit)
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 1


def _cvd(devices: tuple[int, ...]) -> str:
    """CUDA_VISIBLE_DEVICES string from GLOBAL physical device indices."""
    return ",".join(str(d) for d in devices)


def trainer_cuda_visible_devices(split: RolloutSplit) -> str:
    """The CUDA_VISIBLE_DEVICES the trainer process must use (its train devices, global indices)."""
    return _cvd(split.train_devices)


def build_vllm_serve_cmd(
    model_id: str,
    split: RolloutSplit,
    *,
    max_model_len: int,
    port: int,
    gpu_memory_util: float = DEFAULT_SERVER_GPU_UTIL,
    quant: str = "bf16",
    trl_bin: str | None = None,
    trust_remote_code: bool = True,
    enable_prefix_caching: bool = True,
    kv_cache_dtype: str | None = None,
    parallel: str = "tp",
    extra: list[str] | None = None,
) -> list[str]:
    """The ``trl vllm-serve`` argv for the rollout server.

    PARALLELISM across the inference GPUs (after CUDA_VISIBLE_DEVICES pins the server to
    ``split.infer_devices``, those cards re-index to 0..infer_gpus-1):

    * ``parallel="tp"`` (DEFAULT) -> ``--tensor_parallel_size infer_gpus``: shard ONE model across
      the inference GPUs. For a generation-bound GRPO step the decode phase is memory-bandwidth
      bound, so TP gives ~the aggregate HBM bandwidth of all inference cards (weights + KV split
      across them) -> larger concurrent batches + faster decode, which is how higher inference
      ratios (1:2, 1:3) clear the rollout bottleneck. Works for DENSE and MoE alike.
    * ``parallel="dp"`` -> ``--data_parallel_size infer_gpus --tensor_parallel_size 1``: each
      inference GPU is a FULL replica and the server load-balances across them. **vLLM REJECTS
      offline data-parallel for DENSE models** ("Offline data parallel mode is not supported/useful
      for dense models" — ParallelConfig validation), so this is ONLY usable for the MoE model
      (Qwen3.6-35B-A3B), where DP shards experts. Do NOT default to it: every dense model in the
      catalog (MiniCPM, the Qwen3.5 dense line) must use TP.

    ``max_model_len`` bounds the KV cache to the GRPO need (prompt+completion) so the server starts
    on a consumer card instead of sizing for the model's full context.

    Only flags that ``trl vllm-serve``'s ScriptArguments actually defines are emitted (verified
    against trl/scripts/vllm_serve.py): model, tensor_parallel_size, data_parallel_size, host, port,
    gpu_memory_utilization, max_model_len, enable_prefix_caching, trust_remote_code. The CLI has NO
    quantization/load_format option, so a ``4bit-qlora`` model is served bf16/auto on its OWN
    inference card (in disaggregated mode the inference GPU holds ONLY the rollout, not a colocated
    trainer, so the full-precision weights fit a big enough card). The trainer still loads the base
    4-bit (its model_init_kwargs are independent); TRL merges the LoRA into bf16 before each weight
    sync, so trainer-4-bit + server-bf16 stays consistent. ``quant`` is accepted for signature
    stability / sizing notes but does not change the server command.
    """
    bin_ = trl_bin or os.environ.get("FLASH_TRL_BIN", "trl")
    if parallel == "dp":
        # DP replicas: server load-balances generation across infer_gpus full copies. vLLM ONLY
        # allows offline DP for MoE models -> reserve this for the 35B-A3B; dense models must use TP.
        par_flags = [
            "--data_parallel_size",
            str(split.infer_gpus),
            "--tensor_parallel_size",
            "1",
        ]
    else:
        # TP (DEFAULT): shard one model across the inference GPUs -> aggregate HBM bandwidth for a
        # faster, larger-batch decode (the win for higher ratios on a generation-bound step).
        par_flags = ["--tensor_parallel_size", str(split.infer_gpus)]
    cmd = [
        bin_,
        "vllm-serve",
        "--model",
        model_id,
        *par_flags,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu_memory_utilization",
        str(gpu_memory_util),
        "--max_model_len",
        str(max_model_len),
    ]
    if trust_remote_code:
        cmd += ["--trust_remote_code", "true"]
    # verl-validated rollout-engine levers (mirror the colocate path): PREFIX CACHING reuses the
    # shared GRPO prompt-prefix KV across a group's completions (the dominant rollout win); fp8 KV
    # CACHE (Ada/Hopper/Blackwell, compute capability >= 8.9) ~halves the KV pool so a bigger
    # generation batch fits the inference card. CUDA graphs are on by default (enforce_eager unset).
    if enable_prefix_caching:
        cmd += ["--enable_prefix_caching", "true"]
    if kv_cache_dtype:
        cmd += ["--kv_cache_dtype", kv_cache_dtype]
    if extra:
        cmd += list(extra)
    return cmd


# The flags `trl vllm-serve` (TRL 1.6 ScriptArguments) actually accepts. HfArgumentParser REJECTS
# any unknown --flag, failing server launch — so build_vllm_serve_cmd must only emit these (see the
# allowlist test). Mirrors trl/scripts/vllm_serve.py:ScriptArguments.
TRL_VLLM_SERVE_FLAGS = frozenset(
    {
        "--model",
        "--revision",
        "--tensor_parallel_size",
        "--data_parallel_size",
        "--host",
        "--port",
        "--gpu_memory_utilization",
        "--dtype",
        "--max_model_len",
        "--enable_prefix_caching",
        "--enforce_eager",
        "--kv_cache_dtype",
        "--trust_remote_code",
        "--log_level",
        "--vllm_model_impl",
        "--distributed_executor_backend",
        "--speculative_config",
    }
)


def build_accelerate_launch_cmd(
    split: RolloutSplit,
    *,
    worker_module: str = "flash.engine.worker",
    mixed_precision: str = "bf16",
    use_fsdp: bool = True,
) -> list[str]:
    """``accelerate launch`` argv that runs the GRPO trainer across the TRAIN devices.

    For a train_gpus>1 ratio (2:1, 3:1, 2:2) the trainer must be a *distributed* process group, not
    a single process spanning many GPUs (that silently becomes nn.DataParallel, which TRL's
    weight-sync gather does not support). The disaggregated launcher brings the vLLM rollout server
    up on the inference GPUs first, then re-execs the worker's RL phase under this command so
    ``accelerate`` runs the trainer across ``split.train_gpus`` and the trainer connects to the
    already-running server (``FLASH_RL_TRAINER_ONLY=1`` in the child env).

    ``--gpu_ids`` pins the child group to the GLOBAL train device indices (so it never touches the
    inference card), and ``--num_processes`` == train_gpus (one rank per train GPU).

    SHIPPED PATH: the worker calls this with ``use_fsdp=False`` (DDP — replicate the policy per
    rank), because TRL's per-step LoRA-merge weight sync (``merge_adapter()`` ->
    ``get_delta_weight``) breaks under FSDP param sharding. Every model that runs a multi-trainer
    ratio (1-9B) fits replicated on one trainer card, so DDP covers all the 2:1/3:1/2:2 ratios; the
    only model that wouldn't (35B) takes the single-trainer 1:1 path. The ``use_fsdp=True`` branch
    (FSDP SHARD_GRAD_OP, for a base too big to replicate) is kept for a future TRL patch that gathers
    LoRA before the merge, but is not currently exercised by the launcher.
    """
    cmd = [
        "accelerate",
        "launch",
        "--num_machines",
        "1",
        "--num_processes",
        str(split.train_gpus),
        "--gpu_ids",
        _cvd(split.train_devices),
        "--mixed_precision",
        mixed_precision,
        "--dynamo_backend",
        "no",
    ]
    if use_fsdp:
        # FSDP full-shard with transformer auto-wrap: shards params+grads+optimizer across ranks so
        # the trainer scales past one card. CPU-offload off (the train GPUs have room; offload would
        # tank throughput). NB: `--use_fsdp` already implies multi-GPU — accelerate REJECTS
        # `--multi_gpu` alongside it ("You can only use one of --cpu/--multi_gpu/--use_fsdp ..."), so
        # we must NOT pass --multi_gpu here. These are accelerate-launch FSDP flags (accelerate>=1.4).
        cmd += [
            "--use_fsdp",
            # SHARD_GRAD_OP (ZeRO-2: shard gradients+optimizer, REPLICATE parameters) — NOT FULL_SHARD.
            # TRL's per-step weight sync calls peft merge_adapter() -> get_delta_weight() ->
            # weight_B @ weight_A; under FULL_SHARD the LoRA weights are param-sharded so each rank
            # holds a slice and the matmul fails ("inconsistent tensor size [32768] vs [24576]").
            # SHARD_GRAD_OP keeps params whole on every rank so the merge sees full LoRA weights, while
            # still sharding the optimizer/grads. Fine for the dense 1-9B bases (replicated base fits);
            # a base too big to replicate (35B) would need FULL_SHARD + a TRL patch that gathers LoRA
            # before merge — out of scope here.
            "--fsdp_sharding_strategy",
            "SHARD_GRAD_OP",
            "--fsdp_auto_wrap_policy",
            "TRANSFORMER_BASED_WRAP",
            # use_orig_params keeps the original (un-flattened) parameter tensors so peft can read
            # weight_A/weight_B with their real shapes during the merge.
            "--fsdp_use_orig_params",
            "true",
            # FULL_STATE_DICT: transformers' Trainer rejects save_only_model (set by GRPOConfig)
            # alongside SHARDED_STATE_DICT; FULL gathers the small LoRA adapter on rank 0 at save time.
            "--fsdp_state_dict_type",
            "FULL_STATE_DICT",
        ]
    else:
        # Plain DDP across the train GPUs (replicate, no sharding) — needs the explicit --multi_gpu.
        cmd.insert(2, "--multi_gpu")
    cmd += ["-m", worker_module]
    return cmd


def is_main_rank(env: dict | None = None) -> bool:
    """True on the rank that must write artifacts (adapter/metrics/DONE/upload).

    Single-process paths (colocate, train_gpus==1) set no RANK -> "0" -> main, so existing behavior
    is unchanged; under ``accelerate launch`` only RANK 0 writes, avoiding N concurrent HF uploads.
    """
    env = env if env is not None else os.environ
    return str(env.get("RANK", "0")) == "0"


def trainer_only_mode(env: dict | None = None) -> bool:
    """True inside the accelerate-launched trainer child (server already up; skip the launcher)."""
    env = env if env is not None else os.environ
    return str(env.get("FLASH_RL_TRAINER_ONLY", "")) in ("1", "true", "True")


def server_subprocess_env(base_env: dict, split: RolloutSplit) -> dict:
    """Env for the vllm-serve subprocess: pin it to the GLOBAL inference device indices.

    Built from a COPY so the parent (trainer) can independently set its own CUDA_VISIBLE_DEVICES to
    the train devices without affecting the already-launched server. The group port is exported so
    the server's weight-sync extension and the trainer's client agree on the NCCL rendezvous.
    """
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = _cvd(split.infer_devices)
    # Assign (NOT setdefault): the canonical group port is whatever group_port() resolves to, which
    # already reads + SANITIZES FLASH_VLLM_GROUP_PORT (blank / whitespace / non-numeric -> default).
    # setdefault would leave an existing INVALID value untouched (the [worker_env] stringification
    # quirk this whole module guards against), so the server would parse a blank port while the
    # trainer side uses the sanitized default -> NCCL weight-sync rendezvous mismatch. Overwriting
    # with the sanitized value keeps both sides in lockstep AND still honours a valid operator
    # override (group_port() returns it unchanged).
    env["FLASH_VLLM_GROUP_PORT"] = str(group_port())
    # Force vLLM's worker/inspection subprocesses to SPAWN (not fork). `trl vllm-serve` runs its
    # llm_worker as a multiprocessing.Process and vLLM forks a further subprocess to inspect the
    # model architecture; a fork after the parent has touched CUDA/NVML corrupts the NVML handle in
    # the child -> `NVMLError_InvalidArgument` during ModelConfig inspection ("architectures failed
    # to be inspected"). Spawn gives each child a clean CUDA/NVML state. (The colocate path is
    # in-process so it never hit this.) ASSIGN (not setdefault): a stale/incompatible value carried
    # in base_env (e.g. `fork` from the host/image or the [worker_env] TOML stringification) would
    # otherwise be preserved and reintroduce the exact NVML corruption this guard prevents — the
    # same reason the group-port line above overwrites rather than setdefaults.
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return env


def _server_log_tail(log_path: str | None, n: int = 60) -> str:
    """Last ``n`` lines of the rollout server's log (its model-load crash lives here). The TRL
    server only binds its HTTP port AFTER its llm_worker child loads the model + reports ready
    (trl/scripts/vllm_serve.py lifespan), so a load crash shows as 'connection refused' on the
    port while the real error is ONLY in this log — surface it so error_rl.txt is diagnosable."""
    if not log_path:
        return ""
    try:
        with open(log_path, errors="replace") as f:
            lines = f.read().splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"(server log unreadable: {e})"


def wait_for_server_health(
    port: int,
    *,
    timeout: float,
    proc: subprocess.Popen | None = None,
    log_path: str | None = None,
    interval: float = 3.0,
    on_wait=None,
    on_wait_every: float = 60.0,
) -> None:
    """Block until the rollout server answers ``/health/`` 200, the timeout elapses, or the
    subprocess dies (fail fast — a dead server would otherwise hang the trainer at first generation).
    On any failure the server log tail is appended (the HTTP port stays unbound until the worker
    finishes loading the model, so a load/OOM crash surfaces only as 'connection refused').

    ``on_wait`` (called every ``on_wait_every`` s while waiting) lets the caller emit a liveness
    heartbeat during a long boot: a BIG model (35B bf16 ~70 GB + tilelang/CUDA-graph JIT) can take
    >20 min to bind the port, and the control plane's no-heartbeat STALL detector (~25 min) would
    otherwise kill the run mid-boot even though it is healthy-progressing."""
    url = f"http://127.0.0.1:{port}/health/"
    deadline = time.time() + timeout
    last = None
    # Fire the FIRST boot heartbeat promptly at loop entry, not one full ``on_wait_every`` in:
    # the poller's stall clock may already be near its limit after the work between rl_start and
    # server launch, so a 60s-late first ping could let it kill the run before any rl_server_boot.
    _next_ping = time.time()
    while time.time() < deadline:
        if on_wait is not None and time.time() >= _next_ping:
            with contextlib.suppress(Exception):
                on_wait()
            _next_ping = time.time() + on_wait_every
        # ``on_wait`` (the worker's heartbeat) can BLOCK on a slow Hugging Face upload, so re-check
        # liveness AND the deadline AFTER it returns: otherwise a blocking heartbeat would both mask
        # a dead vllm-serve process and let the loop overrun ``timeout`` (the while-condition only
        # re-checks at the top of the NEXT iteration).
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vllm-serve exited (code {proc.returncode}) before becoming healthy.\n"
                f"--- vllm-serve log tail ---\n{_server_log_tail(log_path)}"
            )
        if time.time() >= deadline:
            break
        try:
            with urllib.request.urlopen(url, timeout=interval) as r:
                if 200 <= r.status < 300:
                    return
        except urllib.error.HTTPError as e:
            # Only a 2xx on /health/ means ready. vLLM routes (and returns 4xx on) endpoints while
            # the engine is still loading, so a 4xx is NOT healthy — keep polling until a real 200
            # (or the timeout / a dead subprocess), rather than handing the trainer a server that
            # then fails at the first generation.
            last = f"HTTP {e.code}"
        except Exception as e:  # connection refused while still loading
            last = str(e)[:80]
        time.sleep(interval)
    raise TimeoutError(
        f"vllm-serve not healthy after {timeout:.0f}s (last: {last}).\n"
        f"--- vllm-serve log tail ---\n{_server_log_tail(log_path)}"
    )


def launch_vllm_server(
    cmd: list[str], env: dict, *, log_path: str = "/tmp/vllm_serve.log"
) -> subprocess.Popen:
    """Start the rollout server subprocess, streaming its stdout/stderr to ``log_path`` (so a
    server-side load/OOM is recoverable from the run log on a rented node)."""
    log = open(log_path, "wb")  # noqa: SIM115 — handle lives with the long-running subprocess
    print(f"[rl][disagg] launching rollout server: {' '.join(cmd)}")
    print(f"[rl][disagg] server CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')} log={log_path}")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def terminate_server(proc: subprocess.Popen | None, *, timeout: float = 15.0) -> None:
    """Best-effort shutdown of the rollout server after training (free its GPU + port)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        with contextlib.suppress(Exception):
            proc.kill()
