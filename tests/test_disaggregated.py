"""Pure command/env builders for the disaggregated (multi-GPU async) GRPO rollout server."""

from __future__ import annotations

import pytest

from flash.engine.disaggregated import (
    DEFAULT_GROUP_PORT,
    TRL_VLLM_SERVE_FLAGS,
    build_accelerate_launch_cmd,
    build_vllm_serve_cmd,
    detect_total_gpus,
    is_main_rank,
    server_subprocess_env,
    sm120_vllm_backend,
    trainer_cuda_visible_devices,
    trainer_only_mode,
)
from flash.engine.rollout_bench import select_rollout_split


def test_build_cmd_defaults_to_tensor_parallel():
    # DEFAULT parallel="tp": shard the model across the inference GPUs so decode gets their
    # aggregate HBM bandwidth (the lever that makes 1:2 / 1:3 clear the rollout bottleneck). Works
    # for dense AND MoE. tensor_parallel_size == infer_gpus, no data_parallel_size emitted.
    split = select_rollout_split(3, 2)  # 1 train : 2 infer
    cmd = build_vllm_serve_cmd("some/model", split, max_model_len=1024, port=8000)
    assert cmd[cmd.index("--tensor_parallel_size") + 1] == "2"
    assert "--data_parallel_size" not in cmd
    assert cmd[cmd.index("--model") + 1] == "some/model"
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert cmd[cmd.index("--max_model_len") + 1] == "1024"
    assert cmd[0:2] == ["trl", "vllm-serve"]


def test_build_cmd_data_parallel_replicas_for_moe():
    # parallel="dp": replicas (data_parallel_size == infer_gpus, tensor_parallel_size == 1). vLLM
    # only accepts offline DP for MoE models, so this is reserved for the 35B-A3B.
    split = select_rollout_split(3, 2)  # 1 train : 2 infer
    cmd = build_vllm_serve_cmd("moe/model", split, max_model_len=1024, port=8000, parallel="dp")
    assert cmd[cmd.index("--data_parallel_size") + 1] == "2"
    assert cmd[cmd.index("--tensor_parallel_size") + 1] == "1"


def test_build_cmd_never_emits_unsupported_quant_flags():
    # `trl vllm-serve` (TRL 1.6) has NO --quantization/--load_format option; emitting them makes
    # HfArgumentParser reject the args and the server launch fail. A QLoRA model is served bf16 on
    # its own inference card in disaggregated mode (the trainer keeps the 4-bit base separately).
    for q in ("4bit-qlora", "bf16"):
        cmd = build_vllm_serve_cmd("m/x", select_rollout_split(2, 1), max_model_len=2048, port=8000, quant=q)
        assert "--quantization" not in cmd
        assert "--load_format" not in cmd


def test_build_cmd_only_emits_trl_supported_flags():
    # Guard against the exact blocker the verify pass caught: every --flag must be a real TRL
    # ScriptArguments field, or `trl vllm-serve` rejects the launch. Covers bf16 + qlora + extras.
    for q in ("bf16", "4bit-qlora"):
        cmd = build_vllm_serve_cmd("m/x", select_rollout_split(3, 2), max_model_len=1024, port=9000, quant=q)
        flags = [tok for tok in cmd[2:] if tok.startswith("--")]  # skip the 'trl vllm-serve' prefix
        unknown = [f for f in flags if f not in TRL_VLLM_SERVE_FLAGS]
        assert not unknown, f"unsupported trl vllm-serve flags emitted: {unknown}"


def test_build_cmd_respects_trl_bin_and_extra():
    split = select_rollout_split(2, 1)
    cmd = build_vllm_serve_cmd(
        "m", split, max_model_len=512, port=9001, trl_bin="/opt/trl", extra=["--enforce_eager", "true"]
    )
    assert cmd[0] == "/opt/trl"
    assert cmd[-2:] == ["--enforce_eager", "true"]


def test_server_env_pins_inference_to_device_zero():
    # 4-GPU node, 1 infer : 3 train -> server pinned to device 0 (valid NVML index), trainer 1,2,3
    split = select_rollout_split(4, 1)
    env = server_subprocess_env({"PATH": "/usr/bin", "FOO": "bar"}, split)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["FOO"] == "bar"  # base env preserved
    assert trainer_cuda_visible_devices(split) == "1,2,3"


def test_server_env_is_a_copy():
    split = select_rollout_split(2, 1)
    base = {"CUDA_VISIBLE_DEVICES": "7"}
    env = server_subprocess_env(base, split)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"  # server gets device 0
    assert base["CUDA_VISIBLE_DEVICES"] == "7"  # caller's base dict untouched
    assert trainer_cuda_visible_devices(split) == "1"  # trainer gets the rest


def test_server_env_forces_spawn():
    # vLLM nested-multiprocessing fork after CUDA/NVML init -> NVMLError_InvalidArgument on model
    # inspection; the server env must force spawn so each child gets a clean NVML state.
    env = server_subprocess_env({}, select_rollout_split(2, 1))
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_server_env_overrides_stale_multiproc_method():
    # Re-review guard: an incompatible value carried in base_env (e.g. `fork` from the host/image or
    # the [worker_env] TOML stringification) must be OVERWRITTEN, not preserved (setdefault would
    # keep it and reintroduce the NVML-corruption fork crash this guard exists to prevent).
    env = server_subprocess_env(
        {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}, select_rollout_split(2, 1)
    )
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_server_env_multi_infer():
    split = select_rollout_split(3, 2)  # infer (0,1), train (2,)
    env = server_subprocess_env({}, split)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert trainer_cuda_visible_devices(split) == "2"


def test_server_env_overwrites_blank_group_port(monkeypatch):
    # The [worker_env] TOML stringification can leave FLASH_VLLM_GROUP_PORT as a blank/whitespace
    # string. The trainer side resolves it via group_port() -> sanitized default, so the server env
    # MUST be forced to the same sanitized value (NOT setdefault, which would keep the blank value
    # and break the NCCL weight-sync rendezvous between trainer and server).
    monkeypatch.delenv("FLASH_VLLM_GROUP_PORT", raising=False)
    split = select_rollout_split(2, 1)
    for blank in ("", "   ", "not-a-port"):
        env = server_subprocess_env({"FLASH_VLLM_GROUP_PORT": blank}, split)
        assert env["FLASH_VLLM_GROUP_PORT"] == str(DEFAULT_GROUP_PORT)


def test_server_env_honours_valid_group_port_override(monkeypatch):
    # A VALID operator override must reach the server unchanged (and matches the trainer's group_port).
    monkeypatch.setenv("FLASH_VLLM_GROUP_PORT", "51999")
    env = server_subprocess_env({"FLASH_VLLM_GROUP_PORT": "51999"}, select_rollout_split(2, 1))
    assert env["FLASH_VLLM_GROUP_PORT"] == "51999"


def _fake_nvidia_smi_cap(caps: list[str]):
    """A subprocess.run stub that mimics `nvidia-smi --query-gpu=compute_cap` for the given caps."""

    class _Out:
        stdout = "\n".join(caps)

    def _run(cmd, *a, **k):
        return _Out()

    return _run


def test_sm120_backend_returns_flashinfer_on_blackwell(monkeypatch):
    # A 5090 (compute_cap 12.0) must get FLASHINFER so the disaggregated server doesn't boot the
    # flash-attn PTX backend that silently produces empty rollouts on consumer Blackwell hosts.
    import flash.engine.disaggregated as _d

    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(_d.subprocess, "run", _fake_nvidia_smi_cap(["12.0"]))
    assert sm120_vllm_backend({}) == "FLASHINFER"


def test_sm120_backend_none_off_blackwell(monkeypatch):
    import flash.engine.disaggregated as _d

    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(_d.subprocess, "run", _fake_nvidia_smi_cap(["9.0"]))  # Hopper
    assert sm120_vllm_backend({}) is None


def test_sm120_backend_honours_operator_override(monkeypatch):
    # If a backend is already pinned (operator or caller), never overwrite it — even on a 5090.
    import flash.engine.disaggregated as _d

    monkeypatch.setattr(_d.subprocess, "run", _fake_nvidia_smi_cap(["12.0"]))
    assert sm120_vllm_backend({"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"}) is None


def test_sm120_backend_best_effort_when_probe_fails(monkeypatch):
    import flash.engine.disaggregated as _d

    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi missing")

    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(_d.subprocess, "run", _boom)
    assert sm120_vllm_backend({}) is None


def test_server_env_pins_flashinfer_on_sm120(monkeypatch):
    # The disaggregated server env must carry FLASHINFER on a 5090 (CUDA-free probe), mirroring the
    # colocate path's force_vllm_backend_for_sm120 but without creating a CUDA context in the launcher.
    import flash.engine.disaggregated as _d

    monkeypatch.setattr(_d.subprocess, "run", _fake_nvidia_smi_cap(["12.0", "12.0"]))
    env = server_subprocess_env({}, select_rollout_split(2, 1))
    assert env["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"


def test_server_env_no_backend_off_sm120(monkeypatch):
    import flash.engine.disaggregated as _d

    monkeypatch.setattr(_d.subprocess, "run", _fake_nvidia_smi_cap(["8.0"]))  # Ampere
    env = server_subprocess_env({}, select_rollout_split(2, 1))
    assert "VLLM_ATTENTION_BACKEND" not in env


def test_detect_total_gpus_prefers_explicit_env():
    assert detect_total_gpus({"FLASH_GPU_COUNT": "4"}) == 4
    assert detect_total_gpus({"FLASH_GPU_COUNT": "1"}) == 1
    # a garbage value falls through (does not crash); without nvidia-smi/torch it returns >=1
    assert detect_total_gpus({"FLASH_GPU_COUNT": "not-a-number"}) >= 1


def _fake_nvidia_smi_l(n: int):
    """A subprocess.run stub that mimics `nvidia-smi -L` listing ``n`` GPUs."""

    class _Out:
        stdout = "\n".join(f"GPU {i}: NVIDIA Whatever (UUID: GPU-{i})" for i in range(n))

    def _run(cmd, *a, **k):
        return _Out()

    return _run


def test_detect_total_gpus_caps_to_hint_when_overexposed(monkeypatch):
    # The host exposes MORE GPUs than [gpu].count sized/billed (e.g. a shared 8-GPU box rented as 2).
    # The split topology was fixed at submit time, so HONOR the requested count and ignore the extra
    # cards — never silently expand into a 7:1 DDP run the experiment was not sized for.
    import flash.engine.disaggregated as _d

    monkeypatch.setattr(_d.subprocess, "run", _fake_nvidia_smi_l(8))
    assert detect_total_gpus({"FLASH_GPU_COUNT": "2"}) == 2


def test_detect_total_gpus_uses_real_when_underprovisioned(monkeypatch):
    # The provider UNDER-provisioned (fewer visible GPUs than rented, observed on Vast): use the real
    # count — the missing devices don't exist, so assigning them would target a nonexistent card.
    import flash.engine.disaggregated as _d

    monkeypatch.setattr(_d.subprocess, "run", _fake_nvidia_smi_l(1))
    assert detect_total_gpus({"FLASH_GPU_COUNT": "2"}) == 1


def test_detect_total_gpus_matches_real_when_hint_agrees(monkeypatch):
    import flash.engine.disaggregated as _d

    monkeypatch.setattr(_d.subprocess, "run", _fake_nvidia_smi_l(2))
    assert detect_total_gpus({"FLASH_GPU_COUNT": "2"}) == 2


def test_accelerate_launch_cmd_2to1():
    # 2:1 split (2 trainer GPUs : 1 infer). accelerate launches one rank per TRAIN GPU, pinned to
    # the global train device ids, FSDP-sharded, running the worker module.
    split = select_rollout_split(3, 1)  # total 3, infer 1 -> train_gpus=2
    # FSDP is opt-in (default is the shipped DDP path); pass use_fsdp=True to exercise this branch.
    cmd = build_accelerate_launch_cmd(split, use_fsdp=True)
    assert cmd[0:2] == ["accelerate", "launch"]
    # FSDP implies multi-GPU; accelerate REJECTS --multi_gpu alongside --use_fsdp, so it must be absent
    assert "--multi_gpu" not in cmd
    assert "--use_fsdp" in cmd
    assert cmd[cmd.index("--num_processes") + 1] == "2"  # == train_gpus
    # train devices are the GLOBAL ids after the infer device(s); select_rollout_split puts infer
    # first, so train devices are 1,2 for a 3-GPU node with 1 infer GPU.
    assert cmd[cmd.index("--gpu_ids") + 1] == "1,2"
    assert cmd[-2:] == ["-m", "flash.engine.worker"]


def test_accelerate_launch_cmd_no_fsdp():
    # DDP path: --multi_gpu present, no FSDP flags.
    split = select_rollout_split(3, 1)
    cmd = build_accelerate_launch_cmd(split, use_fsdp=False)
    assert "--multi_gpu" in cmd
    assert "--use_fsdp" not in cmd
    assert "--fsdp_sharding_strategy" not in cmd


def test_is_main_rank_defaults_true_without_rank():
    assert is_main_rank({}) is True  # colocate / single-GPU: no RANK -> main
    assert is_main_rank({"RANK": "0"}) is True
    assert is_main_rank({"RANK": "1"}) is False


def test_trainer_only_mode_flag():
    assert trainer_only_mode({}) is False
    assert trainer_only_mode({"FLASH_RL_TRAINER_ONLY": "1"}) is True
    assert trainer_only_mode({"FLASH_RL_TRAINER_ONLY": "0"}) is False


# ---------------------------------------------------------------------------
# wait_for_server_health boot-heartbeat timing
# ---------------------------------------------------------------------------


def _scripted_clock(monkeypatch, values):
    """Patch disaggregated's time.time to return successive ``values`` (last one repeats), and
    no-op its sleep. Lets a test place the deadline crossing at an exact point in the loop."""
    from flash.engine import disaggregated as d

    seq = list(values)
    idx = {"i": 0}

    def _time():
        i = idx["i"]
        v = seq[i] if i < len(seq) else seq[-1]
        idx["i"] = i + 1
        return v

    monkeypatch.setattr(d.time, "time", _time)
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    return _time


def _ok_urlopen(url, timeout=None):
    import contextlib

    @contextlib.contextmanager
    def _cm():
        class _R:
            status = 200

        yield _R()

    return _cm()


def test_wait_for_server_health_fires_first_ping_promptly(monkeypatch):
    # The first on_wait (boot heartbeat) must fire at loop ENTRY, not one full on_wait_every later,
    # so a poller whose stall clock is already near its limit still sees an rl_server_boot quickly.
    from flash.engine import disaggregated as d

    # Reads: deadline-base(0), _next_ping(0), while-cond(1), ping-gate(1), ... server healthy 1st probe.
    _scripted_clock(monkeypatch, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    pings = []
    monkeypatch.setattr(d.urllib.request, "urlopen", _ok_urlopen)

    d.wait_for_server_health(
        8000,
        timeout=600.0,
        on_wait=lambda: pings.append(True),
        on_wait_every=60.0,
    )
    # The server is healthy on the first probe; the ping must already have fired before it because
    # _next_ping is seeded to loop-entry time (not entry + on_wait_every).
    assert pings, "first boot heartbeat did not fire promptly at loop entry"


def test_wait_for_server_health_deadline_checked_after_on_wait(monkeypatch):
    # A blocking on_wait (slow HF upload) must not let the loop overrun the deadline: after on_wait
    # returns, the deadline is re-checked, so a timeout is raised rather than an extra HTTP probe.
    from flash.engine import disaggregated as d

    # Reads in order: deadline-base(0)->deadline=10; _next_ping(0); while-cond(5)<10 enter loop;
    # ping-gate(5)>=0 fire on_wait; post-on_wait poll has no proc; deadline re-check(20)>=10 -> break
    # -> raise TimeoutError WITHOUT ever probing HTTP. If the re-check were missing, urlopen would run.
    _scripted_clock(monkeypatch, [0.0, 0.0, 5.0, 5.0, 20.0])

    def _boom_urlopen(url, timeout=None):
        raise AssertionError("HTTP probe ran; deadline was not re-checked after on_wait")

    monkeypatch.setattr(d.urllib.request, "urlopen", _boom_urlopen)

    with pytest.raises(TimeoutError):
        d.wait_for_server_health(
            8000,
            timeout=10.0,
            on_wait=lambda: None,
            on_wait_every=60.0,
        )
