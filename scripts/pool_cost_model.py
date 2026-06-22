"""Cost model: shared rollout pool vs origin/dev's one-GPU-per-job (colocate), at scale.

Grounded in dev's OWN GRPO cost model (flash/cost/analytical.py + facts.py):
  per GRPO step = gen_s (vLLM decode, MFU 0.12) + reward_s (grading) + update_s (train, MFU 0.35)
  gen_s    = GEN_FLOPS(2)  * params * gen_tokens / (peak_FLOPS * MFU_DECODE)
  update_s = UPDATE_FLOPS(8)* params * gen_tokens / (peak_FLOPS * MFU_TRAIN)
  reward_s = ceil(completions / 16) * reward_latency

dev (colocate): every run rents ONE GPU that does gen + reward + update sequentially. During
reward_s the GPU is IDLE (grading is CPU/judge/sandbox), and gen runs at decode MFU on an
expensive training-class card. N runs => N expensive GPUs.

pool: the trainer GPU does ONLY update_s (gen+reward are prefetched/overlapped off it); gen runs on
a SHARED pool of cheaper, bandwidth-class inference GPUs (decode is bandwidth-bound, not compute-
bound) holding the base once + many LoRA adapters (multi-LoRA); reward runs on cheap CPU workers.
At scale, continuous batching across concurrent runs raises decode efficiency, so each inference GPU
serves more runs.

Pure stdlib; constants are dev's. Run: python scripts/pool_cost_model.py
"""

from __future__ import annotations

import math

# ---- dev cost-model constants (flash/cost/analytical.py + facts.py) ----
GEN_FLOPS, UPDATE_FLOPS = 2.0, 8.0  # per token per param
MFU_TRAIN, MFU_DECODE = 0.35, 0.12
REWARD_CONCURRENCY = 16.0

# ---- GPU facts (flash/cost/facts.py TFLOPS + flash/providers/base.py $/hr) ----
A100 = {"tflops": 312.0, "usd_hr": 1.39}   # training-class (colocate + pool trainer); A100 PCIe 80GB $/hr from flash/providers/base.py (SXM is 1.49)
A40 = {"tflops": 150.0, "usd_hr": 0.44}    # bandwidth-class inference card (pool rollout)
CPU_REWARD_USD_HR = 0.10                    # a CPU reward worker (serves REWARD_CONCURRENCY slots)


def phase_seconds(params_b, completions, resp_len, reward_latency, decode_mfu=MFU_DECODE, peak_tflops=A100["tflops"]):
    params = params_b * 1e9
    gen_tokens = completions * resp_len
    peak = peak_tflops * 1e12
    gen_s = GEN_FLOPS * params * gen_tokens / (peak * decode_mfu)
    update_s = UPDATE_FLOPS * params * gen_tokens / (peak * MFU_TRAIN)
    reward_s = math.ceil(completions / REWARD_CONCURRENCY) * reward_latency
    return gen_s, reward_s, update_s


def batched_decode_mfu(n_runs):
    """Continuous batching across concurrent runs raises effective decode MFU (more in-flight
    sequences amortize weight loads). Conservative: 0.12 (N=1) -> caps at 0.24 (2x) by ~N=16."""
    return min(0.24, MFU_DECODE * (1.0 + 0.30 * math.log2(max(1, n_runs))))


def compare(params_b=4.0, batch=8, group=8, resp_len=512, steps=30, reward_latency=1.0, n_runs=(1, 4, 8, 16, 32, 64)):
    completions = batch * group
    print(f"# Qwen-{params_b:g}B GRPO | batch {batch} x group {group} = {completions} completions/step | "
          f"resp {resp_len} tok | {steps} steps | reward {reward_latency}s/completion")
    # dev colocate per-step on A100 (gen at base decode MFU on the expensive card)
    g0, r0, u0 = phase_seconds(params_b, completions, resp_len, reward_latency, MFU_DECODE, A100["tflops"])
    step_colo = g0 + r0 + u0
    dev_run_usd = steps * step_colo * A100["usd_hr"] / 3600.0
    # PR #4 (disaggregated, per-run): 1 trainer A100 + 1 dedicated inference A40, SYNCHRONOUS server
    # mode (trainer blocks on gen; reward inline on the trainer) -> wall = gen(A40) + reward + update,
    # and BOTH GPUs are billed the whole wall (each idle during the other's phases). Reward NOT
    # off-GPU. This is the disaggregation done PER RUN with DEDICATED inference (no sharing).
    g_inf, _, _ = phase_seconds(params_b, completions, resp_len, reward_latency, MFU_DECODE, A40["tflops"])
    pr4_step = g_inf + r0 + u0
    pr4_run_usd = steps * pr4_step * (A100["usd_hr"] + A40["usd_hr"]) / 3600.0
    print(f"  dev (colocate)  $/run = ${dev_run_usd:.3f} (1xA100, step {step_colo:.1f}s incl {r0:.1f}s idle reward)")
    print(f"  PR#4 (disagg)   $/run = ${pr4_run_usd:.3f} (1xA100+1xA40 dedicated, sync, step {pr4_step:.1f}s)")
    hdr = f"  {'N':>4} | {'dev $':>8} | {'PR#4 $':>8} | {'pool $':>8} | {'pool GPUs':>22} | {'vs dev':>7} | {'vs PR#4':>7}"
    print(hdr)
    for n in n_runs:
        dec_mfu = batched_decode_mfu(n)
        gN, _, _ = phase_seconds(params_b, completions, resp_len, reward_latency, dec_mfu, A40["tflops"])
        dev_usd = n * dev_run_usd                         # N x 1 expensive GPU, full step
        pr4_usd = n * pr4_run_usd                         # N x (A100+A40) dedicated, synchronous
        # pool: trainers (A100, update only) + shared inference (A40, gen at batched MFU) + CPU reward
        trainer_wall_s = steps * u0
        trainer_usd = n * trainer_wall_s * A100["usd_hr"] / 3600.0
        gen_gpu_seconds = n * steps * gN
        m_infer = max(1, math.ceil(gen_gpu_seconds / max(trainer_wall_s, 1e-9)))
        infer_usd = m_infer * trainer_wall_s * A40["usd_hr"] / 3600.0
        reward_gpu = max(1, math.ceil(n / REWARD_CONCURRENCY))
        reward_usd = reward_gpu * trainer_wall_s * CPU_REWARD_USD_HR / 3600.0
        pool_usd = trainer_usd + infer_usd + reward_usd
        s_dev = (1 - pool_usd / dev_usd) * 100
        s_pr4 = (1 - pool_usd / pr4_usd) * 100
        gpus = f"{n}xA100+{m_infer}xA40+{reward_gpu}cpu"
        print(f"  {n:>4} | {f'${dev_usd:.2f}':>8} | {f'${pr4_usd:.2f}':>8} | {f'${pool_usd:.2f}':>8} | "
              f"{gpus:>22} | {s_dev:>6.0f}% | {s_pr4:>6.0f}%")


if __name__ == "__main__":
    print("=" * 78)
    print("SCENARIO A — light reward (regex/math grader, 0.1s/completion)")
    compare(reward_latency=0.1)
    print("\n" + "=" * 78)
    print("SCENARIO B — default reward (1.0s/completion)")
    compare(reward_latency=1.0)
    print("\n" + "=" * 78)
    print("SCENARIO C — heavy reward (LLM-judge / code-exec, 3.0s/completion)")
    compare(reward_latency=3.0)
