# Flash cost-estimator prompt sweep

Grid: **8 runs** (models x {SFT, GRPO} x step counts), priced at **6 prompt versions**. Ground truth is the analytical model in `flash.cost.analytical`; error is MAPE (mean absolute percentage error) of the LLM's dollar figure vs. that reference.
From **v1 base** (47.6% MAPE) to **v6 +formula** (0.2% MAPE) — a **100% error reduction** as the prompt gains Flash framework knowledge.

## Accuracy by prompt version

MAPE = mean absolute percentage error vs. the analytical reference, also split by method (SFT / GRPO). Mean signed error shows the bias direction (+ = over-estimate).

| Version | What it adds | MAPE | SFT | GRPO | Mean signed | GPU pick acc | ok |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 base | base task only | 47.6% | 63.3% | 31.9% | -37.2% | 87.5% | 8/8 |
| v2 +pricing | + GPU price/VRAM catalog | 48.8% | 39.5% | 58.2% | -14.4% | 100.0% | 8/8 |
| v3 +GPU pick | + cheapest-fit GPU selection | 33.7% | 47.7% | 19.7% | -24.2% | 100.0% | 8/8 |
| v4 +timing | + per-step compute/throughput model | 118.7% | 1.1% | 236.4% | +118.7% | 100.0% | 8/8 |
| v5 +GRPO/MoE | + GRPO rollout split, MoE, QLoRA | 32.6% | 1.4% | 63.7% | +32.6% | 100.0% | 8/8 |
| v6 +formula | + cold-start + exact cost formula | 0.2% | 0.3% | 0.1% | -0.2% | 100.0% | 8/8 |

```
MAPE by prompt version  (shorter bar = closer to ground truth)

v1 base        |████████████████ 47.6%
v2 +pricing    |████████████████ 48.8%
v3 +GPU pick   |███████████ 33.7%
v4 +timing     |████████████████████████████████████████ 118.7%
v5 +GRPO/MoE   |███████████ 32.6%
v6 +formula    |█ 0.2%
```

## Per-run: ground truth vs v1 base vs v6 +formula

| Run | Truth $ | v1 base $ | v6 +formula $ | Truth GPU | v6 +formula GPU |
|---|---:|---:|---:|---|---|
| 0.8B grpo / A5000 / g4 comp256 | 1.58 | 0.95 | 1.58 | RTX A5000 | RTX A5000 |
| 0.8B sft / RTX4090 / seq2048 r16 | 0.37 | 0.12 | 0.37 | RTX 4090 | RTX 4090 |
| 1B sft / A5000 / seq2048 b16 | 0.42 | 0.14 | 0.42 | RTX A5000 | RTX A5000 |
| 2B grpo / RTX5090 / r64 | 7.79 | 4.20 | 7.79 | RTX 5090 | RTX 5090 |
| 2B sft / RTX3090 / thinking | 4.11 | 0.42 | 4.11 | RTX 3090 | RTX 3090 |
| 4B grpo / Pro6000WK / comp512 | 11.65 | 13.50 | 11.63 | RTX Pro 6000 WK | RTX Pro 6000 WK |
| 9B grpo / A100 / thinking g16 | 33.36 | 42.00 | 33.36 | A100 PCIe | A100 PCIe |
| 9B sft / H100NVL / seq4096 | 6.31 | 4.50 | 6.31 | H100 NVL | H100 NVL (pinned) |

_Ground truth = `flash.cost.estimate_cost` (wall-clock hours x GPU $/hr, built on Flash's own pricing/VRAM/allocator primitives). The estimator is Claude (`claude-opus-4-8`) under each prompt version._
