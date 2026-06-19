# Flash cost-estimator prompt sweep (claude-opus-4-8, 24-run grid)

Grid: **24 runs** (models x {SFT, GRPO} x step counts), priced at **6 prompt versions**. Ground truth is the analytical model in `flash.cost.analytical`; error is MAPE (mean absolute percentage error) of the LLM's dollar figure vs. that reference.
From **v1 base** (257.4% MAPE) to **v6 +formula** (5.7% MAPE) — a **98% error reduction** as the prompt gains Flash framework knowledge.

## Accuracy by prompt version

MAPE = mean absolute percentage error vs. the analytical reference, also split by method (SFT / GRPO). Mean signed error shows the bias direction (+ = over-estimate).

| Version | What it adds | MAPE | SFT | GRPO | Mean signed | GPU pick acc | ok |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 base | base task only | 257.4% | 350.4% | 164.4% | +239.6% | 0.0% | 24/24 |
| v2 +pricing | + GPU price/VRAM catalog | 170.6% | 259.5% | 81.8% | +140.7% | 0.0% | 24/24 |
| v3 +GPU pick | + cheapest-fit GPU selection | 59.0% | 73.4% | 44.7% | -0.9% | 70.8% | 24/24 |
| v4 +timing | + per-step compute/throughput model | 169.1% | 12.1% | 326.0% | +157.2% | 75.0% | 24/24 |
| v5 +GRPO/MoE | + GRPO rollout split, MoE, QLoRA | 17.4% | 13.2% | 21.5% | +0.7% | 70.8% | 24/24 |
| v6 +formula | + cold-start + exact cost formula | 5.7% | 10.8% | 0.5% | -5.1% | 66.7% | 24/24 |

```
MAPE by prompt version  (shorter bar = closer to ground truth)

v1 base        |████████████████████████████████████████ 257.4%
v2 +pricing    |███████████████████████████ 170.6%
v3 +GPU pick   |█████████ 59.0%
v4 +timing     |██████████████████████████ 169.1%
v5 +GRPO/MoE   |███ 17.4%
v6 +formula    |█ 5.7%
```

## Per-run: ground truth vs v1 base vs v6 +formula

| Run | Truth $ | v1 base $ | v6 +formula $ | Truth GPU | v6 +formula GPU |
|---|---:|---:|---:|---|---|
| MiniCPM5-1B grpo x100 | 0.92 | 2.00 | 0.92 | RTX 5090 | A100 PCIe 80GB |
| MiniCPM5-1B grpo x1000 | 8.13 | 17.50 | 7.73 | RTX 5090 | A100 PCIe 80GB |
| MiniCPM5-1B grpo x500 | 4.12 | 9.50 | 4.12 | RTX 5090 | RTX 5090 |
| MiniCPM5-1B sft x100 | 0.07 | 0.85 | 0.07 | RTX A5000 | RTX A5000 |
| MiniCPM5-1B sft x1000 | 0.52 | 4.50 | 0.52 | RTX A5000 | RTX A5000 |
| MiniCPM5-1B sft x500 | 0.27 | 1.85 | 0.27 | RTX A5000 | RTX A5000 |
| Qwen3.5-0.8B grpo x100 | 0.42 | 2.85 | 0.42 | RTX A5000 | RTX A5000 |
| Qwen3.5-0.8B grpo x1000 | 3.90 | 18.50 | 3.89 | RTX A5000 | RTX A5000 |
| Qwen3.5-0.8B grpo x500 | 1.96 | 8.50 | 1.96 | RTX A5000 | RTX A5000 |
| Qwen3.5-0.8B sft x100 | 0.06 | 0.45 | 0.06 | RTX A5000 | RTX A5000 |
| Qwen3.5-0.8B sft x1000 | 0.40 | 2.10 | 0.40 | RTX A5000 | RTX A5000 |
| Qwen3.5-0.8B sft x500 | 0.21 | 0.85 | 0.21 | RTX A5000 | RTX A5000 |
| Qwen3.5-4B grpo x100 | 3.13 | 6.50 | 3.13 | A100 PCIe | A100 PCIe 80GB |
| Qwen3.5-4B grpo x1000 | 29.82 | 11.50 | 29.80 | A100 PCIe | A100 PCIe 80GB |
| Qwen3.5-4B grpo x500 | 14.99 | 14.50 | 15.00 | A100 PCIe | A100 PCIe 80GB |
| Qwen3.5-4B sft x100 | 0.40 | 0.85 | 0.42 | RTX 5090 | A100 PCIe 80GB |
| Qwen3.5-4B sft x1000 | 3.12 | 4.20 | 2.99 | RTX 5090 | A100 PCIe 80GB |
| Qwen3.5-4B sft x500 | 1.61 | 3.20 | 1.56 | RTX 5090 | A100 PCIe 80GB |
| Qwen3.5-9B grpo x100 | 6.60 | 2.80 | 6.60 | RTX 5090 | RTX 5090 |
| Qwen3.5-9B grpo x1000 | 23.76 | 16.50 | 23.76 | RTX 5090 | RTX 5090 |
| Qwen3.5-9B grpo x500 | 23.76 | 9.50 | 23.76 | RTX 5090 | RTX 5090 |
| Qwen3.5-9B sft x100 | 0.72 | 1.85 | 0.43 | RTX 5090 | RTX A5000 |
| Qwen3.5-9B sft x1000 | 6.34 | 8.50 | 4.05 | RTX 5090 | RTX A5000 |
| Qwen3.5-9B sft x500 | 3.22 | 4.50 | 2.04 | RTX 5090 | RTX A5000 |

_Ground truth = `flash.cost.estimate_cost` (wall-clock hours x GPU $/hr, built on Flash's own pricing/VRAM/allocator primitives). The estimator is Claude (`claude-opus-4-8`) under each prompt version._
