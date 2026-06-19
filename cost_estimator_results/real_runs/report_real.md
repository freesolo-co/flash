# Cost estimator vs MEASURED real-run cost

13 real training runs on the `freesolo-co/flash-bench` reward env (SFT + GRPO, RTX 5090 / A100 PCIe, varied group/steps/model). Ground truth is the actual control-plane `cost_usd`, not the analytical equation.

## Analytical equation vs measured

`static` uses the registry's fallback $/hr; `actual rate` re-prices the same estimated wall-clock at the rate the run was actually billed (isolates the wall-clock model from the spot-vs-static pricing gap).

| Run | Measured $ | Analytical $ (static) | Analytical $ (actual rate) | Meas wall | Est wall | GPU meas/est |
|---|---:|---:|---:|---:|---:|---|
| q4b-sft-A100-linkd | 0.7664 | 0.5909 | 0.5909 | 2165s | 1530s | A100 PCIe/A100 PCIe |
| cm1b-sft-A100-ticket | 0.0650 | 0.2020 | 0.2020 | 324s | 523s | A100 PCIe/A100 PCIe |
| q4b-sft-A100-ticket | 0.2368 | 0.4194 | 0.4194 | 798s | 1086s | A100 PCIe/A100 PCIe |
| q9b-grpo-A100-linkd | 0.1865 | 0.3370 | 0.3370 | 788s | 873s | A100 PCIe/A100 PCIe |
| q9b-sft-4090-gsm8k | 0.1183 | 0.1041 | 0.0663 | 1001s | 543s | RTX 4090/RTX 4090 |
| q2b-sft-3090-gsm8k | 0.0632 | 0.0559 | 0.0301 | 955s | 437s | RTX 3090/RTX 3090 |
| q08b-sft-3090-gsm8k | 0.0406 | 0.0475 | 0.0256 | 591s | 372s | RTX 3090/RTX 3090 |
| q4b-sft-5090-gsm8k | 0.0605 | 0.1707 | 0.1041 | 364s | 621s | RTX 5090/RTX 5090 |
| q2b-sft-4090-gsm8k | 0.0371 | 0.0729 | 0.0729 | 354s | 380s | RTX 4090/RTX 4090 |
| q08b-grpo-5090-gsm8k | 0.0882 | 0.1212 | 0.0739 | 546s | 441s | RTX 5090/RTX 5090 |
| cm1b-grpo-5090-bench | 0.0473 | 0.1948 | 0.0922 | 365s | 708s | RTX 5090/RTX 5090 |
| cm1b-sft-5090-bench | 0.0344 | 0.1691 | 0.0896 | 243s | 615s | RTX 5090/RTX 5090 |
| cm1b-grpo-5090-bench-s6 | 0.0846 | 0.1175 | 0.0569 | 667s | 427s | RTX 5090/RTX 5090 |

Analytical MAPE vs measured: **115%** (static rate), **77%** (actual rate), **54%** (actual rate + cold-start calibrated to these runs). Median APE: 77% / 72% / 57%.

## LLM estimator vs measured cost (by prompt version)

Median APE is the robust headline (mean MAPE is inflated by v4, whose SFT-timing-without-rollout estimate explodes on tiny-cost GRPO runs).

| Version | MAPE | Median APE | SFT | GRPO | GPU-pick acc |
|---|---:|---:|---:|---:|---:|
| v1 base | 479% | 410% | 539% | 343% | 92% |
| v2 +pricing | 598% | 385% | 615% | 561% | 92% |
| v3 +GPU pick | 496% | 417% | 576% | 315% | 92% |
| v4 +timing | 106% | 42% | 74% | 177% | 92% |
| v5 +GRPO/MoE | 83% | 55% | 84% | 80% | 92% |
| v6 +formula | 1882% | 77% | 114% | 5859% | 92% |

```
MAPE by prompt version  (shorter bar = closer to ground truth)

v1 base        |██████████ 478.7%
v2 +pricing    |█████████████ 598.2%
v3 +GPU pick   |███████████ 495.5%
v4 +timing     |██ 105.7%
v5 +GRPO/MoE   |██ 82.6%
v6 +formula    |████████████████████████████████████████ 1881.6%
```
