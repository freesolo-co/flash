# Matched H100/B200 quality and economics comparison — 2026-10-23

This experiment asks whether a B200 and its qualified worker image reduce time or infrastructure cost for the same Flash workload compared with an H100 and its qualified image. It also repeats fresh-model-cache and warm-model-cache training. The benchmark PR adds measurement and comparison tooling; these are **not before/after speedups from the PR**.

Both use Qwen3.5-9B, BF16, one GPU, seed 17, fixed `us-east`, eight reserved/maximum CPU cores, 32 GiB reserved / 128 GiB maximum RAM, the same 128 training examples and 128 disjoint held-out custom MMLU-Pro “other” questions. SFT uses 128 optimizer steps, batch 1; GRPO uses 16 steps, eight prompts per step and four samples per prompt. All remaining settings and immutable revisions are retained. No quality threshold was changed after observing results.

## Training and observed infrastructure charges

| Workload | Cache | H100 job time (s) | B200 job time (s) | B200 / H100 time | H100 final quality | B200 final quality | H100 accrued cost | B200 accrued cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SFT | cold | 377.7 | 744.7 | 1.972 | 56.25% | 55.47% | $1.011 | $1.476 |
| SFT | warm | 391.4 | 588.0 | 1.502 | 55.47% | 57.03% | $1.056 | $0.863 |
| GRPO | cold | 1138.1 | 799.7 | 0.703 | 51.56% | 51.56% | $3.263 | $2.827 |
| GRPO | warm | 626.5 | 1086.9 | 1.735 | 51.56% | 52.34% | $1.781 | $2.350 |

Base accuracy is **52.34% on H100** and **50.78% on B200**. Architecture-specific kernels can change numerical outputs; identical seeds do not establish bitwise equivalence. Each held-out question is 0.78125 percentage points. This custom split is not an official MMLU-Pro score.

**No evaluated checkpoint reached the predeclared 65% target. Time and cost to that target remain null.** Fixed-horizon job-time ratios above cannot establish which GPU is cheaper or faster to equal quality. They describe completed work at the displayed quality, with one training seed and separate cache conditions.

Ready evaluators checked published checkpoints during seven runs, including artifact publication/download and evaluation delays. Both B200 SFT warm checkpoint grades were recovered after the original evaluator timed out; their recorded receipt delays include recovery. Evaluator startup and idle GPU cost are recorded separately as benchmark overhead. Some asynchronous checkpoint evaluations finish out of step order; target attainment uses observation order, while the table’s final quality uses the final training step. Job time ends after final artifact publication and cleanup. A benchmark measurement completes after both the job and its required checkpoint evaluations complete.

The original evaluator campaign failed after provisioning delays consumed its bounded lifetime. All eight training jobs succeeded. Initial B200 serving completed; initial H100 serving was interrupted. We retained those outputs and repeated serving on both GPUs using the same adapters and settings and persisting each wave. H100 capacity delays prompted an explicit scheduling amendment: the ready B200 evaluator started first to avoid further idle cost. The starts were not simultaneous. Two additional evaluator allocations were capped at 1,500 seconds each; no training was retried. The recomputed conservative total compute exposure was $72.57, within the original $83 ceiling (plus the original $2 storage/egress allowance). B200 completed all recovery work. H100 stalled after 8 of 18 serving waves and was stopped after more than five minutes without a completed wave; no recovery agent episode ran on H100. Diagnostics retained a live process and 100% GPU utilization near 120 W; the root cause is unresolved. No further retry was launched. Missing work remains a failure/incomplete outcome, not a latency sample.

Recovery-engine base accuracies were 52.34% on H100 and 51.56% on B200. Process/backend variability is visible even with fixed seeds; small score differences should not be treated as established quality improvements.

## Checkpoint quality curve

| Run | Step | Held-out quality | Verified receipt since submission (s) | Recovery grade |
|---|---:|---:|---:|---|
| h100-sft-cold | 64 | 56.25% | 321.9 | False |
| h100-sft-cold | 128 | 56.25% | 349.9 | False |
| h100-sft-warm | 64 | 57.81% | 332.1 | False |
| h100-sft-warm | 128 | 55.47% | 369.5 | False |
| h100-grpo-cold | 8 | 50.78% | 1102.5 | False |
| h100-grpo-cold | 16 | 51.56% | 1112.0 | False |
| h100-grpo-warm | 8 | 51.56% | 612.2 | False |
| h100-grpo-warm | 16 | 51.56% | 605.9 | False |
| b200-sft-cold | 64 | 55.47% | 682.7 | False |
| b200-sft-cold | 128 | 55.47% | 720.1 | False |
| b200-sft-warm | 64 | 55.47% | 2418.0 | True |
| b200-sft-warm | 128 | 57.03% | 2418.1 | True |
| b200-grpo-cold | 8 | 51.56% | 800.4 | False |
| b200-grpo-cold | 16 | 51.56% | 791.0 | False |
| b200-grpo-warm | 8 | 50.78% | 1063.6 | False |
| b200-grpo-warm | 16 | 52.34% | 1057.3 | False |

## Time breakdown and architectural implications

| Run | Provision API (s) | Model download (s) | First optimizer completion from submission (s) | Worker training envelope (s) | Final CPU publication (s) |
|---|---:|---:|---:|---:|---:|
| h100-sft-cold | 1.50 | 105.70 | 247.31 | 187.07 | 13.44 |
| h100-sft-warm | 1.19 | 0.70 | 264.11 | 286.48 | 12.44 |
| h100-grpo-cold | 1.14 | 183.10 | 785.13 | 854.25 | 12.76 |
| h100-grpo-warm | 1.21 | 0.00 | 438.56 | 550.63 | 13.97 |
| b200-sft-cold | 354.93 | 107.60 | 623.77 | 198.71 | 12.51 |
| b200-sft-warm | 337.89 | 0.00 | 473.77 | 167.33 | 12.63 |
| b200-grpo-cold | 81.49 | 92.00 | 669.07 | 560.87 | 13.53 |
| b200-grpo-warm | 478.79 | 0.00 | 955.16 | 535.06 | 12.33 |

Provision API time includes provisioning/image work, not only scheduler allocation. First optimizer completion includes all preceding startup plus the first step, so it is an upper bound on startup. The worker training envelope includes framework/model initialization and checkpoint work; it is not optimizer-only time. These columns overlap and must not be added.

| GRPO run | Generation (s) | Old log-probs (s) | Actor update (s) | Rollout weight update (s) | Checkpoint save (s) | Whole steps (s) | Observed reward grading (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| h100-grpo-cold | 209.16 | 55.56 | 98.22 | 45.76 | 8.00 | 453.17 | 0.005621 |
| h100-grpo-warm | 79.13 | 43.09 | 77.79 | 28.02 | 5.25 | 255.83 | 0.006235 |
| b200-grpo-cold | 40.35 | 41.94 | 95.96 | 15.33 | 4.83 | 216.17 | 0.003551 |
| b200-grpo-warm | 37.01 | 35.55 | 82.41 | 15.45 | 4.01 | 194.74 | 0.003143 |

| GRPO first-step behavior | First step (s) | Median subsequent step (s) |
|---|---:|---:|
| h100-grpo-cold | 160.61 | 18.69 |
| h100-grpo-warm | 112.35 | 9.22 |
| b200-grpo-cold | 127.99 | 5.38 |
| b200-grpo-warm | 112.33 | 5.16 |

Native GRPO timers are component totals and can overlap. Reward grading is separately observed inside generation; its log coverage is not complete. Checkpoint-save time excludes some asynchronous exports/uploads. Publication heartbeat samples provide explicit lower bounds in `summary.json`, not exact upload spans. SFT optimizer-only and checkpoint-capture timings remain unmeasured, and the normalized suite leaves these gaps unattributed rather than fabricating a complete exclusive breakdown.

| Evaluator startup | H100 (s) | B200 (s) |
|---|---:|---:|
| Model download | 75.5 | 82.6 |
| Engine initialization | 503.6 | 122.8 |

Recovery engine initialization, using the retained model cache, took 513.0s on H100 and 133.4s on B200. These are separate fresh-process observations. The recovery provision API took 872.3s on H100 and 22.9s on B200; serving latencies below start after engine readiness.

The retained H100 diagnostic shows CPU CUDA compilation while GPU utilization is 0%. This identifies compiled-kernel coverage and process reuse as concrete optimization candidates. It does not quantify the savings of an untested cache or serving-pool change. The evaluator uses the pinned training image and an eager BF16 engine, not Flash’s production serving deployment profile.

## Architectural decisions this comparison can inform

- **Investigate H100 mixed-adapter progress before claiming a serving win.** The recovery stalled at concurrency 8 after eight completed waves. Its partial throughput is conditional on completion. Request progress watchdogs and backend diagnostics would make this failure easier to bound; the present evidence does not establish a root cause.
- **Test prepared capacity and engine reuse first for short jobs.** Provisioning took several minutes in multiple B200 training jobs and over 14 minutes in the H100 recovery. Both retained H100 startup diagnostics show CPU CUDA compilation with 0% GPU utilization. These are measured latency sources; the benefit and idle cost of a pool or compiled-kernel cache still require a controlled before/after experiment.
- **Evaluate B200 for sustained GRPO separately from submission latency.** Median steps after the first were 5.38/5.16s on B200 versus 18.69/9.22s on H100 for cold/warm runs. Yet B200 warm GRPO took longer end to end because provisioning dominated. Hardware plus its qualified image is the treatment; these are single-seed observations, not an isolated silicon speedup.
- **Give publication work its measured priority.** Final CPU publication was about 12–14s per run. It matters, but the observed queue/startup delays were much larger. Native checkpoint and rollout-weight timers are retained separately so a future publication change can be checked without hiding its training impact.
- **Resolve the quality bottleneck before claiming cheaper useful training.** Neither hardware configuration reached 65%; agent protocol compliance did not imply correct task completion. A faster fixed horizon is insufficient evidence for a lower cost to the desired result.

## Concurrent adapters: completed observations and H100 stall

Both GPUs load the **same two immutable H100-trained final adapters**, verified by weight hashes. The benchmark uses two synthetic customer streams, 64 requests each per wave, three planned counterbalanced repetitions of isolated and mixed waves at each global concurrency. B200 completed 18/18 waves; H100 completed 8/18 before stalling. H100 has three repetitions at concurrency 1, one at concurrency 8, and none at concurrency 16. Prefix caching is disabled. This measures local engine latency; hosted HTTP, authorization, routing and real tenant isolation are excluded. Queue-inclusive latency and answer accuracy are reported below and retained per request in the evidence.

| Global concurrency | Mode | Customer | H100 median wave p95 (ms) | B200 median wave p95 (ms) | H100 median requests/s | B200 median requests/s |
|---:|---|---|---:|---:|---:|---:|
| 1 | isolated | customer-a | 271.98 | 191.65 | 4.01 | 5.26 |
| 1 | isolated | customer-b | 264.07 | 190.94 | 4.01 | 5.26 |
| 1 | concurrent | customer-a | 252.67 | 190.63 | 4.11 | 5.28 |
| 1 | concurrent | customer-b | 251.28 | 190.45 | 4.11 | 5.28 |
| 8 | isolated | customer-a | 347.62 | 277.83 | 22.86 | 28.86 |
| 8 | isolated | customer-b | 365.36 | 277.99 | 22.86 | 28.86 |
| 8 | concurrent | customer-a | 360.83 | 278.26 | 22.63 | 28.84 |
| 8 | concurrent | customer-b | 360.58 | 278.21 | 22.63 | 28.84 |
| 16 | isolated | customer-a | unmeasured | 281.44 | unmeasured | 56.80 |
| 16 | isolated | customer-b | unmeasured | 281.30 | unmeasured | 56.80 |
| 16 | concurrent | customer-a | unmeasured | 283.30 | unmeasured | 56.65 |
| 16 | concurrent | customer-b | unmeasured | 282.94 | unmeasured | 56.65 |

Throughput is for the entire two-customer wave; it is repeated in each customer row for context. Cross-customer overlap is verified for completed mixed waves above concurrency 1. H100 has no completed concurrency-16 wave. Concurrency 1 is a serial control. All completed observations and each repetition’s values are retained. These medians are conditional on completion, with unequal repetition counts; they do not establish a successful overall H100/B200 serving comparison. The H100 stall is part of the result. Repetitions are not independent model-quality samples.

Mixed-stream admission queue and quality (median across available completed waves; 64 distinct questions per customer, repeated):

| Concurrency | Customer | H100 queue-inclusive p95 (ms) | B200 queue-inclusive p95 (ms) | H100 accuracy | B200 accuracy |
|---:|---|---:|---:|---:|---:|
| 1 | customer-a | 29446.63 | 22898.63 | 62.50% | 62.50% |
| 1 | customer-b | 29684.31 | 23087.93 | 56.25% | 56.25% |
| 8 | customer-a | 5563.94 | 4375.73 | 62.50% | 60.94% |
| 8 | customer-b | 5643.17 | 4431.52 | 54.69% | 56.25% |
| 16 | customer-a | unmeasured | 2246.96 | unmeasured | 60.94% |
| 16 | customer-b | unmeasured | 2246.97 | unmeasured | 56.25% |

All 128 requests are submitted as a batch per mixed wave. Queue-inclusive latency therefore measures this burst admission policy; it is not an independently sampled production arrival process. Repeating the same questions measures runtime variation, not additional independent quality samples.

## Long multi-turn agents

| GPU | Episodes | Correct final answers | Valid tool protocol | Model turns | Episode p50 (s) | Episode p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| H100 | 0 | 0 | 0 | 0 | unmeasured | unmeasured |
| B200 | 16 | 0 | 16 | 528 | 19.87 | 20.88 |

Sixteen episodes were planned per GPU. H100 completed none because its preceding serving wave stalled; its missing episodes are unmeasured, not successful zero-duration episodes. B200 completed all sixteen. Each episode requires 32 ordered lookup tool calls and a final sum, with full history retained and two concurrent episodes. Final answers are independently regraded. Latencies include incorrect/failed episodes, which remain failures in the quality denominator; lower latency alone does not mean a better agent. This is a deterministic long-agent inference microbenchmark, not long-agent reinforcement-learning training or a customer environment.

## Actual infrastructure cost versus customer charges

| Cost owner | H100 accrued USD | B200 accrued USD |
|---|---:|---:|
| Training apps including final CPU publication | $7.112 | $7.516 |
| Original evaluators: startup + idle + inference | $10.090 | $13.986 |
| Recovery evaluators: startup + grades + inference | $3.052 | $3.734 |

Modal reports **$45.490 accrued** as of 2026-10-23T04:17:05.301861+00:00. These are provider-reported costs from separate owned apps, not customer quotes or rate-derived estimates. Metering is provisional and can lag; this is not a finalized invoice. Customer charges remain a separate **unknown/null** ledger. Long-term artifact/volume storage outside these app receipts is not reconciled. Benchmark observer overhead is fully disclosed rather than folded into one training case or omitted.

The preflight exposure bound used [Modal sandbox pricing](https://modal.com/pricing) and the [1.75× narrow-region multiplier](https://modal.com/docs/guide/region-selection). Region selection is held constant across both GPUs. No list-price estimate is labeled an actual charge.

## Revisions, replay and limits

- Executed Flash code: `a6bfe171dd1879177d8f5787b4d37b41441a9927`.
- Model: `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- H100 base image: `ghcr.io/freesolo-co/flash-worker@sha256:c794abb352bb99c8a90e58f4f1f8f4b56012a1c00fcfe607dd8a037d0f119fb8`.
- B200 base image: `ghcr.io/freesolo-co/flash-worker@sha256:341e0724f1f1730ff8f8ea0492ee56b16df3e374bf3464528a59c55cc67025d3`.
- Data SHA-256: `a8cafd7e29d30a1349101bd2a5469613b57d09dfd2ce2b0eff25e1d75a359895`.
- Environment SHA-256: `59547d9594768767982ce96997e17754012eaf5a46e733a67ebb726aa49cbd1b`.
- Original evaluator SHA-256: `40fbf0c654fab809145f5874dc55a0223f82f8b2b6c5d06983f4bda8905e00b6`; recovery evaluator SHA-256: `34616c312dffa950a0439b823334fd45ae6dbd1d36211c348104a0b13507a14f`.
- Exact job specifications, source archive, evaluator source/configuration, checkpoint revisions and weight digests, request/episode records, native metrics, provider receipts, owned-resource cleanup and full manifests are retained in the evidence bundle.
- The sealed training journals are an explicit offline evidence import, not a claim that the failed evaluator supervisor succeeded. Native training job outcomes and original campaign failure are retained in each imported execution record. They use total measurement-completion time, including required evaluation and any recovery. Their wall ratios differ from the training-job-only ratios in the table above; both are explicitly retained.
- `tools/summarize.py` and `tools/render_report.py` regenerate this report offline. `tools/import_suite.py` produces sealed standard-suite journals and `paired-suite-comparison.json`; both replay through `python -m flash.benchmarks compare`. The normalized comparison explicitly declares the GPU, image and SFT warm evaluation-mode differences; recovery is a confound for measurement-completion timing. The normalized journals intentionally retain timing/accounting gaps.
- This is one training seed with cold/warm conditions. Image caches and underlying hosts are not fixed; cold/warm controls the dedicated base-model cache only. All observed training regions are verified `us-east`.
- Hardware and architecture-specific image change together; underlying host and kernel/cache state are not fixed. These results qualify this measured configuration comparison, not an isolated silicon-only causal effect or a production-wide performance promise.
- `production_claims_eligible` remains false: complete exclusive phase tracing, finalized billing and hosted customer isolation are not established.

## Retained evidence and verification

The [immutable evidence ZIP](https://huggingface.co/datasets/DavidBShan/flash-compare-ada8a16f6c/resolve/d57812a2977f4ec3a8174e9e2fc62c2de0ac7552/evidence/matched-gpu-20261023.zip) is in the private Hugging Face dataset repository `DavidBShan/flash-compare-ada8a16f6c` and requires access to that repository. It contains 560 files, including the complete raw observations, failed-run diagnostics, exact job specifications, executed and analysis source archives, input fixtures, evaluator sources, provider receipts and cleanup records. Large adapter weights remain in that repository at the pinned checkpoint revisions.

- Evidence revision: `d57812a2977f4ec3a8174e9e2fc62c2de0ac7552`.
- ZIP SHA-256: `48bc2b27e6de5121f75435a139f38177605ebb7269e9c2352d5ae1b417459d45`.
- Offline analysis code: `f9061decf3bb9bee22d06e0d0c8b5d8d06dbdb7f`.
- [Machine-readable summary](summary.json), [sealed training comparison](paired-suite-comparison.json), [independent checks](validation.json), and [replay instructions](REPLAY.txt).

The immutable ZIP was downloaded again and its SHA-256 and every manifest entry verified (560 archived files); see [publication verification](publication-check.json).

All eight worker source attestations and positive optimizer gradients were verified. Independent checks regraded 2,560 held-out predictions, 3,328 retained serving responses and 16 completed agent episodes. The two sealed training reports replay; rebuilding the report, summary, comparison and validation in a copied evidence tree is byte-for-byte identical. Missing H100 serving waves and agent episodes remain explicitly incomplete.

The full offline suite passed **10,814 tests (182 skipped)** at `6583f6ad`; the final execution-contract comparison hardening at `f9061dec` passed all **51 benchmark tests**, lint and format checks. Serial wall-clock tests passed (25 passed, 2 skipped); file/function/layout gates and benchmark-package mypy also passed. Full validation logs and revision-specific results are retained in the ZIP. No training-engine behavior was changed by the comparison tooling.

All owned GPU/CPU sandboxes were stopped, six model-cache volumes and checkpoint handoff volumes removed, and a final inventory found no owned sandboxes. The private artifact repository is retained. Costs are the accrued snapshot reported above, not a finalized invoice.
