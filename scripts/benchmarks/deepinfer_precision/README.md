# Native FP4 versus expanded FP8

This comparison uses serving commit `ee68f7a9` on one four-B200 Modal sandbox.
The precision arms run in FP4, FP8, FP8, FP4 order. Each arm restarts all workers.
Both use the same pinned model, tokenizer, compiled runtime, native prefix cache,
Engram transport, scheduler, sampling, request set, and GPU allocation.

Only the expert weight execution path changes. FP4 restores the pinned upstream
Python expert loader; FP8 applies the current conversion and layout patches.
The already-compiled prefix-detachment binding is shared by both arms. Actual
kernel selection is checked from the workers' logs, with loaded binary hashes
retained in each health record.

The harness records three repeats per arm at concurrency 1, 8, and 24, for short
and approximately 5,000-token prompts, cold and cached. It records complete
outputs, token counts, time to first token, completion latency, throughput,
cache hits, startup time, and sampled GPU memory. Kernel warmups are excluded
from latency comparisons. First-use/startup costs are reported separately.
Mixed traffic adds new long prefills while four streams are already decoding.
Twenty-four deterministic arithmetic, retrieval, and code-evaluation questions
provide a limited correctness check; they are not a general capability benchmark.

The decision rule is fixed before measurement: prefer FP4 if it has no task
accuracy regression, throughput is no more than 5% lower, and p95 first-token
and completion latency are no more than 10% worse. Report every repeat and any
output differences. Broader quality checks are required if the paths diverge.

The sandbox has 16 CPUs and 512 GiB host RAM for both arms, and a two-hour hard
lifetime. This isolates precision on the same GPU hardware; it does not qualify
the production 256-GiB host-memory limit or public-router performance.
The controller downloads evidence and terminates the sandbox in `finally`.

Run from the repository root with an authenticated Modal client:

```sh
python scripts/benchmarks/deepinfer_precision/control.py
```

Results are saved under the ignored `results/deepinfer-precision/` directory.
Extract `evidence.tar.gz` into an empty directory, then run:

```sh
python scripts/benchmarks/deepinfer_precision/analyze.py /path/to/extracted/evidence
```

## Results: 2026-10-24

**Decision: retain expanded FP8 as the serving default.** Native FP4 met the
throughput and sampled correctness criteria, but missed the predeclared 10%
first-token p95 limit in two of twelve workload/concurrency combinations.
No production precision setting or conversion patch was changed.

Every table value below is the median across six measured waves per precision
(three waves in each of two independently loaded arms), at concurrency 24.
TTFT is time to first token. Latencies are measured against localhost HTTP.

| Workload | FP4 output tok/s | FP8 output tok/s | FP4 p95 TTFT | FP8 p95 TTFT |
| --- | ---: | ---: | ---: | ---: |
| Short, cold | 2,471 | 2,557 | 274 ms | 241 ms |
| Short, repeated | 2,518 | 2,557 | 255 ms | 251 ms |
| Long, cold | 1,283 | 1,231 | 2,245 ms | 2,225 ms |
| Long, cached | 2,478 | 2,436 | 348 ms | 260 ms |

The short prompt is below the minimum retained-prefix size, so its repeated
workload has zero GPU cache hits. The long cached workload reports 99.64% cached
input tokens. Across all twelve combinations, FP4 throughput ranged from 3.7%
lower to 5.3% higher than FP8. Completion-latency p95 remained within the 10%
limit everywhere. First-token p95 was 13.6% worse on short/cold/c24 and 33.9%
worse on long/cached/c24 (absolute differences about 33 ms and 88 ms).
These are measured differences, not statistical proof of a universal ranking.

Model allocations after loading were 77.520 GiB per GPU for FP4 versus
140.802 GiB for FP8: approximately 45% less with FP4. The runtime fills much of
the released memory with KV capacity, so physical GPU allocation falls by less:
peak sampled usage was 155,114 MiB versus 163,060 MiB. FP4 reserved approximately
649,478 KV blocks per rank versus 47,994 for FP8. This campaign held admission
and cache-retention limits fixed; it did not test the concurrency advantage that
larger usable KV capacity might enable.

FP4 answered 23/24 deterministic questions correctly in both arms; FP8 answered
22/24 in both arms. Both missed the same Python string-expression question,
and FP8 additionally missed a max-minus-min expression. There was no observed
per-question regression from FP8 to FP4 on this small set. The full generated
outputs differ between precisions; this is not a bit-identical transformation
of model execution or a broad model-quality certification.

All 1,584 measured performance requests, 96 correctness requests, and 96
mixed-traffic requests completed without transport/native errors. Kernel
warmups and cache priming add 520 requests outside those measured counts.
All four arms ended with healthy ranks and zero active requests. Mixed-traffic
results and every individual request are retained in the raw archive.

The Modal controller lost access to the long-running exec status, which caused
its local process to report failure after the remote benchmark had completed.
The recovered archive contains `DONE`, all 144 measured waves, all four complete
quality/mixed results, and all four final health records. `verification.json`
records those checks. Preserve `shutdown.json`'s original controller status;
it is not the benchmark outcome. The controller now polls a durable exit-code
file to avoid depending on Modal's ten-minute exec-result retention.

Each server was force-killed after its 25-second shutdown allowance; this was
the same harness cleanup policy for both modes. The four-GPU sandbox was
terminated and its app stopped. Allocation-to-termination was approximately
55 minutes; measurements and worker shutdown finished after approximately
40 minutes, with the remaining time attributable to controller-status recovery.
No benchmark GPU allocation remains active.

Evidence:

- [Full summary](results/summary.json), including all 12 combinations and all repeat throughput values.
- [Raw requests, logs, health records, memory samples, and source hashes](results/raw.tar.gz).
- [Completion and cleanup verification](results/verification.json).
- [Allocation](results/allocation.json) and [original shutdown record](results/shutdown.json).

A follow-up should investigate the two first-token latency cases and measure
whether FP4's extra KV capacity improves useful serving capacity. This result
supports keeping FP4 as a serious candidate, but does not satisfy the automatic
switch rule chosen for this comparison.
