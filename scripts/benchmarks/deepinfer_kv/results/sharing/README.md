# Sharing A/B results

Three alternating paired repeats. Synthetic workload; native command timing, not public API latency.

| Context / traffic | A seconds | B seconds | B/A output throughput | B/A elapsed time | Exact outputs | Retrieval A; B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 1.118 | 6.952 | 0.161× | 6.217× | 21/48 | 12/12; 12/12 |
| 8192 | 1.527 | 12.833 | 0.116× | 8.440× | 18/48 | 12/12; 12/12 |
| 16384 | 2.096 | 9.787 | 0.220× | 4.678× | 18/48 | 12/12; 12/12 |

Values are medians. Lower elapsed-time ratio is better; higher throughput ratio is better.
Output token counts can differ. `summary.json` includes TTFT p50/p95, output counts, paired ranges and correctness for every repeat.

Raw JSON SHA-256 (before gzip): `5465015c5257ff0f4927bae3d515106ea7c541604e68ca2ccda23c2677793bdb`.

See the shared [methodology](../../README.md) for model pins, configuration and limits.

A recomputes on the destination while the source retains matching prefixes. B exports complete source snapshots through an authenticated CPU HTTPS RAM relay and restores them on the destination. Both arms reserve the same two independent four-B200 groups. Request time includes source export/serialization, upload, download/deserialization, destination restore and generation. The source's historical prefix creation is excluded in both arms.

Recommendation: do not use this HTTPS relay in the request path. Transport dominates the savings from avoiding prefill, with a large 8K upload outlier preserved in the raw results. The mechanism may still be useful for recovery. This does not evaluate RDMA/GPU-direct networking or asynchronous transfer overlap. All 36 retrieval probes pass per arm; cold recompute and cached generation are not token-identical.

`repeat-stability.json` also compares each arm against itself across repeats. The accelerated baseline enables variable-length MTP; fixed seeds do not guarantee fully deterministic output. Exact-match differences alone do not measure semantic quality.
