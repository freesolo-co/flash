# Disaggregation A/B results

Three alternating paired repeats. Synthetic workload; native command timing, not public API latency.

| Context / traffic | A seconds | B seconds | B/A output throughput | B/A elapsed time | Exact outputs | Retrieval A; B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 1.036 | 12.047 | 0.086× | 11.598× | 48/48 | 12/12; 12/12 |
| 8192 | 1.205 | 19.242 | 0.063× | 15.975× | 48/48 | 12/12; 12/12 |
| 16384 | 1.516 | 22.617 | 0.067× | 14.875× | 48/48 | 12/12; 12/12 |

Values are medians. Lower elapsed-time ratio is better; higher throughput ratio is better.
Output token counts can differ. `summary.json` includes TTFT p50/p95, output counts, paired ranges and correctness for every repeat.

Raw JSON SHA-256 (before gzip): `a77765ee992da1b9f6f05cd482268749d3ea9daffc598bd91a3ee28cae370183`.

See the shared [methodology](../../README.md) for model pins, configuration and limits.

A runs eight requests on each of two four-B200 groups concurrently. B uses one four-B200 group for prefill, transfers all 16 complete GPU prefix snapshots via the HTTPS RAM relay, then uses the other group for the final four prompt tokens and decoding. Both arms reserve eight B200s and process the same 16 requests. A's native wall time is the maximum of the two concurrent group times; B's request path sums prefill, export/upload, download/restore and generation. Client orchestration gaps are excluded.

Recommendation: do not promote this sequential HTTPS implementation. It is roughly 12–16× slower in request time despite matching all 144 output sequences exactly and passing all 36 retrieval probes per arm. The transfer path dominates. This is a burst test without pipeline overlap; it does not establish the performance of steady-state disaggregation over RDMA or GPU-direct transport.

The original one-token handoff failed because the native decode path required speculative draft state absent from a prefix snapshot. The corrected receiver replays four prompt tokens to initialize that state through prefill. `failed-attempt.json` records this failure and the recovery; failed/reload time is included in campaign spend but excluded from measured A/B request time. The first corrected pair also retains compilation/warm-up effects; all three paired repeats are reported.
