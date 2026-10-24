# Routing A/B results

Three alternating paired repeats. Synthetic workload; native command timing, not public API latency.

| Context / traffic | A seconds | B seconds | B/A output throughput | B/A elapsed time | Exact outputs | Retrieval A; B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 / uniform | 1.374 | 1.256 | 1.092× | 0.914× | 61/96 | 19/19; 19/19 |
| 2048 / skewed | 1.442 | 1.393 | 0.987× | 1.018× | 70/96 | 25/25; 25/25 |
| 8192 / uniform | 2.343 | 1.487 | 1.544× | 0.648× | 61/96 | 19/19; 19/19 |
| 8192 / skewed | 2.730 | 2.183 | 1.275× | 0.784× | 72/96 | 25/25; 25/25 |
| 16384 / uniform | 3.682 | 1.581 | 2.179× | 0.458× | 39/96 | 19/19; 19/19 |
| 16384 / skewed | 4.355 | 2.461 | 1.701× | 0.599× | 61/96 | 25/25; 25/25 |

Values are medians. Lower elapsed-time ratio is better; higher throughput ratio is better.
Output token counts can differ. `summary.json` includes TTFT p50/p95, output counts, paired ranges and correctness for every repeat.

Raw JSON SHA-256 (before gzip): `dd7fafe4b7db57a4f506785070fc19c436093a514ac409d634178f786e66e3fb`.

See the shared [methodology](../../README.md) for model pins, configuration and limits.

A assigns each arrival to the least-active rank. B adds a verified cache hit and queue-work score, with a fixed decode cost estimate of 32 prefill-token equivalents per output token. Each arm handles 32 requests with the same prompts; all four ranks participate in every collective. Capacity is 24 requests/rank. This does not implement cross-replica cache discovery.

The table reports a second full three-repeat round after first-use compilation. `qualification-raw.json.gz` preserves the complete initial round, including JIT outliers. No individual repeat was removed from either round.

Recommendation: prioritize a cache-and-queue-aware routing integration, with the existing admission and tenant-isolation rules. Benefit grows with reusable context. The short skewed case has no benefit; cache affinity is not a reason to overload a hot rank. The policy is opt-in and is not wired into the hosted router. Cached and uncached native execution can differ: all 132 retrieval probes in each arm pass, but only 364/576 paired full outputs match.
