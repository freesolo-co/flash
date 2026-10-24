# Compression A/B results

Three alternating paired repeats. Synthetic workload; native command timing, not public API latency.

| Context / traffic | A seconds | B seconds | B/A output throughput | B/A elapsed time | Exact outputs | Retrieval A; B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 0.913 | 1.094 | 0.839× | 1.200× | 12/48 | 12/12; 12/12 |
| 8192 | 0.976 | 1.537 | 0.587× | 1.608× | 12/48 | 12/12; 12/12 |
| 16384 | 1.004 | 1.589 | 0.626× | 1.582× | 12/48 | 12/12; 12/12 |

Values are medians. Lower elapsed-time ratio is better; higher throughput ratio is better.
Output token counts can differ. `summary.json` includes TTFT p50/p95, output counts, paired ranges and correctness for every repeat.

Raw JSON SHA-256 (before gzip): `f4c510ba6cda55c96b5f31d9961778f75d1ca59f1989f230cfcb1292c756cfe5`.

See the shared [methodology](../../README.md) for model pins, configuration and limits.

A restores native packed FP4 bytes. B maps the main-cache E2M1 values to a custom signed four-magnitude, three-bit codebook and restores native FP4 bytes before execution. Indexer data, scales and complete recurrent/sliding-window state are unchanged. Timed B includes encoding, decoding, restore and generation. Total snapshot bytes fall by only 4.3%, 9.8% and 12.6% at 2K/8K/16K, respectively; the 25% reduction applies only to the selected value bytes. Active GPU cache allocation is identical.

Recommendation: do not promote this implementation. It slows the measured request path and changes all free-form outputs versus the exact native-snapshot control. Passing the 36 retrieval checks does not establish acceptable lossy-model quality. This is not SnapKV token eviction and is not a GPU FP3 kernel.

`repeat-stability.json` also compares each arm against itself across repeats. The accelerated baseline enables variable-length MTP; fixed seeds do not guarantee fully deterministic output. Exact-match differences alone do not measure semantic quality.
