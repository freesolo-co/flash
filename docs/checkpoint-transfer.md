# Checkpoint transfer tuning

Worker images can set `FLASH_HF_UPLOAD_THREADS` from 1 to 16 (default 8) to bound
parallel Hugging Face file transfers. Each folder still becomes visible through
one atomic commit. A failed commit does not advance checkpoint publication or
prune the previous recovery point. Streaming LFS remains enabled to avoid Xet's
additional local disk staging.

This parallelizes independent rank/model/optimizer files. It cannot accelerate
a single 19 GB file by itself. The native transport still splits files only when
required by the Hub's file-size limit; lowering that threshold adds a disk copy
and must be benchmarked with available disk headroom. Keep the worker and its
weight-cache Volume in compatible placement; do not move data between regions
just to increase concurrency. Measure upload time and egress before increasing
the bound: more threads are not a guaranteed speedup.
