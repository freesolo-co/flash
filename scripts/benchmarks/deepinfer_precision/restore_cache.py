import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

root = Path("/weights/compile-cache/deepinfer-9bd9cf2-sm100a-cu132")
m = json.loads((root / "manifest.json").read_text())
roots = {
    "deep_gemm": Path("/root/.dj/cache"),
    "triton": Path("/root/.triton/cache"),
    "tilelang": Path("/root/.tilelang/cache"),
}
jobs = []
for k, v in m["inventories"].items():
    for name, sha in v["sha256"].items():
        jobs.append((root / k / name, roots[k] / name, sha))


def copy(job):
    src, dst, sha = job
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == sha:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha, str(src)
    dst.write_bytes(data)
    return len(data)


start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
    for i, _result in enumerate(pool.map(copy, jobs), 1):
        if i % 250 == 0:
            print("RESTORED", i, len(jobs), round(time.time() - start), flush=True)
print("RESTORE_COMPLETE", len(jobs), time.time() - start, flush=True)
