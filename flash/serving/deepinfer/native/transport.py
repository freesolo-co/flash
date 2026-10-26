"""Configure pinned DeepEP for Modal's single-machine B200:4 allocation."""

import hashlib
from pathlib import Path

BUCKET_SHA256 = "7657ac15c1673d365e54c215d879c0d505c284c35b913c775bf67c01c4fbbc1d"


def patch_bucket(path: Path):
    source = path.read_text()
    before = "        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect the NVLink or RDMA bandwidth'"
    if hashlib.sha256(source.encode()).hexdigest() != BUCKET_SHA256:
        raise RuntimeError("Unrecognized pinned DeepEP bucket source")
    if source.count(before) != 1:
        raise RuntimeError("Ambiguous DeepEP transport patch")
    # Modal places all four GPUs on one physical machine, and all driver/CP
    # processes are children of this container. DeepEP's RDMA path is unused.
    # Its bandwidth parameter must remain positive even without an RDMA device.
    # Preserve detected bandwidth and the native NVLink availability check.
    after = (
        "        rdma_gbs = rdma_gbs or 1.0\n"
        "        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect communication bandwidth'"
    )
    path.write_text(source.replace(before, after))
