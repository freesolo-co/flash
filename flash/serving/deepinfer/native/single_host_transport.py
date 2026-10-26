"""Allow missing RDMA only when DeepEP proves an entirely local NVLink topology."""

import hashlib
import socket
from pathlib import Path

BUCKET_SHA256 = "7657ac15c1673d365e54c215d879c0d505c284c35b913c775bf67c01c4fbbc1d"


def local_bandwidth(groups, rdma_gbs):
    if rdma_gbs != 0:
        return rdma_gbs, False
    import torch.distributed as dist

    for group in groups:
        hosts = [None] * group.size()
        dist.all_gather_object(hosts, socket.gethostname(), group=group)
        if len(set(hosts)) != 1:
            raise RuntimeError("Missing RDMA bandwidth for a multi-host collective")
    # DeepEP requires a positive bandwidth even when its RDMA path is unused.
    return 1.0, True


def validate_local_domains(contexts, fallback):
    if fallback and any(context.get_physical_domain_size()[0] != 1 for context in contexts):
        raise RuntimeError("RDMA fallback is only valid for NVLink-only domains")


def patch_bucket(path: Path):
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != BUCKET_SHA256:
        raise RuntimeError("Unrecognized pinned DeepEP bucket source")
    changes = [
        (
            "        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect the NVLink or RDMA bandwidth'",
            (
                "        from flash.serving.deepinfer.native.single_host_transport import local_bandwidth, validate_local_domains\n"
                "        rdma_gbs, local_only_fallback = local_bandwidth(self.groups, rdma_gbs)\n"
                "        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect communication bandwidth'"
            ),
        ),
        (
            "        self.contexts = self.runtime.contexts",
            (
                "        self.contexts = self.runtime.contexts\n"
                "        validate_local_domains(self.contexts, local_only_fallback)"
            ),
        ),
    ]
    for before, after in changes:
        if source.count(before) != 1:
            raise RuntimeError("Ambiguous DeepEP local-collective patch")
        source = source.replace(before, after)
    path.write_text(source)
