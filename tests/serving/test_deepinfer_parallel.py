"""Distributed prefill decisions and single-host transport must fail safely."""

import hashlib

import pytest

from flash.serving.deepinfer.native import transport
from flash.serving.deepinfer.native.transport import patch_bucket


@pytest.mark.parametrize(
    ("nvl_gbs", "rdma_gbs", "expected"),
    [(900, 0, (900, 1.0)), (900, 50, (900, 50)), (0, 0, None), (-1, 50, None), (900, -1, None)],
)
def test_single_host_patch_preserves_bandwidth_and_requires_nvlink(
    tmp_path, monkeypatch, nvl_gbs, rdma_gbs, expected
):
    source = (
        "class Bucket:\n"
        "    def __init__(self, nvl_gbs, rdma_gbs):\n"
        "        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect the NVLink or RDMA bandwidth'\n"
        "        self.bandwidth = (nvl_gbs, rdma_gbs)\n"
    )
    monkeypatch.setattr(transport, "BUCKET_SHA256", hashlib.sha256(source.encode()).hexdigest())
    path = tmp_path / "bucket.py"
    path.write_text(source)
    patch_bucket(path)
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    if expected is None:
        with pytest.raises(AssertionError, match="communication bandwidth"):
            namespace["Bucket"](nvl_gbs, rdma_gbs)
    else:
        assert namespace["Bucket"](nvl_gbs, rdma_gbs).bandwidth == expected


@pytest.mark.parametrize("matches", [0, 2])
def test_ambiguous_bandwidth_patch_does_not_modify_source(tmp_path, monkeypatch, matches):
    source = (
        "        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect the NVLink or RDMA bandwidth'\n"
    ) * matches
    monkeypatch.setattr(transport, "BUCKET_SHA256", hashlib.sha256(source.encode()).hexdigest())
    path = tmp_path / "bucket.py"
    path.write_text(source)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        patch_bucket(path)
    assert path.read_text() == source


def test_unknown_deep_ep_source_is_never_modified(tmp_path):
    path = tmp_path / "bucket.py"
    source = "unreviewed upstream code"
    path.write_text(source)
    with pytest.raises(RuntimeError, match="Unrecognized"):
        patch_bucket(path)
    assert path.read_text() == source
