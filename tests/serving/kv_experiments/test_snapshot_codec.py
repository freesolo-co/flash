import numpy as np

from scripts.benchmarks.deepinfer_kv.snapshot_codec import decode, encode


def test_fp3_roundtrip_preserves_scales_and_non_main_bytes():
    raw = np.arange(256, dtype=np.uint8).reshape(4, 64)
    spans = [(8, 2, 16, 8)]
    encoded = encode(raw, spans)
    restored = decode(encoded)
    columns = np.r_[8:16, 24:32]
    other = np.setdiff1d(np.arange(64), columns)
    np.testing.assert_array_equal(restored[:, other], raw[:, other])
    assert not np.array_equal(restored[:, columns], raw[:, columns])
    assert len(encoded["data"]) == raw[:, columns].size * 3 // 4
    # A second conversion must not introduce further rounding.
    np.testing.assert_array_equal(decode(encode(restored, spans)), restored)
