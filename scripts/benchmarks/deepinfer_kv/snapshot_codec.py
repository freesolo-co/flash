"""Experimental 3-bit main-KV storage codec. GPU execution still uses native FP4.

Only packed E2M1 values are reduced to signed {0, 1, 3, 6}. Scale bytes,
indexer cache and all recurrent/SWA state remain unchanged. This is lossy and
must never be enabled based on storage savings alone.
"""

import numpy as np

MAGNITUDE = np.array([0, 0, 1, 1, 1, 2, 2, 3], dtype=np.uint8)
RESTORE = np.array([0, 2, 5, 7], dtype=np.uint8)


def columns_from_spans(spans):
    return np.concatenate(
        [
            offset
            + np.arange(tokens, dtype=np.int64)[:, None] * stride
            + np.arange(data_bytes)[None, :]
            for offset, tokens, stride, data_bytes in spans
        ],
        axis=None,
    )


def encode(values, spans):
    if values.dtype != np.uint8 or values.ndim != 2:
        raise ValueError("Expected two-dimensional raw cache bytes")
    columns = columns_from_spans(spans)
    if (
        len(set(columns.tolist())) != len(columns)
        or columns.min() < 0
        or columns.max() >= values.shape[1]
    ):
        raise ValueError("Overlapping or invalid FP4 spans")
    remaining = np.setdiff1d(np.arange(values.shape[1]), columns)
    packed = values[:, columns].reshape(-1)
    nibbles = np.stack([packed & 15, packed >> 4], axis=1).reshape(-1)
    codes = MAGNITUDE[nibbles & 7] | ((nibbles >> 3) << 2)
    bits = ((codes[:, None] >> np.arange(3)) & 1).astype(np.uint8).reshape(-1)
    data = np.packbits(bits, bitorder="little").tobytes()
    return {
        "shape": list(values.shape),
        "spans": spans,
        "data": data,
        "remaining": values[:, remaining].tobytes(),
        "code_count": int(codes.size),
    }


def decode(payload):
    shape = tuple(payload["shape"])
    columns = columns_from_spans(payload["spans"])
    remaining = np.setdiff1d(np.arange(shape[1]), columns)
    bits = np.unpackbits(np.frombuffer(payload["data"], dtype=np.uint8), bitorder="little")
    bits = bits[: payload["code_count"] * 3].reshape(-1, 3)
    codes = (bits * (1 << np.arange(3))).sum(axis=1).astype(np.uint8)
    nibbles = RESTORE[codes & 3] | ((codes >> 2) << 3)
    packed = nibbles[::2] | (nibbles[1::2] << 4)
    values = np.empty(shape, dtype=np.uint8)
    values[:, columns] = packed.reshape(shape[0], len(columns))
    values[:, remaining] = np.frombuffer(payload["remaining"], dtype=np.uint8).reshape(
        shape[0], len(remaining)
    )
    return values
