import time

import torch


def execute(native, command):
    from snapshot_codec import decode, encode

    snapshot = native.snapshots[command["key"]]
    spans = []
    seen = set()
    for layer in native.cache.gpu_cache.cache_list:
        view = layer.ca_cache
        if view is None:
            continue
        offset = view.data_ptr() - native.cache.gpu_blocks.data_ptr()
        if offset in seen:
            continue
        seen.add(offset)
        assert view.shape[-1] == 288, "This codec only supports native 512-element FP4 main KV"
        spans.append((offset, view.shape[1], view.stride(1), 256))
    started = time.perf_counter()
    encoded = encode(snapshot["kv"].numpy(), spans)
    encoded_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reconstructed = torch.from_numpy(decode(encoded))
    decoded_seconds = time.perf_counter() - started
    changed = int((reconstructed != snapshot["kv"]).sum().item())
    result = native.restore(command["key"], dict(snapshot, kv=reconstructed))
    return dict(
        result,
        encode_seconds=encoded_seconds,
        decode_seconds=decoded_seconds,
        original_bytes=snapshot["kv"].numel() + snapshot["state"].numel(),
        encoded_bytes=len(encoded["data"]) + len(encoded["remaining"]) + snapshot["state"].numel(),
        changed_bytes=changed,
        spans=spans,
    )
