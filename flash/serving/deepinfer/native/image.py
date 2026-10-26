"""Verify the compiled snapshot and apply the bounded short-context indexer repair."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path

from flash.serving.deepinfer.contract.config import RUNTIME_REVISION, RUNTIME_ROOT
from flash.serving.deepinfer.native.patches import apply_runtime_patches

INDEX_PATH = "deepinfer/model_executor/models/v4_modules/index_attn.py"
INDEX_SHA256 = "e269196c3c8d2db7041f7f1073597b8023dc4ecd58b40303ebf392171c02d462"
OLD = "assert 0 < topk_blocks <= (p_sum.shape[1] + block_size - 1) // block_size"
NEW = "assert 0 < topk_blocks <= 4096  # DeepSelect pads candidates past the valid KV extent."


def patch_indexer(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != INDEX_SHA256 or source.count(OLD) != 1:
        raise RuntimeError("DeepInfer indexer source does not match the pinned release")
    return source.replace(OLD, NEW)


def main():
    root = Path(RUNTIME_ROOT)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != RUNTIME_REVISION:
        raise RuntimeError("Compiled DeepInfer snapshot has the wrong source revision")
    indexer = root / INDEX_PATH
    indexer.write_text(patch_indexer(indexer.read_text()))
    apply_runtime_patches(root)
    from flash.serving.deepinfer.native.nvlink_compat import patch_bucket

    package = importlib.util.find_spec("deep_ep")
    if package is None or package.origin is None:
        raise RuntimeError("Compiled runtime is missing DeepEP")
    patch_bucket(Path(package.origin).parent / "buffers/bucket.py")
    subprocess.run(
        ["/opt/deepinfer-venv/bin/python", "setup.py", "build_ext", "--inplace"],
        cwd=root,
        check=True,
        env=dict(os.environ, MAX_JOBS="8", CXX="/usr/bin/g++-13"),
    )


if __name__ == "__main__":
    main()
