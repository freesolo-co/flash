"""Prepare identical serving sources with only expert execution precision changed."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from flash.serving.deepinfer.contract.config import RUNTIME_REVISION
from flash.serving.deepinfer.native.patches import PATCHES
from scripts.prepare_deepinfer_image import patch_indexer

root = Path("/opt/DeepInfer")
assert (
    subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    == RUNTIME_REVISION
)
# This reused image already contains the compiled prefix-detachment binding.
# Restore and verify the pinned Python sources before constructing each arm.
for mode in ("fp4", "fp8"):
    target = Path("/root") / mode
    shutil.copytree("/root/flash", target / "flash", dirs_exist_ok=True)
    worker = target / "flash/serving/deepinfer/runtime/worker.py"
    source = worker.read_text()
    if mode == "fp4":
        source = source.replace(
            "from flash.serving.deepinfer.native.fp8 import CONVERTED_EXPERT_TENSORS",
            "CONVERTED_EXPERT_TENSORS = 0",
        )
        source = source.replace(
            "if CONVERTED_EXPERT_TENSORS == 0:", "if CONVERTED_EXPERT_TENSORS != 0:"
        )
        source = source.replace('expert_dtype="float8_e4m3fn"', 'expert_dtype="native_fp4"')
        source = source.replace(
            'expert_source="losslessly-expanded-fp4"', 'expert_source="native-fp4"'
        )
    worker.write_text(source)
    native = Path("/opt") / ("DeepInfer-" + mode)
    shutil.copytree(root, native, symlinks=True, dirs_exist_ok=True)
    for name, (expected, changes) in PATCHES.items():
        if not name.endswith(".py"):
            continue
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", "HEAD:" + name], text=True
        )
        assert hashlib.sha256(original.encode()).hexdigest() == expected
        modified = original
        if mode == "fp8":
            for before, after in changes:
                assert modified.count(before) == 1
                modified = modified.replace(before, after)
        (native / name).write_text(modified)
    indexer = "deepinfer/model_executor/models/v4_modules/index_attn.py"
    original = subprocess.check_output(
        ["git", "-C", str(root), "show", "HEAD:" + indexer], text=True
    )
    (native / indexer).write_text(patch_indexer(original))
    config = target / "flash/serving/deepinfer/contract/config.py"
    config.write_text(
        config.read_text()
        .replace('RUNTIME_ROOT = "/opt/DeepInfer"', f'RUNTIME_ROOT = "{native}"')
        .replace('MODEL_CACHE = "/weights/hub"', 'MODEL_CACHE = "/tmp/model-hub"')
    )
    print("STAGED", mode, flush=True)
cache = Path("/tmp/model-hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots")
cache.mkdir(parents=True, exist_ok=True)
(cache / "1bae0e1088993b9b82de08f59de5be5a34e42170").symlink_to("/weights/model")
manifest = {}
for directory in ("/root/fp4", "/root/fp8"):
    manifest[directory] = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(directory).rglob("*.py")
    }
Path("/results/source-manifest.json").write_text(json.dumps(manifest, indent=2))
