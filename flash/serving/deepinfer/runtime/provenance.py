"""Evidence from each loaded native GPU process, without request or credential data."""

import hashlib
import subprocess
import sys
from pathlib import Path

from flash.serving.deepinfer.contract.config import RUNTIME_REVISION, RUNTIME_ROOT


def loaded_runtime(llm, rank):
    import deepinfer._C as native
    import torch

    root = Path(RUNTIME_ROOT).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    engine_module = type(llm.llm_engine).__module__
    source = Path(sys.modules[engine_module].__file__).resolve()
    if revision != RUNTIME_REVISION or not engine_module.startswith("deepinfer."):
        raise RuntimeError("Loaded inference engine is not the pinned DeepSeek DeepInfer")
    if not source.is_relative_to(root):
        raise RuntimeError("DeepInfer engine loaded outside the verified source tree")
    binary = Path(native.__file__).resolve()
    if not binary.is_relative_to(root):
        raise RuntimeError("DeepInfer native binary loaded outside the verified source tree")
    with binary.open("rb") as handle:
        binary_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "rank": rank,
        "upstream": "https://github.com/deepseek-ai/DeepInfer",
        "revision": revision,
        "engine_class": f"{engine_module}.{type(llm.llm_engine).__name__}",
        "engine_source": str(source),
        "native_binary": str(binary),
        "native_sha256": binary_sha,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
