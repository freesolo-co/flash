"""explicit process initialization; dependency repair belongs to environment construction."""

import importlib.util
import os
from pathlib import Path


def validate_environment() -> None:
    spec = importlib.util.find_spec("tilelang")
    locations = list(spec.submodule_search_locations or ()) if spec else []
    for location in locations:
        stub = Path(location) / "lib" / "libcudart_stub.so"
        if os.path.lexists(stub) and not (stub.is_symlink() and stub.exists()):
            raise RuntimeError(
                "trainer environment has an unrepaired TileLang CUDA stub; rebuild the verl "
                "environment before starting training"
            )


def initialize_process(seed: int | None = None) -> None:
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if seed is not None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
