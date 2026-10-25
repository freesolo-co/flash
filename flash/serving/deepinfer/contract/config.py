"""Pinned identity and bounded capacity for the DeepSeek serving pool."""

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
APP_NAME = "freesolo-deepseek-serving"
MODEL_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
RUNTIME_REVISION = "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
# Immutable CPU-built CUDA 13.2 / Python 3.12 snapshot in the freesolo Modal workspace.
# This snapshot contains compiled DeepInfer, not model weights or serving credentials.
RUNTIME_IMAGE_ID = "im-01M5GXCE5NCMK4TBY608BEPZ1Y"
RUNTIME_PYTHON = "/opt/deepinfer-venv/bin/python"
RUNTIME_ROOT = "/opt/DeepInfer"
MODEL_CACHE = "/weights/hub"
GPU_COUNT = 2
GPU = f"B200:{GPU_COUNT}"
MAX_MODEL_LEN = 32768
SEQUENCES_PER_RANK = 8
PREFILL_TOKENS = 4096
PREFILL_ANY_INTERVAL = 8
DELIVERY_TOKENS = 16
DELIVERY_SECONDS = 0.05
# Keep each stream above 100 decode tokens/s instead of filling all native slots.
# The native scheduler retains headroom for cache affinity and startup qualification.
MAX_INPUTS = 4
TARGET_INPUTS = 3
# Only the explicitly selected main environment keeps one EP=2 replica warm.
MAIN_MIN_CONTAINERS = 1
MAIN_MAX_CONTAINERS = 6
MAIN_BUFFER_CONTAINERS = 1
MAX_CONTAINERS = 4
BUFFER_CONTAINERS = 0
SCALEDOWN_SECONDS = 600
STARTUP_SECONDS = 2700
REQUEST_SECONDS = 600
PORT = 8000
