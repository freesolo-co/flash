"""Pinned identity and bounded capacity for the DeepSeek serving pool."""

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
APP_NAME = "freesolo-deepseek-serving"
MODEL_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
RUNTIME_REVISION = "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
# Immutable CUDA 13.2 / Python 3.12 snapshot with B200 compiler artifacts from the
# complete startup warmup. It contains no model weights, prompt KV, or credentials.
RUNTIME_IMAGE_ID = "im-22oaWkg4YF8JC0J9nM5RjE"
RUNTIME_PYTHON = "/opt/deepinfer-venv/bin/python"
RUNTIME_ROOT = "/opt/DeepInfer"
MODEL_CACHE = "/weights/hub"
# Modal guarantees that a multi-GPU container receives GPUs on one physical host.
GPU_COUNT = 4
CONTEXT_PARALLEL_SIZE = 2
RANK_COUNT = GPU_COUNT // CONTEXT_PARALLEL_SIZE
GPU = f"B200:{GPU_COUNT}"
MAX_MODEL_LEN = 1048576
SEQUENCES_PER_RANK = 64
PREFILL_TOKENS = 16384
PREFILL_ANY_INTERVAL = 8
PREFILL_COALESCING_SECONDS = 0.42
PREFILL_COALESCING_MIN_TOKENS = 8192
GPU_PREFIX_CACHE_TOKENS = 2 * MAX_MODEL_LEN
GPU_PREFIX_CACHE_ENTRIES = SEQUENCES_PER_RANK
DELIVERY_TOKENS = 16
DELIVERY_SECONDS = 0.05
# Two CP=2 driver ranks share four GPUs; bound HTTP admission to native slots.
MAX_INPUTS = 128
# Development retains Modal's demand-driven single-container policy.
TARGET_INPUTS = 8
# Planning capacity per four-GPU container, not a measured per-request SLO.
# Main adds capacity when the whole pool has 64 request slots remaining.
POOL_STREAM_CAPACITY = 100
POOL_SPARE_STREAMS = 64
POOL_SAMPLE_SECONDS = 5
POOL_SAMPLE_MAX_AGE = 20
POOL_STATE_NAME = "freesolo-deepseek-pool-load"
# Main can scale to twelve B200s; other environments remain one group, idle at zero.
MAIN_MIN_CONTAINERS = 1
MAIN_MAX_CONTAINERS = 3
MAIN_BUFFER_CONTAINERS = 0
MAX_CONTAINERS = 1
BUFFER_CONTAINERS = 0
SCALEDOWN_SECONDS = 600
SCALEUP_SECONDS = 1
STARTUP_SECONDS = 2700
# A 350K-token completion takes almost an hour at 100 tokens/s. Leave room for
# prefill and rate variation while retaining a finite generation deadline.
REQUEST_SECONDS = 7200
WORKER_STALL_SECONDS = 600
STREAM_IDLE_SECONDS = 60
PORT = 8000
