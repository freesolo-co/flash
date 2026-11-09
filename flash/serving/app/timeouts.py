"""Shared serving and promotion deadlines, without deployment dependencies."""

from flash.serve.deployment.readiness import CAPACITY_WAIT_SECONDS
from flash.serve.deployment.readiness import (
    ENGINE_REQUEST_TIMEOUT_SECONDS as REQUEST_TIMEOUT_SECONDS,
)
from flash.serve.deployment.readiness import (
    ENGINE_STARTUP_TIMEOUT_SECONDS as STARTUP_TIMEOUT_SECONDS,
)
from flash.serving.deepinfer.contract.config import (
    REQUEST_SECONDS as DEEPINFER_REQUEST_SECONDS,
)

CAPACITY_POLL_SECONDS = 5 * 60
LAMBDA_STARTUP_TIMEOUT_SECONDS = CAPACITY_WAIT_SECONDS + STARTUP_TIMEOUT_SECONDS

# LoRA Functions can wait for cold startup. DeepInfer Servers reject requests
# until ready, so their 55-minute request budget does not include model startup.
ROUTER_TIMEOUT_SECONDS = max(
    LAMBDA_STARTUP_TIMEOUT_SECONDS + REQUEST_TIMEOUT_SECONDS,
    DEEPINFER_REQUEST_SECONDS,
)

# Let the client receive the router's final response before declaring a failed promotion.
CANARY_NETWORK_HEADROOM_SECONDS = 60
CANARY_TIMEOUT_SECONDS = (
    LAMBDA_STARTUP_TIMEOUT_SECONDS + REQUEST_TIMEOUT_SECONDS + CANARY_NETWORK_HEADROOM_SECONDS
)
