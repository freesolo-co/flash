"""Shared serving and promotion deadlines, without deployment dependencies."""

from flash.serve.deployment.readiness import (
    ENGINE_REQUEST_TIMEOUT_SECONDS as REQUEST_TIMEOUT_SECONDS,
)
from flash.serve.deployment.readiness import (
    ENGINE_STARTUP_TIMEOUT_SECONDS as STARTUP_TIMEOUT_SECONDS,
)
from flash.serving.deepinfer.contract.config import (
    REQUEST_SECONDS as DEEPINFER_REQUEST_SECONDS,
)
from flash.serving.deepinfer.contract.config import (
    STARTUP_SECONDS as DEEPINFER_STARTUP_SECONDS,
)

# The router waits for both a cold engine and its request to finish.
ROUTER_TIMEOUT_SECONDS = max(
    STARTUP_TIMEOUT_SECONDS + REQUEST_TIMEOUT_SECONDS,
    DEEPINFER_STARTUP_SECONDS + DEEPINFER_REQUEST_SECONDS,
)

# Let the client receive the router's final response before declaring a failed promotion.
CANARY_NETWORK_HEADROOM_SECONDS = 60
CANARY_TIMEOUT_SECONDS = (
    STARTUP_TIMEOUT_SECONDS + REQUEST_TIMEOUT_SECONDS + CANARY_NETWORK_HEADROOM_SECONDS
)
