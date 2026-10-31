"""Shared serving and promotion deadlines, without deployment dependencies."""

from flash.serve.deployment.readiness import (
    DEPLOYMENT_SMOKE_TIMEOUT_SECONDS,
)
from flash.serve.deployment.readiness import (
    ENGINE_REQUEST_TIMEOUT_SECONDS as REQUEST_TIMEOUT_SECONDS,
)
from flash.serve.deployment.readiness import (
    ENGINE_STARTUP_TIMEOUT_SECONDS as STARTUP_TIMEOUT_SECONDS,
)
from flash.serving.deepinfer.contract.config import (
    REQUEST_SECONDS as DEEPINFER_REQUEST_SECONDS,
)

# LoRA Functions can wait for cold startup. DeepInfer Servers reject requests
# until ready, so their 55-minute request budget does not include model startup.
ROUTER_TIMEOUT_SECONDS = max(
    STARTUP_TIMEOUT_SECONDS + REQUEST_TIMEOUT_SECONDS,
    DEEPINFER_REQUEST_SECONDS,
)

# The canary sends a short request, not a full native-context batch. Keep its short smoke
# allowance independent of the long-request budget so promotion retains time to roll back.
CANARY_NETWORK_HEADROOM_SECONDS = 60
CANARY_TIMEOUT_SECONDS = (
    STARTUP_TIMEOUT_SECONDS + DEPLOYMENT_SMOKE_TIMEOUT_SECONDS + CANARY_NETWORK_HEADROOM_SECONDS
)
