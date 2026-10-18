"""Shared serving and promotion deadlines, without deployment dependencies."""

# The slowest measured B200 cold boot was 1821s. Allow image loading and compilation
# to finish before treating an engine as stuck.
STARTUP_TIMEOUT_SECONDS = 2700
REQUEST_TIMEOUT_SECONDS = 600

# The router waits for both a cold engine and its request to finish.
ROUTER_TIMEOUT_SECONDS = STARTUP_TIMEOUT_SECONDS + REQUEST_TIMEOUT_SECONDS

# Let the client receive the router's final response before declaring a failed promotion.
CANARY_NETWORK_HEADROOM_SECONDS = 60
CANARY_TIMEOUT_SECONDS = ROUTER_TIMEOUT_SECONDS + CANARY_NETWORK_HEADROOM_SECONDS
