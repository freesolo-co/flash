"""shared primitives for the runpod provisioning lifecycle.

`runpod.py` reached the 1000-line file limit, and the group that came out cleanest is the one
nothing else in that module depends on structurally: input validation, transport construction, and
the two call wrappers that decide whether a failed provider call is merely failed or leaves the
outcome unknown. They are pure, provider-call-shaped, and referenced from every phase of the
lifecycle, so they read better as a named foundation than as the first two hundred lines of the
file that uses them.

The split is deliberately one-way: this module imports nothing from `runpod.py`, so there is no
cycle to reason about.
"""

from __future__ import annotations

from collections.abc import Callable

from flash.serve.control import RunPodCredentials

# `Clock`, `Sleeper`, `LifecycleFailure` and `validate_deadline` are provider-neutral and live in
# `_common`. They are re-exported here so the lifecycle modules stay the single import site for
# their own callers, which is what `_modal_lifecycle` does too.
from ._common import (
    Clock,
    LifecycleFailure,
    ServingRuntimeSecrets,
    Sleeper,
    validate_deadline,
)
from ._runpod_transport import RunPodTransport, RunPodTransportFailure

__all__ = [
    "Clock",
    "LifecycleFailure",
    "Sleeper",
    "TransportFactory",
    "identity",
    "mutation_call",
    "open_transport",
    "read_call",
    "validate_control_inputs",
    "validate_deadline",
    "validate_runtime_inputs",
]

TransportFactory = Callable[[str], RunPodTransport]


def validate_runtime_inputs(
    credentials: RunPodCredentials,
    runtime_secrets: ServingRuntimeSecrets,
    deadline_at: float,
    clock: Clock,
) -> None:
    if type(credentials) is not RunPodCredentials:
        raise ValueError("runpod credentials must use the exact credential type")
    if type(runtime_secrets) is not ServingRuntimeSecrets:
        raise ValueError("runtime secrets must use the exact secret boundary")
    validate_deadline(deadline_at, clock)


def validate_control_inputs(
    credentials: RunPodCredentials,
    deadline_at: float,
    clock: Clock,
) -> None:
    if type(credentials) is not RunPodCredentials:
        raise ValueError("runpod credentials must use the exact credential type")
    validate_deadline(deadline_at, clock)


def open_transport(factory: TransportFactory, credentials: RunPodCredentials) -> RunPodTransport:
    try:
        return factory(credentials.reveal())
    except Exception:
        raise RunPodTransportFailure("transport_failed") from None


def read_call(operation: Callable[[], object], parser: Callable[[object], object]):
    """run one read, whose failure leaves no provider state behind."""

    try:
        return parser(operation())
    except RunPodTransportFailure:
        raise
    except Exception:
        raise RunPodTransportFailure("transport_failed") from None


def mutation_call(operation: Callable[[], object], parser: Callable[[object], object]):
    """run one mutation, whose failure may still have applied.

    `resource_ambiguous` rather than `transport_failed`: a create or delete that raised locally may
    already have reached runpod, so the caller must reconcile rather than retry blindly.
    """

    try:
        return parser(operation())
    except RunPodTransportFailure:
        raise
    except Exception:
        raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True) from None


def identity(value: object) -> object:
    """the parser for a mutation whose response has nothing to parse.

    `mutation_call` always parses, so a delete -- which returns no body worth reading -- still has
    to hand it something. Passing this keeps the "every mutation goes through one wrapper" rule
    intact instead of adding a second, parserless call path for one case.
    """

    return value
