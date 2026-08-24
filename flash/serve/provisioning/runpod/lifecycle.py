"""input validation, transport construction, and the runpod call wrappers.

The two call wrappers decide whether a failed provider call is merely failed or leaves the outcome
unknown, which is the distinction every phase of the lifecycle branches on. This module imports
nothing from `runpod.py`, so the dependency stays one-way.
"""

from __future__ import annotations

from collections.abc import Callable

from flash.serve.control import RunPodCredentials
from flash.serve.provisioning.common.records import Clock, ServingRuntimeSecrets, validate_deadline
from flash.serve.provisioning.runpod.transport import RunPodTransport, RunPodTransportFailure

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
