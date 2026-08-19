"""validation for provider http statuses safe to expose across teacher boundaries."""

from __future__ import annotations

# statuses a worker may treat as transient from the status line alone, with no response body.
#
# the broker describes retryability in a structured json body, but a body does not survive the
# trip. any intermediary between the broker and the worker -- load balancer, reverse proxy, cdn --
# may replace the body of a 5xx with its own text; a structured 502 was observed reaching a worker
# as 16 bytes of `error code: 502` with the content type rewritten to text/plain, which cost a
# paid opd run. the worker cannot fall back to "5xx means retry", because a 5xx can arrive after
# the provider already began work and redispatching would bill the same teacher request twice.
#
# so retryability travels in the status code, which every intermediary preserves. 409 marks an
# in-progress replay the broker already owns, and 429 proves rejection before execution; neither
# can be produced by an intermediary in a way that hides completed provider work, so both are safe
# to retry without double-spending. the broker reports every retryable failure with one of these,
# and the worker rescues only these when the body is missing. both sides read this definition.
#
# a status belongs here ONLY if a worker seeing it bare -- with no body, from an unknown hop --
# can still conclude the provider did not begin work. that excludes 5xx, the ambiguous case that
# bills twice on retry. it also excludes 408: the broker's own ingress timeout is pre-dispatch and
# retryable, but a proxy between the broker and the worker emits 408 for its own timeout after
# dispatch, and the worker cannot tell those apart. so the broker reports its ingress timeout as
# 429 (the condition is a concurrency limit) rather than asking the worker to trust a bare 408.
BODY_INDEPENDENT_TRANSIENT_STATUSES = frozenset({409, 429})
DEFAULT_TRANSIENT_STATUS = 429


def validated_provider_status(value: object) -> int | None:
    if type(value) is int and 100 <= value <= 599:
        return value
    return None
