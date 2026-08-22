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
# so retryability travels in the status code, which every intermediary preserves. 429 proves
# rejection before execution and is safe to retry without double-spending. 409 is overloaded by
# the broker: it carries both an in-progress replay and terminal ledger conflicts, so it is
# retryable only when its structured body says so.
#
# a status belongs here ONLY if a worker seeing it bare -- with no body, from an unknown hop --
# can still conclude the provider did not begin work. that excludes 409, 5xx, and 408. 5xx can
# follow completed provider work. 408 can come from a proxy after dispatch. 409 can represent an
# ambiguous or terminal ledger outcome. the broker reports retryable ingress limits as 429 rather
# than asking the worker to trust one of those ambiguous statuses.
BODY_INDEPENDENT_TRANSIENT_STATUSES = frozenset({429})

# structured broker errors may preserve 409 for request_in_progress compatibility. unlike a bare
# 409, its transient classification is authoritative and retries the same ledger request.
BROKER_TRANSIENT_STATUSES = BODY_INDEPENDENT_TRANSIENT_STATUSES | {409}
DEFAULT_TRANSIENT_STATUS = 429


def validated_provider_status(value: object) -> int | None:
    if type(value) is int and 100 <= value <= 599:
        return value
    return None
