"""What a lambda seed owes on an interrupt during launch. MUST NOT import the ``jobs`` package ``__init__``.

Split out of that ``__init__`` to keep it under the file-size gate. Nothing here reads a name the
tests patch on ``jobs``, so it moves without changing any resolution: the run-label sweep and the
exact terminate both stay behind their callers there.
"""

from __future__ import annotations

import contextlib

# Stamped on the propagating exception once some path has terminated the exact instance it rented.
# The coarse run-label reap must stay armed until exact ownership is taken, and must then stand
# down: layering terminate_run_instances(run_id) on top of an exact cleanup would also reap every
# other concurrently-launched attempt of this run.
_EXACT_CLEANUP_ATTR = "_flash_exact_cleanup_done"


class _CoarseReapGuard:
    """What this attempt owes on an interrupt: a run-label sweep, an exact terminate, or nothing.

    Armed for exactly the window a launch request is in flight with no instance id in hand, when
    the label reap is the ONLY thing that can find a box that is rented but not yet named. It is
    an object rather than a local so each region can arm it around its own request instead of the
    caller arming across the whole walk.

    ``owns`` narrows the guard the instant an id exists. Ownership transfer would otherwise span
    two statements (the publication helper returns, then the caller disarms), and an interrupt
    landing between them reaps by run label and terminates every other concurrent seed sharing
    that label. Disarming before the handle is returned would close that window by opening a
    worse one: an interrupt would then leave the box rented with nothing to clean it. Holding the
    id keeps the window covered by cleanup that names exactly one instance.
    """

    def __init__(self) -> None:
        self.armed = False
        self.instance_id: str | None = None

    def arm(self) -> None:
        self.armed = True
        self.instance_id = None

    def owns(self, instance_id: str) -> None:
        """Take exact ownership of a launched box, retiring the label reap for it."""
        self.armed = True
        self.instance_id = instance_id

    def disarm(self) -> None:
        self.armed = False
        self.instance_id = None


def _launch_failed_before_the_request(error: BaseException) -> bool:
    """Whether a ``launch_instance`` failure provably happened before the create was issued.

    ``launch_instance`` repeats ``require_create_allowance`` after the caller's own check and
    before it builds a body or issues the POST, so near the 60-second threshold the caller's check
    can pass and the API's repeat can fail. That raises a bare ``RuntimeError`` carrying the
    allowance message, and nothing downstream of the POST raises that type with that text: the
    request path raises ``LambdaApiError``. Treat it as proof that no box was rented, so the guard
    stands down instead of sweeping the run label for a create that never left the process.
    """
    return type(error) is RuntimeError and "provider allowance remaining" in str(error)


def _mark_exact_cleanup(error: BaseException) -> None:
    with contextlib.suppress(BaseException):  # builtin exceptions accept attributes; be safe anyway
        setattr(error, _EXACT_CLEANUP_ATTR, True)


def _exact_cleanup_taken(error: BaseException) -> bool:
    return bool(getattr(error, _EXACT_CLEANUP_ATTR, False))
