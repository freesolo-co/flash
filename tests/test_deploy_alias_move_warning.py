"""Deploying a second checkpoint of a run must say it is moving the shared alias.

`deploy` registers every checkpoint under the bare run id, so deploying `RUN/step-50` after
`RUN/step-100` repoints the one alias rather than standing up a second endpoint. Everyone
chatting bare `RUN` changes model with no signal, which is what made a redeploy read as a
serving regression instead of the deploy that caused it.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from types import SimpleNamespace

import pytest

import flash.cli.commands as commands
import flash.cli.commands.deploy as deploy_module
from flash.cli.commands.deploy import _alias_move_warning, cmd_deploy
from flash.client import ApiError, ClientError


class _Client:
    """A client whose deployment read records how it was asked."""

    def __init__(self, current=None, *, raises=None) -> None:
        self._current = current
        self._raises = raises
        self.deploy_calls: list[tuple] = []
        self.read_calls: list[tuple] = []
        self.read_timeouts: list[float | None] = []
        self.read_deadlines: list[float | None] = []

    def deployed_checkpoint(self, run_id, timeout=None, *, body_deadline=None):
        self.read_calls.append(run_id)
        self.read_timeouts.append(timeout)
        self.read_deadlines.append(body_deadline)
        if self._raises is not None:
            raise self._raises
        return self._current

    def deploy(self, run_id, dry_run=False):
        self.deploy_calls.append((run_id, dry_run))
        return {"state": "queued", "run_id": run_id}


def _args(run_id: str, *, dry_run: bool = False):
    return SimpleNamespace(run_id=run_id, dry_run=dry_run, wait=None)


def _ready(step):
    return {"run_id": "flash-1", "state": "ready", "checkpoint_step": step}


def test_warns_that_deploying_another_step_moves_the_shared_alias() -> None:
    """The warning must name both checkpoints: which one is lost and which one takes over."""
    client = _Client(_ready(100))

    warning = _alias_move_warning(client, "flash-1", 50)

    assert warning is not None
    assert "step-100" in warning
    assert "step-50" in warning
    assert "flash-1" in warning


def test_warning_reads_the_deployment_without_the_step_filter() -> None:
    """The displaced record is exactly the one `deployment_for`'s step filter would hide."""
    client = _Client(_ready(100))

    _alias_move_warning(client, "flash-1", 50)

    assert client.read_calls == ["flash-1"]


def test_redeploying_the_same_checkpoint_displaces_nothing() -> None:
    """Re-deploying the live checkpoint moves the alias onto what it already serves."""
    assert _alias_move_warning(_Client(_ready(100)), "flash-1", 100) is None


def test_final_adapter_is_named_final_not_step_none() -> None:
    """`checkpoint_step: None` is the final adapter; a user never typed `step-None`."""
    warning = _alias_move_warning(_Client(_ready(None)), "flash-1", 50)

    assert warning is not None
    assert "serves final" in warning
    assert "None" not in warning


def test_deploying_final_over_a_checkpoint_warns_in_the_other_direction() -> None:
    """The final adapter replacing step-100 moves the alias just as much."""
    warning = _alias_move_warning(_Client(_ready(100)), "flash-1", None)

    assert warning is not None
    assert "serves step-100" in warning
    assert "deploying final" in warning


def test_a_displaced_final_adapter_is_reachable_by_revision_not_a_step_selector() -> None:
    """`step-N` cannot address the displaced final adapter: there is no `/final` selector.

    `_parse_chat_target` accepts a bare run id, `RUN/step-N`, or a full immutable revision. Once
    the alias moves, `step-N` names the INCOMING checkpoint and the bare id is the thing that just
    changed -- so the generic "compare with step-N" advice reaches everything except the model the
    user just lost. Its immutable revision is the one selector that still resolves.
    """
    current = {
        "run_id": "flash-1",
        "state": "ready",
        "checkpoint_step": None,
        "adapter_revision": "flash-1@abc123",
    }

    warning = _alias_move_warning(_Client(current), "flash-1", 50) or ""

    assert "flash-1@abc123" in warning
    assert "no `/final` selector" in warning


def test_a_displaced_final_adapter_without_a_revision_admits_it_is_unreachable() -> None:
    """An older plane can answer without `adapter_revision`; never print an empty selector.

    And never defer the user to a later command for it. `cmd_deploy` POSTs as soon as this prints,
    after which the deployments listing carries only the NEW record -- `public_deployment` strips
    `previous_deployment` -- so the displaced revision is not in it. Promising a selector the next
    command cannot produce is worse than saying there is none.
    """
    warning = _alias_move_warning(_Client(_ready(None)), "flash-1", 50) or ""

    assert "no `/final` selector" in warning
    assert "not be addressable after this deploy" in warning
    assert "``" not in warning
    # the listing does not contain the displaced revision, so it must not be offered as the way
    # to recover one.
    assert "models deployments" not in warning


def test_the_step_selector_advice_is_qualified_by_plane_support() -> None:
    """`ApiClient.chat` refuses a step target on a plane without `chat_step_selector_v1`.

    The capability is only knowable via /v1/health, which an advisory warning must not spend, and
    `_chat_step_selector_available` starts False -- so a negative reading here cannot distinguish
    "unsupported" from "not checked yet". The warning therefore states the condition rather than
    asserting the command works, and names the selector that works on every plane.
    """
    warning = _alias_move_warning(_Client(_ready(100)), "flash-1", 50) or ""

    assert "step-N" in warning
    assert "immutable adapter revision" in warning
    assert "does not support step selectors" in warning


def test_a_displaced_checkpoint_needs_no_revision_hint() -> None:
    """`step-100` still addresses a displaced CHECKPOINT, so the extra hint would be noise."""
    warning = _alias_move_warning(_Client(_ready(100)), "flash-1", 50) or ""

    assert "/final" not in warning


def test_a_displaced_checkpoint_names_its_revision_when_the_record_carries_one() -> None:
    """`step-N` is not enough on its own: a plane without `chat_step_selector_v1` rejects it.

    The qualified advice tells such a user to reach the checkpoint by immutable revision, which is
    useless without the revision itself. It is on this record, and this is the last moment it can
    be printed -- after the deploy, the listing carries only the new record, because
    `public_deployment` strips `previous_deployment`.
    """
    current = {
        "run_id": "flash-1",
        "state": "ready",
        "checkpoint_step": 100,
        "adapter_revision": "flash-1@abc123",
    }

    warning = _alias_move_warning(_Client(current), "flash-1", 50) or ""

    assert "flash-1@abc123" in warning
    # the final-adapter wording must not leak into a numbered checkpoint, which keeps `step-N`
    # wherever the plane supports it.
    assert "/final" not in warning


def test_a_run_with_no_deployment_is_not_a_replacement() -> None:
    """A first deploy displaces nothing, so it must stay silent."""
    assert _alias_move_warning(_Client(None), "flash-1", 50) is None


@pytest.mark.parametrize("state", ["queued", "failed", "reconciling"])
def test_only_a_servable_revision_can_be_displaced(state: str) -> None:
    """Nothing is serving off the alias yet, so moving it costs nobody their model."""
    current = {"run_id": "flash-1", "state": state, "checkpoint_step": 100}

    assert _alias_move_warning(_Client(current), "flash-1", 50) is None


def test_a_failed_revocation_leaves_the_alias_ambiguous_not_idle() -> None:
    """`revocation_failed` is backend cleanup that did NOT finish, so the alias may still resolve.

    Local authority is revoked, but the target the backend serves is exactly what could not be
    confirmed -- and `_validate_deploy_request` rejects only the busy set, so a deploy proceeds
    from here and can replace a target that is still live. Reading it as "not serving" suppressed
    the warning in a case where the alias is already ambiguous.
    """
    current = {"run_id": "flash-1", "state": "revocation_failed", "checkpoint_step": 100}

    warning = _alias_move_warning(_Client(current), "flash-1", 50) or ""

    assert "revocation did not complete" in warning
    assert "cannot be determined from here" in warning
    # the same reason the unsettled arm names no checkpoint: this record's step describes the
    # deployment whose revocation failed, not a confirmed live alias target.
    assert "step-100" not in warning


def test_a_failed_revocation_warns_even_on_the_same_step() -> None:
    """The step comparison cannot clear it: what the alias holds is the unknown."""
    current = {"run_id": "flash-1", "state": "revocation_failed", "checkpoint_step": 50}

    assert _alias_move_warning(_Client(current), "flash-1", 50) is not None


def _unsettled(step):
    """A `reconciling` record whose activation outcome was never recorded.

    Its `checkpoint_step` is the INCOMING attempt. The plane resolves what the alias really holds
    from `adapter_alias_target` / `previous_deployment`, neither of which reaches the client.
    """
    return {
        "run_id": "flash-1",
        "state": "reconciling",
        "checkpoint_step": step,
        "activation_outcome_unknown": True,
    }


def test_an_unsettled_activation_may_already_hold_the_alias() -> None:
    """`reconciling` + `activation_outcome_unknown` is the case the alias is MOST likely to move.

    The plane permits replacing exactly that record instead of rejecting it as busy, and resolves
    the authoritative target through `_activation_predecessor` when it does. Reading the state
    alone as "not serving" silenced the warning precisely where it is needed.
    """
    warning = _alias_move_warning(_Client(_unsettled(100)), "flash-1", 50)

    assert warning is not None
    assert "activation never settled" in warning
    assert "step-50" in warning


def test_an_unsettled_activation_does_not_name_a_checkpoint_it_cannot_know() -> None:
    """`checkpoint_step` there is the incoming attempt, so naming it would name the WRONG one.

    `tests/test_server_api.py` covers an attempted final revision while the alias still serves
    step-20: the record's own step describes neither what is live nor what will be displaced, and
    the predecessor that does is stripped by `public_deployment`. Saying "serves step-N" from this
    field is a confident wrong answer, which is worse than declining to name one.
    """
    warning = _alias_move_warning(_Client(_unsettled(100)), "flash-1", 50)

    assert warning is not None
    assert "step-100" not in warning
    assert "serves step-" not in warning
    assert "cannot be determined" in warning


def test_an_unsettled_activation_warns_even_on_the_same_step() -> None:
    """Equality with an untrustworthy step proves nothing, so it cannot buy silence.

    A settled record suppresses the warning on a same-step redeploy because the step is then a
    fact. Here it is the incoming attempt, so the alias may still be on something else entirely.
    """
    warning = _alias_move_warning(_Client(_unsettled(50)), "flash-1", 50)

    assert warning is not None
    assert "activation never settled" in warning


def test_a_settled_ready_record_is_stated_as_fact() -> None:
    """A confirmed `ready` revision is hedged by nothing: it IS what the id serves."""
    warning = _alias_move_warning(_Client(_ready(100)), "flash-1", 50)

    assert warning is not None
    assert "currently serves step-100" in warning
    assert "may currently serve" not in warning


def test_a_numeric_string_step_still_compares_as_a_number() -> None:
    """A plane that JSON-encodes the step as a string must not read as a different checkpoint."""
    current = {"run_id": "flash-1", "state": "ready", "checkpoint_step": "50"}

    assert _alias_move_warning(_Client(current), "flash-1", 50) is None


@pytest.mark.parametrize("failure", [ApiError(500, "boom"), ClientError("unreachable")])
def test_an_unreadable_plane_produces_no_warning_instead_of_an_error(failure) -> None:
    """The read is advisory: a plane that cannot answer must not fail the deploy it decorates."""
    assert _alias_move_warning(_Client(raises=failure), "flash-1", 50) is None


@pytest.mark.parametrize(
    "failure",
    [
        # the client translates urllib's HTTPError and URLError and documents that anything else
        # propagates, so a connection dropped mid-read arrives as the bare OSError urllib raised.
        ConnectionResetError("connection reset by peer"),
        http.client.IncompleteRead(b"{"),
        # a plane answering a shape no version of this client models still must not break deploy.
        TypeError("unexpected payload"),
    ],
)
def test_an_untranslated_transport_failure_is_contained_not_printed(failure, monkeypatch) -> None:
    """The advisory read runs on a worker thread, where an escaping exception is not just silent.

    `ApiError`/`ClientError` do not cover every way a read fails: an untranslated one reaches the
    thread hook and prints a traceback to stderr across a deploy that is otherwise proceeding
    normally, so the warning nobody asked for becomes noise on a command that worked.

    Asserted at the thread hook rather than on captured stderr: the hook is what would print, and
    pytest installs its own, so a stderr assertion passes whether or not the exception escaped.
    """
    escaped = []
    monkeypatch.setattr(threading, "excepthook", lambda args: escaped.append(args.exc_type))

    assert _alias_move_warning(_Client(raises=failure), "flash-1", 50) is None

    # the worker is a daemon thread, so wait for it before judging whether anything escaped.
    for thread in threading.enumerate():
        if thread is not threading.current_thread():
            thread.join(5.0)
    assert escaped == []


def test_the_advisory_read_is_bounded_well_under_the_client_default() -> None:
    """An advisory line must not hold a real deploy for the client's full 60s default.

    Unbounded, a stalled status GET delays every deploy behind a warning nobody asked for.
    """
    client = _Client(_ready(100))

    _alias_move_warning(client, "flash-1", 50)

    assert client.read_timeouts == [deploy_module._ALIAS_WARNING_READ_SECONDS]
    assert deploy_module._ALIAS_WARNING_READ_SECONDS < 60.0


def test_the_advisory_read_is_bounded_in_wall_clock_not_just_per_socket_read() -> None:
    """A socket timeout alone does not bound the read, and this read must be bounded.

    `timeout` restarts on every byte that arrives, so a proxy trickling a response just inside
    it holds the GET open for as long as it likes -- unbounded in exactly the way the advisory
    read must not be, since the deploy the user actually asked for waits behind it. Only the
    wall-clock body deadline bounds the whole read.
    """
    client = _Client(_ready(100))

    _alias_move_warning(client, "flash-1", 50)

    assert client.read_deadlines == [deploy_module._ALIAS_WARNING_READ_SECONDS]


@pytest.mark.wallclock
def test_a_read_that_overruns_its_budget_is_abandoned_rather_than_waited_out() -> None:
    """Neither client bound covers every phase, so the caller has to bound the total itself.

    `timeout` restarts on each socket operation and the body deadline is only consulted once
    `urlopen` has parsed the headers, so a peer trickling the status line or headers just inside
    the socket timeout stalls for timeout x however many pauses it takes -- measured at 12s
    against a 2s budget. The deploy the user actually asked for is waiting behind this, so an
    overrunning advisory read has to be given up on, not waited out.
    """

    released = threading.Event()
    returned_late = []

    class _Stalling:
        def deployed_checkpoint(self, run_id, timeout=None, *, body_deadline=None):
            # stands in for a peer trickling headers: nothing to read, and no bound the client
            # applies covers it. released at the end so the thread cannot outlive the test.
            returned_late.append(released.wait(30))
            return _ready(100)

    start = time.monotonic()
    warning = _alias_move_warning(_Stalling(), "flash-1", 50)
    elapsed = time.monotonic() - start
    # snapshot BEFORE releasing the worker. releasing first lets it wake and append between the
    # release and the read, so the abandonment assertion below could fail on a correct
    # implementation -- the list only means "still in flight" while the worker is still blocked.
    still_in_flight = list(returned_late)
    released.set()

    # no warning is the documented outcome for a plane that cannot answer in time.
    assert warning is None
    # generous against CI scheduling noise, and still far below the 30s the read would take.
    assert elapsed < deploy_module._ALIAS_WARNING_READ_SECONDS + 5.0, f"waited {elapsed:.2f}s"
    # the read had NOT finished when the caller gave up: it was abandoned, not merely slow.
    assert still_in_flight == []


def test_a_read_that_answers_in_time_still_warns() -> None:
    """Bounding the read must not cost the warning on a plane that answers normally."""
    assert "step-100" in (_alias_move_warning(_Client(_ready(100)), "flash-1", 50) or "")


@pytest.mark.parametrize(
    "bad_step",
    [
        "abc",
        "",
        [],
        {},
        object(),
        # `json.loads` accepts the non-standard `NaN`/`Infinity` literals, so a single 2xx body
        # reaches int() with each of these. they raise ValueError and OverflowError respectively,
        # and a guard that catches only the obvious exceptions still crashes the deploy.
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_an_unreadable_checkpoint_step_cannot_fail_the_deploy(bad_step) -> None:
    """A 2xx carrying a step this client cannot parse is an unreadable record, not a crash.

    A proxy or older plane can answer with a nonnumeric `checkpoint_step`. Converting it outside
    the guarded read let the advisory warning traceback out of the command it decorates.
    """
    current = {"run_id": "flash-1", "state": "ready", "checkpoint_step": bad_step}

    assert _alias_move_warning(_Client(current), "flash-1", 50) is None


def test_a_non_finite_step_is_reachable_from_a_real_response_body() -> None:
    """Anchor the case above in the decoder the client actually uses, not an invented value.

    `float("inf")` in a parametrize list only proves the guard catches a value a test made up.
    The client decodes with `json.loads`, which accepts the non-standard `Infinity` and `NaN`
    literals, so a control plane emitting either really does hand this code a non-finite step.
    """
    decoded = json.loads(b'{"state": "ready", "checkpoint_step": Infinity}')

    assert decoded["checkpoint_step"] == float("inf")
    with pytest.raises(OverflowError):
        int(decoded["checkpoint_step"])
    assert _alias_move_warning(_Client({**decoded, "run_id": "flash-1"}), "flash-1", 50) is None


@pytest.mark.parametrize("lossy_step", [True, False, 1.5, 0.5, -2.5])
def test_a_lossy_checkpoint_step_is_unreadable_rather_than_coerced(lossy_step) -> None:
    """`int()` does not only reject: it silently coerces, and both outcomes are wrong.

    `json.loads` produces `True` from `true` and `1.5` from a fractional number, and `int()`
    answers 1 for both without raising -- so the earlier exception guard never sees them. Read as
    a number, `true` suppresses the warning entirely when deploying step-1, and any other target
    gets told step-1 is live when nothing of the sort is. Neither is a checkpoint, so the record
    is unreadable.
    """
    current = {"run_id": "flash-1", "state": "ready", "checkpoint_step": lossy_step}

    # step 1 is the value int() coerces to, so it is the target that would be silenced.
    assert _alias_move_warning(_Client(current), "flash-1", 1) is None
    # and any other target is the one that would be told step-1 is live.
    assert _alias_move_warning(_Client(current), "flash-1", 50) is None


@pytest.mark.parametrize("whole_step", [100.0, 0.0])
def test_a_whole_float_step_is_still_read_as_that_checkpoint(whole_step) -> None:
    """Rejecting lossy conversions must not reject a lossless one.

    A plane that JSON-encodes the step as `100.0` loses nothing by converting; only a fractional
    value does. Rejecting every float would turn a readable record into a silent no-warning.
    """
    current = {"run_id": "flash-1", "state": "ready", "checkpoint_step": whole_step}

    assert _alias_move_warning(_Client(current), "flash-1", int(whole_step)) is None
    warning = _alias_move_warning(_Client(current), "flash-1", 50) or ""
    assert f"serves step-{int(whole_step)}" in warning


def test_a_numeric_string_step_is_still_compared_as_a_number() -> None:
    """Normalizing must not turn `"100"` into a spurious warning against step 100."""
    current = {"run_id": "flash-1", "state": "ready", "checkpoint_step": "100"}

    assert _alias_move_warning(_Client(current), "flash-1", 100) is None
    warning = _alias_move_warning(_Client(current), "flash-1", 50)
    assert warning is not None
    assert "serves step-100" in warning


def test_a_malformed_step_does_not_stop_the_deploy_itself(monkeypatch, capsys) -> None:
    """End to end: the deploy still goes through, with no traceback and no warning."""
    client = _Client({"run_id": "flash-1", "state": "ready", "checkpoint_step": "abc"})
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    assert cmd_deploy(_args("flash-1/step-50")) == 0

    assert "currently serves" not in capsys.readouterr().err
    assert client.deploy_calls == [("flash-1/step-50", False)]


def test_deployed_state_counts_as_servable() -> None:
    """`deployed` serves just like `ready`; missing it would silence a real replacement."""
    current = {"run_id": "flash-1", "state": "deployed", "checkpoint_step": 100}

    assert _alias_move_warning(_Client(current), "flash-1", 50) is not None


def test_deploy_warns_before_it_moves_the_alias(monkeypatch, capsys) -> None:
    """After the POST the alias has already moved; the warning has to precede it."""
    client = _Client(_ready(100))
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    assert cmd_deploy(_args("flash-1/step-50")) == 0

    err = capsys.readouterr().err
    assert "warning:" in err
    assert "step-100" in err
    # the read happened, and the deploy still went through: this warns, it does not block.
    assert client.read_calls == ["flash-1"]
    assert client.deploy_calls == [("flash-1/step-50", False)]


def test_deploy_points_at_the_explicit_checkpoint_selector(monkeypatch, capsys) -> None:
    """Naming the fix matters: `RUN/step-N` compares checkpoints without fighting over the alias."""
    monkeypatch.setattr(commands, "client_from_config", lambda: _Client(_ready(100)))
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    cmd_deploy(_args("flash-1/step-50"))

    assert "models chat flash-1/step-N" in capsys.readouterr().err


def test_a_dry_run_registers_nothing_so_it_warns_about_nothing(monkeypatch, capsys) -> None:
    """`--dry-run` never touches the alias, so a replacement warning there is false."""
    client = _Client(_ready(100))
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    cmd_deploy(_args("flash-1/step-50", dry_run=True))

    assert "currently serves" not in capsys.readouterr().err
    assert client.read_calls == []


def test_first_deploy_of_a_run_says_nothing_about_aliases(monkeypatch, capsys) -> None:
    """A run with nothing deployed must not gain a warning about a checkpoint that does not exist."""
    monkeypatch.setattr(commands, "client_from_config", lambda: _Client(None))
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    assert cmd_deploy(_args("flash-1/step-50")) == 0

    assert "currently serves" not in capsys.readouterr().err
