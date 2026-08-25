"""Behavior tests for the pure formatting/theme/table helpers in ``flash.cli.ui.render``.

These target the render helpers the existing suite (test_cli_render_theme / test_cli_help)
doesn't exercise: the tri-state env flag, the isatty fallback, the identity/login cards, the
256-color fallback + COLORFGBG parse guard, the interactive ``select`` prompt, colored-JSON leaf
types, provider redaction over lists, timestamp/age humanizers, the cost panel, object panel, and
the small confirmation cards. Color is dropped via ``NO_COLOR`` wherever we assert on contiguous
text (the same discipline the sibling theme tests use); the two tests that must see SGR codes force
color on with an explicit ``TERM``/``COLORTERM``/``COLORFGBG`` so they stay deterministic.
"""

from __future__ import annotations

import re
import types

import pytest

from flash.cli.ui import cost, env_panels, heartbeat, render, tables
from flash.cli.ui import heartbeat as ui_heartbeat


@pytest.fixture
def styled_plain(monkeypatch):
    """Themed layout on, color off — contiguous text we can substring-match."""
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")


# --------------------------------------------------------------------------- flags / gating


def test_flag_is_tristate(monkeypatch) -> None:
    monkeypatch.delenv("FLASH_STYLE", raising=False)
    assert render._flag("FLASH_STYLE") is None  # unset -> None (line 29)
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("FLASH_STYLE", truthy)
        assert render._flag("FLASH_STYLE") is True
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("FLASH_STYLE", falsy)
        assert render._flag("FLASH_STYLE") is False
    monkeypatch.setenv("FLASH_STYLE", "maybe")  # unrecognized -> None (line 35)
    assert render._flag("FLASH_STYLE") is None


def test_styled_falls_back_to_isatty_when_unforced(monkeypatch) -> None:
    monkeypatch.delenv("FLASH_STYLE", raising=False)

    class _Stdout:
        encoding = "utf-8"

        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(render.sys, "stdout", _Stdout(True))
    assert render.styled() is True  # line 43
    monkeypatch.setattr(render.sys, "stdout", _Stdout(False))
    assert render.styled() is False


# --------------------------------------------------------------------------- identity / login cards


def test_format_identity_and_whoami(styled_plain) -> None:
    me = {
        "email": "a@b.co",
        "org_id": "org1",
        "user_id": "u1",
        "project_id": "p1",
        "training_agent_job_id": "job1",
        "key_prefix": "sk_ab",
        "kind": "internal",
    }
    card = render.format_identity(me)
    for label in ("account", "org", "user", "project", "job", "key"):
        assert label in card
    assert "a@b.co" in card
    assert "org1" in card
    assert "job1" in card
    assert "sk_ab" in card
    assert "internal key" in card  # _KIND_LABEL mapping applied

    who = render.whoami(me)
    assert "logged in to flash" in who
    assert "a@b.co" in who


def test_format_identity_unknown_kind_and_no_prefix(styled_plain) -> None:
    # unknown kind falls through to the raw kind string; missing prefix -> bare kind, no ellipsis
    card = render.format_identity({"email": "x@y.z", "kind": "weird_kind"})
    assert "weird_kind" in card
    assert "..." not in card
    assert "…" not in card
    # no kind at all -> the default "api key" label
    card2 = render.format_identity({"email": "x@y.z"})
    assert "api key" in card2


def test_login_ok_and_failed(styled_plain) -> None:
    assert "account details unavailable" in render.login_ok(None)
    ok = render.login_ok({"email": "a@b.co", "kind": "internal"})
    assert "logged in to flash" in ok
    assert "a@b.co" in ok

    failed = render.login_failed("bad api key")
    assert "login failed" in failed
    assert "bad api key" in failed
    assert "try again" in failed


# --------------------------------------------------------------------------- theme / color codes


def test_theme_colorfgbg_non_integer_defaults_dark(monkeypatch) -> None:
    monkeypatch.setenv("COLORFGBG", "foo;bar")  # last field not an int -> ValueError guard
    assert render._theme() == "dark"


def test_sgr_uses_256_color_fallback_without_truecolor(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")  # not "dumb" -> color stays on
    monkeypatch.delenv("COLORTERM", raising=False)  # not truecolor -> 256 fallback
    monkeypatch.setenv("COLORFGBG", "15;0")  # light text on dark background -> dark theme
    out = render.badge("done")  # green/dark 256 fallback is 84
    assert "38;5;84" in out
    assert "38;2;" not in out  # no truecolor triple


# --------------------------------------------------------------------------- interactive select


def _feed_input(monkeypatch, answers):
    it = iter(answers)

    def fake_input(_prompt=""):
        try:
            return next(it)
        except StopIteration:  # pragma: no cover - defensive
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


def test_select_empty_answer_takes_default(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "0")
    opts = [("a", "Alpha", "first"), ("b", "Beta", "")]
    _feed_input(monkeypatch, [""])
    assert render.select("pick", opts, default=1) == "b"
    out = capsys.readouterr().out
    assert "Alpha" in out
    assert "Beta" in out
    assert "(default)" in out  # the default option is marked


def test_select_valid_digit_and_out_of_range_then_retry(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "0")
    opts = [("a", "Alpha", ""), ("b", "Beta", ""), ("c", "Gamma", "")]
    # a direct valid pick
    _feed_input(monkeypatch, ["2"])
    assert render.select("pick", opts) == "b"
    # out-of-range digit, then a non-digit, then a valid pick -> loops until valid
    _feed_input(monkeypatch, ["9", "x", "3"])
    assert render.select("pick", opts) == "c"
    assert "enter 1-3" in capsys.readouterr().out  # the retry hint was shown


def test_select_required_rejects_empty_and_requires_numeric_choice(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "0")
    opts = [("a", "Alpha", "uuid-a"), ("b", "Beta", "uuid-b")]
    _feed_input(monkeypatch, ["", "x", "2"])
    assert render.select_required("pick", opts) == "b"
    assert "empty input is not accepted" in capsys.readouterr().out


def test_select_required_rejects_eof(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "0")
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError()))
    with pytest.raises(ValueError, match="explicit numeric project choice"):
        render.select_required("pick", [("a", "Alpha", "uuid-a")])


def test_select_eof_returns_default(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "0")
    opts = [("a", "Alpha", ""), ("b", "Beta", "")]

    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    assert render.select("pick", opts, default=0) == "a"  # closed stdin -> default, no hang


# --------------------------------------------------------------------------- colored JSON leaves


def test_color_json_covers_all_leaf_types(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    obj = {
        "n": None,
        "t": True,
        "f": False,
        "i": 7,
        "x": 1.5,
        "s": "hi",
        "list": [1, "two"],
        "empty_list": [],
        "empty_dict": {},
    }
    out = render._json(obj)
    # leaf tokens survive the syntax highlighting (as substrings inside SGR wrappers)
    assert "null" in out
    assert "true" in out
    assert "false" in out
    assert "7" in out
    assert "1.5" in out
    assert '"hi"' in out
    assert "[]" in out
    assert "{}" in out
    assert "\x1b[" in out  # it really is the colored path


def test_kv_returns_empty_when_no_rows(styled_plain) -> None:
    assert render._kv([]) == ""
    assert render._kv([("only", None)]) == ""  # all-None rows filtered out


# --------------------------------------------------------------------------- provider redaction


def test_hide_provider_metadata_over_lists_and_bare_strings() -> None:
    # a bare provider name anywhere in the tree becomes "managed"; provider/flash_arm keys drop
    got = render._hide_provider_metadata(
        ["runpod", {"provider": "vast", "flash_arm": "x", "keep": 1}, 5, "other"]
    )
    assert got == ["managed", {"keep": 1}, 5, "other"]
    assert render._hide_provider_metadata("RunPod") == "managed"  # case-insensitive


# --------------------------------------------------------------------------- timestamp / age


def test_humanize_ts_formats_epoch_and_rejects_non_numbers() -> None:
    assert render._humanize_ts(1700000000) == "2023-11-14 22:13 UTC"
    assert render._humanize_ts(0) is None
    assert render._humanize_ts(-5) is None
    assert render._humanize_ts("nope") is None


def test_humanize_age_buckets(monkeypatch) -> None:
    # the age panel composes these two: _heartbeat_age_seconds turns a heartbeat ts into an age,
    # _humanize_age_seconds buckets it. asserted as that pair rather than through a one-line
    # wrapper, so the bucket edges stay pinned to the composition the panel actually renders.
    def humanize(value: object) -> str | None:
        return heartbeat._humanize_age_seconds(heartbeat._heartbeat_age_seconds(value))

    now = 1_000_000.0
    # patch the clock in the module that READS it: both helpers live in `flash.cli.ui.heartbeat`
    # and are re-exported through `render`, so patching a `time` on `render` would miss them (and
    # `render` no longer imports `time` at all).
    monkeypatch.setattr(ui_heartbeat.time, "time", lambda: now)
    assert humanize(now - 30) == "30s ago"  # < 90s
    assert humanize(now - 600) == "10m ago"  # < 5400s -> minutes
    assert humanize(now - 7200) == "2.0h ago"  # >= 5400s -> hours
    assert humanize(0) is None
    assert humanize("x") is None


# --------------------------------------------------------------------------- panels / tables


def test_deployments_table_truncates_long_detail(styled_plain) -> None:
    long_err = "E" * 100
    out = tables.deployments_table(
        [{"run_id": "r1", "deployment": {"state": "failed", "error": long_err}}]
    )
    assert ("E" * 61 + "...") in out  # 61 chars + ellipsis
    assert long_err not in out  # the full 100-char string was truncated away
    assert "failed" in out


def test_deployments_table_formats_verified_at_as_utc(styled_plain) -> None:
    out = tables.deployments_table(
        [{"run_id": "r1", "deployment": {"state": "ready", "verified_at": 1700000000}}]
    )
    assert "2023-11-14 22:13 UTC" in out  # epoch rendered as UTC, not a raw number
    assert "1700000000" not in out


def test_deployments_table_surfaces_base_url_and_handles_missing_value(styled_plain) -> None:
    out = tables.deployments_table(
        [
            {
                "run_id": "r1",
                "deployment": {
                    "state": "ready",
                    "openai_base_url": "https://serve.example/v1",
                },
            },
            {"run_id": "r2", "deployment": {"state": "ready"}},
        ]
    )
    assert "OPENAI BASE URL" in out
    assert "https://serve.example/v1" in out
    assert "r2" in out
    assert "-" in out


def test_run_status_shows_realized_cost_and_artifacts(styled_plain) -> None:
    obj = {
        "run_id": "flash-1",
        "state": "done",
        "spec": {
            "project": "11111111-1111-4111-8111-111111111111",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
        },
        "cost_usd": 0.10,
        "realized_cost_usd": 0.42,
        "artifacts_dir": "/tmp/run/artifacts",
        "error": "something broke",
    }
    out = render.run_status(obj)
    assert "realized" in out
    assert "$0.4200" in out  # money() with default 4 decimals
    assert "artifacts" in out
    assert "/tmp/run/artifacts" in out
    assert "something broke" in out


def test_object_panel_with_and_without_state(styled_plain) -> None:
    with_state = render.object_panel("train", {"state": "done", "run_id": "flash-1", "keep": 2})
    assert "train" in with_state
    assert "done" in with_state  # state badge line
    assert "flash-1" in with_state
    assert '"keep": 2' in with_state  # lossless JSON below

    no_state = render.object_panel("train", {"alpha": 3})
    assert '"alpha": 3' in no_state
    # no run/state badge line was emitted for a stateless object
    assert "●" not in no_state
    assert "done" not in no_state


def test_cost_panel_grpo_with_teacher_and_wall_cap(styled_plain) -> None:
    est = types.SimpleNamespace(
        method="grpo",
        model_id="Qwen/Qwen3.5-9B",
        steps=100,
        gpu="RTX 5090",
        gpu_vram_gb=32,
        # a stub standing in for CostEstimate has to carry the fields the panel reads; one card
        # offers exactly itself, so offered_vram_gb matches gpu_vram_gb here and the joined width
        # matches the billed count.
        offered_vram_gb=32,
        gpu_count=1,
        joined_gpu_count=1,
        required_vram_gb=24,
        gpu_hourly_usd=0.99,
        setup_seconds=300.0,
        seconds_per_step=2.5,
        train_seconds=1800.0,
        wall_capped=True,
        wall_clock_hours=0.6,
        billable_hours=0.5,
        teacher_api_usd=1.25,
        total_usd=1.75,
        notes=["heads up: capped"],
    )
    out = render.cost_panel(est)
    assert "pre-flight cost estimate" in out
    assert "vLLM init" in out  # grpo-only setup extra
    assert "capped at wall-clock limit" in out  # wall_capped branch
    assert "teacher api" in out
    assert "$1.25" in out
    assert "TOTAL" in out
    assert "$1.75" in out
    assert "heads up: capped" in out  # notes section rendered


def test_server_backed_sft_total_uses_billing_round_half_up(styled_plain) -> None:
    assert f"{1.005:.2f}" == "1.00", "the fixture must expose half-even formatting"
    assert "$1.01" in render.sft_cost_panel([("run", "model  [SFT]")], 1.005)


def test_cost_panel_sft_omits_optional_sections(styled_plain) -> None:
    est = types.SimpleNamespace(
        method="sft",
        model_id="Qwen/Qwen3.5-9B",
        steps=50,
        gpu="RTX 4090",
        gpu_vram_gb=24,
        offered_vram_gb=24,
        gpu_count=1,
        joined_gpu_count=1,
        required_vram_gb=20,
        gpu_hourly_usd=0.50,
        setup_seconds=120.0,
        seconds_per_step=1.0,
        train_seconds=600.0,
        wall_capped=False,
        wall_clock_hours=0.3,
        billable_hours=0.25,
        teacher_api_usd=0.0,
        total_usd=0.30,
        notes=[],
    )
    out = render.cost_panel(est)
    assert "vLLM init" not in out  # sft has no vLLM init in setup
    assert "teacher api" not in out  # zero teacher spend -> no row
    assert "capped at wall-clock limit" not in out
    assert "notes" not in out  # empty notes -> no notes section
    assert "TOTAL" in out
    assert "$0.30" in out


# --------------------------------------------------------------------------- misc small cards


def test_version_and_chat_label(styled_plain) -> None:
    assert "v1.2.3" in render.version("1.2.3")
    assert "assistant" in render.chat_label()


def test_env_list_with_local_sources(styled_plain) -> None:
    out = env_panels.env_list(["envs/one", "envs/two"])
    assert "local sources" in out
    assert "envs/one" in out
    assert "envs/two" in out
    assert f"{render.CLI_NAME} env push --project <project-uuid> --name <name> <path>" in out


def test_projects_table_lists_name_and_id(styled_plain) -> None:
    out = tables.projects_table(
        [{"id": "11111111-1111-4111-8111-111111111111", "name": "Test project"}]
    )
    assert "Test project" in out
    assert "11111111-1111-4111-8111-111111111111" in out


def test_env_published_and_pulled(styled_plain) -> None:
    pub = render.env_published("my-env")
    assert "published" in pub
    assert 'id = "my-env"' in pub  # the copy-pasteable config snippet

    pulled = render.env_pulled("./dest", "3 files")
    assert "pulled" in pulled
    assert "./dest" in pulled
    assert "3 files" in pulled

    bare = render.env_pulled("./dest")  # no detail line
    assert "pulled" in bare
    assert "./dest" in bare
    assert re.search(r"pulled.*dest", bare)


# --------------------------------------------------------------------------- live-run cost


def test_settled_cost_states_cover_every_runner_terminal_state() -> None:
    """render's copy of the settled set must not drift from the runner's.

    render.py is deliberately stdlib-only, so it restates which states mean "cost_usd is final"
    instead of importing them. If the runner ever adds a terminal state, this fails here rather
    than silently showing a settled charge as an estimate forever.
    """
    from flash.runner.lifecycle.state import TERMINAL_STATES

    assert TERMINAL_STATES | {"deployed"} == cost.SETTLED_COST_STATES


def test_run_cost_prefers_the_quote_while_a_run_is_live() -> None:
    # a running run has cost_usd 0.0 until the terminal transition writes it; showing that as the
    # cost tells the user a billing GPU is free.
    amount, is_estimate = cost.run_cost(
        {"state": "running", "cost_usd": 0.0, "estimated_cost_usd": 3.5}
    )
    assert (amount, is_estimate) == (3.5, True)


def test_run_cost_uses_the_settled_charge_once_terminal() -> None:
    for state in ("done", "failed", "cancelled", "dry_run", "deployed"):
        amount, is_estimate = cost.run_cost(
            {"state": state, "cost_usd": 1.25, "estimated_cost_usd": 3.5}
        )
        assert (amount, is_estimate) == (1.25, False), state


def test_run_cost_prefers_measured_spend_over_the_quote_while_live() -> None:
    # once a live run has accrued a real number, that beats the submit-time guess -- but it is
    # still not the settled charge, so it stays flagged as an estimate.
    amount, is_estimate = cost.run_cost(
        {"state": "running", "cost_usd": 0.75, "estimated_cost_usd": 3.5}
    )
    assert (amount, is_estimate) == (0.75, True)


def test_run_cost_without_a_quote_reports_zero_unflagged() -> None:
    # nothing to show is not the same as an estimate of zero; don't decorate a bare 0.0.
    assert cost.run_cost({"state": "queued", "cost_usd": 0.0}) == (0.0, False)
    assert cost.run_cost({"state": "queued", "estimated_cost_usd": None}) == (0.0, False)


def test_a_failed_run_that_never_measured_a_charge_is_not_reported_as_settled_zero() -> None:
    """A terminal 0.0 is the absence of a measurement, not a measurement of zero.

    ``cost_usd`` comes from the worker's metrics, so a run whose every attempt died before the
    worker produced any never gets one, and that is exactly the run most likely to have rented
    hardware over and over. one historical failure rented 47 instances, failed to confirm teardown
    on 44, and printed $0.0000 with no estimate marker, which reads as "this cost nothing".
    """
    amount, is_estimate = cost.run_cost(
        {"state": "failed", "cost_usd": 0.0, "estimated_cost_usd": 3.5}
    )
    assert (amount, is_estimate) == (3.5, True)


def test_realized_cogs_is_never_shown_as_the_customers_cost() -> None:
    """``realized_cost_usd`` is provider COGS, not what the customer is charged.

    ``runner_state.RunStatus`` says so directly: it is pulled from the provider's billing API by
    reconciliation and is "distinct from ``cost_usd`` (the flash.cost ESTIMATE we charge the
    customer)". Promoting it into the cost slot would bill the user our internal spend, and
    ``run_status`` already prints it on its own dedicated ``realized`` row.
    """
    amount, _ = cost.run_cost(
        {
            "state": "failed",
            "cost_usd": 0.0,
            "estimated_cost_usd": 3.5,
            "realized_cost_usd": 1.75,
        }
    )
    assert amount == 3.5


def test_a_failed_run_with_no_evidence_at_all_still_reports_a_bare_zero() -> None:
    """Without a quote there is nothing to show; do not invent a number."""
    assert cost.run_cost({"state": "failed", "cost_usd": 0.0}) == (0.0, False)


def test_a_settled_zero_that_is_not_a_failure_keeps_its_zero() -> None:
    """Only ``failed`` has an unmeasured zero. The other settled states earn theirs.

    A ``dry_run`` rents nothing and a ``cancelled``/``done`` run with no charge went through the
    normal accounting path, so resurfacing the submit quote for them would invent a charge nobody
    incurred -- the mirror image of the bug this fix is for.
    """
    for state in ("dry_run", "cancelled", "done", "deployed"):
        assert cost.run_cost({"state": state, "cost_usd": 0.0, "estimated_cost_usd": 3.5}) == (
            0.0,
            False,
        ), state


def test_a_terminal_estimate_is_not_labelled_run_in_progress(styled_plain) -> None:
    """The failed-run quote is flagged, but the run is over -- do not claim it is still running."""
    out = render.run_status(
        {
            "run_id": "flash-1",
            "state": "failed",
            "cost_usd": 0.0,
            "estimated_cost_usd": 3.5,
            "spec": {"model": "m", "algorithm": "sft"},
        }
    )
    assert "estimate, not measured" in out
    assert "run in progress" not in out


def test_run_status_marks_a_live_cost_as_an_estimate(styled_plain) -> None:
    obj = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo"},
        "cost_usd": 0.0,
        "estimated_cost_usd": 2.5,
    }
    out = render.run_status(obj)
    assert "$2.5000" in out
    assert "estimate, run in progress" in out
    assert "$0.0000" not in out


def test_runs_table_marks_live_rows_with_a_tilde(styled_plain) -> None:
    out = tables.runs_table(
        [
            {
                "run_id": "flash-live",
                "state": "running",
                "updated_at": 2,
                "spec": {"model": "m", "algorithm": "grpo"},
                "cost_usd": 0.0,
                "estimated_cost_usd": 2.5,
            },
            {
                "run_id": "flash-done",
                "state": "done",
                "updated_at": 1,
                "spec": {"model": "m", "algorithm": "sft"},
                "cost_usd": 1.25,
                "estimated_cost_usd": 2.5,
            },
        ]
    )
    assert "~$2.5000" in out  # live: the quote, flagged
    assert "$1.2500" in out  # settled: the real charge
    assert "~$1.2500" not in out


def test_runs_table_renders_an_ordered_gpu_pin_that_has_not_allocated_yet(styled_plain) -> None:
    """An unallocated ordered pin renders as a label without breaking the table."""
    out = tables.runs_table(
        [
            {
                "run_id": "flash-ordered",
                "state": "provisioning",
                "updated_at": 2,
                "spec": {
                    "model": "m",
                    "algorithm": "sft",
                    "gpu": {"type": ["A100 PCIe", "A100 SXM"]},
                },
                "remote": {},
            },
            {
                "run_id": "flash-other",
                "state": "done",
                "updated_at": 1,
                "spec": {"model": "m", "algorithm": "sft", "gpu": {"type": "H200"}},
                "cost_usd": 1.25,
            },
        ]
    )
    assert "A100 PCIe | A100 SXM" in out
    # the unrelated run is still listed, which is what the crash used to prevent.
    assert "flash-other" in out
    assert "H200" in out
