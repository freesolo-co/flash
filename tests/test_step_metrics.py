"""verl GRPO per-step metrics reconstructed from the trainer's stdout.

an in-process trainer would feed `flash runs log -f` from a TrainerCallback. verl's trainer runs
out of process and cannot host one, so rl_train reconstructs the same `metrics_last` backlog from
verl's own per-step stdout line. these tests pin the parse against the
line format verl actually prints (LocalLogger.concat_dict_to_str) under the interpreter flash
actually ships (Dockerfile.worker installs numpy 2.2.6 into /opt/verl-venv), plus the lifecycle
sites that carry the backlog to the console.
"""

import ast
import inspect
import textwrap

from flash.engine.worker.backend_common import (
    append_step_metrics,
    parse_verl_metric,
    parse_verl_step_metrics,
)

# a realistic verl step line: ray tags worker stdout with a pid prefix, and reduce_metrics returns
# numpy scalars that pprint renders as np.float64(...) under numpy>=2.
_RAY_PREFIX = "(TaskRunner pid=3125) "
_STEP_LINE = (
    "step:7 - critic/rewards/mean:0.625 - actor/pg_loss:-0.0134 - actor/grad_norm:1.5 - "
    "actor/kl_loss:0.002 - actor/entropy:0.91 - response_length/mean:213.5 - "
    "response_length/clip_ratio:0.125"
)


def test_parses_every_rendered_field_from_a_step_line():
    metrics = parse_verl_step_metrics(_STEP_LINE)

    assert metrics == {
        "step": 7,
        "reward": 0.625,
        "grad_norm": 1.5,
        "kl": 0.002,
        "entropy": 0.91,
        "mean_completion_tokens": 213.5,
        "truncation_rate": 0.125,
    }


def test_parses_ray_prefixed_numpy2_line():
    # the two production realities a synthetic test would miss: ray's "(TaskRunner pid=N) " stdout
    # prefix (so an anchored regex parses nothing) and numpy>=2's np.float64(...) repr (so a
    # plain-float regex drops the column). both are live in the shipped worker image.
    line = _RAY_PREFIX + (
        "step:12 - critic/rewards/mean:np.float64(0.4) - actor/grad_norm:np.float32(2.25) - "
        "response_length/mean:np.float64(180.0)"
    )

    metrics = parse_verl_step_metrics(line)

    assert metrics == {
        "step": 12,
        "reward": 0.4,
        "grad_norm": 2.25,
        "mean_completion_tokens": 180.0,
    }


def test_non_finite_metric_is_dropped_without_losing_its_siblings():
    # a diverged run prints nan; rendering it as a column is meaningless and it can poison the json
    # payload downstream, so drop that field only -- the step and the healthy fields must survive.
    line = "step:3 - critic/rewards/mean:nan - actor/grad_norm:inf - actor/entropy:0.5"

    metrics = parse_verl_step_metrics(line)

    assert metrics == {"step": 3, "entropy": 0.5}


def test_missing_metrics_are_absent_rather_than_zero():
    # verl has no frac_reward_zero_std analogue and omits some keys on non-actor steps. an absent
    # metric must not render as 0.0, which would read as a real measurement.
    metrics = parse_verl_step_metrics("step:1 - critic/rewards/mean:0.5")

    assert metrics == {"step": 1, "reward": 0.5}


def test_non_step_lines_are_ignored():
    for line in (
        "",
        "Training Progress:  12%|#2        | 6/50",
        "(TaskRunner pid=3125) validation metrics: val-core/mean:0.4",
        "step:not-a-number - critic/rewards/mean:0.5",
    ):
        assert parse_verl_step_metrics(line) is None


def test_validation_only_record_yields_no_row():
    # verl logs its pre-training validation pass as its own record at the current step counter
    # (ray_trainer.py `logger.log(data=val_metrics, step=self.global_steps)`), and every key on it
    # is namespaced val-core/ or val-aux/. emitting {"step": n} for it would render a row carrying a
    # step number and nothing else, and on a resumed run it would displace a real training row
    # because the backlog deduplicates by step.
    line = (
        "(TaskRunner pid=3125) step:0 - val-core/openai/gsm8k/reward/mean@1:0.31"
        " - val-aux/num_turns/mean:1.0"
    )

    assert parse_verl_step_metrics(line) is None


def test_metric_key_does_not_cross_match_a_longer_sibling():
    # response_length_non_aborted/mean ENDS WITH response_length/mean, so an unanchored substring
    # search would report the non-aborted value under the wrong column.
    line = "step:4 - response_length_non_aborted/mean:999.0 - response_length/mean:213.5"

    assert parse_verl_metric(line, "response_length/mean") == 213.5
    assert parse_verl_metric(line, "response_length_non_aborted/mean") == 999.0


def test_replayed_step_replaces_rather_than_duplicates():
    # verl reprints a step on a validation pass and a resumed run replays its resume step; appending
    # blindly would render the same step twice in the cli table.
    backlog: list[dict] = []
    append_step_metrics(backlog, {"step": 1, "reward": 0.1}, limit=8)
    append_step_metrics(backlog, {"step": 2, "reward": 0.2}, limit=8)
    append_step_metrics(backlog, {"step": 1, "reward": 0.9}, limit=8)

    assert backlog == [{"step": 2, "reward": 0.2}, {"step": 1, "reward": 0.9}]


def test_backlog_is_bounded_to_the_most_recent_steps():
    backlog: list[dict] = []
    for step in range(10):
        append_step_metrics(backlog, {"step": step}, limit=4)

    assert backlog == [{"step": 6}, {"step": 7}, {"step": 8}, {"step": 9}]


def test_backlog_is_mutated_in_place_for_the_heartbeat_reader():
    # the liveness thread closes over this list while the stdout loop writes it. rebinding would
    # leave that reader pinned to a stale, permanently empty list.
    backlog: list[dict] = []
    alias = backlog
    for step in range(6):
        append_step_metrics(backlog, {"step": step}, limit=3)

    assert alias is backlog
    assert alias == [{"step": 3}, {"step": 4}, {"step": 5}]


def _verl_rl_tree() -> ast.Module:
    from flash.engine.worker import rl_train

    return ast.parse(textwrap.dedent(inspect.getsource(rl_train.run_rl_train)))


def test_verl_rl_lifecycle_heartbeats_carry_latest_metrics():
    # mirrors test_worker_init_heartbeat.test_rl_lifecycle_heartbeats_carry_latest_metrics, which
    # pins only rl.run_rl (trl). that blind spot is exactly why verl shipped without metrics_last:
    # the parity test could not see the backend that lacked it.
    tree = _verl_rl_tree()

    terminal_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "heartbeat"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "rl_trained"
    ]
    assert len(terminal_calls) == 1
    terminal_keywords = {kw.arg: kw.value for kw in terminal_calls[0].keywords}
    assert "metrics_last" in terminal_keywords

    liveness_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "liveness_heartbeat"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in {"rl_step", "rl_finalizing"}
    ]
    assert len(liveness_calls) == 2
    for call in liveness_calls:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert "fields" in keywords
        assert "metrics_last" in ast.unparse(keywords["fields"])

    write_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_train_meta"
    ]
    assert len(write_calls) == 1
    write_keywords = {kw.arg: kw.value for kw in write_calls[0].keywords}
    assert "heartbeat_fields" in write_keywords
    assert "metrics_last" in ast.unparse(write_keywords["heartbeat_fields"])


def test_verl_rl_publishes_backlog_to_the_error_path_global():
    # worker/__init__.py:_err_metrics reads LATEST_GRPO_METRICS_LAST so a run that dies mid-training
    # still reports the steps it completed. trl's callback writes it; verl must too.
    tree = _verl_rl_tree()

    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "LATEST_GRPO_METRICS_LAST"
            for target in node.targets
        )
    ]
    assert assigns, "verl rl must publish its backlog to LATEST_GRPO_METRICS_LAST"


def test_train_meta_series_are_collected_only_from_step_lines():
    # verl's sole console metric sink is LocalLogger, which always prints "step:N - ...", so a line
    # without a step carries no metric. collecting the train_meta series outside the step branch
    # would re-scan every rollout/log line for keys that cannot be there.
    tree = _verl_rl_tree()

    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "step_metrics"
    ]
    assert len(guards) == 1, "expected exactly one step_metrics guard"
    collected = ast.unparse(guards[0])

    for verl_key in ("critic/rewards/mean", "actor/pg_loss", "response_length/mean"):
        assert verl_key in collected, f"{verl_key} must be collected inside the step branch"


def test_first_backlog_is_forced_past_the_rl_step_throttle():
    # rl_train_start arms the 900s rl_step throttle and the liveness daemon never passes force=True,
    # so without an explicit forced ping the first metrics row stays invisible for 15 minutes -- the
    # exact window trl covers with force_first_samples (heartbeat.py). the retry-until-committed
    # shape matters too: the daemon can claim the step first, and a bare `sent = True` would then
    # mark a heartbeat that never committed as sent.
    tree = _verl_rl_tree()

    forced = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "rl_step"
        and any(kw.arg == "force" for kw in node.keywords)
    ]
    assert len(forced) == 1, "verl must force exactly one first rl_step metrics heartbeat"
    keywords = {kw.arg for kw in forced[0].keywords}
    assert "metrics_last" in keywords, "the forced ping must carry the backlog it exists to surface"

    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "sent_first_metrics" in ast.unparse(node.test)
    )
    assert isinstance(guard.body[0], ast.Assign), "retry until a forced ping actually commits"
    assert "heartbeat" in ast.unparse(guard.body[0].value)


def test_verl_rl_renders_the_same_metric_fields_the_cli_shows():
    # the payload schema belongs to the cli, not verl: a key the renderer does not know is dead
    # weight, so keep the mapping's flash-side names inside the rendered set.
    from flash.cli.commands import _FOLLOW_METRIC_FIELDS
    from flash.engine.worker.backend_common import _VERL_METRIC_FIELDS

    rendered = {name for name, *_ in _FOLLOW_METRIC_FIELDS}
    emitted = {flash_key for _, flash_key in _VERL_METRIC_FIELDS}

    assert emitted <= rendered, f"verl emits fields the cli never renders: {emitted - rendered}"
