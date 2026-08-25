import flash.engine.worker.train.entry.rl_train_runner as rl_train_runner

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

from flash.engine.worker.train.entry.backend_common import (
    append_step_metrics,
    parse_verl_metric,
    parse_verl_step_metrics,
    verl_step_number,
)
from flash.engine.worker.train.entry.rl_train_runner import _ingest_step_metrics, _StepMetricState

# a realistic verl step line: ray tags worker stdout with a pid prefix, and reduce_metrics returns
# numpy scalars that pprint renders as np.float64(...) under numpy>=2.
_RAY_PREFIX = "(TaskRunner pid=3125) "
_STEP_LINE = (
    "step:7 - critic/rewards/mean:0.625 - critic/advantages/min:-0.25 - "
    "critic/advantages/max:0.75 - actor/pg_loss:-0.0134 - actor/grad_norm:1.5 - "
    "actor/kl_loss:0.002 - actor/entropy:0.91 - response_length/mean:213.5 - "
    "response_length/clip_ratio:0.125"
)


def test_parses_every_rendered_field_from_a_step_line():
    metrics = parse_verl_step_metrics(_STEP_LINE)

    assert metrics == {
        "step": 7,
        "reward": 0.625,
        "advantage_min": -0.25,
        "advantage_max": 0.75,
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
        "step:12 - critic/rewards/mean:np.float64(0.4) - "
        "critic/advantages/min:np.float32(-0.5) - "
        "critic/advantages/max:np.float64(1.25) - actor/grad_norm:np.float32(2.25) - "
        "response_length/mean:np.float64(180.0)"
    )

    metrics = parse_verl_step_metrics(line)

    assert metrics == {
        "step": 12,
        "reward": 0.4,
        "advantage_min": -0.5,
        "advantage_max": 1.25,
        "grad_norm": 2.25,
        "mean_completion_tokens": 180.0,
    }


def test_non_finite_metric_is_dropped_without_losing_its_siblings():
    # a diverged run prints nan; rendering it as a column is meaningless and it can poison the json
    # payload downstream, so drop that field only -- the step and the healthy fields must survive.
    line = (
        "step:3 - critic/rewards/mean:nan - critic/advantages/min:-0.5 - "
        "critic/advantages/max:inf - actor/grad_norm:inf - actor/entropy:0.5"
    )

    metrics = parse_verl_step_metrics(line)

    assert metrics == {"step": 3, "entropy": 0.5}


def test_incomplete_or_reversed_advantage_bounds_are_rejected_as_a_pair():
    for line in (
        "step:3 - critic/rewards/mean:0.5 - critic/advantages/min:-0.5",
        "step:3 - critic/rewards/mean:0.5 - critic/advantages/min:1.0 - critic/advantages/max:0.0",
        "step:3 - critic/rewards/mean:0.5 - critic/advantages/min:-1e308 - critic/advantages/max:1e308",
    ):
        metrics = parse_verl_step_metrics(line)
        assert metrics == {"step": 3, "reward": 0.5}


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


def test_step_line_is_parsed_when_a_tqdm_bar_is_flushed_in_front_of_it():
    # VERL-134. verl's LocalLogger shares its stream with tqdm, which ends a bar with "]" and no
    # trailing newline, so the metric line arrives glued to it. anchoring the left edge on
    # whitespace matched step 1 and missed every step after it: the sft heartbeat froze on step 1's
    # metrics and the zero-grad guard never armed, so a run that trained nothing reported done.
    line = (
        "Epoch 1/1:  25%|##        | 1/4 [01:21<04:04, 81.49s/it]"
        "step:2 - critic/rewards/mean:0.5 - actor/grad_norm:0.0"
    )

    assert parse_verl_step_metrics(line) == {"step": 2, "reward": 0.5, "grad_norm": 0.0}


def test_step_key_does_not_match_a_longer_word_or_a_path():
    # the left edge is widened to "not part of a longer word", not dropped: global_step: is a
    # different counter and a checkpoint path ending in step: is not a metric line at all. asserted
    # on both patterns because they share that edge -- the trainers gate on verl_step_number and the
    # metrics row on parse_verl_step_metrics.
    for line in (
        "global_step:9 - critic/rewards/mean:0.5",
        "/tmp/flash/checkpoints/step:9 - critic/rewards/mean:0.5",
    ):
        assert parse_verl_step_metrics(line) is None
        assert verl_step_number(line) is None

    # ray tags worker stdout with a pid prefix, so the edge must stay permissive enough for it.
    assert verl_step_number("(TaskRunner pid=123) step:7 - actor/grad_norm:1.0") == 7


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


def test_exact_advantage_bounds_are_retained_in_the_forced_step_heartbeat(monkeypatch):
    calls = []
    outcomes = iter((False, True))

    def heartbeat(stage, **fields):
        calls.append((stage, fields))
        return next(outcomes)

    monkeypatch.setattr(rl_train_runner._worker_heartbeat, "heartbeat", heartbeat)
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    for step, minimum, maximum in ((1, -0.25, 0.75), (2, -0.5, 1.5)):
        _ingest_step_metrics(
            f"step:{step} - critic/rewards/mean:0.5 - "
            f"critic/advantages/min:{minimum} - critic/advantages/max:{maximum} - "
            "actor/grad_norm:1.0",
            {"max_completion": 512},
            state,
            dict,
        )

    assert len(calls) == 2
    assert calls[-1][1]["metrics_last"] == [
        {
            "step": 1,
            "reward": 0.5,
            "advantage_min": -0.25,
            "advantage_max": 0.75,
            "grad_norm": 1.0,
            "max_completion_tokens": 512,
        },
        {
            "step": 2,
            "reward": 0.5,
            "advantage_min": -0.5,
            "advantage_max": 1.5,
            "grad_norm": 1.0,
            "max_completion_tokens": 512,
        },
    ]
    assert state.adv_spread_history == [1.0, 2.0]
    rl_train_runner._validate_rl_child(0, state, 0, 2, None)
    assert state.advantage_bounds_evidence == [
        {"step": 1, "min": -0.25, "max": 0.75, "spread": 1.0},
        {"step": 2, "min": -0.5, "max": 1.5, "spread": 2.0},
    ]
    assert state.grad_norm_evidence == [
        {"step": 1, "grad_norm": 1.0},
        {"step": 2, "grad_norm": 1.0},
    ]


def test_masked_truncation_sequence_uses_grad_norm_as_publication_evidence(monkeypatch):
    rewards = [1.0] * 14 + [0.0] * 2
    response_mask = [1] * 14 + [0] * 2
    assert sum(rewards) / len(rewards) == 0.875
    assert 1.0 - sum(response_mask) / len(response_mask) == 0.125

    monkeypatch.setattr(
        rl_train_runner._worker_heartbeat, "heartbeat", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    lines = (
        (
            "step:1 - critic/rewards/mean:0.875 - critic/advantages/min:0.125 - "
            "critic/advantages/max:0.125 - actor/grad_norm:0.0076509942300617695 - "
            "response_length/clip_ratio:0.125"
        ),
        (
            "step:2 - critic/rewards/mean:1.0 - critic/advantages/min:0.0 - "
            "critic/advantages/max:0.0 - actor/grad_norm:0.0 - response_length/clip_ratio:0.0"
        ),
    )
    for line in lines:
        _ingest_step_metrics(line, {"max_completion": 512}, state, dict)

    rl_train_runner._validate_rl_child(0, state, 0, 2, None)

    assert state.reward_history == [0.875, 1.0]
    assert state.advantage_bounds_evidence == [
        {"step": 1, "min": 0.125, "max": 0.125, "spread": 0.0},
        {"step": 2, "min": 0.0, "max": 0.0, "spread": 0.0},
    ]
    assert state.grad_norm_evidence == [
        {"step": 1, "grad_norm": 0.0076509942300617695},
        {"step": 2, "grad_norm": 0.0},
    ]


def test_replayed_step_replaces_gradient_evidence(monkeypatch):
    monkeypatch.setattr(
        rl_train_runner._worker_heartbeat, "heartbeat", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    common = (
        "step:1 - critic/rewards/mean:0.5 - critic/advantages/min:0.0 - critic/advantages/max:0.0"
    )
    _ingest_step_metrics(common + " - actor/grad_norm:0.0", {"max_completion": 32}, state, dict)
    _ingest_step_metrics(common + " - actor/grad_norm:0.25", {"max_completion": 32}, state, dict)

    assert state.grad_norms == {1: 0.25}
    rl_train_runner._validate_rl_child(0, state, 0, 1, None)


def test_replayed_step_without_grad_norm_removes_stale_evidence(monkeypatch):
    monkeypatch.setattr(
        rl_train_runner._worker_heartbeat, "heartbeat", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    common = (
        "step:1 - critic/rewards/mean:0.5 - critic/advantages/min:0.0 - critic/advantages/max:0.0"
    )
    _ingest_step_metrics(common + " - actor/grad_norm:0.25", {"max_completion": 32}, state, dict)
    _ingest_step_metrics(common, {"max_completion": 32}, state, dict)

    assert state.grad_norms == {}
    assert state.advantage_bounds == {1: (0.0, 0.0)}


def test_replayed_step_without_bounds_clears_stale_bounds_atomically(monkeypatch):
    monkeypatch.setattr(
        rl_train_runner._worker_heartbeat, "heartbeat", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    full = (
        "step:1 - critic/rewards/mean:0.5 - critic/advantages/min:-0.5 - "
        "critic/advantages/max:0.5 - actor/grad_norm:0.25"
    )
    _ingest_step_metrics(full, {"max_completion": 32}, state, dict)
    _ingest_step_metrics(
        "step:1 - critic/rewards/mean:0.5 - actor/grad_norm:0.5",
        {"max_completion": 32},
        state,
        dict,
    )

    assert state.grad_norms == {1: 0.5}
    assert state.advantage_bounds == {}
    assert state.adv_spread_history == []


def test_pg_loss_only_training_replay_clears_all_stale_terminal_evidence(monkeypatch):
    monkeypatch.setattr(
        rl_train_runner._worker_heartbeat, "heartbeat", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    full = (
        "step:1 - critic/rewards/mean:0.5 - critic/advantages/min:-0.5 - "
        "critic/advantages/max:0.5 - actor/grad_norm:0.25"
    )
    _ingest_step_metrics(full, {"max_completion": 32}, state, dict)
    _ingest_step_metrics("step:1 - actor/pg_loss:0.125", {"max_completion": 32}, state, dict)

    assert state.grad_norms == {}
    assert state.advantage_bounds == {}
    assert state.adv_spread_history == []
    assert state.loss_curve == [0.125]


def test_validation_only_replay_does_not_clear_training_evidence(monkeypatch):
    monkeypatch.setattr(
        rl_train_runner._worker_heartbeat, "heartbeat", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    full = (
        "step:1 - critic/rewards/mean:0.5 - critic/advantages/min:-0.5 - "
        "critic/advantages/max:0.5 - actor/grad_norm:0.25"
    )
    _ingest_step_metrics(full, {"max_completion": 32}, state, dict)
    _ingest_step_metrics("step:1 - val-core/reward/mean:0.8", {"max_completion": 32}, state, dict)

    assert state.grad_norms == {1: 0.25}
    assert state.advantage_bounds == {1: (-0.5, 0.5)}
    assert state.adv_spread_history == [1.0]


def _verl_rl_tree() -> ast.Module:
    from flash.engine.worker.train.entry import rl_train

    source = "\n".join(
        inspect.getsource(fn)
        for fn in (
            rl_train.run_rl_train,
            rl_train_runner._ingest_step_metrics,
            rl_train._write_terminal_metadata,
        )
    )
    return ast.parse(textwrap.dedent(source))


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
    # actor/pg_loss must also detect a training replay whose other metrics are unrenderable, but it
    # still parses only after verl_step_number proves this is a step-tagged LocalLogger record.
    source = inspect.getsource(_ingest_step_metrics)
    assert 'parse_verl_metric(line, "actor/pg_loss") if step_number is not None else None' in source
    structured = source[source.index("if step_metrics is not None:") :]
    for verl_key in ("critic/rewards/mean", "response_length/mean"):
        assert verl_key in structured, (
            f"{verl_key} must be collected inside the structured step branch"
        )


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
    from flash.cli.commands.ops.log_follow import _FOLLOW_METRIC_FIELDS
    from flash.engine.worker.train.entry.backend_common import _VERL_METRIC_FIELDS

    rendered = {name for name, *_ in _FOLLOW_METRIC_FIELDS}
    emitted = {flash_key for _, flash_key in _VERL_METRIC_FIELDS}

    assert emitted <= rendered, f"verl emits fields the cli never renders: {emitted - rendered}"
