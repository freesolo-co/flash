"""`flash train --cost`: map a training config to a pre-flight cost."""

from __future__ import annotations

import copy
import dataclasses
import types

import pytest

from flash.cli.commands.ops.train import cmd_train
from flash.cost.spec import UnknownPromptPoolSize
from flash.cost.spec import runconfig_from_spec as _runconfig_from_spec
from flash.cost.spec import spec_steps as _spec_steps
from flash.cost.types import RunConfig
from flash.engine.plan.recipe import RECIPE
from flash.engine.profiling.workload_profile import WorkloadProfileMismatch
from flash.schema import spec_from_dict
from tests._helpers.profile import attach_sft_profile

GRPO_RAW = {
    "model": "Qwen/Qwen3.5-9B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {
        "epochs": 1,
        "max_examples": 800,
        "group_size": 8,
        "prompts_per_step": 16,
        "max_completion_tokens": 512,
        "max_context_tokens": 2048,
    },
    "gpu": {},
}


def _spec(**overrides):
    raw = copy.deepcopy(GRPO_RAW)
    for key, value in overrides.items():
        section, _, leaf = key.partition(".")
        if leaf:
            raw.setdefault(section, {})[leaf] = value
        else:
            raw[section] = value
    return spec_from_dict(raw)


def test_runconfig_from_grpo_spec_maps_fields():
    spec = _spec()
    cfg = _runconfig_from_spec(spec)
    assert cfg.model_id == "Qwen/Qwen3.5-9B"
    assert cfg.method == "grpo"
    assert cfg.steps == 50
    assert cfg.batch_size == 16
    assert cfg.group_size == 8
    assert cfg.completion_len == 512  # GRPO max_completion_tokens
    assert cfg.seq_len == 2048
    assert cfg.environment == "github:freesolo-co/envs@main:gsm8k/environment.py"


def test_grpo_default_epochs_mirror_recipe():
    spec = _spec()
    object.__setattr__(spec.train, "epochs", None)
    assert RECIPE.rl.num_epochs == 1
    assert _spec_steps(spec) == 50


def test_grpo_epochs_derive_steps_from_max_examples():
    spec = _spec(**{"train.epochs": 2, "train.max_examples": 33})
    assert _spec_steps(spec) == 5  # ceil(33 rows * 2 epochs / batch_size 16)


def test_an_unbounded_prompt_pool_refuses_to_quote_instead_of_pricing_one_step():
    """No stated row count used to mean "the pool is one step wide", which is never true.

    The worker sizes the horizon from ``len(prompts)`` -- every row the environment yields -- so a
    config that bounded nothing was quoted at one step regardless of dataset size. Against a
    1153-row pool at batch 8 that is 145 real steps priced as 1, and the opd teacher capability in
    ``capability_limits_for_spec`` is sized off the same number. The quote now refuses rather than
    reporting a horizon nothing stated.
    """
    spec = _spec(**{"train.max_examples": None, "train.epochs": 2})

    with pytest.raises(UnknownPromptPoolSize, match="without a prompt-pool size") as refusal:
        _spec_steps(spec)
    # only the knob the WORKER applies is advertised; see the [environment.params] note below.
    assert "[environment.params]" not in str(refusal.value)

    # the two the message names: a row count, or a horizon that needs no row count at all.
    assert _spec_steps(_spec(**{"train.max_examples": 1153, "train.epochs": 1})) == 73
    assert _spec_steps(_spec(**{"train.max_examples": None, "train.max_steps": 145})) == 145

    # [environment.params] max_examples does NOT answer it. It reaches the environment as an opaque
    # load_environment(**params) kwarg that the starter templates ignore, so it is wrong in both
    # directions: an environment that honours it trains fewer rows than a horizon derived from the
    # authored batch, and one that ignores it yields every row, making a 1153 here the pool size of
    # a run that may really iterate far more. A number nothing enforces is not a stated horizon, so
    # this stays the same refusal as a spec that says nothing at all.
    env_bounded = _spec(**{"train.max_examples": None, "train.epochs": 1})
    object.__setattr__(env_bounded.environment, "params", {"max_examples": 1153})
    with pytest.raises(UnknownPromptPoolSize, match="without a prompt-pool size"):
        _spec_steps(env_bounded)


def test_grpo_positive_max_steps_is_authoritative():
    assert _spec_steps(_spec(**{"train.max_steps": 73})) == 73
    assert _spec_steps(_spec(**{"train.max_steps": 0})) == 50


def test_required_save_density_adds_wall_time_and_cost_without_changing_steps():
    from flash.cost.analytical import estimate_cost

    for method in ("sft", "grpo"):
        common = {
            "model_id": "Qwen/Qwen3.5-9B",
            "method": method,
            "steps": 10,
            "seq_len": 1024,
            "batch_size": 4,
        }
        if method == "grpo":
            common.update(completion_len=128, group_size=2)
        baseline = RunConfig(**common)
        sparse = RunConfig(**common, save_at_steps=(5,))
        dense = RunConfig(**common, save_at_steps=(2, 4, 6, 8))

        baseline_estimate = estimate_cost(baseline)
        sparse_estimate = estimate_cost(sparse)
        dense_estimate = estimate_cost(dense)

        assert baseline.steps == sparse.steps == dense.steps == 10
        assert baseline_estimate.train_seconds < sparse_estimate.train_seconds
        assert sparse_estimate.train_seconds < dense_estimate.train_seconds
        assert baseline_estimate.total_usd < sparse_estimate.total_usd < dense_estimate.total_usd


def test_required_save_overhead_uses_contractual_commit_counts():
    from flash.cost.analytical import (
        REQUIRED_SAVE_COMMIT_FLOOR_S,
        REQUIRED_SAVE_S_PER_MODEL_B_AT_RANK32,
        required_save_overhead_seconds,
    )
    from flash.cost.facts import total_params_b

    model_id = "Qwen/Qwen3.5-9B"
    save_at_steps = (2, 4, 6)
    common = {
        "model_id": model_id,
        "steps": 10,
        "seq_len": 1024,
        "batch_size": 4,
        "lora_rank": 32,
        "save_at_steps": save_at_steps,
    }
    serialize_per_save = REQUIRED_SAVE_S_PER_MODEL_B_AT_RANK32 * total_params_b(model_id)

    for method, commits_per_save in (("sft", 2), ("grpo", 2), ("opd", 1)):
        config = RunConfig(method=method, **common)
        expected = len(save_at_steps) * (
            commits_per_save * REQUIRED_SAVE_COMMIT_FLOOR_S + serialize_per_save
        )
        assert required_save_overhead_seconds(config) == pytest.approx(expected)


def test_opd_required_saves_add_overhead_without_changing_steps():
    from flash.cost.analytical import estimate_cost

    common = {
        "model_id": "Qwen/Qwen3.5-9B",
        "method": "opd",
        "steps": 10,
        "seq_len": 1024,
        "batch_size": 4,
        "completion_len": 128,
        "group_size": 1,
    }
    baseline = RunConfig(**common)
    withsave = RunConfig(**common, save_at_steps=(2, 4, 6))

    # opd publishes a deployable adapter at each exact save, so exact saves cost wall/dollars too.
    assert baseline.steps == withsave.steps == 10
    assert estimate_cost(withsave).train_seconds > estimate_cost(baseline).train_seconds
    assert estimate_cost(withsave).total_usd > estimate_cost(baseline).total_usd


def test_partial_reprice_counts_reached_saves_and_drops_future_saves():
    from flash.runner.accounting.costs import charge_usd_for_spec

    def partial_charge(save_at_steps):
        raw = copy.deepcopy(GRPO_RAW)
        raw["train"].update({"max_steps": 100, "save_at_steps": save_at_steps})
        return charge_usd_for_spec(spec_from_dict(raw), steps=10, fallback=-1.0)

    # cancel at step 10: step 5 landed and remains priced, while steps 50/100 are dropped before the
    # reduced run config is built so neither estimate falls back.
    reached_save_charge = partial_charge([5, 50, 100])
    future_only_charge = partial_charge([50, 100])

    assert reached_save_charge != -1.0
    assert future_only_charge != -1.0
    assert reached_save_charge > future_only_charge > 0.0


def test_opd_epochs_derive_steps_from_max_examples():
    raw = copy.deepcopy(GRPO_RAW)
    raw["algorithm"] = "opd"
    raw["train"].update({"epochs": 2, "max_examples": 17, "prompts_per_step": 8, "group_size": 1})
    spec = spec_from_dict(raw)
    assert _spec_steps(spec) == 5  # ceil(17 rows * 2 epochs / batch_size 8)


def test_opd_positive_max_steps_is_authoritative():
    raw = copy.deepcopy(GRPO_RAW)
    raw["algorithm"] = "opd"
    raw["train"].update(
        {"epochs": 2, "max_examples": 17, "prompts_per_step": 8, "group_size": 1, "max_steps": 31}
    )
    assert _spec_steps(spec_from_dict(raw)) == 31


def test_opd_runconfig_carries_selected_teacher_and_prices_it():
    from flash.cost.analytical import estimate_cost

    def _opd(teacher=None):
        raw = copy.deepcopy(GRPO_RAW)
        raw["model"] = "Qwen/Qwen3.5-9B"
        raw["algorithm"] = "opd"
        raw["train"].update(
            {"epochs": 1, "max_examples": 40, "prompts_per_step": 8, "group_size": 1}
        )
        if teacher is not None:
            raw["train"]["teacher_model"] = teacher
        return spec_from_dict(raw)

    assert _runconfig_from_spec(_opd()).teacher_model == "glm-5.2"
    qwen_cfg = _runconfig_from_spec(_opd("qwen3.5-397b-a17b"))
    assert qwen_cfg.teacher_model == "qwen3.5-397b-a17b"

    default_teacher_usd = estimate_cost(_runconfig_from_spec(_opd())).teacher_api_usd
    qwen_teacher_usd = estimate_cost(qwen_cfg).teacher_api_usd
    assert 0 < qwen_teacher_usd < default_teacher_usd

    # sft/grpo carry no teacher.
    assert _runconfig_from_spec(_spec()).teacher_model == ""


def test_sft_cost_requires_the_measured_workload_rather_than_deriving_one():
    """sft has no analytical step count left: without a profile the quote refuses to exist.

    The old path derived steps from ``max_examples`` and priced a 128-token-per-row prior. Both are
    gone, so the failure has to be an explicit mismatch, not a plausible number.
    """
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "environment": {"id": "github:acme/envs@main:sft-data/environment.py"},
            "train": {"max_examples": 320, "batch_size": 16, "epochs": 2},
            "gpu": {},
        }
    )

    assert spec.train.max_examples == 320
    for call in (_spec_steps, _runconfig_from_spec):
        with pytest.raises(WorkloadProfileMismatch):
            call(spec)


def test_sft_cost_reads_the_horizon_and_tokens_the_profile_measured():
    """Every exact field reaches ``RunConfig`` from the profile rather than from the config.

    Scope: this is the wiring, not the measurement. ``tests/test_sft_workload.py`` owns whether the
    measured numbers are right, and the helper's profile mirrors the config's own shape, so equality
    with a config-derived number here would prove nothing on its own. What it does pin is that the
    estimator reads ``authoritative_steps``/token/packing fields that only a profile carries -- and
    ``seq_len``, which the config leaves unset entirely.
    """
    spec = attach_sft_profile(
        spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "sft",
                "environment": {"id": "github:acme/envs@main:sft-data/environment.py"},
                "train": {"max_examples": 320, "batch_size": 16, "epochs": 2},
                "gpu": {},
            }
        )
    )
    profile = spec.workload_profile

    cfg = _runconfig_from_spec(spec)

    assert cfg.method == "sft"
    assert cfg.steps == _spec_steps(spec) == profile["authoritative_steps"]
    assert cfg.batch_size == profile["examples_per_update"]
    assert cfg.seq_len == profile["max_length"]
    assert cfg.train_tokens == profile["authoritative_compute_tokens"]
    assert cfg.supervised_train_tokens == profile["authoritative_supervised_tokens"]
    assert cfg.sft_packing_mode == profile["packing_mode"]
    assert cfg.sft_packed_blocks == profile["packed_blocks"]
    assert cfg.group_size is None  # sft carries no completions-per-prompt
    assert cfg.completion_len is None


def test_sft_cost_rejects_a_profile_keyed_to_a_different_workload():
    """Editing a workload-shaping field after the profile was measured must not reprice silently."""
    spec = attach_sft_profile(
        spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "sft",
                "environment": {"id": "github:acme/envs@main:sft-data/environment.py"},
                "train": {"max_examples": 320, "batch_size": 16, "epochs": 2},
                "gpu": {},
            }
        )
    )
    retuned = dataclasses.replace(spec, train=dataclasses.replace(spec.train, epochs=3))

    with pytest.raises(WorkloadProfileMismatch, match="input digest"):
        _runconfig_from_spec(retuned)


def test_runconfig_preserves_positional_seq_len_compatibility():
    cfg = RunConfig("Qwen/Qwen3.5-9B", "sft", 10, 2048)
    assert cfg.seq_len == 2048
    assert cfg.train_tokens is None


def test_cmd_train_cost_prints_server_quote_without_submitting(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 800\n"
        "prompts_per_step = 16\n"
        "[gpu]\n"
        ""
    )
    args = types.SimpleNamespace(
        config=str(cfg),
        overrides=[],
        extra_configs=[],
        cost=True,
        dry_run=False,
        background=False,
    )
    client = _use_client(monkeypatch, _QuotingClient({"estimated_cost_usd": 4.25}))

    rc = cmd_train(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert client.calls[0]["dry_run"] is True
    assert client.calls[0]["spec"]["algorithm"] == "grpo"
    assert "TOTAL" in out
    assert "$4.25" in out
    assert "GPU" not in out  # the server response does not expose allocation details


SFT_TOML = (
    'model = "Qwen/Qwen3.5-9B"\n'
    'project = "11111111-1111-4111-8111-111111111111"\n'
    'algorithm = "sft"\n'
    "[environment]\n"
    'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
    "[train]\n"
    "epochs = 1\n"
    "batch_size = 8\n"
    "[gpu]\n"
)

EXACT_PROFILE = {
    "environment_id": "github:freesolo-co/envs@main:gsm8k/environment.py",
    "environment_revision": "a" * 40,
    "source_examples": 10,
    "authoritative_steps": 7,
    "selected_examples": 10,
    "retained_examples": 8,
    "dropped_examples": 2,
    "authoritative_compute_tokens": 4096,
    "authoritative_supervised_tokens": 2048,
    "packing_mode": "packed",
    "architecture_mode": "pure-attention",
    "authored_reasoning_turns": 0,
    "rendered_reasoning_spans": 0,
    "truncated_reasoning_spans": 0,
    "reasoning_rows": 8,
    "content_digest": "b" * 64,
}


def _sft_args(tmp_path, body: str = SFT_TOML, **overrides):
    cfg = tmp_path / "sft.toml"
    cfg.write_text(body)
    return types.SimpleNamespace(
        config=str(cfg),
        overrides=[],
        extra_configs=[],
        cost=True,
        dry_run=False,
        background=False,
        **overrides,
    )


class _QuotingClient:
    """server that answers the dry-run with its packaged-dataset estimate."""

    def __init__(self, response: dict | None = None):
        self.calls: list[dict] = []
        self.response = (
            {"estimated_cost_usd": 1.25, "workload_profile": dict(EXACT_PROFILE)}
            if response is None
            else response
        )

    def create_run(self, spec, runtime_secrets=None, dry_run=False, client_train_schema=None):
        self.calls.append(
            {
                "spec": spec,
                "runtime_secrets": runtime_secrets,
                "dry_run": dry_run,
                "client_train_schema": client_train_schema,
            }
        )
        return dict(self.response)


class _MissingDatasetClient:
    """server refusal when the pinned package has no readable sft dataset file."""

    def __init__(self):
        from flash.client import ApiError

        self.error = ApiError(
            400,
            "environment package has no readable dataset for split 'train'. "
            "Add dataset/train.jsonl to the environment package.",
        )
        self.calls: list[dict] = []

    def create_run(self, spec, runtime_secrets=None, dry_run=False, client_train_schema=None):
        self.calls.append({"spec": spec, "dry_run": dry_run})
        raise self.error


def _use_client(monkeypatch, client):
    from flash.cli.commands.ops import train as commands

    monkeypatch.setattr(commands, "client_from_config", lambda *a, **k: client)
    monkeypatch.setenv("FLASH_STYLE", "0")
    return client


def test_sft_cost_asks_the_server_for_the_quote_without_creating_a_training_run(
    tmp_path, monkeypatch, capsys
):
    """sft ``--cost`` is a dry-run submit because the server builds the dataset estimate.

    the public payload must stay public: the internal profile carrier is attached by preparation,
    so a client that could send one could fabricate the workload its own quote is derived from.
    """
    client = _use_client(monkeypatch, _QuotingClient())

    rc = cmd_train(_sft_args(tmp_path))

    assert rc == 0
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["dry_run"] is True
    assert not any(key.startswith("workload_profile") for key in call["spec"])


@pytest.mark.parametrize(
    ("environment_id", "expected", "absent"),
    [
        (
            "github:freesolo-co/envs@main:gsm8k/environment.py",
            "make the named remote branch or tag point at the new commit",
            "full new commit SHA",
        ),
        (
            "github:freesolo-co/envs@v1.2.3:gsm8k/environment.py",
            "A tag may instead need a new tag and an updated [environment] id",
            "full new commit SHA",
        ),
        (
            f"github:freesolo-co/envs@{'c' * 40}:gsm8k/environment.py",
            "publish the new commit, then update [environment] id to the full new commit SHA",
            "named remote branch or tag",
        ),
    ],
    ids=("branch", "tag", "sha"),
)
def test_github_republish_advice_covers_branch_tag_and_sha(environment_id, expected, absent):
    from flash.cli.commands.ops import train

    advice = train._republish_advice(environment_id)

    assert expected in advice
    assert absent not in advice


def test_managed_hub_github_ref_uses_env_push_advice():
    from flash.cli.commands.ops import train

    advice = train._republish_advice(
        "github:freesolo-co/environment-hub@main:owner/project/env/environment.py"
    )

    assert "env push" in advice
    assert "named remote branch or tag" not in advice


def test_managed_republish_advice_prints_required_env_push_arguments():
    from flash.cli.commands.ops import train

    advice = train._republish_advice("owner/project/env")

    assert f"{train.CLI_NAME} env push --name NAME --project PROJECT_UUID [path]" in advice


def test_published_environment_note_ignores_unknown_environment_ids(monkeypatch, capsys):
    from flash.cli.commands.ops import train

    monkeypatch.setenv("FLASH_STYLE", "0")
    train._print_published_sft_environment_note(
        {
            "workload_profile": {
                "environment_id": "local-environment",
                "environment_revision": "a" * 40,
                "source_examples": 10,
            }
        },
        types.SimpleNamespace(environment=types.SimpleNamespace(params={})),
    )

    assert capsys.readouterr().err == ""


def test_sft_cost_reports_the_dataset_estimate_and_no_invented_hardware(
    tmp_path, monkeypatch, capsys
):
    """the panel prints the dataset estimate without inventing unavailable timing fields.

    the analytical breakdown named a gpu, an hourly rate, and a per-step time. the server-side quote
    has none of those details, so reprinting that layout here would invent every number in it.
    """
    _use_client(monkeypatch, _QuotingClient())

    rc = cmd_train(_sft_args(tmp_path))
    captured = capsys.readouterr()

    assert rc == 0
    assert "$1.25" in captured.out
    assert "7 steps" in captured.out
    assert "github:freesolo-co/envs@main:gsm8k/environment.py" in captured.out
    assert "aaaaaaaaaaaa" in captured.out
    assert "published environment github:freesolo-co/envs@main:gsm8k/environment.py" in captured.out
    assert "aaaaaaaaaaaa (published commit)" in captured.out
    assert "8 trained of 10 selected from 10 source rows in published copy" in captured.out
    assert "(2 dropped)" in captured.out
    assert "4,096 compute, 2,048 supervised" in captured.out
    assert "packed (pure-attention)" in captured.out
    assert "bbbbbbbbbbbb" in captured.out
    assert (
        "SFT dataset counts come from this resolved published copy, not local files" in captured.err
    )
    for invented in ("/hr", "setup", "per-step", "train_seconds"):
        assert invented not in captured.out
    assert "nothing was charged for training" in captured.err


UNPACKED_MULTIMODAL_PROFILE = {
    **EXACT_PROFILE,
    "packing_mode": "exact-unpacked",
    "architecture_mode": "multimodal",
    "examples_per_update": 1,
}


def test_sft_cost_warns_that_an_unpacked_run_ignores_the_configured_batch_size(
    tmp_path, monkeypatch, capsys
):
    """The batch override is a quote-time fact, so the user must hear it before submitting.

    A multimodal run is never packed, so ``batch_size = 8`` in the config buys nothing: verl takes
    one optimizer step per example.
    """
    _use_client(
        monkeypatch,
        _QuotingClient(
            {"estimated_cost_usd": 1.25, "workload_profile": dict(UNPACKED_MULTIMODAL_PROFILE)}
        ),
    )

    rc = cmd_train(_sft_args(tmp_path))
    err = capsys.readouterr().err

    assert rc == 0
    assert "sequence packing is OFF" in err
    # the reason is the architecture label the packing decision froze on the profile
    assert "multimodal" in err
    assert "the configured batch_size 8 no longer groups examples into an update" in err
    # batch_size is not inert: it still keys the profile and sizes an auto-picked gpu
    assert "sizes the gpu" in err
    # max_steps outranks epochs over rows, so the warning must claim no step count
    assert "per epoch" not in err
    assert "learning_rate" in err


def test_a_real_unpacked_submit_warns_before_the_run_starts(tmp_path, monkeypatch, capsys):
    """The submit path is the one that spends money, so it cannot be the one that stays quiet.

    ``prepare_sft_workload`` warns too, but only into the remote worker log, and for a foreground
    submit that is after the training gpu is already allocated.
    """
    _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "run_id": "run-unpacked",
                "workload_profile": dict(UNPACKED_MULTIMODAL_PROFILE),
            }
        ),
    )

    args = _sft_args(tmp_path)
    args.cost = False
    args.background = True

    rc = cmd_train(args)
    err = capsys.readouterr().err

    assert rc == 0
    assert "sequence packing is OFF" in err


def test_sft_cost_stays_quiet_about_batching_when_the_run_is_packed(tmp_path, monkeypatch, capsys):
    """A packed run honours ``batch_size``, so the warning must not cry wolf."""
    _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "estimated_cost_usd": 1.25,
                "workload_profile": {**EXACT_PROFILE, "examples_per_update": 8},
            }
        ),
    )

    rc = cmd_train(_sft_args(tmp_path))
    err = capsys.readouterr().err

    assert rc == 0
    assert "sequence packing is OFF" not in err


def test_sft_cost_warns_the_client_that_the_template_dropped_reasoning(
    tmp_path, monkeypatch, capsys
):
    """The quote is where the user can still act on it, so the quote has to say it.

    Control-plane profiling runs inside the server process, so the measurement's own stderr is not
    the submitting client's. The counts ride on the returned profile and the line is rendered here,
    before any training GPU is allocated.
    """
    _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "estimated_cost_usd": 1.25,
                "workload_profile": {
                    **EXACT_PROFILE,
                    "authored_reasoning_turns": 4,
                    "rendered_reasoning_spans": 1,
                },
            }
        ),
    )

    rc = cmd_train(_sft_args(tmp_path))
    err = capsys.readouterr().err

    assert rc == 0
    assert "dropped 3 of 4 authored reasoning blocks" in err
    # the actionable half: what to do about it
    assert "K single-turn rows" in err


def test_a_quote_still_renders_when_the_plane_omits_the_reasoning_counts(
    tmp_path, monkeypatch, capsys
):
    """A field this build expects can be absent from an OLDER plane's reply, and that is not fatal.

    The CLI does not ship with the control plane, so during a rolling upgrade a quote can arrive
    from a producer that predates `truncated_reasoning_spans` / `reasoning_rows`. This is one
    advisory warning line and its call sites are unwrapped, so indexing the profile strictly would
    abort `train --cost` and the SFT dry run AFTER the server already returned a valid quote --
    trading a missing warning for a failed command.
    """
    profile = {key: value for key, value in EXACT_PROFILE.items() if key != "reasoning_rows"}
    profile.pop("truncated_reasoning_spans", None)
    _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "estimated_cost_usd": 1.25,
                "workload_profile": {
                    **profile,
                    "authored_reasoning_turns": 4,
                    "rendered_reasoning_spans": 1,
                },
            }
        ),
    )

    rc = cmd_train(_sft_args(tmp_path))
    out = capsys.readouterr()

    assert rc == 0, "an older plane's profile must not fail the quote it arrived on"
    assert "1.25" in out.out
    # the warning is what degrades, not the command.
    assert "authored reasoning blocks" not in out.err


def test_the_client_warning_counts_rows_over_the_same_horizon_as_the_counts(
    tmp_path, monkeypatch, capsys
):
    """The percentage has to describe ONE population, and the profile carries two row figures.

    The reasoning counts on the profile cover the rows the update horizon reaches. Pairing them
    with ``retained_examples`` -- the whole retained dataset, which sizes the allocation -- would
    divide a bounded numerator by an unbounded denominator and understate the survival rate in the
    line the user reads. Here 2 updates at 2 rows each reach 4 of the 8 retained rows.
    """
    _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "estimated_cost_usd": 1.25,
                "workload_profile": {
                    **EXACT_PROFILE,
                    "authoritative_steps": 2,
                    "examples_per_update": 2,
                    "authored_reasoning_turns": 4,
                    "rendered_reasoning_spans": 1,
                    "reasoning_rows": 4,
                },
            }
        ),
    )

    rc = cmd_train(_sft_args(tmp_path))
    err = capsys.readouterr().err

    assert rc == 0
    assert "dropped 3 of 4 authored reasoning blocks" in err
    # the horizon reaches 4 rows, not the 8 that were retained
    assert "across 4 SFT rows" in err
    assert "across 8 SFT rows" not in err


@pytest.mark.parametrize(
    "absent",
    [
        "authored_reasoning_turns",
        "rendered_reasoning_spans",
        "truncated_reasoning_spans",
        "reasoning_rows",
    ],
)
def test_sft_cost_survives_any_single_missing_reasoning_count(
    tmp_path, monkeypatch, absent, capsys
):
    """No single absent count may fail the quote it arrived on.

    This deliberately replaces a `pytest.raises(KeyError)` contract of my own: requiring a complete
    profile here is right for the in-process producer and wrong for this reader, which parses the
    reply of a control plane the CLI does not ship with. A newer CLI routinely talks to an older
    plane, and the strict version turned that into a failed command rather than a missing advisory
    line.
    """
    profile = dict(EXACT_PROFILE)
    profile.pop(absent)
    _use_client(
        monkeypatch,
        _QuotingClient({"estimated_cost_usd": 1.25, "workload_profile": profile}),
    )

    assert cmd_train(_sft_args(tmp_path)) == 0
    assert "1.25" in capsys.readouterr().out


@pytest.mark.parametrize(
    "malformed",
    [
        "authored_reasoning_turns",
        "rendered_reasoning_spans",
        "truncated_reasoning_spans",
        "reasoning_rows",
    ],
)
def test_sft_cost_survives_a_malformed_reasoning_count(tmp_path, monkeypatch, malformed, capsys):
    """A malformed count degrades the same way an absent one does, for the same reason.

    A peer that sends `null` is no more actionable to the user than one that omits the key, and the
    remedy is identical: skip the advisory line, keep the quote.
    """
    _use_client(
        monkeypatch,
        _QuotingClient(
            {"estimated_cost_usd": 1.25, "workload_profile": {**EXACT_PROFILE, malformed: None}}
        ),
    )

    assert cmd_train(_sft_args(tmp_path)) == 0
    assert "1.25" in capsys.readouterr().out


def test_a_real_sft_submit_warns_about_dropped_reasoning_before_the_run_starts(
    tmp_path, monkeypatch, capsys
):
    """The submit path spends the money, so it cannot be the one that stays quiet."""
    _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "run_id": "run-thinking",
                "workload_profile": {
                    **EXACT_PROFILE,
                    "authored_reasoning_turns": 4,
                    "rendered_reasoning_spans": 1,
                },
            }
        ),
    )

    args = _sft_args(tmp_path)
    args.cost = False
    args.background = True

    rc = cmd_train(args)
    err = capsys.readouterr().err

    assert rc == 0
    assert "dropped 3 of 4 authored reasoning blocks" in err


def test_a_real_submit_reports_its_run_id_even_when_the_profile_is_malformed(
    tmp_path, monkeypatch, capsys
):
    """The run is already created and billing here, so a bad profile must not take the id with it.

    the quote and dry-run paths reject a malformed profile loudly, which is where nothing has been
    allocated yet. past this line the money is spent, and a user who never sees the run id cannot
    name it to cancel.
    """
    _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "run_id": "run-malformed-profile",
                "workload_profile": {
                    **{k: v for k, v in EXACT_PROFILE.items() if k != "reasoning_rows"},
                    "authored_reasoning_turns": 4,
                    "rendered_reasoning_spans": 1,
                },
            }
        ),
    )

    args = _sft_args(tmp_path)
    args.cost = False
    args.background = True

    rc = cmd_train(args)
    out = capsys.readouterr().out

    assert rc == 0
    assert "run-malformed-profile" in out


def test_sft_cost_stays_quiet_when_every_authored_block_survives(tmp_path, monkeypatch, capsys):
    """A dataset the template renders whole must not be told to restructure itself."""
    _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "estimated_cost_usd": 1.25,
                "workload_profile": {
                    **EXACT_PROFILE,
                    "authored_reasoning_turns": 4,
                    "rendered_reasoning_spans": 4,
                },
            }
        ),
    )

    rc = cmd_train(_sft_args(tmp_path))
    err = capsys.readouterr().err

    assert rc == 0
    assert "authored reasoning blocks" not in err


def test_sft_cost_omits_aggregates_the_profile_did_not_report(tmp_path, monkeypatch, capsys):
    """A partial profile drops rows rather than defaulting them to zero."""
    _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "estimated_cost_usd": 0.5,
                "workload_profile": {
                    "authoritative_steps": 3,
                    "authored_reasoning_turns": 0,
                    "rendered_reasoning_spans": 0,
                    "truncated_reasoning_spans": 0,
                    "reasoning_rows": 0,
                },
            }
        ),
    )

    rc = cmd_train(_sft_args(tmp_path))
    out = capsys.readouterr().out

    assert rc == 0
    assert "3 steps" in out
    assert "$0.50" in out
    for absent in ("examples", "tokens", "workload", "digest"):
        assert f"{absent}  " not in out


@pytest.mark.parametrize("quote", [None, True, "1.25"])
def test_sft_cost_refuses_to_print_a_total_the_server_did_not_quote(
    tmp_path, monkeypatch, quote, capsys
):
    """No estimate means no panel. ``True`` matters on its own: bool is an int subclass, so an
    unchecked numeric test would render a JSON ``true`` as ``$1.00``."""
    from flash.client import ClientError

    response = {"workload_profile": dict(EXACT_PROFILE)}
    if quote is not None:
        response["estimated_cost_usd"] = quote
    _use_client(monkeypatch, _QuotingClient(response))

    with pytest.raises(ClientError, match="no cost estimate"):
        cmd_train(_sft_args(tmp_path))
    assert "$" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("cost", "dry_run"),
    [(True, False), (False, True), (False, False)],
    ids=("cost", "dry-run", "submit"),
)
def test_sft_commands_refuse_a_missing_packaged_dataset(tmp_path, monkeypatch, cost, dry_run):
    """cost, dry-run, and submit surface the same packaged-dataset remediation."""
    from flash.client import ApiError

    client = _use_client(monkeypatch, _MissingDatasetClient())
    args = _sft_args(tmp_path)
    args.cost, args.dry_run = cost, dry_run

    with pytest.raises(ApiError, match=r"dataset/train\.jsonl") as excinfo:
        cmd_train(args)

    assert excinfo.value.status == 400
    assert len(client.calls) == 1
    assert client.calls[0]["dry_run"] is (cost or dry_run)


def test_sft_cost_leaves_unrelated_api_errors_alone(tmp_path, monkeypatch):
    """unrelated server rejections surface unchanged."""
    from flash.client import ApiError

    class _Rejecting:
        def create_run(self, spec, runtime_secrets=None, dry_run=False, client_train_schema=None):
            raise ApiError(402, "insufficient balance")

    _use_client(monkeypatch, _Rejecting())

    with pytest.raises(ApiError, match="insufficient balance"):
        cmd_train(_sft_args(tmp_path))


def test_sft_cost_forwards_declared_secrets_without_printing_them(tmp_path, monkeypatch, capsys):
    """A declared secret reaches the server out of band and appears in neither the spec nor output."""
    monkeypatch.setenv("MY_ENV_TOKEN", "s3cret-value")
    client = _use_client(monkeypatch, _QuotingClient())
    body = SFT_TOML.replace(
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n',
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\nsecrets = ["MY_ENV_TOKEN"]\n',
    )

    assert cmd_train(_sft_args(tmp_path, body)) == 0

    call = client.calls[0]
    assert call["runtime_secrets"]["MY_ENV_TOKEN"] == "s3cret-value"
    assert "s3cret-value" not in repr(call["spec"])
    captured = capsys.readouterr()
    assert "s3cret-value" not in captured.out + captured.err


def test_sft_cost_warns_when_an_env_key_shadows_the_saved_login(tmp_path, monkeypatch, capsys):
    """the cost command warns after parsing because every quote reaches an organization."""
    from flash.cli.commands.ops import train as commands

    monkeypatch.setattr(commands, "shadowed_login_warning", lambda: "shadowed!")
    _use_client(monkeypatch, _QuotingClient())

    assert cmd_train(_sft_args(tmp_path)) == 0
    assert "shadowed!" in capsys.readouterr().err


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_warm_start_non_sft_cost_uses_the_authoritative_server_quote(
    tmp_path, monkeypatch, capsys, algorithm
):
    """warm-start grpo/opd use submit preparation so the resolved source rank prices the quote."""
    from flash.cli.commands.ops import train as commands

    client = _use_client(
        monkeypatch,
        _QuotingClient(
            {
                "estimated_cost_usd": 1.005,
                "prompt_budget": {
                    "algorithm": algorithm,
                    "context_source": "recipe_default",
                    "prompt_budget_is_upper_bound": True,
                    "engine_len": 4096,
                    "max_completion": 512,
                    "prompt_budget": 3584,
                    "warm_start_context": 2048,
                },
            }
        ),
    )
    monkeypatch.setattr(commands, "shadowed_login_warning", lambda: "shadowed!")
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("WARM_START_TOKEN", "server-only-secret")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    # group_size 2 is grpo's floor and is valid for opd too.
    body = (
        SFT_TOML.replace('algorithm = "sft"', f'algorithm = "{algorithm}"')
        .replace(
            'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n',
            'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
            'secrets = ["WARM_START_TOKEN"]\n',
        )
        .replace(
            "batch_size = 8\n",
            "prompts_per_step = 8\nmax_examples = 40\ngroup_size = 2\n"
            'init_from_adapter = "source-run/final"\n',
        )
    )

    assert cmd_train(_sft_args(tmp_path, body)) == 0
    captured = capsys.readouterr()

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["dry_run"] is True
    assert call["spec"]["train"]["init_from_adapter"] == "source-run/final"
    assert call["runtime_secrets"] == {"WARM_START_TOKEN": "server-only-secret"}
    assert call["client_train_schema"]["version"]
    assert "init_from_adapter" in call["client_train_schema"]["authored_keys"]
    assert f"{1.005:.2f}" == "1.00", "the fixture must expose half-even formatting"
    assert "$1.01" in captured.out
    assert "TOTAL" in captured.out
    assert "server-only-secret" not in captured.out + captured.err + repr(call["spec"])
    assert "published environment" not in captured.out
    assert "source rows in published copy" not in captured.out
    assert "selected from" not in captured.out
    assert "shadowed!" in captured.err
    assert f"{algorithm.upper()} derives a prompt budget of at most 3584 tokens" in captured.err
    assert "warm-start source was configured with max_context_tokens=2048" in captured.err
    assert "provisional rank-1" not in captured.err


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_plain_non_sft_cost_uses_the_authoritative_server_quote(
    tmp_path, monkeypatch, capsys, algorithm
):
    """ordinary grpo/opd use preparation so resolved revision and disk cannot change the quote."""
    from flash.cli.commands.ops import train as commands

    client = _use_client(monkeypatch, _QuotingClient({"estimated_cost_usd": 3.25}))
    monkeypatch.setattr(commands, "shadowed_login_warning", lambda: "shadowed!")
    monkeypatch.setenv("FLASH_STYLE", "0")
    body = SFT_TOML.replace('algorithm = "sft"', f'algorithm = "{algorithm}"').replace(
        "batch_size = 8\n", "prompts_per_step = 8\nmax_examples = 40\ngroup_size = 2\n"
    )

    assert cmd_train(_sft_args(tmp_path, body)) == 0
    captured = capsys.readouterr()
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["dry_run"] is True
    assert call["spec"]["algorithm"] == algorithm
    assert call["client_train_schema"]["version"]
    assert "$3.25" in captured.out
    assert "TOTAL" in captured.out
    assert "shadowed!" in captured.err
    assert "warm-start" not in captured.err


def test_warm_start_exact_card_cost_uses_the_server_prepared_quote(tmp_path, capsys, monkeypatch):
    """an exact card pin is priced only after preparation resolves the source adapter rank."""
    client = _use_client(monkeypatch, _QuotingClient({"estimated_cost_usd": 6.5}))
    body = SFT_TOML.replace('algorithm = "sft"', 'algorithm = "grpo"').replace(
        "batch_size = 8\n",
        "prompts_per_step = 8\nmax_examples = 40\ngroup_size = 2\n"
        'init_from_adapter = "source-run/final"\n',
    )
    # sft_toml ends with an empty `[gpu]`, so the pin is appended into that section.
    body = body.replace('model = "Qwen/Qwen3.5-9B"', 'model = "Qwen/Qwen3.6-35B-A3B"')
    body += 'type = "B200"\ncount = 1\n'

    assert cmd_train(_sft_args(tmp_path, body)) == 0
    captured = capsys.readouterr()

    assert client.calls[0]["dry_run"] is True
    assert client.calls[0]["spec"]["gpu"]["type"] == "B200"
    assert client.calls[0]["spec"]["gpu"]["count"] == 1
    assert "$6.50" in captured.out
    assert "--gpus 2" not in captured.err
    assert "provisional" not in captured.err


def _warm_start_rank_boundary_config():
    from flash.cost.spec import runconfig_from_spec

    spec = spec_from_dict(
        {
            "project": "00000000-0000-0000-0000-000000000001",
            "model": "Qwen/Qwen3.6-35B-A3B",
            "algorithm": "grpo",
            "environment": {"id": "my-org/my-proj/my-env"},
            "train": {"prompts_per_step": 8, "max_examples": 40, "group_size": 2},
        }
    )
    return runconfig_from_spec(spec)


def test_higher_warm_start_rank_crosses_hardware_shape_boundary():
    """Catalog geometry, not mutable prices, owns the rank-dependent hardware boundary."""
    from dataclasses import replace

    from flash.cost import estimate_cost
    from flash.providers.core.base import GPU_INFO

    config = _warm_start_rank_boundary_config()
    rank_1 = estimate_cost(replace(config, lora_rank=1))
    rank_8 = estimate_cost(replace(config, lora_rank=8))

    assert GPU_INFO["B200"].vram_gb == 180
    assert GPU_INFO["H200"].vram_gb == 141
    assert (rank_1.required_vram_gb, rank_1.gpu_count, rank_1.gpu) == (180, 1, "B200")
    assert (rank_8.required_vram_gb, rank_8.gpu_count, rank_8.gpu) == (182, 2, "H200")


def test_higher_warm_start_rank_can_select_cheaper_hardware(monkeypatch):
    """Controlled prices prove that crossing the shape boundary can lower total cost."""
    from dataclasses import replace

    from flash.cost import estimate_cost
    from flash.providers.core import base

    b200 = replace(base.GPU_INFO["B200"], hourly_usd=5.50)
    h200 = replace(base.GPU_INFO["H200"], hourly_usd=4.00)
    monkeypatch.setattr(base, "GPU_INFO", {"B200": b200, "H200": h200})

    config = _warm_start_rank_boundary_config()
    rank_1 = estimate_cost(replace(config, lora_rank=1))
    rank_8 = estimate_cost(replace(config, lora_rank=8))

    assert (rank_1.gpu, rank_1.gpu_count, rank_1.gpu_hourly_usd) == ("B200", 1, 5.50)
    assert (rank_8.gpu, rank_8.gpu_count, rank_8.gpu_hourly_usd) == ("H200", 2, 4.00)
    assert rank_8.total_usd < rank_1.total_usd


def test_cmd_train_cost_rejects_context_above_serving_cap(tmp_path):
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "sft"\n'
        "[environment]\n"
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 8\n"
        "max_context_tokens = 33000\n"
    )
    args = types.SimpleNamespace(
        config=str(cfg), overrides=[], extra_configs=[], cost=True, dry_run=False, background=False
    )

    with pytest.raises(
        ValueError,
        match=r"train\.max_context_tokens=33000 exceeds Qwen/Qwen3\.5-9B's serving max_model_len=32768",
    ):
        cmd_train(args)


def test_cmd_train_cost_rejects_unlisted_model(tmp_path):
    """Cost is catalog-only: ``--cost`` on a non-catalog model errors cleanly (no
    sizing)."""
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "some-org/unlisted-7b"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 10\n"
    )
    args = types.SimpleNamespace(
        config=str(cfg), overrides=[], extra_configs=[], cost=True, dry_run=False, background=False
    )
    with pytest.raises((KeyError, ValueError)):
        cmd_train(args)
