"""`flash train --cost`: map a training config to a pre-flight cost."""

from __future__ import annotations

import copy
import dataclasses
import types

import pytest

from flash.cli.commands import cmd_train
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
        "batch_size": 16,
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


def test_grpo_epochs_need_max_examples_for_cost():
    spec = _spec(**{"train.max_examples": None, "train.epochs": 2})
    assert _spec_steps(spec) == 2


def test_grpo_positive_max_steps_is_authoritative():
    assert _spec_steps(_spec(**{"train.max_steps": 73})) == 73
    assert _spec_steps(_spec(**{"train.max_steps": 0})) == 50


def test_required_save_density_adds_wall_time_and_cost_without_changing_steps():
    from flash.cost.analytical import estimate_cost

    for method in ("sft", "grpo"):
        common = {
            "model_id": "Qwen/Qwen3.5-4B",
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

    model_id = "Qwen/Qwen3.5-4B"
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
        "model_id": "Qwen/Qwen3.5-4B",
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
    from flash.runner import charge_usd_for_spec

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
    raw["train"].update({"epochs": 2, "max_examples": 17, "batch_size": 8, "group_size": 1})
    spec = spec_from_dict(raw)
    assert _spec_steps(spec) == 5  # ceil(17 rows * 2 epochs / batch_size 8)


def test_opd_positive_max_steps_is_authoritative():
    raw = copy.deepcopy(GRPO_RAW)
    raw["algorithm"] = "opd"
    raw["train"].update(
        {"epochs": 2, "max_examples": 17, "batch_size": 8, "group_size": 1, "max_steps": 31}
    )
    assert _spec_steps(spec_from_dict(raw)) == 31


def test_opd_runconfig_carries_selected_teacher_and_prices_it():
    from flash.cost.analytical import estimate_cost

    def _opd(teacher=None):
        raw = copy.deepcopy(GRPO_RAW)
        raw["model"] = "Qwen/Qwen3.5-4B"
        raw["algorithm"] = "opd"
        raw["train"].update({"epochs": 1, "max_examples": 40, "batch_size": 8, "group_size": 1})
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
            "model": "Qwen/Qwen3.5-4B",
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
                "model": "Qwen/Qwen3.5-4B",
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
                "model": "Qwen/Qwen3.5-4B",
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
    cfg = RunConfig("Qwen/Qwen3.5-4B", "sft", 10, 2048)
    assert cfg.seq_len == 2048
    assert cfg.train_tokens is None


def test_cmd_train_cost_prints_breakdown_without_submitting(tmp_path, capsys):
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
        "batch_size = 16\n"
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
    # --cost is local: it must NOT touch the control-plane client. GRPO needs no env load, and
    # estimate_cost sizes VRAM offline, so no network is required for a listed model.
    rc = cmd_train(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "TOTAL" in out
    assert "$" in out
    assert "GPU" in out  # the breakdown names the chosen (provisional cheapest-fit) class


SFT_TOML = (
    'model = "Qwen/Qwen3.5-4B"\n'
    'project = "11111111-1111-4111-8111-111111111111"\n'
    'algorithm = "sft"\n'
    "[environment]\n"
    'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
    "[train]\n"
    "epochs = 1\n"
    "batch_size = 8\n"
    "[gpu]\n"
)

PROFILE_RUN_ID = "profile-sft-" + "a" * 64

EXACT_PROFILE = {
    "authoritative_steps": 7,
    "selected_examples": 10,
    "retained_examples": 8,
    "dropped_examples": 2,
    "authoritative_compute_tokens": 4096,
    "authoritative_supervised_tokens": 2048,
    "packing_mode": "packed",
    "architecture_mode": "pure-attention",
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
    """Server that already holds a matching profile and answers the dry-run with an exact quote."""

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


class _PendingClient:
    """Server with no matching profile: it starts one and rejects the quote with 409."""

    def __init__(
        self,
        *,
        state: str = "queued",
        profile_quote: object = 0.25,
        owned: bool | None = True,
        launched: bool | None = True,
    ):
        from flash.client import ApiError

        detail = {
            "code": "workload_profile_pending",
            "profile_run_id": PROFILE_RUN_ID,
            "state": state,
        }
        if owned is not None:
            detail["owned"] = owned
        if launched is not None:
            detail["launched"] = launched
        self.error = ApiError(409, "workload profile pending", detail=detail)
        self.profile_quote = profile_quote
        self.get_run_calls: list[str] = []

    def create_run(self, spec, runtime_secrets=None, dry_run=False, client_train_schema=None):
        raise self.error

    def get_run(self, run_id):
        self.get_run_calls.append(run_id)
        return {"run_id": run_id, "state": "queued", "estimated_cost_usd": self.profile_quote}


def _use_client(monkeypatch, client):
    from flash.cli import commands

    monkeypatch.setattr(commands, "client_from_config", lambda *a, **k: client)
    monkeypatch.setenv("FLASH_STYLE", "0")
    return client


def test_sft_cost_asks_the_server_for_the_exact_quote_without_creating_a_training_run(
    tmp_path, monkeypatch, capsys
):
    """sft ``--cost`` is a dry-run submit, because only the server can hold the profile.

    The public payload must stay public: the internal profile carrier is what the server attaches
    after measuring, so a client that could send one could also fabricate the workload its own
    quote is derived from.
    """
    client = _use_client(monkeypatch, _QuotingClient())

    rc = cmd_train(_sft_args(tmp_path))

    assert rc == 0
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["dry_run"] is True
    assert not any(key.startswith("workload_profile") for key in call["spec"])


def test_sft_cost_reports_the_measured_workload_and_no_invented_hardware(
    tmp_path, monkeypatch, capsys
):
    """The panel prints what was measured. It must not fabricate the timing fields it no longer has.

    The analytical breakdown named a GPU, an hourly rate, and a per-step time. An exact quote is
    computed server-side, so reprinting that layout here would mean inventing every number in it.
    """
    _use_client(monkeypatch, _QuotingClient())

    rc = cmd_train(_sft_args(tmp_path))
    captured = capsys.readouterr()

    assert rc == 0
    assert "$1.25" in captured.out
    assert "7 steps" in captured.out
    assert "8 trained of 10 selected" in captured.out
    assert "(2 dropped)" in captured.out
    assert "4,096 compute, 2,048 supervised" in captured.out
    assert "packed (pure-attention)" in captured.out
    assert "bbbbbbbbbbbb" in captured.out
    for invented in ("/hr", "setup", "per-step", "train_seconds"):
        assert invented not in captured.out
    assert "nothing was charged for training" in captured.err


def test_sft_cost_omits_aggregates_the_profile_did_not_report(tmp_path, monkeypatch, capsys):
    """A partial profile drops rows rather than defaulting them to zero."""
    _use_client(
        monkeypatch,
        _QuotingClient({"estimated_cost_usd": 0.5, "workload_profile": {"authoritative_steps": 3}}),
    )

    rc = cmd_train(_sft_args(tmp_path))
    out = capsys.readouterr().out

    assert rc == 0
    assert "3 steps" in out
    assert "$0.50" in out
    for absent in ("examples", "tokens", "workload", "profile"):
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


def test_sft_cost_on_a_profile_miss_explains_the_separate_charge_and_fails(
    tmp_path, monkeypatch, capsys
):
    """A miss must not read as "your training started". It started a different, billed job.

    The user asked what training would cost and instead incurred a charge. Naming the profile run,
    its own quote, and what was NOT charged is the difference between that and a surprise.
    """
    from flash.client import ClientError

    client = _use_client(monkeypatch, _PendingClient())

    with pytest.raises(ClientError, match=f"workload profile {PROFILE_RUN_ID} is queued"):
        cmd_train(_sft_args(tmp_path))

    err = capsys.readouterr().err
    assert "$0.25" in err
    assert "billed on its own" in err
    assert "no training run was created" in err
    assert f"flash runs status {PROFILE_RUN_ID}" in err
    assert client.get_run_calls == [PROFILE_RUN_ID]


@pytest.mark.parametrize("quote", [True, "0.25", None])
def test_sft_cost_pending_omits_the_charge_it_cannot_read(tmp_path, monkeypatch, quote, capsys):
    """An unreadable profile quote drops the amount rather than inventing one."""
    from flash.client import ClientError

    _use_client(monkeypatch, _PendingClient(profile_quote=quote))

    with pytest.raises(ClientError):
        cmd_train(_sft_args(tmp_path))

    err = capsys.readouterr().err
    assert "billed on its own;" in err
    assert "$" not in err


def test_sft_cost_pending_on_someone_elses_profile_promises_no_charge_and_no_poll(
    tmp_path, monkeypatch, capsys
):
    """A profile another key launched is not readable here, so neither instruction may be repeated.

    The id is deterministic in the workload, so this is ordinary reuse. Telling the user to poll it
    would send them to a 404, and telling them they were billed for it would be false: they were
    not charged, and nothing was started on their behalf.
    """
    from flash.client import ClientError

    client = _use_client(monkeypatch, _PendingClient(owned=False))

    with pytest.raises(ClientError, match=f"workload profile {PROFILE_RUN_ID} is queued"):
        cmd_train(_sft_args(tmp_path))

    err = capsys.readouterr().err
    assert "already being measured" in err
    assert "nothing was started or charged here" in err
    assert "runs status" not in err
    assert "$" not in err
    # no charge was quoted, so the unreadable run is never fetched to price one.
    assert client.get_run_calls == []


def test_sft_cost_pending_without_an_ownership_flag_keeps_the_owner_wording(
    tmp_path, monkeypatch, capsys
):
    """An older server omits ``owned``. Absent must mean "yours", not "someone else's".

    Reading a missing key as not-owned would tell every user of such a server that they were not
    charged for a profile run they in fact own and are paying for.
    """
    from flash.client import ClientError

    client = _use_client(monkeypatch, _PendingClient(owned=None))

    with pytest.raises(ClientError):
        cmd_train(_sft_args(tmp_path))

    err = capsys.readouterr().err
    assert "billed on its own" in err
    assert f"flash runs status {PROFILE_RUN_ID}" in err
    assert client.get_run_calls == [PROFILE_RUN_ID]


def test_sft_cost_pending_on_your_own_running_profile_names_no_second_charge(
    tmp_path, monkeypatch, capsys
):
    """Re-running against your own in-flight profile joins it. It does not start or bill another.

    The profile id is deterministic in the workload, so the natural thing to do after a miss is to
    re-run the same command. Only the request that won the claim launched anything; every later one
    returns the same 409. Repeating the start-and-bill wording there would name a charge per retry
    that the account never sees, and would read as the profile being relaunched each time.
    """
    from flash.client import ClientError

    client = _use_client(monkeypatch, _PendingClient(state="running", launched=False))

    with pytest.raises(ClientError, match=f"workload profile {PROFILE_RUN_ID} is running"):
        cmd_train(_sft_args(tmp_path))

    err = capsys.readouterr().err
    assert "launched nothing and charged nothing" in err
    assert "the server started a separate profile run" not in err
    assert "billed on its own" not in err
    # still the owner's run, so the poll instruction stays.
    assert f"flash runs status {PROFILE_RUN_ID}" in err
    # no charge is quoted for a launch that did not happen, so the run is never priced.
    assert "$" not in err
    assert client.get_run_calls == []


def test_sft_cost_pending_without_a_launched_flag_keeps_the_charge_warning(
    tmp_path, monkeypatch, capsys
):
    """An older server omits ``launched``. Absent must mean "you were charged", not "you weren't".

    The two errors are not symmetric. Warning about a charge that did not happen costs a re-read;
    staying silent about one that did leaves the user paying for a run they were never told about.
    """
    from flash.client import ClientError

    client = _use_client(monkeypatch, _PendingClient(launched=None))

    with pytest.raises(ClientError):
        cmd_train(_sft_args(tmp_path))

    err = capsys.readouterr().err
    assert "billed on its own" in err
    assert "$0.25" in err
    assert client.get_run_calls == [PROFILE_RUN_ID]


def test_sft_dry_run_shares_the_profile_pending_path(tmp_path, monkeypatch, capsys):
    """`train --dry-run` hits the same miss and must explain it identically, not print a traceback."""
    from flash.client import ClientError

    args = _sft_args(tmp_path)
    args.cost, args.dry_run = False, True
    _use_client(monkeypatch, _PendingClient())

    with pytest.raises(ClientError, match="workload profile"):
        cmd_train(args)
    assert "no training run was created" in capsys.readouterr().err


def test_sft_real_submit_shares_the_profile_pending_path(tmp_path, monkeypatch, capsys):
    """A real submit misses the cache like a preview does, and starts the same billed profile run.

    This is the path that actually spends money, so it is the one where a bare 409 is worst: the
    user is charged for a profile run whose id, purpose and cost the error never mentions.
    """
    from flash.client import ClientError

    args = _sft_args(tmp_path)
    args.cost, args.dry_run = False, False
    client = _use_client(monkeypatch, _PendingClient())

    with pytest.raises(ClientError, match=f"workload profile {PROFILE_RUN_ID} is queued"):
        cmd_train(args)

    err = capsys.readouterr().err
    assert "billed on its own" in err
    assert "no training run was created" in err
    assert f"flash runs status {PROFILE_RUN_ID}" in err


def test_sft_cost_leaves_unrelated_api_errors_alone(tmp_path, monkeypatch):
    """Only the profile-pending code is translated; every other rejection surfaces as itself."""
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
    """The generic ``--cost`` hook stays quiet because it cannot know the algorithm yet. sft does
    reach an organization and can start a billed profile there, so it warns for itself."""
    from flash.cli import commands

    monkeypatch.setattr(commands, "shadowed_login_warning", lambda: "shadowed!")
    _use_client(monkeypatch, _QuotingClient())

    assert cmd_train(_sft_args(tmp_path)) == 0
    assert "shadowed!" in capsys.readouterr().err


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_non_sft_cost_stays_offline(tmp_path, monkeypatch, capsys, algorithm):
    """grpo/opd keep the local analytical quote until PR2 profiles their rollouts.

    Asserted by making the client constructor itself fatal: a passing quote then proves no
    authenticated request was possible, not merely that none was observed.
    """
    from flash.cli import commands

    def _forbidden(*_a, **_kw):
        raise AssertionError(f"{algorithm} --cost must not contact the control plane")

    monkeypatch.setattr(commands, "client_from_config", _forbidden)
    monkeypatch.setattr(commands, "shadowed_login_warning", lambda: "shadowed!")
    monkeypatch.setenv("FLASH_STYLE", "0")
    # group_size 2 is grpo's floor (advantages are group-relative) and is valid for opd too.
    body = SFT_TOML.replace('algorithm = "sft"', f'algorithm = "{algorithm}"').replace(
        "batch_size = 8\n", "batch_size = 8\nmax_examples = 40\ngroup_size = 2\n"
    )

    assert cmd_train(_sft_args(tmp_path, body)) == 0
    captured = capsys.readouterr()
    assert "TOTAL" in captured.out
    assert "shadowed!" not in captured.err


def test_cmd_train_cost_rejects_context_above_serving_cap(tmp_path):
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-4B"\n'
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
        match=r"train\.max_context_tokens=33000 exceeds Qwen/Qwen3\.5-4B's serving max_model_len=32768",
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
