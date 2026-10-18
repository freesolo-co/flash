"""prompt-budget reporting across cli, runner, and worker boundaries."""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import flash.cli.commands.ops.train as cli_train
from flash.core.spec import JobSpec
from flash.engine.plan.prompt_budget import rl_prompt_budget
from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.vram import (
    grpo_completion_len,
    grpo_rollout_seq_len,
    opd_completion_len,
    opd_rollout_seq_len,
)
from flash.schema import ConfigError, spec_from_dict


def _budget_spec(algorithm: str, train: dict | None = None, *, thinking: bool = False) -> JobSpec:
    return spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "project": "11111111-1111-4111-8111-111111111111",
            "algorithm": algorithm,
            "thinking": thinking,
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "max_examples": 5, **(train or {})},
        },
        run_id=f"{algorithm}-prompt-budget",
    )


def _serialized_budget(*, warm_start_context: int | None = None) -> dict:
    budget = {
        "algorithm": "grpo",
        "engine_len": 6144,
        "max_completion": 4096,
        "prompt_budget": 2048,
        "context_source": "recipe_default",
        "prompt_budget_is_upper_bound": True,
    }
    if warm_start_context is not None:
        budget["warm_start_context"] = warm_start_context
    return budget


def _train_config(tmp_path, *, init_from_adapter: str = ""):
    config = tmp_path / "train.toml"
    warm_start = f'init_from_adapter = "{init_from_adapter}"\n' if init_from_adapter else ""
    config.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "grpo"\n'
        '[environment]\nid = "owner/project/env"\n'
        "[train]\nepochs = 1\nmax_examples = 5\nmax_completion_tokens = 4096\n"
        f"{warm_start}"
    )
    return config


def _train_args(config, *, dry_run: bool = False):
    return SimpleNamespace(
        cost=False,
        config=str(config),
        overrides=[],
        extra_configs=[],
        dry_run=dry_run,
        background=not dry_run,
    )


def test_reporter_matches_grpo_worker_for_authored_and_default_lengths(monkeypatch) -> None:
    from flash.engine.worker.train.rl.launch import inputs

    monkeypatch.setattr(inputs._worker_state, "THINKING", False)
    monkeypatch.setattr(inputs, "model_max_position_embeddings", lambda *_args: 32768)

    for context in (0, 8192):
        train = {"max_completion_tokens": 512}
        if context:
            train["max_context_tokens"] = context
        spec = _budget_spec("grpo", train)
        lengths = inputs._resolve_sequence_lengths(
            spec.model,
            spec.model_revision,
            spec.train,
            RECIPE.rl,
            {"max_tokens": 512},
            object(),
            False,
        )
        reported = rl_prompt_budget(spec)
        assert reported is not None
        assert reported["engine_len"] == lengths["vllm_max_len"]
        assert reported["max_completion"] == lengths["max_completion"]
        assert reported["prompt_budget"] == lengths["prompt_budget"]


def test_grpo_worker_keeps_clamp_and_value_error_contract(monkeypatch, capsys) -> None:
    from flash.engine.worker.train.rl.launch import inputs

    monkeypatch.setattr(inputs._worker_state, "THINKING", False)
    monkeypatch.setattr(inputs, "model_max_position_embeddings", lambda *_args: 1024)
    spec = _budget_spec("grpo", {"max_context_tokens": 2048, "max_completion_tokens": 512})
    lengths = inputs._resolve_sequence_lengths(
        spec.model,
        spec.model_revision,
        spec.train,
        RECIPE.rl,
        {"max_tokens": 512},
        object(),
        False,
    )
    assert lengths == {"max_completion": 512, "vllm_max_len": 1024, "prompt_budget": 512}
    assert "training at 1024" in capsys.readouterr().out

    with pytest.raises(ValueError, match="engine length leaves no room for the completion"):
        inputs._resolve_sequence_lengths(
            spec.model,
            spec.model_revision,
            SimpleNamespace(max_context_tokens=512),
            RECIPE.rl,
            {"max_tokens": 512},
            object(),
            False,
        )


def _opd_prompt_state(monkeypatch, *, max_length: int, architecture_limit: int = 32768):
    import flash.engine.worker.teacher.client as teacher_client
    from flash.engine.worker.entry.opd import OpdKnobs
    from flash.engine.worker.train.opd.orchestration import prompt_preparation
    from flash.engine.worker.train.opd.orchestration.state import _OpdRequest

    class Tokenizer:
        pad_token = None
        eos_token = "<eos>"

        def apply_chat_template(self, *_args, **_kwargs):
            return list(range(100))

    monkeypatch.setattr(prompt_preparation._worker_state, "THINKING", False)
    monkeypatch.setattr(
        prompt_preparation._worker_hf,
        "load_tokenizer",
        lambda *_args, **_kwargs: Tokenizer(),
    )
    monkeypatch.setattr(prompt_preparation, "_thinking_prefill_text", lambda _tokenizer: "")
    monkeypatch.setattr(
        prompt_preparation._backend,
        "model_max_position_embeddings",
        lambda *_args: architecture_limit,
    )
    monkeypatch.setattr(
        prompt_preparation,
        "liveness_heartbeat",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(teacher_client, "TeacherClient", lambda *_args, **_kwargs: object())
    request = _OpdRequest(
        spec=None,
        env=SimpleNamespace(package_root=None),
        multi_turn=False,
        max_turns=0,
        knobs=OpdKnobs(max_completion=512, max_length=max_length),
        model_id="Qwen/Qwen3.5-9B",
        model_revision="",
    )
    state = prompt_preparation.prepare_prompts(
        request,
        [({}, [{"role": "user", "content": "question"}])],
        False,
        "capability",
        "http://control-plane",
    )
    return request, state


@pytest.mark.parametrize("max_length", [0, 2048])
def test_reporter_matches_opd_worker_for_authored_and_default_lengths(
    monkeypatch, max_length
) -> None:
    request, state = _opd_prompt_state(monkeypatch, max_length=max_length)
    train = {"max_completion_tokens": request.knobs.max_completion}
    if max_length:
        train["max_context_tokens"] = max_length
    reported = rl_prompt_budget(_budget_spec("opd", train))
    assert reported is not None
    assert reported["engine_len"] == state.max_model_len
    assert reported["max_completion"] == request.knobs.max_completion
    assert reported["prompt_budget"] == state.prompt_budget


def test_opd_worker_keeps_clamp_log_and_runtime_error_contract(monkeypatch, capsys) -> None:
    _request, state = _opd_prompt_state(monkeypatch, max_length=2048, architecture_limit=1024)
    assert state.max_model_len == 1024
    assert state.prompt_budget == 512
    assert "training at 1024" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="opd max_context_tokens leaves no room for a prompt"):
        _opd_prompt_state(monkeypatch, max_length=512, architecture_limit=512)


def test_rollout_helpers_preserve_omitted_zero_and_thinking_behavior() -> None:
    assert grpo_completion_len(None, False) == RECIPE.rl.max_completion_len
    assert grpo_completion_len(0, True) == RECIPE.rl.max_completion_len_thinking
    assert grpo_rollout_seq_len(0, 512, False) == RECIPE.rl.max_prompt_len + 512
    assert opd_completion_len(None, False) == RECIPE.opd.max_completion_len
    assert opd_completion_len(0, True) == RECIPE.opd.max_completion_len_thinking
    assert opd_rollout_seq_len(0, 512, False) == RECIPE.opd.max_prompt_len + 512


def test_defaulted_budget_warning_and_authored_silence(capsys) -> None:
    from flash.cli.commands.ops.prompt_budget import (
        print_status_prompt_budget_warning,
        prompt_budget_warning,
    )

    defaulted = rl_prompt_budget(_budget_spec("grpo", {"max_completion_tokens": 4096}))
    assert defaulted is not None
    assert defaulted["prompt_budget"] == RECIPE.rl.max_prompt_len
    message = prompt_budget_warning(defaulted)
    assert message is not None
    assert "at most 2048 tokens" in message
    assert "DROPPED, not truncated" in message
    assert "can be smaller" in message

    authored = rl_prompt_budget(
        _budget_spec("grpo", {"max_context_tokens": 8192, "max_completion_tokens": 4096})
    )
    assert authored is not None
    assert authored["context_source"] == "authored"
    assert authored["prompt_budget_is_upper_bound"] is True
    assert prompt_budget_warning(authored) is None
    print_status_prompt_budget_warning({"prompt_budget": defaulted})
    assert "warning:" in capsys.readouterr().err


def test_malformed_status_budget_is_not_treated_as_validated(capsys) -> None:
    from flash.cli.commands.ops.prompt_budget import (
        print_status_prompt_budget_warning,
        prompt_budget_validation_suffix,
    )

    malformed = {
        "context_source": "recipe_default",
        "prompt_budget": 2048,
        "max_completion": 4096,
    }
    status = {"prompt_budget": malformed}

    assert prompt_budget_validation_suffix(status) == ""
    print_status_prompt_budget_warning(status)
    assert capsys.readouterr().err == ""


def test_sft_has_no_drop_semantics_prompt_budget() -> None:
    from flash.cli.commands.ops.prompt_budget import prompt_budget_warning

    budget = rl_prompt_budget(_budget_spec("sft", {"max_context_tokens": 8192}))
    assert budget is None
    assert prompt_budget_warning(budget) is None


def test_paid_submit_owner_warning_precedes_create_and_names_source(
    monkeypatch, tmp_path, capsys
) -> None:
    from flash.cli.commands.ops import prompt_budget as cli_prompt_budget

    events = []

    class Client:
        def get_run(self, run_id):
            events.append(("get_run", run_id))
            return {"spec": {"train": {"max_context_tokens": 8192}}}

        def create_run(self, *_args, **_kwargs):
            early = capsys.readouterr()
            events.append(("create_run", early.err))
            return {
                "run_id": "child-run",
                "state": "queued",
                "prompt_budget": _serialized_budget(warm_start_context=8192),
            }

    monkeypatch.setattr(cli_train, "client_from_config", lambda: Client())
    monkeypatch.setattr(cli_prompt_budget.render, "styled", lambda: False)
    result = cli_train.cmd_train(
        _train_args(_train_config(tmp_path, init_from_adapter="source-run/final"))
    )
    captured = capsys.readouterr()

    assert result == 0
    assert events[0] == ("get_run", "source-run")
    assert events[1][0] == "create_run"
    assert "prompt budget of at most 2048 tokens" in events[1][1]
    assert "configured with max_context_tokens=8192" in events[1][1]
    assert "Derived locally by this CLI" in events[1][1]
    assert captured.err == ""
    assert json.loads(captured.out)["prompt_budget"]["warm_start_context"] == 8192


def test_paid_submit_org_peer_prints_only_supplement_after_create(
    monkeypatch, tmp_path, capsys
) -> None:
    from flash.cli.commands.ops import prompt_budget as cli_prompt_budget

    events = []

    class Client:
        def get_run(self, run_id):
            events.append(("get_run", run_id))
            raise RuntimeError("the cli key does not own this org-peer run")

        def create_run(self, *_args, **_kwargs):
            early = capsys.readouterr()
            events.append(("create_run", early.err))
            return {
                "run_id": "child-run",
                "state": "queued",
                "prompt_budget": _serialized_budget(warm_start_context=8192),
            }

    monkeypatch.setattr(cli_train, "client_from_config", lambda: Client())
    monkeypatch.setattr(cli_prompt_budget.render, "styled", lambda: False)
    result = cli_train.cmd_train(
        _train_args(_train_config(tmp_path, init_from_adapter="org-peer-run/final"))
    )
    captured = capsys.readouterr()

    assert result == 0
    assert events[0] == ("get_run", "org-peer-run")
    assert events[1][0] == "create_run"
    assert events[1][1].count("prompt budget of at most") == 1
    assert "max_context_tokens=8192" not in events[1][1]
    assert "configured with max_context_tokens=8192" in captured.err
    assert "prompt budget of at most" not in captured.err
    assert json.loads(captured.out)["prompt_budget"]["warm_start_context"] == 8192


def test_source_lookup_failure_is_non_fatal() -> None:
    from flash.cli.commands.ops.prompt_budget import warmstart_source_context

    class Client:
        def get_run(self, _run_id):
            raise RuntimeError("unavailable")

    assert (
        warmstart_source_context(
            Client(), _budget_spec("grpo", {"init_from_adapter": "source-run/final"})
        )
        is None
    )


def test_dry_run_keeps_budget_in_machine_readable_stdout(monkeypatch, tmp_path, capsys) -> None:
    from flash.cli.commands.ops import prompt_budget as cli_prompt_budget

    class Client:
        def create_run(self, *_args, **_kwargs):
            return {
                "run_id": "dry-run",
                "state": "dry_run",
                "prompt_budget": _serialized_budget(),
                "train_schema_compatibility": {"status": "agreement"},
                "affordability_verified": True,
            }

    monkeypatch.setattr(cli_train, "client_from_config", lambda: Client())
    monkeypatch.setattr(cli_prompt_budget.render, "styled", lambda: False)
    result = cli_train.cmd_train(_train_args(_train_config(tmp_path), dry_run=True))
    captured = capsys.readouterr()

    assert result == 0
    payload = json.loads(captured.out)
    assert payload["prompt_budget"] == _serialized_budget()
    assert "prompt budget (upper bound)" in captured.err
    assert "prompt budget of at most 2048 tokens" in captured.err


def test_status_serialization_keeps_prompt_budget_payload_identical() -> None:
    from flash.runner.lifecycle.state import RunStatus

    budget = rl_prompt_budget(_budget_spec("grpo"))
    assert budget is not None
    status = RunStatus(run_id="run", state="queued", spec={}, prompt_budget=budget)
    assert status.to_dict()["prompt_budget"] == budget


def test_authored_context_below_completion_is_still_rejected_at_parse_time() -> None:
    with pytest.raises(ConfigError, match="prompt budget"):
        _budget_spec("grpo", {"max_context_tokens": 512, "max_completion_tokens": 512})
