"""The profile worker: what it must measure, and what it must never touch.

A profile run exists to answer "how much work is this?" before a training run is created or quoted.
Two properties make that safe to bill separately from training, and both are asserted here:

- it is cpu-only. it loads a tokenizer and the pinned environment and then exits, so it can be
  allocated the cheapest card in the fleet rather than the one training will need. if it ever
  reached for weights or cuda, the "cheapest card" quote would be pricing a job that cannot run
  there, and the separate charge would stop being defensible.
- it cannot half-succeed. a profile that could not be produced must end as a failed profile *run*,
  never as a completed run carrying a partial artifact, because preparation trusts any artifact it
  finds and would freeze a quote against it.
"""

from __future__ import annotations

import builtins
import sys
from dataclasses import replace

import pytest

from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec
from flash.workload_profile import SFT_PROFILE_KIND, sft_profile_input_digest

_PRODUCER_VERSION = "9.9.9"


class _Tokenizer:
    eos_token = "|"
    eos_token_id = 2
    pad_token = None
    pad_token_id = 0
    all_special_ids = (0, 1, 2)

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking, **_kwargs
    ):
        assert not tokenize
        text = "".join(str(message.get("content") or "") for message in messages)
        return text + (">" if add_generation_prompt else "")

    def __call__(self, texts, *, truncation, max_length):
        assert truncation
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [
                [self.eos_token_id if ch == self.eos_token else 3 + ord(ch) % 89 for ch in text][
                    :max_length
                ]
                for text in texts
            ]
        }


class _Environment:
    multi_turn = False
    package_root = None

    def dataset(self):
        return [
            {"prompt": "one", "answer": "alpha"},
            {"prompt": "two", "answer": "beta"},
        ]

    def prompt_messages(self, row):
        return [{"role": "user", "content": row["prompt"]}]

    def sft_completion(self, row):
        return [{"role": "assistant", "content": row["answer"]}]


def _profile_spec() -> JobSpec:
    spec = JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(id="team/example", resolved_sha="b" * 40),
        train=TrainSpec(epochs=1, batch_size=2, max_context_tokens=32, max_examples=2),
        gpu=GpuSpec(count=1, max_wall_seconds=600),
        seed=7,
    )
    digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=spec.model_revision,
        producer_version=_PRODUCER_VERSION,
    )
    return replace(
        spec,
        workload_profile_kind=SFT_PROFILE_KIND,
        workload_profile_input_digest=digest,
    )


@pytest.fixture
def profile_worker(monkeypatch):
    """The profile entrypoint with its worker-module boundary stubbed, nothing else.

    Only the four things a worker cannot do inside a test are replaced: the job spec, the loaded
    environment, the tokenizer, and the terminal upload. prepare_sft_workload runs for real, because
    it is the measurement under test.
    """
    import flash.engine.worker as worker
    from flash.engine.worker import sft_profile

    monkeypatch.setattr(worker, "JOB_SPEC", _profile_spec(), raising=False)
    monkeypatch.setattr(worker, "require_active_env", lambda: _Environment(), raising=False)
    monkeypatch.setattr(
        worker, "load_tokenizer", lambda _model, revision=None: _Tokenizer(), raising=False
    )
    monkeypatch.setattr(worker, "heartbeat", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(sft_profile, "__version__", _PRODUCER_VERSION, raising=False)
    monkeypatch.setattr("flash.__version__", _PRODUCER_VERSION, raising=False)

    finalized: list = []
    monkeypatch.setattr(
        worker,
        "_finalize",
        lambda metrics, **kwargs: finalized.append((metrics, kwargs)),
        raising=False,
    )
    return sft_profile, finalized


def test_profile_run_emits_the_measurement_and_stamps_when_it_ran(profile_worker) -> None:
    sft_profile, finalized = profile_worker

    sft_profile.run_sft_profile()

    assert len(finalized) == 1
    metrics, kwargs = finalized[0]
    assert metrics.phase == "profile"
    assert kwargs["heartbeat_fields"]["profile_kind"] == SFT_PROFILE_KIND
    profile = metrics.workload_profile
    assert profile["retained_examples"] == 2
    assert profile["input_digest"] == _profile_spec().workload_profile_input_digest
    # the producer is the only writer of provenance; the training worker recomputes with 0.0.
    assert profile["created_at"] > 0.0


def test_profile_run_loads_no_weights_and_never_imports_torch(profile_worker, monkeypatch) -> None:
    """The whole basis for quoting a profile off the cheapest card in the fleet.

    Guarded at the boundaries that would actually cost money: the weight prefetch (what makes a
    training job need a big disk and a big card) and importing torch at all (the gateway to cuda,
    and the thing that makes the job need a gpu). The import guard is deliberately stricter than a
    cuda-initialization check -- a cuda assertion is vacuous in an environment without torch
    installed, and would report "cpu-only" for a run that had in fact pulled in the model stack.
    Detonators, not counters: a call fails at the call site rather than being tallied and forgotten.
    """
    sft_profile, finalized = profile_worker
    import flash.engine.worker as worker

    def _forbidden(name):
        def _boom(*_a, **_kw):
            raise AssertionError(f"profile run reached {name}")

        return _boom

    monkeypatch.setattr(worker, "prefetch_model", _forbidden("prefetch_model"), raising=False)
    monkeypatch.setattr(worker, "load_model", _forbidden("load_model"), raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("profile run imported torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)

    sft_profile.run_sft_profile()

    assert len(finalized) == 1, "the measurement must still complete without the model stack"
    assert "torch" not in sys.modules


def test_profile_run_refuses_a_spec_whose_digest_does_not_key_it(profile_worker) -> None:
    """The worker re-derives its own key rather than trusting the one it was handed.

    Without this the control plane could store the artifact under a key describing a different
    workload, and preparation would later match that key and freeze a quote from the wrong
    measurement.
    """
    sft_profile, finalized = profile_worker
    import flash.engine.worker as worker

    worker.JOB_SPEC = replace(worker.JOB_SPEC, workload_profile_input_digest="c" * 64)

    with pytest.raises(ValueError, match="input digest does not match"):
        sft_profile.run_sft_profile()
    assert finalized == []


def test_profile_run_refuses_a_training_spec(profile_worker) -> None:
    sft_profile, finalized = profile_worker
    import flash.engine.worker as worker

    worker.JOB_SPEC = replace(worker.JOB_SPEC, workload_profile_kind="")

    with pytest.raises(RuntimeError, match="requires an internal profile job spec"):
        sft_profile.run_sft_profile()
    assert finalized == []


def test_a_profile_that_cannot_be_measured_never_reaches_terminal_success(
    profile_worker, monkeypatch
) -> None:
    """A failed measurement must be a failed run, not a completed run with a partial artifact.

    Preparation trusts any artifact whose key matches, so "completed, but the measurement is
    missing/partial" is the one outcome that would silently pass a wrong quote through the
    fail-closed gate. Raising is what keeps that outcome unreachable.
    """
    sft_profile, finalized = profile_worker

    def _explode(*_args, **_kwargs):
        raise RuntimeError("environment dataset unreadable")

    monkeypatch.setattr(sft_profile, "prepare_sft_workload", _explode)

    with pytest.raises(RuntimeError, match="environment dataset unreadable"):
        sft_profile.run_sft_profile()
    assert finalized == [], "a failed measurement must not publish DONE or metrics"


def test_profile_measurement_is_reproducible_across_two_runs(profile_worker) -> None:
    """Two profile runs of one workload must agree on everything except when they ran.

    This is the property the training worker's parity check depends on: it re-derives the
    measurement and compares it to the stored artifact, so any per-run variation in a measured field
    would fail every training run.
    """
    sft_profile, finalized = profile_worker

    sft_profile.run_sft_profile()
    sft_profile.run_sft_profile()

    first, second = (metrics.workload_profile for metrics, _kw in finalized)
    assert first["content_digest"] == second["content_digest"]
    assert {k: v for k, v in first.items() if k != "created_at"} == {
        k: v for k, v in second.items() if k != "created_at"
    }
