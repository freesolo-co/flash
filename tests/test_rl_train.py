"""verl grpo backend: dispatch, data/config/reward glue, and reward parity (cpu-only, no verl)."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import flash.engine.plan.steps as plan_steps
import flash.engine.worker.train.entry.rl_train_runner as rl_runner
import flash.engine.worker.train.rl.launch.config as worker_config
import flash.engine.worker.train.rl.launch.inputs as rl_inputs
import flash.engine.worker.train.rl.launch.verl_config as rl_verl
import flash.engine.worker.train.rl.rollout.multi_turn as rl_multi
import flash.engine.worker.train.rl.rollout.reward_module as rl_reward
import flash.engine.worker.train.rl.rollout.single_turn as rl_single
from flash.core.grpo import SUPPORTED_GRPO_GROUP_SIZES
from flash.engine.worker.entry import rl
from flash.engine.worker.io.heartbeat import RewardObservabilityBuffer
from flash.engine.worker.train.core.child import runtime as child_runtime
from flash.engine.worker.train.core.child import runtime as verl_child_runtime
from flash.engine.worker.train.entry import backend_common, rl_train, sft_train
from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn
from flash.engine.worker.train.rl.child import patches as verl_patches
from flash.engine.worker.train.rl.child import plugin as grpo_plugin


# ------------------------------- dispatch -------------------------------
@pytest.mark.parametrize("stale", [None, "verl", "trl", "megatron"])
def test_run_rl_always_delegates_to_verl(monkeypatch, stale):
    """run_rl delegates to run_rl_train unconditionally -- no env key selects a backend.

    verl is the only trainer, so a stale FLASH_RL_BACKEND left in a config must be inert rather than
    routing anywhere else or raising.
    """
    called = []
    monkeypatch.setattr(rl_train, "run_rl_train", lambda: called.append(True))
    if stale is None:
        monkeypatch.delenv("FLASH_RL_BACKEND", raising=False)
    else:
        monkeypatch.setenv("FLASH_RL_BACKEND", stale)
    rl.run_rl()
    assert called == [True]


class _FakeGrpoProcess:
    """Stand-in for the verl Popen.

    ``never_exits`` models the real hazard the bounded wait exists for: the direct child is gone
    but a grandchild holds the merged stdout pipe, so the stream ends while ``wait`` would block.
    """

    def __init__(self, lines, *, wait_code, stale_return_code, never_exits=False, poll_code=None):
        self.stdout = iter(lines)
        self.pid = 424242
        self.returncode = stale_return_code
        self._wait_code = wait_code
        self._never_exits = never_exits
        # what the watchdog thread sees. None keeps the child "running" so the watchdog never arms,
        # which is what every test that is not about the watchdog wants.
        self._poll_code = poll_code
        self.wait_calls = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self):
        return self._poll_code

    # matches subprocess.Popen.wait; a stub that omitted `timeout` would make an unbounded wait
    # impossible to write a failing test for.
    def wait(self, timeout=None):
        self.wait_calls += 1
        self.wait_timeouts.append(timeout)
        if self._never_exits:
            if timeout is None:
                raise AssertionError(
                    "waited on the verl child with no timeout; a grandchild holding the merged "
                    "stdout pipe parks the attempt on a paid gpu with nothing left to report"
                )
            raise subprocess.TimeoutExpired(cmd="verl", timeout=timeout)
        return self._wait_code


def test_grpo_subprocess_stream_classifies_the_recorded_nonzero_exit(monkeypatch):
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    terminated = []
    monkeypatch.setattr(
        rl_runner,
        "kill_process_group",
        lambda proc, *, process_group_id: terminated.append((proc, process_group_id)),
    )
    signature = "cudaErrorDevicesUnavailable\n"
    lines = [signature, *(f"filler-{i}\n" for i in range(150))]
    proc = _FakeGrpoProcess(lines, wait_code=17, stale_return_code=0)
    stream = rl_runner._GrpoSubprocessStream(proc)

    assert list(stream) == lines
    with pytest.raises(RetriableInfraError) as exc_info:
        stream.wait_and_classify()

    assert "cudaErrorDevicesUnavailable" in str(exc_info.value)
    assert proc.wait_calls == 1
    assert terminated == [(proc, proc.pid)]


def test_grpo_unclassified_nonzero_exit_still_tears_down_the_group(monkeypatch):
    """The one failing path the classifier lets through must still reach teardown.

    `raise_for_classified_verl_exit` RETURNS rather than raising when a nonzero exit carries no oom
    evidence and no retriable signature, so this path reaches neither the except-clause teardown nor
    the exception the caller would otherwise tear down on. the direct child is gone, but its group
    need not be: a surviving EngineCore holds its cuda context and the next attempt on this worker
    gets an occupied gpu.
    """
    terminated = []
    monkeypatch.setattr(
        rl_runner,
        "kill_process_group",
        lambda proc, *, process_group_id: terminated.append((proc, process_group_id)),
    )
    # deliberately unclassifiable: no oom evidence, no retriable infra signature.
    proc = _FakeGrpoProcess(["unrelated trainer failure\n"], wait_code=9, stale_return_code=0)
    stream = rl_runner._GrpoSubprocessStream(proc)

    assert list(stream) == ["unrelated trainer failure\n"]
    # returned, not raised: the status stays terminal and the caller still sees 9.
    assert stream.wait_and_classify() == 9
    assert terminated == [(proc, proc.pid)], (
        "an unclassified nonzero exit left the process group alive; a surviving EngineCore strands "
        "the gpu for every later attempt on this worker"
    )


def test_grpo_wait_is_bounded_so_a_pipe_holding_grandchild_cannot_park_the_attempt(monkeypatch):
    """The stream ends on stdout EOF, which is not the direct child's exit.

    verl spawns vllm's EngineCore as a grandchild that inherits the same merged pipe. the reverse
    shape -- child dead, grandchild holding the pipe -- ends the iterator early; this one is a child
    that outlives its own stdout, where an unbounded wait blocks forever on a paid gpu with nothing
    left to report. the fake asserts on `timeout is None` directly, so an unbounded wait fails here
    rather than hanging the suite.
    """
    terminated = []
    monkeypatch.setattr(
        rl_runner,
        "kill_process_group",
        lambda proc, *, process_group_id: terminated.append((proc, process_group_id)),
    )
    proc = _FakeGrpoProcess(["step: 1\n"], wait_code=0, stale_return_code=None, never_exits=True)
    stream = rl_runner._GrpoSubprocessStream(proc)

    assert list(stream) == ["step: 1\n"]
    return_code = stream.wait_and_classify()

    assert proc.wait_timeouts == [backend_common._TEARDOWN_GRACE_S], (
        "the wait on the verl child must carry the teardown grace, not block indefinitely"
    )
    assert terminated == [(proc, proc.pid)], "a child that outlived its stdout was never torn down"
    # the stub never collects, standing in for a member wedged in uninterruptible io. that is a
    # failure however it reads, so it must not surface as a success.
    assert return_code != 0


def test_grpo_consumer_inside_one_long_step_is_not_read_as_a_stuck_reader(monkeypatch):
    """This is a generator, so the consumer's work for a line runs while it is suspended at `yield`.

    A grpo step -- generation, reward, an optimizer pass -- easily outlasts the orphaned-pipe grace.
    Progress counted at the arrival of a line cannot advance during it, so the consumer looks
    exactly like a reader blocked on a pipe nobody will close, and the group is torn down under a
    run that is working. The child polls as exited here, which is the state that arms the
    watchdog at all, so the only thing keeping the group alive is the in-flight line.
    """
    monkeypatch.setattr(rl_runner, "_ORPHANED_PIPE_GRACE_S", 0.1)
    terminated = []
    # patch the binding the WATCHDOG resolves, which is backend_common's -- it calls the name from
    # its own module, so patching rl_train's copy here would intercept nothing and the test would
    # pass against the very defect it names.
    monkeypatch.setattr(
        backend_common,
        "kill_process_group",
        lambda proc, *, process_group_id: terminated.append((proc, process_group_id)),
    )
    proc = _FakeGrpoProcess(["step: 1\n"], wait_code=0, stale_return_code=0, poll_code=0)
    stream = rl_runner._GrpoSubprocessStream(proc)

    consumed = []
    for line in stream:
        # one step several graces long, taken while suspended inside the generator. must also clear
        # the watchdog's own 0.5s poll interval, or the arming it is meant to survive never happens.
        time.sleep(1.5)
        consumed.append(line)

    assert consumed == ["step: 1\n"]
    assert terminated == [], (
        "the group was torn down while the consumer was working through a single long step"
    )
    assert not stream._orphaned_pipe, (
        "the watchdog recorded a leaked pipe for a consumer that was working the whole time"
    )
    assert stream.wait_and_classify() == 0, (
        "a successful grpo attempt was failed because one step outlasted the grace"
    )


def test_grpo_subprocess_stream_does_not_classify_a_zero_exit():
    lines = ["cudaErrorDevicesUnavailable\n"]
    proc = _FakeGrpoProcess(lines, wait_code=0, stale_return_code=17)
    stream = rl_runner._GrpoSubprocessStream(proc)

    assert list(stream) == lines
    assert stream.wait_and_classify() == 0
    assert proc.wait_calls == 1


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="requires linux process groups")
def test_grpo_classified_exit_drains_group_after_leader_is_reaped(tmp_path, monkeypatch):
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    if not backend_common.adopt_orphaned_descendants():
        pytest.skip("child subreaper unavailable")
    monkeypatch.setattr(backend_common, "_TEARDOWN_GRACE_S", 0.5)
    marker = tmp_path / "grpo-classified-grandchild.pid"
    grandchild = (
        "import signal,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(300)\n"
    )
    leader = (
        "import pathlib,subprocess,sys\n"
        f"g = subprocess.Popen([sys.executable, '-c', {grandchild!r}], "
        "stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)\n"
        "assert g.stdout.readline().strip() == 'ready'\n"
        "g.stdout.close()\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(g.pid))\n"
        "print('cudaErrorDevicesUnavailable', flush=True)\n"
        "raise SystemExit(1)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", leader],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    grandchild_pid = None
    try:
        stream = rl_runner._GrpoSubprocessStream(proc)
        assert "cudaErrorDevicesUnavailable\n" in list(stream)
        with pytest.raises(RetriableInfraError, match="cudaErrorDevicesUnavailable"):
            stream.wait_and_classify()

        assert proc.poll() == 1, "test did not exercise classification after leader reaping"
        assert marker.exists(), "leader exited before recording its surviving grandchild"
        grandchild_pid = int(marker.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and os.path.exists(f"/proc/{grandchild_pid}"):
            time.sleep(0.05)
        assert not os.path.exists(f"/proc/{grandchild_pid}"), (
            f"classified grpo exit left grandchild {grandchild_pid} alive or unreaped"
        )
    finally:
        if grandchild_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(grandchild_pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(grandchild_pid, 0)
        if proc.poll() is None:  # pragma: no cover - only on an unexpected failure
            backend_common.kill_process_group(proc)


def test_run_rl_train_reaches_the_executable_grpo_subprocess_stream():
    source = inspect.getsource(rl_runner._execute_rl_child)

    assert "child_stream = _GrpoSubprocessStream(" in source
    assert "silence_watchdog=silence_watchdog" in source
    assert "for line in child_stream" in source
    assert "return child_stream.wait_and_classify()" in source


# ------------------------------- data conversion -------------------------------
def test_build_verl_dataset_rows_schema_and_index():
    rows = rl_verl.build_verl_dataset_rows(
        [[{"role": "user", "content": "q0"}], [{"role": "user", "content": "q1"}]],
        [5, 9],
        ["42", ""],
    )
    assert rows[0]["prompt"] == [{"role": "user", "content": "q0"}]
    assert rows[0]["reward_model"] == {"style": "rule", "ground_truth": "42"}
    # the flash rollout index must round-trip through verl's extra_info so the reward maps back.
    assert [r["extra_info"]["index"] for r in rows] == [5, 9]
    assert all(r["data_source"] == rl_verl.DATA_SOURCE for r in rows)


def test_build_verl_dataset_rows_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        rl_verl.build_verl_dataset_rows([[{"role": "user", "content": "q"}]], [1, 2], ["a", "b"])


# ------------------------------- multimodal parquet contract -------------------------------
# verl's RLHFDataset._build_messages re-splits each prompt on "<image>" and then asserts the
# placeholder count equals len(images). the two halves of that invariant are produced in different
# places here (message flattening vs the images column), so they are exactly the kind of pair that
# drifts silently: a row can look well-formed on both sides and still raise inside verl.


def _image_placeholder_count(row) -> int:
    return sum(str(m["content"]).count("<image>") for m in row["prompt"])


def _module_scope_imports(source: str) -> set[str]:
    """top-level packages imported at MODULE scope by ``source`` (nested defs excluded)."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            # the unwrapped tf32 fragment is a bare try/except at module scope; look inside it.
            for inner in node.body:
                if isinstance(inner, ast.Import):
                    names.update(alias.name.split(".")[0] for alias in inner.names)
                elif isinstance(inner, ast.ImportFrom) and inner.module and inner.level == 0:
                    names.add(inner.module.split(".")[0])
    return names


def test_no_grpo_plugin_module_imports_cuda_sensitive_packages_at_module_scope():
    """the regression barrier for the multi-gpu rank-to-device collapse.

    verl loads its external plugin before ray narrows each actor's CUDA visibility. Importing torch,
    verl, or vllm from plugin module scope can initialize cuda against the full device list and leave
    every rank on device zero. Installers may import those packages only after the deferred target has
    completed its real import under the actor's pinned environment.
    """
    sources = {
        "runtime": inspect.getsource(child_runtime),
        "patches": inspect.getsource(verl_patches),
        "plugin": inspect.getsource(grpo_plugin),
    }
    offenders = {
        name: sorted(_module_scope_imports(source) & {"torch", "verl", "vllm"})
        for name, source in sources.items()
    }
    assert {name: imports for name, imports in offenders.items() if imports} == {}


def _clear_deferred_target(target: str) -> None:
    sys.modules.pop(target, None)
    finder = child_runtime._DEFERRED_FINDER
    finder.pending.pop(target, None)
    finder.active_targets.discard(target)
    # the shared finder stays installed while any other target is still armed.
    if not finder.pending:
        finder.uninstall()


def test_a_deferred_fragment_records_its_marker_only_once_the_patch_applies(tmp_path, monkeypatch):
    """the marker must mean patched, not merely queued."""
    target = "flash_marker_probe"
    markers = tmp_path / "applied_shims.txt"
    (tmp_path / f"{target}.py").write_text("PATCHED = False\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_deferred_target(target)

    try:
        child_runtime._arm_deferred(
            name="probe-fragment",
            marker_file=str(markers),
            target=target,
            patch=lambda module: setattr(module, "PATCHED", True),
            required=True,
        )
        assert backend_common.read_applied_shim_markers(str(markers)) == set()
        with pytest.raises(RuntimeError, match="never proved these required runtime patches"):
            backend_common.verify_applied_shim_markers(str(markers), ["probe-fragment"])

        imported = importlib.import_module(target)

        assert imported.PATCHED is True
        assert backend_common.read_applied_shim_markers(str(markers)) == {"probe-fragment"}
        backend_common.verify_applied_shim_markers(str(markers), ["probe-fragment"])
        assert markers.read_text().count("probe-fragment") == 1
    finally:
        _clear_deferred_target(target)


def test_every_required_grpo_patch_is_registered_deferred(monkeypatch, tmp_path):
    """the external plugin may queue required patches but must not execute them while verl imports."""
    config = {
        "marker_file": str(tmp_path / "markers"),
        "dp_cards": 2,
        "reentrant_checkpointing": True,
        "multimodal": True,
        "entropy_quantile": 0.2,
        "per_turn_credit": True,
        "stop_sequences": ["</answer>"],
        "image_pad_token_id": 151655,
        "structured_outputs": {"json": {"type": "object"}},
        "save_at_steps": [3],
        "total_steps": 10,
        "kl_ref_adapter": True,
        "multi_turn": True,
        "gdn_model_type": None,
        "wandb": False,
    }
    registered = []

    def register(name, marker_file, target, installer, *args, **kwargs):
        registered.append((name, marker_file, target, installer, args, kwargs))

    monkeypatch.setattr(child_runtime, "load_plugin_config_file", lambda _name: config)
    monkeypatch.setattr(child_runtime, "install_deferred_required", register)
    monkeypatch.setattr(child_runtime, "install_deferred_lora_rollout_guard", lambda _path: None)

    grpo_plugin.install()

    assert [entry[0] for entry in registered] == [
        "rank-device-assert",
        "nonempty-response-mask",
        "exact-rollout-identity",
        "reentrant-checkpointing",
        "entropy-quantile",
        "per-turn-credit",
        "stop-sequences",
        "image-pad-ban",
        "structured-outputs",
        "exact-save-steps",
        "kl-ref-adapter",
        "multi-turn-loop",
    ]
    assert registered[0][2] == "verl.single_controller.base.worker"
    assert registered[1][2] == "verl.trainer.ppo.rollout_corr_helper"
    assert registered[1][3] is verl_patches.install_nonempty_response_mask
    assert [entry[2] for entry in registered[6:9]] == [
        "verl.experimental.agent_loop.agent_loop",
        "verl.experimental.agent_loop.agent_loop",
        "verl.experimental.agent_loop.agent_loop",
    ]
    assert not Path(config["marker_file"]).exists()


def test_parent_and_child_share_required_patch_registry_and_preserve_zero_entropy(
    monkeypatch, tmp_path
):
    config = {
        "marker_file": str(tmp_path / "markers"),
        "dp_cards": 2,
        "reentrant_checkpointing": False,
        "multimodal": False,
        "entropy_quantile": 0.0,
        "per_turn_credit": False,
        "stop_sequences": [],
        "image_pad_token_id": None,
        "structured_outputs": None,
        "save_at_steps": [],
        "total_steps": 10,
        "kl_ref_adapter": False,
        "multi_turn": True,
        "gdn_model_type": "qwen3_5",
        "wandb": False,
    }
    installed = []

    def deferred(name, _marker_file, _target, _installer, *args, **_kwargs):
        installed.append(name)
        if name == "entropy-quantile":
            assert args == (0.0,)

    monkeypatch.setattr(child_runtime, "load_plugin_config_file", lambda _name: config)
    monkeypatch.setattr(child_runtime, "install_deferred_required", deferred)
    monkeypatch.setattr(
        child_runtime,
        "install_deferred_lora_rollout_guard",
        lambda _marker_file: installed.append(child_runtime.LORA_ROLLOUT_GUARD_SHIM),
    )
    monkeypatch.setattr(
        child_runtime,
        "install_deferred_gdn",
        lambda model_type, _marker_file: installed.append(
            "gdn-varlen" if model_type == "qwen3_5" else "wrong-gdn"
        ),
    )

    grpo_plugin.install()

    expected = grpo_plugin.required_patch_names(config)
    assert installed == expected
    assert expected == [
        "rank-device-assert",
        "nonempty-response-mask",
        "exact-rollout-identity",
        "entropy-quantile",
        "multi-turn-loop",
        "lora-rollout-guard",
        "gdn-varlen",
    ]


def _truncated_thinking_response_mask(monkeypatch):
    monkeypatch.setenv("FLASH_VERL_MULTITURN_URL", "http://bridge.invalid")
    monkeypatch.setenv("FLASH_VERL_MAX_TURNS", "1")
    monkeypatch.setenv("FLASH_VERL_MAX_MODEL_LEN", "512")
    monkeypatch.setenv("FLASH_VERL_MAX_COMPLETION_TOKENS", "32")
    monkeypatch.setenv("FLASH_VERL_THINKING", "1")
    settings = grpo_multiturn._EpisodeSettings()
    episode = grpo_multiturn._EpisodeTranscript(
        [7] * 53,
        response_capacity=459,
        max_model_len=512,
    )
    max_tokens = episode.turn_budget(settings.max_completion_tokens)
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=False: "x" * len(ids))
    turn = grpo_multiturn.prepare_assistant_turn(
        tokenizer,
        list(range(32)),
        stop_reason="completed",
        max_tokens=max_tokens,
        eos_token_ids=frozenset(),
        stop_sequences=(),
    )
    episode.append_model_turn(turn, [0.0] * 32)
    padded_mask = episode.response_mask + [0] * (
        episode.response_capacity - len(episode.response_mask)
    )
    return settings, episode, turn, np.asarray(padded_mask, dtype=np.int64)


def test_thinking_rollout_at_the_authored_32_token_cap_is_fully_masked(monkeypatch):
    settings, episode, turn, response_mask = _truncated_thinking_response_mask(monkeypatch)

    assert settings.thinking is True
    assert settings.max_completion_tokens == 32
    assert episode.response_capacity == 459
    assert turn["max_tokens"] == 32
    assert turn["truncated"] is True
    assert len(episode.response_ids) == 32
    assert len(response_mask) == 459
    assert not response_mask.any(), "truncated response tokens became trainable"


def _install_rollout_corr_test_module(monkeypatch, original):
    helper = types.ModuleType("verl.trainer.ppo.rollout_corr_helper")
    helper.compute_rollout_correction_and_add_to_batch = original
    ppo = types.ModuleType("verl.trainer.ppo")
    ppo.rollout_corr_helper = helper
    trainer = types.ModuleType("verl.trainer")
    trainer.ppo = ppo
    verl = types.ModuleType("verl")
    verl.trainer = trainer
    for name, module in {
        "verl": verl,
        "verl.trainer": trainer,
        "verl.trainer.ppo": ppo,
        "verl.trainer.ppo.rollout_corr_helper": helper,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return helper


def test_nonempty_response_mask_guard_blocks_the_all_empty_batch_before_verl(monkeypatch):
    calls = []

    def poison_original(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("poison original was called")

    helper = _install_rollout_corr_test_module(monkeypatch, poison_original)
    _, _, _, empty = _truncated_thinking_response_mask(monkeypatch)
    batch = SimpleNamespace(batch={"response_mask": np.stack([empty, empty])})
    verl_patches.install_nonempty_response_mask()

    with pytest.raises(RuntimeError) as exc_info:
        helper.compute_rollout_correction_and_add_to_batch(batch, {"rollout_is": "token"})

    assert str(exc_info.value) == (
        "flash: no trainable response tokens remain in this batch: every rollout was truncated or "
        "unusable, so no optimizer update will run. increase train.max_completion_tokens or disable "
        "thinking. child context: FLASH_VERL_THINKING=1, "
        "FLASH_VERL_MAX_COMPLETION_TOKENS=32."
    )
    assert calls == []


def test_nonempty_response_mask_guard_delegates_a_mixed_batch_exactly_once(monkeypatch):
    calls = []
    sentinel = object()

    def original(batch, rollout_corr_config):
        calls.append((batch, rollout_corr_config))
        return sentinel

    helper = _install_rollout_corr_test_module(monkeypatch, original)
    _, _, _, empty = _truncated_thinking_response_mask(monkeypatch)
    mixed = np.stack([empty, empty.copy()])
    mixed[1, 31] = 1
    batch = SimpleNamespace(batch={"response_mask": mixed})
    config = {"rollout_is": "token"}
    verl_patches.install_nonempty_response_mask()

    result = helper.compute_rollout_correction_and_add_to_batch(batch, config)

    assert result is sentinel
    assert calls == [(batch, config)]


def test_nonempty_response_mask_installer_is_idempotent(monkeypatch):
    helper = _install_rollout_corr_test_module(monkeypatch, lambda batch, config: (batch, config))

    verl_patches.install_nonempty_response_mask()
    installed = helper.compute_rollout_correction_and_add_to_batch
    verl_patches.install_nonempty_response_mask()

    assert helper.compute_rollout_correction_and_add_to_batch is installed
    assert installed._flash_nonempty_response_mask is True


def test_nonempty_response_mask_patch_is_required_and_marker_wired(monkeypatch, tmp_path):
    config = {
        "marker_file": str(tmp_path / "markers"),
        "dp_cards": 1,
        "total_steps": 1,
    }
    registered = []

    def register(name, marker_file, target, installer, *args, **kwargs):
        registered.append((name, marker_file, target, installer, args, kwargs))

    monkeypatch.setattr(child_runtime, "load_plugin_config_file", lambda _name: config)
    monkeypatch.setattr(child_runtime, "install_deferred_required", register)
    monkeypatch.setattr(child_runtime, "install_deferred_lora_rollout_guard", lambda _path: None)

    grpo_plugin.install()

    assert registered == [
        (
            "nonempty-response-mask",
            config["marker_file"],
            "verl.trainer.ppo.rollout_corr_helper",
            verl_patches.install_nonempty_response_mask,
            (),
            {},
        ),
        (
            "exact-rollout-identity",
            config["marker_file"],
            "verl.experimental.agent_loop.agent_loop",
            verl_patches.install_exact_rollout_identity,
            (),
            {},
        ),
    ]
    assert grpo_plugin.required_patch_names(config) == [
        "nonempty-response-mask",
        "exact-rollout-identity",
        child_runtime.LORA_ROLLOUT_GUARD_SHIM,
    ]
    assert not Path(config["marker_file"]).exists()


def _install_entropy_test_modules(monkeypatch, observed):
    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def bool(self):
            return FakeTensor(self.values.astype(bool))

        def float(self):
            return FakeTensor(self.values.astype(float))

        def reshape(self, *shape):
            return FakeTensor(self.values.reshape(*shape))

        def numel(self):
            return self.values.size

        def clone(self):
            return FakeTensor(self.values.copy())

        def tolist(self):
            return self.values.tolist()

        def __getitem__(self, index):
            if isinstance(index, FakeTensor):
                index = index.values
            return FakeTensor(self.values[index])

        def __mul__(self, other):
            values = other.values if isinstance(other, FakeTensor) else other
            return FakeTensor(self.values * values)

        def __ge__(self, other):
            return FakeTensor(self.values >= other)

        def __and__(self, other):
            return FakeTensor(self.values & other.values)

    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: False
    distributed.is_initialized = lambda: False
    distributed.get_world_size = lambda: 1
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.distributed = distributed
    torch.bool = bool
    torch.tensor = lambda values: FakeTensor(values)
    torch.zeros_like = lambda tensor, dtype=None: FakeTensor(
        np.zeros_like(tensor.values, dtype=bool)
    )
    torch.quantile = lambda tensor, threshold: float(np.quantile(tensor.values, threshold))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    losses = types.ModuleType("verl.workers.utils.losses")

    def get_policy_loss_fn(_loss_mode):
        def policy_loss(*_args, **kwargs):
            observed["policy_mask"] = kwargs["response_mask"].clone()
            return "policy"

        return policy_loss

    def ppo_loss(_config, _model_output, data, _dp_group=None):
        observed["other_term_mask"] = data["response_mask"].clone()
        return losses.get_policy_loss_fn("vanilla")(response_mask=data["response_mask"])

    losses.get_policy_loss_fn = get_policy_loss_fn
    losses.ppo_loss = ppo_loss
    padding = types.ModuleType("verl.workers.utils.padding")
    padding.no_padding_2_padding = lambda entropy, _data: entropy
    utils = types.ModuleType("verl.workers.utils")
    utils.losses = losses
    modules = {
        "verl": types.ModuleType("verl"),
        "verl.workers": types.ModuleType("verl.workers"),
        "verl.workers.utils": utils,
        "verl.workers.utils.losses": losses,
        "verl.workers.utils.padding": padding,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    return torch, losses


@pytest.mark.parametrize(
    ("authored_quantile", "expected_threshold"),
    [(0.2, 0.8), (0.0, 1.0)],
)
def test_entropy_installer_masks_only_policy_loss_and_preserves_authored_threshold(
    monkeypatch, authored_quantile, expected_threshold
):
    observed = {}
    torch, losses = _install_entropy_test_modules(monkeypatch, observed)
    real_quantile = torch.quantile
    thresholds = []

    def quantile(values, threshold):
        thresholds.append(float(threshold))
        return real_quantile(values, threshold)

    monkeypatch.setattr(torch, "quantile", quantile)
    verl_patches.install_entropy_quantile(authored_quantile)
    installed = losses.ppo_loss
    response_mask = torch.tensor([[True, True, True, True]])
    entropy = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    assert losses.ppo_loss(None, {"entropy": entropy}, {"response_mask": response_mask}) == "policy"
    assert thresholds == pytest.approx([expected_threshold])
    assert observed["policy_mask"].tolist() == [[False, False, False, True]]
    assert observed["other_term_mask"].tolist() == [[True, True, True, True]]
    assert response_mask.tolist() == [[True, True, True, True]]

    verl_patches.install_entropy_quantile(authored_quantile)
    assert losses.ppo_loss is installed


def test_exact_save_installer_keeps_authored_and_final_steps_and_is_idempotent(
    monkeypatch,
):
    calls = []

    class RayPPOTrainer:
        def __init__(self):
            self.global_steps = 0

        def _save_checkpoint(self):
            calls.append(self.global_steps)
            return self.global_steps

    ray_trainer = types.ModuleType("verl.trainer.ppo.ray_trainer")
    ray_trainer.RayPPOTrainer = RayPPOTrainer
    ppo = types.ModuleType("verl.trainer.ppo")
    ppo.ray_trainer = ray_trainer
    modules = {
        "verl": types.ModuleType("verl"),
        "verl.trainer": types.ModuleType("verl.trainer"),
        "verl.trainer.ppo": ppo,
        "verl.trainer.ppo.ray_trainer": ray_trainer,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    config = {"save_at_steps": (), "total_steps": 20}
    assert "exact-save-steps" not in grpo_plugin.required_patch_names(config)
    verl_patches.install_exact_save_steps((7, 13), 20)
    installed = RayPPOTrainer._save_checkpoint
    trainer = RayPPOTrainer()
    for step in (1, 7, 10, 13, 20):
        trainer.global_steps = step
        trainer._save_checkpoint()

    assert calls == [7, 13, 20]
    verl_patches.install_exact_save_steps((7, 13), 20)
    assert RayPPOTrainer._save_checkpoint is installed


def test_the_deferred_registry_runs_patches_at_the_targets_real_import(tmp_path, monkeypatch):
    """callbacks sharing one target run in registration order after the module finishes importing."""
    target = "flash_defer_probe"
    marker_file = str(tmp_path / "markers")
    (tmp_path / f"{target}.py").write_text("VALUE = 'imported'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    fired = []
    _clear_deferred_target(target)

    try:
        for name in ("first", "second"):
            child_runtime._arm_deferred(
                name=name,
                marker_file=marker_file,
                target=target,
                patch=lambda module, patch_name=name: fired.append((patch_name, module.VALUE)),
                required=True,
            )
        assert fired == []

        importlib.import_module(target)

        assert fired == [("first", "imported"), ("second", "imported")]
        assert Path(marker_file).read_text().splitlines() == ["first", "second"]
        assert target not in child_runtime._DEFERRED_FINDER.pending
        assert child_runtime._DEFERRED_FINDER not in sys.meta_path
    finally:
        _clear_deferred_target(target)


def test_a_deferred_patch_registered_after_its_target_is_imported_applies_immediately(
    monkeypatch, tmp_path
):
    # ordering must not decide whether a patch happens. an imported target cannot be intercepted.
    target = "flash_defer_present"
    module = types.ModuleType(target)
    fired = []
    monkeypatch.setitem(sys.modules, target, module)

    child_runtime._arm_deferred(
        name="present",
        marker_file=str(tmp_path / "markers"),
        target=target,
        patch=lambda imported: fired.append(imported),
        required=True,
    )

    assert fired == [module]


def test_a_deferred_body_that_raises_hard_exits_and_cannot_be_retried_around(tmp_path):
    """a required deferred installer kills the child before an importer can retry unpatched."""
    target = "flash_defer_boom"
    (tmp_path / f"{target}.py").write_text("VALUE = 1\n")
    rl_multi.copy_grpo_child_modules(str(tmp_path))
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import flash_verl_runtime as runtime\n"
        "def boom():\n"
        "    raise RuntimeError('fragment could not apply')\n"
        f"runtime.install_deferred_required('boom-fragment', {str(tmp_path / 'markers')!r}, "
        f"{target!r}, boom)\n"
        "try:\n"
        f"    import {target}\n"
        "except BaseException:\n"
        "    pass\n"
        f"import {target} as retried\n"
        "print('REACHED_RETRY', retried.VALUE)\n"
    )

    done = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=120)

    assert done.returncode == backend_common.SHIM_FRAGMENT_FAILED_EXIT_CODE, done.stdout
    assert "REACHED_RETRY" not in done.stdout
    assert "boom-fragment" in done.stderr


def test_the_rank_device_assert_is_inert_on_a_single_card_run(monkeypatch):
    # one rank cannot collide with itself, so the installer must return before importing torch/verl.
    imported = []
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "verl"}:
            imported.append(name)
            raise AssertionError(f"single-card rank check imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    verl_patches.install_rank_device_assert(0)
    verl_patches.install_rank_device_assert(1)
    assert imported == []


def test_the_rank_device_assert_compares_uuids_after_verls_own_init():
    """the check runs after verl pins the worker and compares physical uuids, not ordinals."""
    source = inspect.getsource(verl_patches.install_rank_device_assert)
    installer = ast.parse(source).body[0]
    wrapper = next(
        node
        for node in ast.walk(installer)
        if isinstance(node, ast.FunctionDef) and node.name == "init"
    )

    def statement_calling(name: str) -> int:
        for index, node in enumerate(wrapper.body):
            if any(
                isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == name
                for inner in ast.walk(node)
            ):
                return index
        raise AssertionError(f"{name} is not called in rank-device init wrapper")

    assert statement_calling("original_init") < statement_calling("check")
    assert "get_device_properties" in source
    assert "uuid" in source
    assert "raise RuntimeError(" in source


def _run_rank_device_check(tmp_path, monkeypatch, bindings, *, env_rank=None):
    """run the installer once per rank against fake torch and verl modules sharing one claims file."""
    claims = tmp_path / "rank_device_claims.txt"
    monkeypatch.setenv("FLASH_RANK_DEVICE_CLAIMS", str(claims))
    results = []
    for rank, ordinal, uuid in bindings:
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda ordinal=ordinal: ordinal,
            get_device_properties=lambda _o, uuid=uuid: SimpleNamespace(uuid=uuid),
        )

        class _Worker:
            def __init__(self, rank=rank):
                # verl's Worker.__init__ stores the rank on the instance before it returns, so the
                # wrapper that runs after it can read the same value verl will use downstream.
                self._rank = rank

        worker_module = types.ModuleType("verl.single_controller.base.worker")
        worker_module.Worker = _Worker
        base = types.ModuleType("verl.single_controller.base")
        base.worker = worker_module
        for name, module in {
            "torch": torch_stub,
            "verl": types.ModuleType("verl"),
            "verl.single_controller": types.ModuleType("verl.single_controller"),
            "verl.single_controller.base": base,
            "verl.single_controller.base.worker": worker_module,
        }.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setenv("RANK", str(rank if env_rank is None else env_rank))
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(ordinal))
        verl_patches.install_rank_device_assert(len(bindings))
        try:
            _Worker()
            results.append(None)
        except RuntimeError as exc:
            results.append(str(exc))
    return results


def test_the_rank_device_assert_passes_when_every_rank_opened_its_own_card(tmp_path, monkeypatch):
    # the healthy multi-card shape: distinct uuids, so nothing is refused. both ranks report
    # ordinal 0 because ray gave each actor its own CUDA_VISIBLE_DEVICES -- which is exactly why
    # ordinals cannot be the thing compared.
    errors = _run_rank_device_check(tmp_path, monkeypatch, [(0, 0, "GPU-aaaa"), (1, 0, "GPU-bbbb")])
    assert errors == [None, None]


def test_the_rank_device_assert_refuses_when_two_ranks_land_on_one_card(tmp_path, monkeypatch):
    # the failure that killed 7/7 multi-gpu runs. the second rank finds rank 0's uuid already
    # claimed and refuses, minutes before nccl would abort with a pci id and no rank mapping.
    errors = _run_rank_device_check(tmp_path, monkeypatch, [(0, 0, "GPU-same"), (1, 0, "GPU-same")])
    assert errors[0] is None
    assert errors[1] is not None
    # the message has to carry the mapping: naming the colliding ranks is the entire diagnostic
    # value over the nccl abort it front-runs.
    assert "GPU-same" in errors[1]
    assert "[0, 1]" in errors[1]


def test_the_rank_device_assert_reads_the_rank_verl_resolved_not_the_environment(
    tmp_path, monkeypatch
):
    # the claims file is keyed by rank, so a rank that every actor agrees on collapses the set to
    # one entry and the collision stops being visible. pin the source: RANK is held at 0 for both
    # actors while the workers carry 0 and 1, and the check must still refuse.
    errors = _run_rank_device_check(
        tmp_path,
        monkeypatch,
        [(0, 0, "GPU-same"), (1, 0, "GPU-same")],
        env_rank=0,
    )
    assert errors[0] is None
    assert errors[1] is not None
    assert "[0, 1]" in errors[1]


def test_the_rank_device_assert_has_no_rank_zero_default_to_fall_back_on():
    # a getenv("RANK", "0") would make an unset RANK report every actor as rank 0, which is the one
    # value that turns this check into a no-op. verl reads os.environ["RANK"] unconditionally in the
    # same __init__, so an absent RANK is already fatal there -- this keeps it fatal here too.
    source = inspect.getsource(verl_patches.install_rank_device_assert)
    assert 'environ.get("RANK"' not in source
    assert 'environ["RANK"]' in source


def test_multimodal_rows_match_verl_placeholder_assertion():
    rows = rl_verl.build_verl_dataset_rows(
        [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "q0"}, {"type": "image"}],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "q1"},
                    ],
                }
            ],
        ],
        [0, 1],
        ["a", "b"],
        image_uris=[["file:///w/0-0.png"], ["file:///w/1-0.png", "file:///w/1-1.png"]],
    )
    # this is verl's own assertion, restated: image_offset == len(images) or the dataset raises.
    for row in rows:
        assert _image_placeholder_count(row) == len(row["images"])
    assert rows[1]["images"] == [
        {"image": "file:///w/1-0.png"},
        {"image": "file:///w/1-1.png"},
    ]
    # content is flattened to a string; verl re-expands it. leaving content blocks in place would
    # make the re-split find zero placeholders while images is non-empty.
    assert all(isinstance(m["content"], str) for row in rows for m in row["prompt"])


def test_text_rows_carry_no_images_column():
    # the control: without image_uris the rows must stay exactly as before. an unconditional images
    # column would make every text job take verl's multimodal dataset path.
    rows = rl_verl.build_verl_dataset_rows([[{"role": "user", "content": "q"}]], [0], ["a"])
    assert "images" not in rows[0]


def test_multimodal_rows_reject_a_mismatched_uri_list():
    with pytest.raises(ValueError, match="image_uris length mismatch"):
        rl_verl.build_verl_dataset_rows(
            [[{"role": "user", "content": "q"}]], [0], ["a"], image_uris=[[], []]
        )


def test_multimodal_rows_reject_a_literal_image_placeholder_in_text():
    # verl splits prompt text on "<image>" and re-expands each hit into a real image block, so a
    # prompt that merely TALKS about the token consumes an image the row does not have. verl would
    # abort dataset loading with a bare offset assertion; catching it here names the example.
    with pytest.raises(ValueError, match="reserved by verl"):
        rl_verl.build_verl_dataset_rows(
            [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "the <image> token marks an image"},
                            {"type": "image"},
                        ],
                    }
                ]
            ],
            [7],
            ["a"],
            image_uris=[["file:///w/0-0.png"]],
        )


def test_text_only_row_of_a_mixed_job_rejects_a_literal_placeholder():
    # the row itself has no images, but verl's split is driven by the row's OWN modality columns and
    # a mixed job writes an images column on every row -- so a text row with a literal "<image>"
    # asserts against its empty list. this is the case a per-job (rather than per-row) check misses.
    with pytest.raises(ValueError, match="reserved by verl"):
        rl_verl.build_verl_dataset_rows(
            [
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "describe <image> please"}],
                    }
                ],
                [{"role": "user", "content": [{"type": "image"}]}],
            ],
            [0, 1],
            ["a", "b"],
            image_uris=[[], ["file:///w/1-0.png"]],
        )


@pytest.mark.parametrize("reserved", ["<video>", "<audio>"])
def test_multimodal_rows_reject_other_reserved_media_placeholders(reserved):
    # _build_messages splits on all three markers. flash never writes a videos/audios column, so a
    # single literal occurrence asserts against an empty list -- the count check on <image> alone
    # would pass this row straight through.
    with pytest.raises(ValueError, match="reserves as a media placeholder"):
        rl_verl.build_verl_dataset_rows(
            [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"a {reserved} marker"},
                            {"type": "image"},
                        ],
                    }
                ]
            ],
            [3],
            ["a"],
            image_uris=[["file:///w/0-0.png"]],
        )


def test_text_job_does_not_police_reserved_placeholders():
    # the control: without an images column verl never splits, so "<image>" is ordinary text and
    # rejecting it would break text jobs that legitimately discuss the token.
    rows = rl_verl.build_verl_dataset_rows(
        [[{"role": "user", "content": "what does <image> mean?"}]], [0], ["a"]
    )
    assert rows[0]["prompt"] == [{"role": "user", "content": "what does <image> mean?"}]


def test_text_block_grpo_rows_flatten_content_without_python_repr():
    rows = rl_verl.build_verl_dataset_rows(
        [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ],
                }
            ]
        ],
        [0],
        ["a"],
    )

    assert rows[0]["prompt"] == [{"role": "user", "content": "hello world"}]


def test_mixed_job_parquet_round_trips_the_images_column(tmp_path):
    # Dataset.from_list infers ONE type per column across all rows. in a mixed job the text rows
    # have an empty images list, and inference on an all-empty-or-partly-empty column can land on a
    # type verl cannot read back as a struct. this asserts the round trip, not the schema object,
    # because the schema is only interesting insofar as the read-back works.
    rows = rl_verl.build_verl_dataset_rows(
        [
            [{"role": "user", "content": [{"type": "text", "text": "text only"}]}],
            [{"role": "user", "content": [{"type": "image"}]}],
        ],
        [0, 1],
        ["a", "b"],
        image_uris=[[], ["file:///w/1-0.png"]],
    )
    path = str(tmp_path / "train.parquet")
    rl_verl.write_verl_grpo_parquet(rows, path)

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert table.num_rows == 2
    images = table.column("images").to_pylist()
    assert images == [[], [{"image": "file:///w/1-0.png"}]]
    # the empty row must still be a LIST OF STRUCTS, not a null column: verl indexes row["images"]
    # per element, so a null-typed column fails on read rather than on the empty row. asserted
    # structurally because arrow spells the list field "item" or "element" by version.
    import pyarrow as pa

    images_type = table.schema.field("images").type
    assert pa.types.is_list(images_type)
    assert [f.name for f in images_type.value_type] == ["image"]
    assert pa.types.is_string(images_type.value_type.field("image").type)
    assert table.column("extra_info").to_pylist()[1]["index"] == 1
    assert table.column("prompt").to_pylist() == [row["prompt"] for row in rows]


def test_text_only_parquet_omits_multimodal_and_reasoning_fields_when_unauthored(tmp_path):
    rows = rl_verl.build_verl_dataset_rows([[{"role": "user", "content": "q"}]], [0], ["a"])
    path = str(tmp_path / "train.parquet")
    rl_verl.write_verl_grpo_parquet(rows, path)

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert "images" not in table.schema.names
    prompt_type = table.schema.field("prompt").type.value_type
    assert "reasoning_content" not in [field.name for field in prompt_type]


def test_text_only_parquet_preserves_reasoning_first_authored_on_a_later_row(tmp_path):
    rows = rl_verl.build_verl_dataset_rows(
        [
            [{"role": "user", "content": "first"}],
            [
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "answer", "reasoning_content": "old"},
            ],
        ],
        [0, 1],
        ["a", "b"],
    )
    path = str(tmp_path / "reasoning.parquet")

    rl_verl.write_verl_grpo_parquet(rows, path)

    datasets = pytest.importorskip("datasets")
    restored = datasets.Dataset.from_parquet(path)
    assert restored[1]["prompt"][1]["reasoning_content"] == "old"
    assert "reasoning_content" in restored.features["prompt"].feature


def test_grpo_reasoning_only_assistant_round_trips_with_empty_content(tmp_path):
    rows = rl_verl.build_verl_dataset_rows(
        [[{"role": "assistant", "content": None, "reasoning_content": "working"}]],
        [0],
        ["a"],
    )
    path = str(tmp_path / "reasoning-only.parquet")

    rl_verl.write_verl_grpo_parquet(rows, path)

    datasets = pytest.importorskip("datasets")
    restored = datasets.Dataset.from_parquet(path)
    assert restored[0]["prompt"] == [
        {"role": "assistant", "content": "", "reasoning_content": "working"}
    ]


def test_grpo_rows_reject_non_string_authored_reasoning():
    with pytest.raises(ValueError, match="reasoning_content must be text"):
        rl_verl.build_verl_dataset_rows(
            [[{"role": "assistant", "content": "answer", "reasoning_content": ["old"]}]],
            [0],
            ["a"],
        )


# ------------------------------- override generation -------------------------------
def _overrides_cfg(**over):
    cfg = {
        # required: run_rl_train always resolves this from the capability probe.
        "fused_ce_backend": "torch",
        "train_files": "/w/train.parquet",
        "val_files": "/w/val.parquet",
        "model_path": "Qwen/Qwen3-4B",
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": "all-linear",
        "exclude_modules": None,
        "fsdp_generation": 2,
        "lr": 1e-5,
        "group_size": 8,
        "prompts_per_step": 16,
        "max_prompt_len": 2048,
        "max_model_len": 2368,
        "max_token_len_per_gpu": 2368,
        # single-turn: the response tensor holds one completion, so it is max_completion wide.
        "max_completion": 320,
        "max_response_len": 320,
        "multi_turn": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "kl_coef": 0.0,
        "entropy_quantile": None,
        "stop_sequences": (),
        "structured_outputs": None,
        "thinking": False,
        "loss_agg_mode": "seq-mean-token-sum-norm",
        "seed": 42,
        "ppo_epochs": 1,
        "steps": 60,
        # an arbitrary passthrough value for the override-rendering tests below, NOT the budget the
        # worker launches -- that is sized per run by resolve_gpu_mem_util.
        "gpu_mem_util": 0.5,
        "n_gpus": 1,
        "loggers": ["console"],
        "fp8_kv": False,
        "enforce_eager": False,
        "warmstart_adapter": "",
        "reward_path": "/w/reward.py",
        "reward_name": "compute_score",
        "mask_truncated_completions": True,
        "total_epochs": 1,
        "save_freq": 20,
        "ckpt_to_keep": 1,
        "local_dir": "/w/ckpt",
        "project_name": "flash",
        "experiment_name": "flash-rl-run123",
    }
    cfg.update(over)
    return cfg


def test_build_verl_overrides_enables_layered_summon_for_fused_expert_targets():
    # end-to-end through the real builder: a fused-expert model must reach verl with the layered
    # weight-sync gather, and must keep load_format=safetensors, which verl requires as
    # base_sync_done before it will allow the layered walk (fsdp_utils.py:684 raises otherwise).
    o = rl_verl.build_verl_overrides(
        _overrides_cfg(target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])
    )
    assert "actor_rollout_ref.rollout.layered_summon=true" in o
    assert "actor_rollout_ref.rollout.load_format=safetensors" in o


def test_build_verl_overrides_omits_layered_summon_for_dense_models():
    o = rl_verl.build_verl_overrides(_overrides_cfg(target_parameters=None))
    assert not [x for x in o if "layered_summon" in x]


@pytest.mark.parametrize(
    "target_parameters",
    [None, ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]],
)
def test_build_verl_overrides_puts_every_actor_on_fsdp2(target_parameters):
    o = rl_verl.build_verl_overrides(_overrides_cfg(target_parameters=target_parameters))
    assert "actor_rollout_ref.actor.strategy=fsdp2" in o
    # exactly one bare writer of the key, so the value cannot be ambiguous at merge time
    strategy_overrides = [x for x in o if "actor_rollout_ref.actor.strategy=" in x]
    assert strategy_overrides == ["actor_rollout_ref.actor.strategy=fsdp2"]


@pytest.mark.parametrize("reshard_after_forward", [False, True])
def test_grpo_fsdp2_does_not_change_zero2_or_zero3(reshard_after_forward):
    for target_parameters in (None, ["mlp.experts.gate_up_proj"]):
        overrides = dict(
            value.split("=", 1)
            for value in rl_verl.build_verl_overrides(
                _overrides_cfg(
                    target_parameters=target_parameters,
                    reshard_after_forward=reshard_after_forward,
                )
            )
        )
        assert overrides["actor_rollout_ref.actor.strategy"] == "fsdp2"
        assert (
            overrides["actor_rollout_ref.actor.fsdp_config.reshard_after_forward"]
            == str(reshard_after_forward).lower()
        )


def test_build_verl_overrides_carries_dr_grpo_recipe():
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert "algorithm.adv_estimator=grpo" in o
    # dr-grpo: no std normalization + constant-length loss aggregation.
    assert "algorithm.norm_adv_by_std_in_grpo=False" in o
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm" in o
    assert "actor_rollout_ref.model.lora_rank=32" in o
    assert "actor_rollout_ref.rollout.n=8" in o
    assert "actor_rollout_ref.rollout.load_format=safetensors" in o
    assert "actor_rollout_ref.rollout.top_p=0.95" in o
    # constant lr, on-policy updates, gradient checkpointing, seed, max-steps horizon, save schedule.
    assert "actor_rollout_ref.actor.optim.weight_decay=0.0" in o
    assert "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0" in o
    assert "actor_rollout_ref.actor.ppo_epochs=1" in o
    assert "actor_rollout_ref.model.enable_gradient_checkpointing=True" in o
    assert "data.seed=42" in o
    # the rollout engine seed rides engine_kwargs, not `rollout.seed`, which verl 0.8.0's
    # RolloutConfig does not declare. see build_verl_overrides for the full reasoning.
    assert "++actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42" in o
    assert "actor_rollout_ref.rollout.seed=42" not in o
    assert "trainer.total_training_steps=60" in o
    assert "trainer.save_freq=20" in o
    assert "trainer.max_actor_ckpt_to_keep=1" in o
    assert "trainer.logger=[console]" in o
    assert "data.train_batch_size=16" in o
    assert "+data.apply_chat_template_kwargs.enable_thinking=false" in o
    assert "+data.apply_chat_template_kwargs.preserve_thinking=false" in o
    # truncated importance sampling: token-level, cap 2.0 (matches flash's tis recipe).
    assert "algorithm.rollout_correction.rollout_is=token" in o
    assert "algorithm.rollout_correction.rollout_is_threshold=2.0" in o
    # ...and the rollout logprobs that make those two do anything. verl only applies the correction
    # when "rollout_log_probs" is on the batch, and only a sampler asked for logprobs writes it.
    # without this the two overrides above are inert. see build_verl_overrides.
    assert "actor_rollout_ref.rollout.calculate_log_probs=True" in o


def test_tis_overrides_are_not_inert_without_rollout_logprobs():
    """the tis knobs and their enabling flag must never be separated.

    this is the regression that shipped: `rollout_is=token` and the 2.0 threshold composed fine,
    cost nothing, and applied no correction, because verl gates on the logprobs key being present
    rather than on the tis config being set. asserting the pair together is what makes a future
    edit that drops calculate_log_probs fail here instead of silently in training.
    """
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    tis_configured = any(item.startswith("algorithm.rollout_correction.rollout_is=") for item in o)
    logprobs_enabled = "actor_rollout_ref.rollout.calculate_log_probs=True" in o
    assert tis_configured is logprobs_enabled, (
        "rollout correction is configured but rollout logprobs are not enabled (or vice versa); "
        "verl needs both or the correction silently does nothing"
    )


def test_build_verl_overrides_carries_fused_expert_target_parameters():
    o = rl_verl.build_verl_overrides(
        _overrides_cfg(
            target_parameters=[
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
            ]
        )
    )

    assert (
        "++actor_rollout_ref.model.target_parameters="
        "[mlp.experts.gate_up_proj,mlp.experts.down_proj]"
    ) in o


def test_build_verl_training_cfg_resolves_expert_targets_from_the_catalog_id():
    """The fused-expert lookup must use the catalog id, NOT the local snapshot path.

    The builder is handed a snapshot dir to load weights from, and handing that same string to
    ``lora_target_parameters`` (which matches an exact hf repo id) resolves to None: PEFT then
    adapts only ``all-linear`` and the routed-expert parameters are never trained, on a run that
    completes, checkpoints, and reports healthy metrics. The test above renders already-resolved
    targets, so it cannot see that; this one goes through the real builder with a path that looks
    like what ``_cached_model_path`` actually returns.
    """
    inp = {**_mem_util_inp(model_id="Qwen/Qwen3.6-35B-A3B", engine_len=8192)}
    inp.update(
        {
            "lora_rank": 32,
            "lora_alpha": 64,
            "lr": 1e-5,
            "prompts_per_step": 16,
            "mask_truncated_completions": True,
            "max_prompt_len": 1024,
            "max_completion": 1024,
            "max_response_len": 1024,
            "multi_turn": False,
            "temperature": 1.0,
            "top_p": 0.95,
            "kl_coef": 0.0,
            "entropy_quantile": None,
            "stop_sequences": (),
            "structured_outputs": None,
            "seed": 42,
            "ppo_epochs": 1,
            "steps": 60,
            "warmstart_adapter": "",
            "fsdp_generation": 2,
            "verl_total_epochs": 1,
            "save_freq": 20,
            "ckpt_to_keep": 1,
        }
    )
    snapshot = "/cache/models--Qwen--Qwen3.6-35B-A3B/snapshots/deadbeefcafe"

    cfg = rl_verl._build_verl_training_cfg(
        inp,
        train_files="/w/t.parquet",
        val_files="/w/v.parquet",
        model_path=snapshot,
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
        ce_backend="torch",
        reward_path="/w/r.py",
        local_dir="/w/ckpt",
        project_name="p",
        experiment_name="e",
    )

    # the weights still load from the snapshot dir ...
    assert cfg["model_path"] == snapshot
    # ... but the fused expert targets resolve from the catalog id.
    assert cfg["target_parameters"] == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    assert cfg["exclude_modules"] is None


def test_build_verl_overrides_does_not_emit_inert_drop_last_override():
    # this guards only against flash emitting a misleading no-op; it does not prove verl reads the key.
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert not any("drop_last" in override for override in o)


def test_build_verl_overrides_limits_grpo_rollouts_to_four_images():
    overrides = rl_verl.build_verl_overrides(_overrides_cfg())
    assert "++actor_rollout_ref.rollout.limit_images=4" in overrides


def test_build_verl_overrides_sizes_agent_loop_workers_to_the_rollout_batch():
    # verl chunks prompts_per_step * group_size across agent.num_workers and asserts exact
    # divisibility; its default of 8 aborts before the first step on e.g. 2 x 2 = 4.
    o = rl_verl.build_verl_overrides(_overrides_cfg(prompts_per_step=2, group_size=2))
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in o
    # the common case still gets the full worker pool.
    big = rl_verl.build_verl_overrides(_overrides_cfg(prompts_per_step=64, group_size=8))
    assert "actor_rollout_ref.rollout.agent.num_workers=8" in big


def test_build_verl_overrides_halves_agent_loop_workers_for_multimodal_runs():
    # every agent worker is a ray actor with its own processor copy. on an image run that is a
    # full image processor, and eight of them beside the vllm engine exhausted the worker
    # container: `libgomp: Thread creation failed`. the actor died mid-grading, so the rollouts
    # generated (reward_completions=8) but none were graded (reward_grading_depth=0) and the run
    # hung until the 1200s child-silence watchdog killed it.
    batch = {"prompts_per_step": 2, "group_size": 4}
    text = rl_verl.build_verl_overrides(_overrides_cfg(**batch))
    assert "actor_rollout_ref.rollout.agent.num_workers=8" in text
    image = rl_verl.build_verl_overrides(_overrides_cfg(**batch, multimodal=True))
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in image
    # the authored knobs are untouched: only scheduling parallelism narrows.
    assert "actor_rollout_ref.rollout.n=4" in image
    # and the cap still yields an exact divisor of the rollout batch, which verl asserts on.
    workers = int(next(o for o in image if "agent.num_workers=" in o).rsplit("=", 1)[1])
    assert (batch["prompts_per_step"] * batch["group_size"]) % workers == 0


@pytest.mark.parametrize(("count", "expected"), [(None, 1), (1, 1), (2, 2), (8, 8)])
def test_run_rl_train_sizes_the_run_from_the_spec_gpu_count(count, expected):
    # the wiring, not just the builder: a spec that rents N cards must configure verl for N.
    # gpu_count_of is the same reader the runpod rental path uses, so the rented shape and the
    # trained shape cannot drift apart.
    from flash.core.spec import GpuSpec, JobSpec, gpu_count_of

    project = "11111111-1111-4111-8111-111111111111"
    spec = (
        JobSpec(project=project)
        if count is None
        else JobSpec(project=project, gpu=GpuSpec(count=count))
    )
    assert gpu_count_of(spec) == expected


def test_build_verl_overrides_single_gpu_is_the_unchanged_default():
    o = rl_verl.build_verl_overrides(_overrides_cfg(n_gpus=1))
    assert "+ray_kwargs.ray_init.num_gpus=1" in o
    assert "trainer.n_gpus_per_node=1" in o
    assert "trainer.nnodes=1" in o
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in o
    assert "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1" in o


@pytest.mark.parametrize("n_gpus", [2, 4, 8])
def test_build_verl_overrides_shards_every_card_by_data(n_gpus):
    # ulysses is pinned OFF at every width, so verl's mesh leaves every rank a DATA-parallel rank.
    # sequence parallelism would slice the sequence at fixed offsets and hand each rank's slice
    # straight to the GatedDeltaNet linear-attention and conv layers, whose state runs ALONG the
    # sequence -- every rank but rank 0 would start its recurrence from zero state. the global batch
    # survives anyway because use_dynamic_bsz token-balances it across the dp ranks.
    o = rl_verl.build_verl_overrides(_overrides_cfg(n_gpus=n_gpus))
    assert f"+ray_kwargs.ray_init.num_gpus={n_gpus}" in o
    assert f"trainer.n_gpus_per_node={n_gpus}" in o
    assert "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1" in o
    # no card-count-dependent sp override may survive anywhere in the command.
    assert not [
        s
        for s in o
        if "ulysses_sequence_parallel_size" in s
        and not s.endswith("ulysses_sequence_parallel_size=1")
    ]
    # rollout TENSOR parallelism is a separate width and still tracks the cards: tp splits attention
    # heads, not the sequence, so it has no recurrent-state problem. this is also why the allocator's
    # head-divisibility cap outlives ulysses -- vllm raises at engine init when heads % tp != 0.
    assert f"actor_rollout_ref.rollout.tensor_model_parallel_size={n_gpus}" in o
    # one worker, many cards: nnodes stays 1 so verl's replica_rank offset (and the rollout seed)
    # is unaffected by the card count.
    assert "trainer.nnodes=1" in o
    # remove-padding is independent of sp and still required: the no_padding loss reads log_prob
    # values off the flattened batch.
    assert "actor_rollout_ref.model.use_remove_padding=True" in o


def test_grpo_launches_the_width_the_step_can_fill():
    """`n_gpus` handed to verl is the executed dp width, never the rented card count.

    Pinning ulysses off makes every rank a dp rank, so verl now chunks the step's sequences across
    them and asserts exact divisibility (`DataProto.chunk`, and `_balance_batch` with
    `equal_size=True`). At sp = card count the dp width was 1 and this could not fire; that is why
    this guard arrives with the pin rather than before it. Launching wider than the sequences divide
    aborts at step 0 on a box already rented.

    The width is resolved ONCE, in `_assemble_grpo_inputs`, because two phases consume it: the child
    config here and the resume probe in `_prepare_rl_files`, which discards a checkpoint whose shard
    count disagrees with the width it is handed. Deriving it separately let them drift -- resume
    compared the RENTED count while the child launched the clamped one, so a checkpoint written at the
    executed width read as the wrong topology and every retry restarted from step 0.

    Asserted on the source: both call sites write child files or download weights, so neither is
    drivable offline. The rule itself is covered by
    `test_rl_width_never_exceeds_the_sequences_one_step_holds` in the opd suite, and the allocator
    and quote agree through `executed_gpu_count`.
    """
    import inspect

    from flash.engine.worker.train.entry import rl_train_runner
    from flash.engine.worker.train.rl.launch import inputs as rl_inputs

    # one derivation, in the builder that assembles every other resolved grpo knob.
    resolver = inspect.getsource(rl_inputs._assemble_grpo_inputs)
    assert '"dp_cards": rl_data_parallel_cards(' in resolver
    assert 'int(schedule["prompts_per_step"]) * int(options["group_size"])' in resolver

    # and both consumers read that one value rather than recomputing it.
    child = inspect.getsource(rl_train._configure_rl_child)
    line = next(ln.strip() for ln in child.splitlines() if ln.strip().startswith("n_gpus="))
    assert line == 'n_gpus=int(inp["dp_cards"]),', line

    resume = inspect.getsource(rl_train_runner._prepare_rl_files)
    assert 'world_size=int(inp["dp_cards"])' in resume
    # the rented count must not reach either phase, or the drift returns.
    assert "gpu_count_of" not in child
    assert "gpu_count_of" not in resume


def test_build_verl_overrides_batch_shape_is_identical_across_gpu_counts():
    # the guard that matters: adding cards must not change what the optimizer sees. anything that
    # would alter the effective batch (or the per-gpu micro batch) is a silent recipe change.
    batch_keys = (
        "data.train_batch_size=",
        "actor_rollout_ref.rollout.n=",
        "actor_rollout_ref.actor.ppo_mini_batch_size=",
        # the per-gpu token budget is part of the shape too: it is PER-GPU, so emitting a
        # card-dependent value would change each rank's micro-batch as cards are added.
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=",
    )
    one = rl_verl.build_verl_overrides(_overrides_cfg(n_gpus=1))
    for n_gpus in (2, 4, 8):
        many = rl_verl.build_verl_overrides(_overrides_cfg(n_gpus=n_gpus))
        for key in batch_keys:
            assert [o for o in one if o.startswith(key)] == [o for o in many if o.startswith(key)]


def test_one_optimizer_step_consumes_exactly_the_requested_unique_prompts():
    # train_batch_size and ppo_mini_batch_size count unique prompts; rollout.n carries the group.
    # folding group_size into either silently trains on 1/group_size of the intended data.
    for prompts, group in ((64, 8), (5, 8), (2, 2), (1, 6)):
        o = rl_verl.build_verl_overrides(_overrides_cfg(prompts_per_step=prompts, group_size=group))
        assert f"data.train_batch_size={prompts}" in o
        assert f"actor_rollout_ref.actor.ppo_mini_batch_size={prompts}" in o
        assert f"actor_rollout_ref.rollout.n={group}" in o


def test_build_verl_overrides_sets_truncation_mask_when_enabled():
    o = rl_verl.build_verl_overrides(_overrides_cfg(mask_truncated_completions=True))
    # `++` (append-or-override), because the key exists in the fork's rollout.yaml but not stock's.
    assert "++actor_rollout_ref.rollout.mask_truncated_completions=true" in o


def test_build_verl_overrides_omits_truncation_mask_when_disabled():
    # stock verl rejects the unknown key at dataclass conversion, and not masking is already its
    # behavior, so emitting `=false` would break stock runs while changing nothing.
    o = rl_verl.build_verl_overrides(_overrides_cfg(mask_truncated_completions=False))
    assert not any("mask_truncated_completions" in override for override in o)


def test_build_verl_overrides_pins_both_blackwell_attention_backends():
    o = rl_verl.build_verl_overrides(
        _overrides_cfg(attention_backend="FLASHINFER", mm_encoder_attn_backend="TORCH_SDPA")
    )
    # verl spreads engine_kwargs.vllm straight into AsyncEngineArgs, where both are real fields in
    # the pinned vllm 0.19.1. `+` appends under the existing struct, as kv_cache_dtype does.
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASHINFER" in o
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend=TORCH_SDPA" in o


def test_build_verl_overrides_leaves_attention_backends_to_vllm_off_blackwell():
    # off blackwell vllm's own capability-ordered defaults are correct, and pinning a backend there
    # would override a working choice. the resolver returns None/None, so nothing may be emitted.
    o = rl_verl.build_verl_overrides(
        _overrides_cfg(attention_backend=None, mm_encoder_attn_backend=None)
    )
    assert not any("attention_backend" in override for override in o)


def test_build_verl_overrides_sizes_engine_to_the_job_not_the_architecture():
    # left unset, verl substitutes the model's full max_position_embeddings and hands it to vllm,
    # so a short job on a long-context model reserves kv cache it can never use. the emitted length
    # must be the job's own engine length.
    o = rl_verl.build_verl_overrides(_overrides_cfg(max_model_len=2368))
    assert "actor_rollout_ref.rollout.max_model_len=2368" in o


def test_engine_len_clamped_to_model_limit():
    # verl raises ValueError when max_model_len exceeds max_position_embeddings, so a job asking
    # for more context than the architecture has must train shorter, not die at rollout startup.
    assert backend_common.clamp_engine_len(32768, 8192) == 8192
    # under the limit is untouched, and an unknown limit leaves verl's own resolution in charge.
    assert backend_common.clamp_engine_len(4096, 40960) == 4096
    assert backend_common.clamp_engine_len(32768, None) == 32768
    assert backend_common.clamp_engine_len(32768, 0) == 32768


def test_token_budget_admits_a_full_length_sequence():
    # dynamic bsz packs micro-batches up to this budget. below one full sequence, the longest
    # rollout the engine can produce fits in no micro-batch at all.
    cfg = _overrides_cfg(max_prompt_len=31744, max_completion=1024, max_token_len_per_gpu=32768)
    o = rl_verl.build_verl_overrides(cfg)
    assert "actor_rollout_ref.actor.use_dynamic_bsz=true" in o
    assert "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true" in o
    key = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu="
    budget = int(next(x for x in o if x.startswith(key)).split("=")[1])
    assert budget >= cfg["max_prompt_len"] + cfg["max_completion"]
    # the engine asserts the actor and log-prob flags match, so both budgets move together.
    assert f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={budget}" in o


def test_dynamic_bsz_replaces_sequence_count_micro_batches():
    # with use_dynamic_bsz on, verl's actor config validation skips the micro-batch checks entirely
    # and asserts the TOKEN budgets are set instead. a leftover sequence-count key would be dead
    # config claiming to bound memory.
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert not any("ppo_micro_batch_size_per_gpu" in x for x in o)
    assert not any("log_prob_micro_batch_size_per_gpu" in x for x in o)


def test_multimodal_overrides_hand_verl_the_images_column():
    # the parquet's images column is inert unless verl is told to read it: without image_key the
    # dataset treats the rows as text, the <image> placeholders never re-expand, and the model
    # trains on the caption alone -- silently, which is the failure this whole port exists to avoid.
    o = rl_verl.build_verl_overrides(_overrides_cfg(multimodal=True))
    assert "data.image_key=images" in o
    # a processor rather than a bare tokenizer, and raw chat so verl owns the expansion.
    assert "actor_rollout_ref.model.trust_remote_code=true" in o
    assert "data.return_raw_chat=true" in o
    # verl RAISES on an over-budget multimodal prompt rather than truncating it, so truncation must
    # be error (never left on a silent trim) and the length filter must be armed.
    assert "data.truncation=error" in o
    assert "data.filter_overlong_prompts=true" in o
    # the processor's image loader is not fork-safe under verl's default dataloader workers.
    assert "data.dataloader_num_workers=0" in o


def test_shared_rollout_cache_override_uses_the_current_vllm_key():
    assert backend_common.rollout_mm_processor_cache_overrides() == [
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0"
    ]


def test_grpo_unconditionally_disables_the_vllm_multimodal_processor_cache():
    overrides = rl_verl.build_verl_overrides(_overrides_cfg())
    assert set(backend_common.rollout_mm_processor_cache_overrides()) <= set(overrides)


def test_text_overrides_omit_every_multimodal_key():
    # the control: these keys must be absent, not merely false. data.image_key=images on a text job
    # points verl at a column the parquet does not have.
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    for key in (
        "data.image_key",
        "data.return_raw_chat",
        "data.return_multi_modal_inputs",
    ):
        assert not any(x.startswith(key) for x in o), key
    assert "data.dataloader_num_workers=0" in o


def test_build_verl_training_cfg_carries_the_multimodal_flag():
    # the flag is resolved in _resolve_grpo_inputs but consumed by build_verl_overrides, so
    # a cfg that dropped it would produce a text-shaped override list from a multimodal parquet.
    source = inspect.getsource(rl_verl._build_verl_training_cfg)
    assert '"multimodal": bool(inp.get("multimodal"))' in source


def _mem_util_inp(**over):
    inp = {
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "",
        "engine_len": 2048,
        "group_size": 8,
        "fsdp_generation": 2,
        "lora_rank": 32,
    }
    inp.update(over)
    return inp


def test_gpu_mem_util_is_the_sized_budget_not_a_constant():
    """VERL-155: the launched executor budget must be the one ``colocate_kv_util`` computes.

    ``gpu_memory_utilization`` shipped as a flat 0.5 while ``estimate_vram_gb``'s admission check
    documented itself as mirroring ``colocate_kv_util``'s cap -- so preflight sized a run against one
    budget and the worker requested another. The assertion is deliberately made against the model's
    OWN output rather than a copied literal: pinning the number here is exactly what let the previous
    constant look intended, and it would have to be hand-edited (i.e. re-decided) on any retune.
    """
    from flash.core.catalog import MODELS
    from flash.engine.plan.vram import colocate_kv_util
    from flash.providers.core.base import get_gpu_info

    info = MODELS["Qwen/Qwen3.5-9B"]
    want = colocate_kv_util(
        float(info.params_b),
        2048,
        float(get_gpu_info("H100").vram_gb),
        sleep_mode=True,
        num_generations=8,
        active_params_b=None,
        fp8_kv=False,
        model_info=info,
        lora_rank=32,
    )
    got = rl_verl.resolve_gpu_mem_util(
        _mem_util_inp(), gpu_type="H100", n_gpus=1, fp8_kv=False, sleep_unsupported=False
    )
    explicit_tp_one = colocate_kv_util(
        float(info.params_b),
        2048,
        float(get_gpu_info("H100").vram_gb),
        sleep_mode=True,
        num_generations=8,
        active_params_b=None,
        fp8_kv=False,
        model_info=info,
        tensor_parallel=1,
        lora_rank=32,
    )
    assert got == want == explicit_tp_one
    # the current 9b geometry, kv reserve, and rank-32 adapter fit inside vllm's 80 gb share.
    assert got * 80 == pytest.approx(31.605060096)
    assert got.hex() == "0x1.948b75ff05902p-2"
    # and it is genuinely NOT the old constant, so the test cannot pass on an unwired build.
    assert got != rl_verl._DEFAULT_GPU_MEM_UTIL
    assert got < rl_verl._DEFAULT_GPU_MEM_UTIL


def test_gpu_mem_util_sizing_reaches_the_launch_config():
    """The sized value must survive into the cfg dict and the override list, not just the helper."""
    cfg = rl_verl._build_verl_training_cfg(
        {
            **_mem_util_inp(),
            "lora_rank": 32,
            "lora_alpha": 64,
            "lr": 1e-5,
            "prompts_per_step": 16,
            "mask_truncated_completions": True,
            "max_prompt_len": 1024,
            "max_completion": 1024,
            "max_response_len": 1024,
            "multi_turn": False,
            "temperature": 1.0,
            "top_p": 0.95,
            "kl_coef": 0.0,
            "entropy_quantile": None,
            "stop_sequences": (),
            "structured_outputs": None,
            "seed": 42,
            "ppo_epochs": 1,
            "steps": 60,
            "warmstart_adapter": "",
            "fsdp_generation": 2,
            "verl_total_epochs": 1,
            "save_freq": 20,
            "ckpt_to_keep": 1,
        },
        train_files="/w/t.parquet",
        val_files="/w/v.parquet",
        model_path="Qwen/Qwen3.5-9B",
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
        ce_backend="torch",
        reward_path="/w/r.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
        gpu_type="H100",
        n_gpus=1,
    )
    want = rl_verl.resolve_gpu_mem_util(
        _mem_util_inp(), gpu_type="H100", n_gpus=1, fp8_kv=False, sleep_unsupported=False
    )
    assert cfg["gpu_mem_util"] == want
    assert (
        f"actor_rollout_ref.rollout.gpu_memory_utilization={want}"
        in rl_verl.build_verl_overrides(cfg)
    )


def test_multigpu_gpu_mem_util_shards_only_weights_and_frees_the_observed_shortfall():
    default = rl_verl._DEFAULT_GPU_MEM_UTIL
    inp = _mem_util_inp(model_id="Qwen/Qwen3.6-35B-A3B", group_size=4)
    for gpu_type, vram_gb in (("H200", 141.0), ("B200", 180.0)):
        got = rl_verl.resolve_gpu_mem_util(
            inp, gpu_type=gpu_type, n_gpus=2, fp8_kv=False, sleep_unsupported=True
        )
        assert got < default
        assert (default - got) * vram_gb > 1.57


def test_multigpu_gpu_mem_util_reserves_rank_local_lora_inside_vllm_without_reducing_kv():
    from flash.core.catalog import MODELS
    from flash.engine.plan.vram import _lora_weight_memory_gb, colocate_kv_util
    from flash.providers.core.base import get_gpu_info

    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    card_gb = float(get_gpu_info("B200").vram_gb)
    adapter_gb = _lora_weight_memory_gb(64, info, tensor_parallel=2)
    assert adapter_gb == pytest.approx(6.591276032)

    without_adapter = colocate_kv_util(
        float(info.params_b),
        2048,
        card_gb,
        sleep_mode=False,
        num_generations=4,
        active_params_b=float(info.active_params_b),
        model_info=info,
        tensor_parallel=2,
    )
    got = rl_verl.resolve_gpu_mem_util(
        _mem_util_inp(
            model_id=info.id,
            group_size=4,
            lora_rank=64,
        ),
        gpu_type="B200",
        n_gpus=2,
        fp8_kv=False,
        sleep_unsupported=True,
    )

    # tp2 gives each rank 35 gb of weights. the resident short-context pool is the 8 gb floor, and
    # the default tp lora layout adds 6.591276032 gb on that same rank.
    assert got * card_gb == pytest.approx(49.591276032)
    assert got == pytest.approx(without_adapter + adapter_gb / card_gb)
    assert got * card_gb - 35.0 - adapter_gb == pytest.approx(8.0)


def test_gpu_mem_util_preserves_current_budget_without_catalog_lora_shapes(monkeypatch):
    from flash.core.catalog import MODELS
    from flash.engine.plan.vram import _lora_weight_memory_gb, colocate_kv_util
    from flash.providers.core.base import get_gpu_info

    model_id = "test/shape-less-model"
    info = SimpleNamespace(
        id=model_id,
        params_b=4.0,
        active_params_b=0.0,
        lora_target_shapes=(),
    )
    monkeypatch.setitem(MODELS, model_id, info)
    card_gb = float(get_gpu_info("H100").vram_gb)
    assert _lora_weight_memory_gb(64, info) is None
    got = rl_verl.resolve_gpu_mem_util(
        _mem_util_inp(model_id=model_id, lora_rank=64),
        gpu_type="H100",
        n_gpus=1,
        fp8_kv=False,
        sleep_unsupported=False,
    )
    current = colocate_kv_util(
        info.params_b,
        2048,
        card_gb,
        sleep_mode=True,
        num_generations=8,
        model_info=info,
    )

    assert got == current


def test_multigpu_gpu_mem_util_caps_the_sizer_at_the_previous_constant(monkeypatch):
    monkeypatch.setattr("flash.engine.plan.vram.colocate_kv_util", lambda *_args, **_kwargs: 0.55)

    got = rl_verl.resolve_gpu_mem_util(
        _mem_util_inp(), gpu_type="H100", n_gpus=2, fp8_kv=False, sleep_unsupported=False
    )

    assert got == rl_verl._DEFAULT_GPU_MEM_UTIL == 0.5


def test_multigpu_gpu_mem_util_never_exceeds_the_previous_constant():
    default = rl_verl._DEFAULT_GPU_MEM_UTIL
    model_ids = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B")
    for model_id in model_ids:
        for gpu_type in ("H100", "H200", "B200"):
            for tensor_parallel in (2, 4, 8):
                for engine_len in (1024, 2048, 8192, 32768):
                    for group_size in SUPPORTED_GRPO_GROUP_SIZES:
                        got = rl_verl.resolve_gpu_mem_util(
                            _mem_util_inp(
                                model_id=model_id,
                                engine_len=engine_len,
                                group_size=group_size,
                            ),
                            gpu_type=gpu_type,
                            n_gpus=tensor_parallel,
                            fp8_kv=False,
                            sleep_unsupported=True,
                        )
                        assert 0.10 <= got <= default, (
                            model_id,
                            gpu_type,
                            tensor_parallel,
                            engine_len,
                            group_size,
                            got,
                        )


def test_gpu_mem_util_keeps_the_constant_where_the_model_does_not_apply(monkeypatch):
    default = rl_verl._DEFAULT_GPU_MEM_UTIL
    # unknown card: the budget is a fraction of the card, so there is nothing to take a fraction of.
    for unknown in ("", "   ", "Nonexistent9000"):
        assert (
            rl_verl.resolve_gpu_mem_util(
                _mem_util_inp(), gpu_type=unknown, n_gpus=1, fp8_kv=False, sleep_unsupported=False
            )
            == default
        )
    # unresolvable model size: the weight copy is the dominant term.
    monkeypatch.setattr("flash.engine.plan.vram.resolve_params_b", lambda *_args, **_kwargs: None)
    assert (
        rl_verl.resolve_gpu_mem_util(
            _mem_util_inp(
                model_id="Qwen/Qwen3.5-9B",
                model_revision="a" * 40,
            ),
            gpu_type="H100",
            n_gpus=1,
            fp8_kv=False,
            sleep_unsupported=False,
        )
        == default
    )
    # any sizing exception keeps launch on the prior constant rather than making sizing fatal.
    monkeypatch.setattr("flash.providers.core.base.get_gpu_info", lambda _gpu_type: 1 / 0)
    assert (
        rl_verl.resolve_gpu_mem_util(
            _mem_util_inp(), gpu_type="H100", n_gpus=2, fp8_kv=False, sleep_unsupported=False
        )
        == default
    )


def test_gpu_mem_util_does_not_credit_an_offload_a_resident_engine_never_takes():
    """A sleep_unsupported model stays RESIDENT, so it must not get the sleep path's 1.5x KV pool.

    ``colocate_kv_util``'s sleep branch budgets a larger pool because the engine is offloaded during
    the backward. For a model flagged sleep_unsupported the engine never leaves the card, so that
    larger pool would be sized against a peak the run does not have.
    """
    resident = rl_verl.resolve_gpu_mem_util(
        _mem_util_inp(), gpu_type="H100", n_gpus=1, fp8_kv=False, sleep_unsupported=True
    )
    sleeping = rl_verl.resolve_gpu_mem_util(
        _mem_util_inp(), gpu_type="H100", n_gpus=1, fp8_kv=False, sleep_unsupported=False
    )
    assert resident < sleeping


def test_gpu_mem_util_raises_the_kv_pool_where_the_constant_starves_it():
    """The sizing is not uniformly a reduction: it must be free to ask for MORE than 0.5.

    A long-context 35B MoE on a 180 GB B200 is the shape the 0.55 cap lift exists for -- the flat
    0.5 caps the KV pool and with it rollout concurrency. A wiring that could only ever lower the
    budget would silently keep that case starved.
    """
    got = rl_verl.resolve_gpu_mem_util(
        _mem_util_inp(model_id="Qwen/Qwen3.6-35B-A3B", engine_len=8192),
        gpu_type="B200",
        n_gpus=1,
        fp8_kv=False,
        sleep_unsupported=True,
    )
    assert got > rl_verl._DEFAULT_GPU_MEM_UTIL


def test_sleep_unsupported_models_keep_the_rollout_engine_resident():
    """VERL-091: a model whose vLLM wake HANGS must never be offloaded between steps.

    verl defaults free_cache_engine and enable_sleep_mode BOTH True and sleeps the rollout engine at
    every step boundary, so without an explicit override a catalog model flagged sleep_unsupported
    wedges on the first wake instead of failing fast. The flag comes from the catalog, so assert
    against a real flagged entry rather than a fabricated one.
    """
    from flash.core.catalog import MODELS

    flagged = [m for m, i in MODELS.items() if getattr(i, "sleep_unsupported", False)]
    assert flagged, "no catalog model is sleep_unsupported; this guard now has no subject"

    def _argv(model_id):
        inp = {
            "lora_rank": 32,
            "lora_alpha": 64,
            "lr": 1e-5,
            "group_size": 8,
            "prompts_per_step": 16,
            "mask_truncated_completions": True,
            "max_prompt_len": 3072,
            "max_completion": 1024,
            "max_response_len": 1024,
            "multi_turn": False,
            "engine_len": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
            "kl_coef": 0.0,
            "entropy_quantile": None,
            "stop_sequences": (),
            "structured_outputs": None,
            "seed": 42,
            "ppo_epochs": 1,
            "steps": 60,
            "warmstart_adapter": "",
            "fsdp_generation": 2,
            "model_id": model_id,
            "verl_total_epochs": 2,
            "save_freq": 20,
            "ckpt_to_keep": 1,
        }
        # go through the real builder so the flag cannot drift out of the cfg it emits.
        cfg = rl_verl._build_verl_training_cfg(
            inp,
            train_files="/w/t.parquet",
            val_files="/w/v.parquet",
            model_path="/w/model",
            thinking=False,
            loggers=["console"],
            fp8_kv=False,
            enforce_eager=False,
            attention_backend=None,
            mm_encoder_attn_backend=None,
            ce_backend="torch",
            reward_path="/w/r.py",
            local_dir="/w/ckpt",
            project_name="flash",
            experiment_name="flash-rl-run123",
        )
        return rl_verl.build_verl_overrides(cfg)

    # the two knobs need DIFFERENT hydra prefixes -- rollout_resident_overrides' docstring has the
    # why. asserted EXACTLY rather than as a substring, because "x=false" is a substring of
    # "+x=false": the obvious assertion passes against the spelling that kills the run at parse,
    # which is how this shipped. see ISSUES.md VERL-148.
    for override in (
        "actor_rollout_ref.rollout.free_cache_engine=false",
        "+actor_rollout_ref.rollout.enable_sleep_mode=false",
    ):
        assert override in _argv(flagged[0])
    # the bare spelling must be ABSENT, not merely accompanied by the prefixed one.
    assert "actor_rollout_ref.rollout.enable_sleep_mode=false" not in _argv(flagged[0])
    # and the override is scoped: an ordinary model keeps verl's own sleep/wake offload, which is
    # what lets a large rollout fit alongside the training weights.
    for key in ("free_cache_engine", "enable_sleep_mode"):
        assert not [a for a in _argv("Qwen/Qwen3.5-9B") if key in a]


def test_build_verl_training_cfg_derives_engine_len_and_budget():
    inp = {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lr": 1e-5,
        "group_size": 8,
        "prompts_per_step": 16,
        "mask_truncated_completions": True,
        "max_prompt_len": 3072,
        "max_completion": 1024,
        "max_response_len": 1024,
        "multi_turn": False,
        "engine_len": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "kl_coef": 0.0,
        "entropy_quantile": None,
        "stop_sequences": (),
        "structured_outputs": None,
        "seed": 42,
        "ppo_epochs": 1,
        "steps": 60,
        "warmstart_adapter": "",
        "fsdp_generation": 2,
        "model_id": "Qwen/Qwen3.5-9B",
        "verl_total_epochs": 2,
        "save_freq": 20,
        "ckpt_to_keep": 1,
    }
    common = {
        "ce_backend": "torch",
        "train_files": "/w/t.parquet",
        "val_files": "/w/v.parquet",
        "model_path": "Qwen/Qwen3-4B",
        "thinking": False,
        "loggers": ["console"],
        "fp8_kv": False,
        "enforce_eager": False,
        "attention_backend": None,
        "mm_encoder_attn_backend": None,
        "reward_path": "/w/r.py",
        "local_dir": "/w/ckpt",
        "project_name": "flash",
        "experiment_name": "flash-rl-run123",
    }
    cfg = rl_verl._build_verl_training_cfg(inp, **common)
    # the engine gets the full prompt+completion length, not the prompt budget alone, and the token
    # budget matches it. the resolver clamps engine_len, so the builder passes it through unchanged.
    assert cfg["max_model_len"] == 4096
    assert cfg["max_token_len_per_gpu"] == 4096


@pytest.mark.parametrize(
    (
        "prompt_count",
        "prompts_per_step",
        "epochs",
        "max_steps",
        "expected_steps",
        "expected_epochs",
    ),
    [
        pytest.param(33, 16, 2, None, 5, 3, id="partial-batch-derived-horizon"),
        pytest.param(32, 16, 2, 7, 7, 4, id="explicit-horizon-beyond-derived"),
        pytest.param(32, 16, 2, None, 4, 2, id="exactly-divisible"),
    ],
)
def test_verl_epoch_capacity_reaches_update_horizon(
    prompt_count, prompts_per_step, epochs, max_steps, expected_steps, expected_epochs
):
    derived_steps = plan_steps.on_policy_steps(
        epochs=epochs, prompt_count=prompt_count, prompts_per_step=prompts_per_step
    )
    steps = plan_steps.resolve_update_horizon(derived_steps, max_steps)
    resolved_epochs = rl_verl._verl_epochs_for_horizon(
        epochs=epochs,
        prompt_count=prompt_count,
        prompts_per_step=prompts_per_step,
        steps=steps,
    )

    assert steps == expected_steps
    assert resolved_epochs == expected_epochs
    assert (prompt_count // prompts_per_step) * resolved_epochs >= steps


@pytest.mark.parametrize(
    ("prompt_count", "prompts_per_step", "message"),
    [
        pytest.param(0, 16, "prompt_count must be positive", id="no-prompts"),
        pytest.param(5, 0, "prompts_per_step must be positive", id="zero-batch"),
        pytest.param(5, 16, "prompt_count must be at least", id="batch-exceeds-prompts"),
    ],
)
def test_verl_epoch_capacity_rejects_invalid_batch_inputs(prompt_count, prompts_per_step, message):
    with pytest.raises(ValueError, match=message):
        rl_verl._verl_epochs_for_horizon(
            epochs=2,
            prompt_count=prompt_count,
            prompts_per_step=prompts_per_step,
            steps=3,
        )


def test_verl_epoch_capacity_invariant_across_valid_inputs():
    for prompt_count in range(1, 34):
        for prompts_per_step in range(1, prompt_count + 1):
            for epochs in (1, 2, 4):
                for steps in (1, epochs, epochs + 5):
                    resolved_epochs = rl_verl._verl_epochs_for_horizon(
                        epochs=epochs,
                        prompt_count=prompt_count,
                        prompts_per_step=prompts_per_step,
                        steps=steps,
                    )
                    assert (prompt_count // prompts_per_step) * resolved_epochs >= steps


def test_resolver_clamps_prompt_budget_with_the_engine(monkeypatch):
    # regression: clamping only the engine let the prompt filter admit prompts sized against the
    # UNCLAMPED context. those prompts plus the completion allowance overflow the engine vllm was
    # actually given, so they die at rollout instead of training on the shorter context. every
    # length must descend from one clamped value.
    # asks for twice the architecture's context (model_max_position_embeddings is pinned to 32768).
    inp = _capability_resolve(monkeypatch, _capability_env(), train={"max_context_tokens": 65536})
    assert inp["engine_len"] == 32768
    # the prompt filter's budget is carved out of the clamped engine, not the requested 65536.
    assert inp["max_prompt_len"] + inp["max_completion"] == 32768
    # and every length the overrides emit agrees with it.
    cfg = rl_verl._build_verl_training_cfg(
        inp,
        ce_backend="torch",
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_path=inp["model_id"],
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    assert cfg["max_model_len"] == 32768
    assert cfg["max_token_len_per_gpu"] == 32768
    assert cfg["max_prompt_len"] + cfg["max_completion"] == cfg["max_model_len"]


def _save_steps_inputs(monkeypatch, *, save_at_steps=None, save_every=None, max_steps=100):
    """resolve grpo verl inputs for a job with (or without) exact save steps."""
    train: dict = {"max_steps": max_steps}
    if save_at_steps is not None:
        train["save_at_steps"] = list(save_at_steps)
    if save_every is not None:
        train["save_every"] = save_every
    return _capability_resolve(monkeypatch, _capability_env(), train=train)


def test_save_freq_is_the_gcd_so_verl_lands_on_every_required_step(monkeypatch):
    # periodic saves land only when global_step % save_freq == 0, so they cannot hit an arbitrary set
    # directly. the gcd is the largest interval every required step divides, so verl writes a superset
    # of the checkpoints and the uploader publishes deployables at exactly the requested ones.
    inp = _save_steps_inputs(monkeypatch, save_at_steps=(10, 25, 100))
    assert inp["save_freq"] == 5
    assert inp["save_at_steps"] == (10, 25, 100)
    for step in inp["save_at_steps"]:
        assert step % inp["save_freq"] == 0


def test_save_freq_falls_back_to_save_every_without_exact_steps(monkeypatch):
    # a long run preserves the customer's interval, so the clamp never increases normal frequency.
    inp = _save_steps_inputs(monkeypatch, save_every=15)
    assert inp["steps"] == 100
    assert inp["save_freq"] == 15
    assert inp["save_at_steps"] == ()


def test_short_derived_horizon_clamps_save_freq_to_the_final_step(monkeypatch):
    inp = _capability_resolve(
        monkeypatch,
        _capability_env(example_count=800),
        train={"max_examples": 800, "epochs": 1, "prompts_per_step": 64},
    )
    assert inp["steps"] == 13
    assert inp["save_freq"] == 13
    assert inp["steps"] % inp["save_freq"] == 0


def test_save_steps_reach_the_horizon_they_were_validated_against(monkeypatch):
    # save_at_steps requires max_steps, and the horizon resolves to exactly that, so every required
    # step is reachable by the run. this is the invariant the uploader's completeness check assumes.
    inp = _save_steps_inputs(monkeypatch, save_at_steps=(10, 25, 100), max_steps=100)
    assert inp["steps"] == 100
    assert inp["save_at_steps"][-1] <= inp["steps"]


def test_final_publish_is_suppressed_when_exact_save_steps_are_set():
    # parity with the retired trl path: with save_at_steps set the customer
    # asked for those steps and nothing else, so the final step must not add an unrequested
    # deployable. without them the final checkpoint is still preserved.
    from flash.engine.plan.steps import final_save_due

    assert not final_save_due(100, (10, 25))
    assert final_save_due(100, ())


def test_checkpoint_retention_outlives_the_export_when_exact_saves_are_set(monkeypatch):
    # verl prunes a checkpoint once the NEXT save completes, so keeping 1 gives the uploader a
    # single save interval to export before its source is deleted. with a gcd of 1 that interval is
    # one update, which races the export and can lose a required deployable.
    exact = _save_steps_inputs(monkeypatch, save_at_steps=(10, 11))
    assert exact["save_freq"] == 1
    assert exact["ckpt_to_keep"] > 1

    # nothing is exported mid-run without exact saves, so retention stays at its cheapest.
    periodic = _save_steps_inputs(monkeypatch, save_every=15)
    assert periodic["ckpt_to_keep"] == 1


def test_verl_resolver_builds_capacity_overrides_and_configured_metadata(monkeypatch):
    from flash.core.spec import JobSpec
    from flash.engine.worker.train.rl.launch import inputs as rl_inputs

    class _Env:
        multi_turn = False
        is_tool_env = False

        def dataset(self):
            return [{"index": i} for i in range(33)]

        def prompt_messages(self, ex):
            return [{"role": "user", "content": f"question {ex['index']}"}]

    class _Tokenizer:
        pad_token = None
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1])

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {"prompts_per_step": 16, "epochs": 2},
        }
    )
    monkeypatch.setattr(rl_inputs._worker_state, "JOB_SPEC", spec)
    monkeypatch.setattr(rl_inputs._worker_state, "SEED", 42)
    monkeypatch.setattr(rl_inputs._worker_state, "THINKING", False)
    monkeypatch.setattr(rl_inputs._worker_state, "require_active_env", lambda: _Env())
    monkeypatch.setattr(rl_inputs._worker_config, "grpo_overrides", dict)
    monkeypatch.setattr(
        rl_inputs._worker_config, "grpo_mask_truncated_completions", lambda _train: False
    )
    monkeypatch.setattr(
        rl_inputs._worker_hf, "load_tokenizer", lambda *args, **kwargs: _Tokenizer()
    )
    monkeypatch.setattr(rl_inputs, "seed_training_rngs", lambda seed: None)
    # the context-limit probe reads the model config off the hub; keep this unit test offline.
    monkeypatch.setattr(rl_inputs, "model_max_position_embeddings", lambda *a, **k: 40960)

    inp = rl_inputs._resolve_grpo_inputs()
    cfg = rl_verl._build_verl_training_cfg(
        inp,
        ce_backend="torch",
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_path=inp["model_id"],
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    overrides = rl_verl.build_verl_overrides(cfg)
    notes = rl_verl._build_verl_train_notes(
        inp,
        steps_run=5,
        retained_prompts=len(inp["prompts"]),
        reward_history=[],
        loss_curve=[],
    )

    assert "trainer.total_training_steps=5" in overrides
    assert "trainer.total_epochs=3" in overrides
    assert notes["epochs"] == 2
    assert notes["grpo_recipe"]["verl_total_epochs"] == 3
    # the builder is called without n_gpus here, exactly as a single-gpu job does: the default must
    # stay 1 so a spec with no gpu.count keeps the historical shape.
    assert cfg["n_gpus"] == 1


def test_build_verl_overrides_wandb_logger_when_enabled():
    o = rl_verl.build_verl_overrides(_overrides_cfg(loggers=["console", "wandb"]))
    assert "trainer.logger=[console,wandb]" in o


def test_build_verl_overrides_warmstart_adapter_path():
    # fresh run: no lora_adapter_path override.
    fresh = rl_verl.build_verl_overrides(_overrides_cfg(warmstart_adapter=""))
    assert not any("lora_adapter_path" in x for x in fresh)
    # warm-start: point verl's lora init at the downloaded source adapter dir.
    warm = rl_verl.build_verl_overrides(_overrides_cfg(warmstart_adapter="/tmp/sft_adapter"))
    assert "actor_rollout_ref.model.lora_adapter_path=/tmp/sft_adapter" in warm


def test_build_verl_overrides_fp8_kv_gated_on_hardware():
    off = rl_verl.build_verl_overrides(_overrides_cfg(fp8_kv=False))
    assert not any("kv_cache_dtype" in x for x in off)
    on = rl_verl.build_verl_overrides(_overrides_cfg(fp8_kv=True))
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8" in on


def test_build_verl_overrides_enforce_eager_gated_on_hardware():
    off = rl_verl.build_verl_overrides(_overrides_cfg(enforce_eager=False))
    assert not any("enforce_eager" in x for x in off)
    on = rl_verl.build_verl_overrides(_overrides_cfg(enforce_eager=True))
    # plain override, not '+': enforce_eager is a declared verl RolloutConfig field
    # (workers/config/rollout.py:195), so appending it would be a duplicate-key error.
    assert "actor_rollout_ref.rollout.enforce_eager=True" in on


def test_the_resolved_eager_flag_reaches_the_verl_config():
    # the string assertions above pass against a resolver whose answer is never carried into the
    # config, which is exactly how the retired trl workaround got dropped. pin the wiring.
    built = inspect.getsource(rl_train.run_rl_train) + inspect.getsource(
        rl_train._configure_rl_child
    )
    assert "enforce_eager = resolve_rollout_enforce_eager(verl_cc)" in built
    assert "enforce_eager=enforce_eager," in built
    # and the capability it decides from is the one batched child probe every question shares.
    assert "caps = probe_verl_capabilities(python_bin, gdn_module)" in built
    assert "verl_cc = verl_device_capability(caps)" in built
    assert (
        "attention_backend, mm_encoder_attn_backend = "
        "resolve_blackwell_attention_backends(caps, verl_cc)" in " ".join(built.split())
    )
    cfg = inspect.getsource(rl_verl._build_verl_training_cfg)
    assert '"enforce_eager": enforce_eager,' in cfg


def test_build_verl_overrides_kl_off_by_default():
    # flash default kl_penalty_coef=0 (dr-grpo, no kl term) -> no reference policy.
    o = rl_verl.build_verl_overrides(_overrides_cfg(kl_coef=0.0))
    assert "actor_rollout_ref.actor.use_kl_loss=False" in o
    assert not any("kl_loss_coef" in x for x in o)
    assert not any("ref.log_prob_micro_batch" in x for x in o)


def test_build_verl_overrides_kl_on_when_requested():
    o = rl_verl.build_verl_overrides(_overrides_cfg(kl_coef=0.02))
    assert "actor_rollout_ref.actor.use_kl_loss=True" in o
    assert "actor_rollout_ref.actor.kl_loss_coef=0.02" in o
    # the ref worker carries no batching keys of its own: ref.yaml resolves
    # log_prob_use_dynamic_bsz / log_prob_max_token_len_per_gpu through oc.select on the actor keys
    # the block above sets, so emitting a sequence-count micro batch here would contradict them.
    assert not any("ref.log_prob_micro_batch" in x for x in o)


def test_verl_uses_canonical_heartbeat_stage_contracts():
    from flash.engine.worker.io.heartbeat import _HB_THROTTLED_STAGES
    from flash.providers._lifecycle.instances.poll import STEP_GATED_STAGES
    from flash.runner.lifecycle.state import _TRAINING_STAGES

    src = inspect.getsource(rl_train.run_rl_train)
    # the stage names are a cross-process contract: the poller, the throttle table and the runner
    # all key off "rl_step"/"rl_finalizing". the tempting mistake is to coin a module-prefixed
    # variant, which reports a stage none of those three recognise. assert against the spelling the
    # CURRENT module name would produce -- pinning the pre-rename "rl_verl_*" spelling here would be
    # asserting the absence of a string nothing can emit any more, which no regression can fail.
    assert "rl_train_training" not in src
    assert "rl_train_finalizing" not in src
    initial_heartbeat = '_worker_heartbeat.heartbeat("rl_step", step=0, initial=True)'
    assert initial_heartbeat in src
    # ordering is read off the ast, not off substring offsets: the liveness call spans several lines
    # once it carries keywords, and a text search for the one-line spelling would report "missing"
    # for a call that is present and correctly placed.
    tree = ast.parse(textwrap.dedent(src))
    stage_linenos = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == "rl_step"):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "liveness_heartbeat":
            stage_linenos["liveness"] = node.lineno
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "heartbeat"
            and any(kw.arg == "initial" for kw in node.keywords)
        ):
            stage_linenos["initial"] = node.lineno
    assert "initial" in stage_linenos
    assert "liveness" in stage_linenos
    assert stage_linenos["initial"] < stage_linenos["liveness"]
    assert '"rl_finalizing"' in src
    assert "rl_step" in _HB_THROTTLED_STAGES
    assert "rl_step" in STEP_GATED_STAGES
    assert "rl_step" in _TRAINING_STAGES
    assert "rl_finalizing" in _HB_THROTTLED_STAGES


# ------------------------------- reward module render -------------------------------
def _rendered_reward_namespace(url_env="FLASH_VERL_REWARD_URL"):
    previous = sys.modules.get("flash_grpo_multiturn")
    sys.modules["flash_grpo_multiturn"] = grpo_multiturn
    try:
        namespace: dict = {}
        exec(compile(rl_reward.render_reward_module(url_env), "<reward>", "exec"), namespace)
        return namespace
    finally:
        if previous is None:
            sys.modules.pop("flash_grpo_multiturn", None)
        else:
            sys.modules["flash_grpo_multiturn"] = previous


def test_render_reward_module_is_valid_and_defines_compute_score():
    src = rl_reward.render_reward_module()
    ns = _rendered_reward_namespace()
    assert callable(ns["compute_score"])
    assert "from flash_grpo_multiturn import post_json" in src
    assert "urllib" not in src


def test_render_reward_module_missing_index_raises():
    ns = _rendered_reward_namespace()
    with pytest.raises(RuntimeError, match="no example index"):
        ns["compute_score"]("flash_env", "answer", "unused", extra_info={})


@pytest.mark.parametrize(
    "index",
    [True, 1.9, np.bool_(True), np.bool_(False), float("nan"), float("inf"), float("-inf")],
    ids=["bool", "fractional", "numpy-true", "numpy-false", "nan", "positive-inf", "negative-inf"],
)
def test_render_reward_module_rejects_invalid_index(monkeypatch, index):
    monkeypatch.setenv("TEST_FLASH_VERL_REWARD_URL", "http://unused")
    ns = _rendered_reward_namespace("TEST_FLASH_VERL_REWARD_URL")
    monkeypatch.setattr(
        grpo_multiturn.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("invalid index must not reach the reward server"),
    )

    with pytest.raises(RuntimeError, match="invalid example index"):
        ns["compute_score"]("flash_env", "answer", "unused", extra_info={"index": index})


@pytest.mark.parametrize("index", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_render_reward_module_accepts_exact_integral_index(monkeypatch, index):
    scored = []
    server, url = rl_multi.start_reward_server(
        lambda idx, solution: scored.append((idx, solution)) or 3.0,
        example_count=2,
    )
    try:
        monkeypatch.setenv("TEST_FLASH_VERL_REWARD_URL", url)
        ns = _rendered_reward_namespace("TEST_FLASH_VERL_REWARD_URL")
        assert (
            ns["compute_score"](
                "flash_env",
                "answer",
                "unused",
                extra_info={
                    "index": index,
                    "flash_rollout_identity": {
                        "optimizer_step": 1,
                        "sample_index": 1,
                        "rollout_ordinal": 0,
                        "validate": False,
                    },
                },
            )
            == 3.0
        )
        assert scored == [(1, "answer")]
    finally:
        server.shutdown()


def test_a_slow_env_call_is_not_cut_off_by_a_client_deadline(monkeypatch):
    # verl fans reward scoring out hard -- RewardLoopManager spawns reward.num_workers ray workers
    # and each asyncio.gathers its whole chunk -- and start_reward_server serializes them behind one
    # lock. a per-request deadline therefore bounds QUEUE WAIT, not the env call, so the Nth caller
    # in line fails for arriving Nth. a wedged env is the stall watchdog's job, not this client's.
    waited = []
    server, url = rl_multi.start_reward_server(
        lambda idx, solution: waited.append(idx) or 7.0, example_count=2
    )
    try:
        ns = _rendered_reward_namespace("TEST_URL")
        real_urlopen = grpo_multiturn.urllib.request.urlopen
        seen = []

        def urlopen_recording_deadline(req, *args, **kwargs):
            seen.append((args, kwargs))
            return real_urlopen(req)

        monkeypatch.setattr(grpo_multiturn.urllib.request, "urlopen", urlopen_recording_deadline)
        ns["_URL"] = url
        assert (
            ns["compute_score"](
                "env",
                "answer",
                "unused",
                extra_info={
                    "index": 0,
                    "flash_rollout_identity": {
                        "optimizer_step": 1,
                        "sample_index": 0,
                        "rollout_ordinal": 0,
                        "validate": False,
                    },
                },
            )
            == 7.0
        )
        assert waited == [0]
        assert seen == [((), {})], f"reward client still carries a deadline: {seen!r}"
    finally:
        server.shutdown()


def test_concurrent_scorers_are_serialized_for_the_env():
    # a flash env is a plain python object with no concurrency contract; the retired trl path only
    # ever called it from one thread. verl's reward workers do not, so the server must impose it.
    import concurrent.futures

    live = []
    peak = []
    lock = threading.Lock()

    def score(idx, solution):
        with lock:
            live.append(idx)
            peak.append(len(live))
        time.sleep(0.02)
        with lock:
            live.remove(idx)
        return float(idx)

    server, url = rl_multi.start_reward_server(score, example_count=8)
    try:
        ns = _rendered_reward_namespace("TEST_URL")
        ns["_URL"] = url
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda i: ns["compute_score"](
                        "env",
                        "a",
                        "u",
                        extra_info={
                            "index": i,
                            "flash_rollout_identity": {
                                "optimizer_step": 1,
                                "sample_index": i,
                                "rollout_ordinal": 0,
                                "validate": False,
                            },
                        },
                    ),
                    range(8),
                )
            )
        assert sorted(results) == [float(i) for i in range(8)]
        assert max(peak) == 1, f"env saw {max(peak)} concurrent calls"
    finally:
        server.shutdown()


def test_reward_server_accept_queue_holds_a_whole_rollout_batch(monkeypatch):
    # verl opens one connection per episode and starts a whole step at once, so the accept queue
    # sees prompts_per_step * group_size connects in a burst. socketserver's default backlog of 5
    # overflows there and the kernel RESETS the excess, which reaches the client as
    # ConnectionResetError at getresponse(); bridge_post does not retry, so that kills the run.
    rollout_batch = 512  # the default 64x8 recipe

    # record the argument the server actually hands to listen(). asserting on request_queue_size
    # alone cannot tell a working fix from a no-op: server_activate() reads that attribute once,
    # so a value set after server_bind() would never reach the socket at all.
    backlogs = []
    real_listen = socket.socket.listen

    def spy_listen(self, *args):
        backlogs.append(args[0] if args else None)
        return real_listen(self, *args)

    monkeypatch.setattr(socket.socket, "listen", spy_listen)

    server, _url = rl_multi.start_reward_server(
        lambda idx, solution: 1.0, example_count=8, rollout_batch=rollout_batch
    )
    try:
        assert server.request_queue_size >= rollout_batch
        assert backlogs, "server never called listen()"
        assert backlogs[-1] is not None, "listen() was called with no backlog argument"
        assert backlogs[-1] >= rollout_batch, f"listen() backlog is {backlogs[-1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("rollout_batch", [0, 32, 64 * 8, 2048])
def test_reward_bridge_backlog_never_falls_below_the_burst(rollout_batch):
    # a fixed constant would only move the cliff, so the queue is sized from the caller's burst.
    # an unspecified batch still keeps a floor well clear of socketserver's default of 5.
    server, _url = rl_multi.start_reward_server(
        lambda idx, solution: 1.0, example_count=1, rollout_batch=rollout_batch
    )
    try:
        assert server.request_queue_size >= max(128, rollout_batch)
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------- reward parity -------------------------------
class _BreakdownEnv:
    def scores_breakdown(self, graded, ex, state):
        return {"total": 1.0 if graded.strip() == ex["gt"] else 0.0}


class _RewardOnlyEnv:
    def reward(self, graded, ex, state):
        return 2.5 if ex["gt"] in graded else 0.0


class _RaisingEnv:
    def scores_breakdown(self, graded, ex, state):
        raise ValueError("boom")


@pytest.fixture
def _identity_graded(monkeypatch):
    import flash.engine.worker.model.decoding as decoding

    monkeypatch.setattr(decoding, "graded_text", lambda text, prompt_opened_thinking=False: text)
    monkeypatch.setattr(decoding, "thinking_text", lambda text, prompt_opened_thinking=False: "")
    monkeypatch.setattr(
        decoding, "think_token_count", lambda text, tok, prompt_opened_thinking=False: 3
    )


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_breakdown_and_reward_env():
    s = rl_single.score_single_turn(
        _BreakdownEnv(),
        "7",
        {"gt": "7"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )
    assert s == 1.0
    s2 = rl_single.score_single_turn(
        _RewardOnlyEnv(),
        "the answer is 7",
        {"gt": "7"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )
    assert s2 == 2.5


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_applies_thinking_penalty():
    # base reward 1.0 minus think_penalty(0.1) * think_token_count(3) = 0.7
    s = rl_single.score_single_turn(
        _BreakdownEnv(),
        "7",
        {"gt": "7"},
        tok=object(),
        thinking=True,
        prompt_opened_thinking=True,
        think_penalty=0.1,
    )
    assert abs(s - 0.7) < 1e-9


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_env_error_is_zero():
    s = rl_single.score_single_turn(
        _RaisingEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )
    assert s == 0.0


class _UnscorableRewardEnv:
    def reward(self, graded, ex, state):
        return float("nan")


class _UnscorableBreakdownEnv:
    def scores_breakdown(self, graded, ex, state):
        return {"total": float("nan"), "judge": 1.0}


@pytest.mark.usefixtures("_identity_graded")
@pytest.mark.parametrize(
    "env", [_UnscorableRewardEnv(), _UnscorableBreakdownEnv()], ids=["reward", "scores_breakdown"]
)
def test_an_unscorable_reward_is_masked_before_it_reaches_verl(env):
    """A non-finite reward must not be forwarded, from EITHER env hook.

    verl's grpo baseline is a plain torch.mean/torch.std over the group (core_algos.py:320-326)
    with no nan-aware variant on its path, so one nan row makes the mean, the std, and all
    `group_size` advantages nan -- the whole group, not just the unscorable row. The retired trl
    path could forward it because it masked nan rows out of the baseline and zeroed their
    advantage (grpo_trainer.py:2171,:2222); nothing downstream of here does that now.
    """
    breakdowns: list[dict[str, float] | None] = []
    s = rl_single.score_single_turn(
        env,
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert s == 0.0
    assert math.isfinite(s)


@pytest.mark.usefixtures("_identity_graded")
def test_an_infinite_reward_is_masked_too_even_with_a_penalty_applied():
    # inf, not just nan: nan is score_rollouts' canonical unscorable marker, but an env returning
    # inf poisons verl's baseline exactly as thoroughly and is not covered by testing nan alone.
    # the penalty is live here because that arithmetic runs between the env's return and the mask;
    # inf minus a finite penalty is still inf, so the mask is what has to catch it.
    class _InfEnv:
        def reward(self, graded, ex, state):
            return float("inf")

    s = rl_single.score_single_turn(
        _InfEnv(),
        "x",
        {"gt": "1"},
        tok=object(),
        thinking=True,
        prompt_opened_thinking=True,
        think_penalty=0.1,
    )
    assert s == 0.0


@pytest.mark.usefixtures("_identity_graded")
def test_an_unscorable_reward_still_re_raises_for_the_latency_profiler():
    # same contract as a raising env: the profiler must tell a real 0.0 apart from a grader that
    # is not returning a usable number, or it reports a broken grader as fast and confident.
    with pytest.raises(ValueError, match="non-finite"):
        rl_single.score_single_turn(
            _UnscorableRewardEnv(),
            "x",
            {"gt": "1"},
            tok=None,
            thinking=False,
            prompt_opened_thinking=False,
            think_penalty=0.0,
            raise_on_error=True,
        )


class _RaisingProbeEnv:
    """An env whose `scores_breakdown` ATTRIBUTE LOOKUP raises, not its call.

    Real shapes that do this: a `@property` that touches a closed handle, or a lazy proxy that
    dials a sidecar on first access. `hasattr` only swallows AttributeError, so anything else
    propagates out of the probe itself.
    """

    def __getattr__(self, name):
        if name == "scores_breakdown":
            raise RuntimeError("scoring sidecar is gone")
        raise AttributeError(name)


@pytest.mark.usefixtures("_identity_graded")
def test_a_capability_probe_that_raises_scores_zero_and_counts_as_a_failed_grading():
    """The probe is env code too, so it has to sit inside the guard that turns env faults into 0.0.

    Preserve ``None`` so failed scoring remains in named-metric denominators as zero.
    """
    breakdowns: list[dict[str, float] | None] = []
    s = rl_single.score_single_turn(
        _RaisingProbeEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )

    assert s == 0.0
    assert breakdowns == [None]


@pytest.mark.usefixtures("_identity_graded")
def test_a_raising_probe_still_re_raises_for_the_latency_profiler():
    # raise_on_error is the profiler's way of telling a real 0.0 apart from a broken grader. the
    # guard must not swallow the probe's fault for that caller either.
    with pytest.raises(RuntimeError, match="scoring sidecar is gone"):
        rl_single.score_single_turn(
            _RaisingProbeEnv(),
            "x",
            {"gt": "1"},
            tok=None,
            thinking=False,
            prompt_opened_thinking=False,
            think_penalty=0.0,
            raise_on_error=True,
        )


# --------------------- reward_metrics: per-name breakdown collection ---------------------
class _NamedBreakdownEnv:
    def scores_breakdown(self, graded, ex, state):
        hit = 1.0 if graded.strip() == ex["gt"] else 0.0
        return {"success": hit, "quality": 0.5, "total": hit}


class _CountingBreakdownEnv:
    """Scores each grading with its own ordinal, so a retained buffer window is identifiable."""

    def __init__(self):
        self.n = 0

    def scores_breakdown(self, graded, ex, state):
        n = float(self.n)
        self.n += 1
        return {"n": n, "total": n}


class _OverflowingBreakdownEnv:
    """A named component too large to be a float, beside a usable one and a usable total.

    `float()` raises OverflowError on this, NOT ValueError -- a distinction the observability pass
    has to make because it runs outside the grading error guard."""

    def scores_breakdown(self, graded, ex, state):
        hit = 1.0 if graded.strip() == ex["gt"] else 0.0
        return {"success": hit, "enormous": 10**400, "total": hit}


class _UnusableComponentEnv:
    """Emits a component whose value never coerces to a finite float, alongside a usable one."""

    def scores_breakdown(self, graded, ex, state):
        return {"broken": None, "diverged": float("nan"), "quality": 0.5, "total": 1.0}


class _BadTotalEnv:
    def scores_breakdown(self, graded, ex, state):
        return {"success": 1.0, "total": "not-a-number"}


class _RaisingRewardOnlyEnv:
    def reward(self, graded, ex, state):
        raise ValueError("grader is down")


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_collects_the_named_breakdown_for_reward_metrics():
    breakdowns: list[dict | None] = []
    score = rl_single.score_single_turn(
        _NamedBreakdownEnv(),
        "7",
        {"gt": "7"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert score == 1.0
    assert breakdowns == [{"success": 1.0, "quality": 0.5, "total": 1.0}]


@pytest.mark.usefixtures("_identity_graded")
def test_a_scalar_reward_env_contributes_no_breakdown_at_all():
    # appending {} for a scores_breakdown-less env would add a denominator under no numerators:
    # RewardObservabilityBuffer divides by every scored completion, so an env mixing the two
    # shapes -- or a run with none at all -- would publish every name shrunk toward 0.
    breakdowns: list[dict | None] = []
    score = rl_single.score_single_turn(
        _RewardOnlyEnv(),
        "the answer is 7",
        {"gt": "7"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert score == 2.5
    assert breakdowns == []


@pytest.mark.usefixtures("_identity_graded")
def test_a_scalar_reward_env_contributes_nothing_when_its_grading_fails_either():
    # the failure path is where the scores_breakdown gate actually bites: without it, a run whose
    # env has no named components at all would still append None per failed completion, and the
    # buffer's outage branch would then republish the LAST run's names as a flat 0 for an env that
    # never reported them.
    breakdowns: list[dict | None] = []
    score = rl_single.score_single_turn(
        _RaisingRewardOnlyEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert score == 0.0
    assert breakdowns == []


@pytest.mark.usefixtures("_identity_graded")
def test_a_failed_grading_records_none_so_it_counts_as_a_zero():
    # trl's contract: a completion that failed to grade still scored 0.0, and must pull the mean of
    # every name the OTHER completions reported down with it. dropping it silently would report the
    # surviving completions' average as if the whole generation had earned it.
    breakdowns: list[dict | None] = []
    score = rl_single.score_single_turn(
        _RaisingEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert score == 0.0
    assert breakdowns == [None]


@pytest.mark.usefixtures("_identity_graded")
def test_an_unusable_total_records_no_named_components():
    # float(total) raising IS a failed grading -- the completion scores 0.0. crediting its named
    # components would report metrics for a completion that earned nothing.
    breakdowns: list[dict | None] = []
    score = rl_single.score_single_turn(
        _BadTotalEnv(),
        "x",
        {"gt": "1"},
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
        breakdowns=breakdowns,
    )
    assert score == 0.0
    assert breakdowns == [None]


# ------------------------------- reward rpc bridge -------------------------------
def test_reward_server_round_trip():
    server, url = rl_multi.start_reward_server(lambda idx, s: float(idx) + len(s), example_count=4)
    try:
        body = json.dumps({"index": 3, "solution_str": "abcd"}).encode()
        req = urllib.request.Request(
            url + "/score", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            got = json.loads(r.read().decode())
        assert got["score"] == 7.0  # 3 + len("abcd")
    finally:
        server.shutdown()


@pytest.mark.parametrize("index", [-1, 2])
def test_reward_server_rejects_out_of_range_index_before_lookup(index):
    examples = [{"name": "first"}, {"name": "last"}]
    scored = []

    def scorer(index, solution_str):
        scored.append(examples[index]["name"])
        return 1.0

    server, url = rl_multi.start_reward_server(scorer, example_count=len(examples))
    try:
        body = json.dumps({"index": index, "solution_str": "answer"}).encode()
        req = urllib.request.Request(
            url + "/score", data=body, headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 400
        assert scored == []
    finally:
        server.shutdown()


def test_reward_bridge_lookup_failure_raises(monkeypatch):
    # the index is IN range, so the scorer's own IndexError is the ENV failing, not a bad request:
    # it is reported as a server fault carrying the cause rather than as a malformed-payload 400.
    def missing_example(idx, solution_str):
        raise IndexError(idx)

    server, url = rl_multi.start_reward_server(missing_example, example_count=100)
    try:
        monkeypatch.setenv("TEST_FLASH_VERL_REWARD_URL", url)
        ns = _rendered_reward_namespace("TEST_FLASH_VERL_REWARD_URL")
        with pytest.raises(RuntimeError, match=r"could not serve the request .*IndexError: 99"):
            ns["compute_score"](
                "flash_env",
                "answer",
                "unused",
                extra_info={
                    "index": 99,
                    "flash_rollout_identity": {
                        "optimizer_step": 1,
                        "sample_index": 99,
                        "rollout_ordinal": 0,
                        "validate": False,
                    },
                },
            )
    finally:
        server.shutdown()


def test_reward_server_reports_thread_exhaustion_as_a_server_fault():
    """Thread exhaustion is 503 WITH its cause, not 400.

    Under rollout concurrency the scoring thread pool can fail to start a thread; the handler used
    to answer 400 for that, which asserts the caller sent something wrong. Here score_episode
    cannot produce a 400 at all (it returns a well-formed result carrying `error`), so a 400 sent
    every reader to the one component provably not at fault.
    """

    def exhausted(index, solution_str):
        raise RuntimeError("can't start new thread")

    server, url = rl_multi.start_reward_server(exhausted, example_count=4)
    try:
        body = json.dumps({"index": 0, "solution_str": "answer"}).encode()
        req = urllib.request.Request(
            url + "/score", data=body, headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 503, "a resource fault must not be reported as a bad request"
        # the cause has to survive to the client: naming the failing resource is the whole point.
        assert "can't start new thread" in exc_info.value.read().decode()
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "env_error",
    [IndexError("list index out of range"), KeyError("grader internal lookup failed")],
)
def test_an_env_raising_indexerror_or_keyerror_is_a_server_fault_not_a_bad_request(env_error):
    """Classifying by exception TYPE reintroduces the bug one layer down.

    The bridge rejects a bad index with IndexError and an unknown session with KeyError, so a tuple
    of those types around the ROUTE CALL also catches the same types raised by a user env deep
    inside its own scoring -- reporting a valid request as malformed. ScoreBatcher re-raises a GRPO
    batch error into every waiter as-is, so one env KeyError would 400 every request in the batch.
    """

    def failing_env_scorer(index, solution_str):
        raise env_error

    server, url = rl_multi.start_reward_server(failing_env_scorer, example_count=4)
    try:
        req = urllib.request.Request(
            url + "/score",
            data=json.dumps({"index": 0, "solution_str": "x"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 503, (
            "the request was valid and the ENV failed; blaming the caller is the original bug"
        )
        assert type(env_error).__name__ in exc_info.value.read().decode()
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"index": 1e309, "solution_str": "x"}, "not an integer"),  # json inf -> int() overflows
        ([], "must be a json object"),  # valid json, invalid request object
        ({"solution_str": "x"}, "missing required field"),
    ],
)
def test_a_malformed_request_shape_is_a_client_error_not_a_server_fault(payload, expected):
    """A body this bridge cannot read is the caller's fault, so it must not read as unavailable."""
    server, url = rl_multi.start_reward_server(lambda i, s: 1.0, example_count=4)
    try:
        req = urllib.request.Request(
            url + "/score",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 400
        body = exc_info.value.read().decode()
        assert expected in body
        # the private class name is not something a caller can act on.
        assert "_Bad" not in body, "an internal exception class leaked into the client message"
    finally:
        server.shutdown()


def test_the_generated_single_turn_reward_module_surfaces_the_bridges_cause(monkeypatch):
    """The REAL single-turn caller must not drop the detail the server now supplies.

    `HTTPError` is a `URLError` subclass, so a single `except URLError` catches it first and never
    reads the body -- leaving "HTTP Error 503: Service Unavailable", which names no cause. Asserting
    on the server response alone would pass while the user-visible path stayed detail-free.
    """

    def exhausted(index, solution_str):
        raise RuntimeError("can't start new thread")

    server, url = rl_multi.start_reward_server(exhausted, example_count=4)
    try:
        monkeypatch.setenv("TEST_FLASH_VERL_REWARD_URL", url)
        ns = _rendered_reward_namespace("TEST_FLASH_VERL_REWARD_URL")
        with pytest.raises(RuntimeError) as exc_info:
            ns["compute_score"](
                "flash_env",
                "answer",
                "unused",
                extra_info={
                    "index": 0,
                    "flash_rollout_identity": {
                        "optimizer_step": 1,
                        "sample_index": 0,
                        "rollout_ordinal": 0,
                        "validate": False,
                    },
                },
            )
        message = str(exc_info.value)
        assert "can't start new thread" in message, "the cause never reached the training loop"
        assert "could not serve" in message
    finally:
        server.shutdown()


def test_an_unrepresentable_env_reply_is_rejected_rather_than_reported_as_a_fault():
    """A refused reply block is permanent, so it must reach the user as 400, not 503.

    The reply validators raise on a block this transcript cannot carry. A bare ValueError there
    falls to the handler's catch-all and answers 503, which the child prints as
    ``could not serve`` -- telling the reader the bridge had a capacity problem when the actual fix
    is to change what ``step_episode`` returns. Retrying produces the identical block forever, so
    this asserts the status the 400/503 split promises rather than only the raise.
    """
    bridge = rl_multi.MultiTurnBridge(
        _BridgeEnv(
            replies=[{"role": "user", "content": [{"type": "audio", "audio": "x"}]}],
            done_after=99,
        ),
        examples=[{"q": "a"}],
        env_prompts=[[{"role": "user", "content": "a"}]],
        max_turns=4,
    )
    server, url = rl_multi.start_reward_server(
        lambda i, s: 1.0, example_count=1, multi_turn_bridge=bridge
    )

    def _post(path, payload):
        request = urllib.request.Request(
            url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=10)

    try:
        _post(
            "/multiturn/start",
            {
                "index": 0,
                "session_id": "a",
                "raw_prompt": [{"role": "user", "content": "a"}],
                "prompt_ids": [],
                "image_count": 0,
                "image_digests": [],
            },
        ).close()
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post(
                "/multiturn/step",
                {
                    "session_id": "a",
                    "turn_ordinal": 0,
                    "accepted_prefix": [],
                    "response_ids": [],
                    "completion_text": "answer",
                    "image_count": 0,
                    "image_digests": [],
                },
            )
        assert exc_info.value.code == 400, (
            "an env reply this transcript cannot carry is permanent, not a capacity fault"
        )
        assert "unsupported content block" in exc_info.value.read().decode()
    finally:
        server.shutdown()
        bridge.shutdown()


def test_reward_server_still_rejects_a_malformed_request_as_a_client_error():
    """The 5xx split must not turn genuine client errors into server faults.

    A non-integer index and an unknown session id are the caller's fault at any capacity, so both
    stay 400 -- otherwise the new status would tell a caller to retry a request that can never work.
    A no-regression guard, not evidence of the split: these were 400 under the old catch-all too.
    """
    bridge = rl_multi.MultiTurnBridge(
        _BridgeEnv(),
        examples=[{"q": "a"}],
        env_prompts=[[{"role": "user", "content": "a"}]],
        max_turns=2,
    )
    server, url = rl_multi.start_reward_server(
        lambda i, s: 1.0, example_count=1, multi_turn_bridge=bridge
    )
    try:
        for path, payload in (
            ("/score", {"index": "not-an-int", "solution_str": "x"}),
            ("/multiturn/step", {"session_id": "nope", "completion_text": "x"}),
        ):
            req = urllib.request.Request(
                url + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req, timeout=10)
            assert exc_info.value.code == 400, f"{path} is a client error at any capacity"
    finally:
        server.shutdown()
        bridge.shutdown()


@pytest.mark.parametrize(
    ("status", "detail", "expected_fault"),
    [
        (503, "RuntimeError: can't start new thread", "could not serve"),
        (400, "IndexError: index 9 is outside [0, 4)", "rejected"),
    ],
)
def test_the_child_surfaces_the_bridges_error_text_not_just_the_status(
    monkeypatch, status, detail, expected_fault
):
    """`suppress(Exception)` around the raise swallowed the detail-bearing error itself.

    The bridge puts its real cause in the response body, the child decoded it, built the
    RuntimeError naming it -- and the suppress ate that raise, so the generic "returned HTTP 400"
    was the ONLY message that could ever escape. The cause must reach the user, and a 5xx must not
    read as a rejected request.
    """
    from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn

    def raise_http_error(*_a, **_k):
        payload = json.dumps({"error": detail}).encode()
        raise urllib.error.HTTPError(
            "http://bridge/multiturn/score", status, "err", {}, io.BytesIO(payload)
        )

    monkeypatch.setattr(grpo_multiturn.urllib.request, "urlopen", raise_http_error)
    with pytest.raises(RuntimeError) as exc_info:
        grpo_multiturn.post_json("http://bridge", "/multiturn/score", {})
    message = str(exc_info.value)
    assert detail in message, "the bridge's own cause never reached the caller"
    assert expected_fault in message


def test_the_child_names_the_status_when_the_body_carries_no_detail(monkeypatch):
    """An undecodable body must still produce a message naming what happened."""

    def raise_http_error(*_a, **_k):
        raise urllib.error.HTTPError(
            "http://bridge/multiturn/score", 503, "err", {}, io.BytesIO(b"not json")
        )

    monkeypatch.setattr(grpo_multiturn.urllib.request, "urlopen", raise_http_error)
    with pytest.raises(RuntimeError, match=r"could not serve .*HTTP 503 with no error detail"):
        grpo_multiturn.post_json("http://bridge", "/multiturn/score", {})


def test_shared_transport_preserves_reward_specific_network_and_decode_errors(monkeypatch):
    def raise_url_error(*_args, **_kwargs):
        raise urllib.error.URLError("bridge down")

    monkeypatch.setattr(grpo_multiturn.urllib.request, "urlopen", raise_url_error)
    with pytest.raises(
        RuntimeError,
        match=r"flash reward bridge request failed: <urlopen error bridge down>",
    ):
        grpo_multiturn.post_json("http://bridge", "/score", {}, error_style="reward")

    class InvalidResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"not-json"

    monkeypatch.setattr(
        grpo_multiturn.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: InvalidResponse(),
    )
    with pytest.raises(
        RuntimeError,
        match="flash reward bridge returned an invalid response",
    ):
        grpo_multiturn.post_json("http://bridge", "/score", {}, error_style="reward")


def test_reward_server_scorer_can_capture_samples():
    # the #607 per-step dump relies on the scoring closure capturing recent completions; verify the
    # reward-server -> scorer -> rolling-buffer path populates in order.
    captured: list = []
    lock = threading.Lock()

    def scorer(idx, sol):
        with lock:
            captured.append((sol, float(len(sol))))
            del captured[:-64]
        return float(len(sol))

    server, url = rl_multi.start_reward_server(scorer, example_count=3)
    try:
        for i in range(3):
            body = json.dumps({"index": i, "solution_str": f"c{i}"}).encode()
            req = urllib.request.Request(
                url + "/score", data=body, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=10).read()
        assert [c[0] for c in captured] == ["c0", "c1", "c2"]
    finally:
        server.shutdown()


@contextlib.contextmanager
def _vision_hook_installer(torch_module):
    """yield the rendered vision helper with a stand-in torch visible to it.

    the helper imports torch INSIDE the function body, so the stand-in has to stay in sys.modules
    for the duration of the call rather than just the exec -- which is also the property that lets
    the real shim run inside verl's child, where torch is imported long after sitecustomize.
    """
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch_module
    try:
        yield verl_child_runtime.install_vision_input_grads
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


class _FakeTensor:
    """the two attributes the hook interrogates, plus the mutation it performs."""

    def __init__(self, *, floating=True):
        self.requires_grad = False
        self._floating = floating

    def is_floating_point(self):
        return self._floating

    def requires_grad_(self, value=True):
        self.requires_grad = value
        return self


class _FakeTorch:
    Tensor = _FakeTensor


class _FakeSubmodule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)

    def forward(self, output):
        for hook in self.hooks:
            hook(self, (), output)
        return output


class _FakeVisionModel:
    """named_modules() the way the hook walks it, at the paths a real vlm exposes."""

    def __init__(self, *, patch_embed_path="visual.patch_embed"):
        self.patch_embed = _FakeSubmodule()
        self.other = _FakeSubmodule()
        self._paths = {patch_embed_path: self.patch_embed, "model.layers.0": self.other}

    def named_modules(self):
        return list(self._paths.items())


def test_the_vision_hook_marks_patch_embed_output_as_requiring_grad():
    # runs the rendered helper for real rather than asserting on its text: a hook registered on the
    # wrong submodule, or one that inspects the output incorrectly, still renders a plausible
    # string. torch is absent from the unit env, so the stand-in reproduces exactly the contract
    # the hook depends on -- `is_floating_point()` and `requires_grad_()` on a `torch.Tensor`.
    model = _FakeVisionModel()
    with _vision_hook_installer(_FakeTorch) as install:
        install(model)
        assert len(model.patch_embed.hooks) == 1
        assert model.other.hooks == [], "the hook was installed on a non-vision submodule"

        output = _FakeTensor()
        assert output.requires_grad is False, "the fixture cannot demonstrate the bug"
        assert model.patch_embed.forward(output).requires_grad is True

        # a tuple output is what a real patch-embed returns; the hook must reach inside it.
        tupled = _FakeTensor()
        model.patch_embed.forward((tupled, None))
        assert tupled.requires_grad is True

        # an integer output (ids, not activations) must be left alone: requires_grad_ on a
        # non-float tensor raises in real torch, taking the run down at the first forward.
        integral = _FakeTensor(floating=False)
        model.patch_embed.forward(integral)
        assert integral.requires_grad is False


def test_the_vision_hook_reports_when_it_finds_no_patch_embed(capsys):
    # a silent no-op is the failure mode being fixed: visual modules training on nothing while the
    # run reports success. if the path ever moves, the log line is what makes that visible.
    with _vision_hook_installer(_FakeTorch) as install:
        install(_FakeVisionModel(patch_embed_path="vision_tower.embeddings"))
    assert "no visual.patch_embed found" in capsys.readouterr().out


def test_the_vision_hook_unwraps_a_peft_model_to_find_the_vision_tower():
    # grpo trains through a peft wrapper, so named_modules() on the wrapper is prefixed. the hook
    # unwraps to the base model first; without that it would find nothing and silently no-op.
    class _PeftWrapped:
        def __init__(self, base):
            self._base = base

        def get_base_model(self):
            return self._base

        def named_modules(self):
            raise AssertionError("the wrapper's own module list must not be walked")

    base = _FakeVisionModel()
    with _vision_hook_installer(_FakeTorch) as install:
        install(_PeftWrapped(base))
    assert len(base.patch_embed.hooks) == 1


@contextlib.contextmanager
def _stubbed_modules(stubs: dict[str, types.ModuleType]):
    """Install worker-image module stubs, then RESTORE whatever was there before.

    Popping unconditionally is wrong: names such as `vllm` are also stubbed process-wide by
    tests/serving/conftest.py, so a pop here deletes that stub for every test the same xdist worker
    runs afterwards, and their `import vllm` raises ModuleNotFoundError.
    """
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_the_reentrant_shim_flips_the_flag_and_leaves_uncheckpointed_models_alone():
    # execute the rendered source against a stand-in engine: asserting on the string alone would not
    # catch a patch that never runs, or one that turns checkpointing ON for a model verl left off.
    class _Cfg:
        def __init__(self, on):
            self.enable_gradient_checkpointing = on

    class _Module:
        def __init__(self):
            self.kwargs = None
            self.input_grads = False

        def enable_input_require_grads(self):
            self.input_grads = True

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.kwargs = gradient_checkpointing_kwargs

    class _Engine:
        def __init__(self, on):
            self.model_config = _Cfg(on)
            self.module = _Module()

        def _build_module(self):
            return self.module

    module_stub = types.ModuleType("verl.workers.engine.fsdp.transformer_impl")
    module_stub.FSDPEngine = _Engine
    parents = [
        "verl",
        "verl.workers",
        "verl.workers.engine",
        "verl.workers.engine.fsdp",
    ]
    stubs = {name: types.ModuleType(name) for name in parents}
    stubs[module_stub.__name__] = module_stub
    with _stubbed_modules(stubs):
        verl_patches.install_reentrant_checkpointing(multimodal=False)
        # checkpointing on -> the flag is put back to reentrant, and the lora-frozen embeddings
        # get input grads so the checkpointed segment actually produces a backward (GRAD-001)
        engine = _Engine(True)
        built = engine._build_module()
        assert built.kwargs == {"use_reentrant": True}
        assert built.input_grads is True
        # checkpointing off -> untouched, so the shim cannot silently raise the memory profile
        off = _Engine(False)
        off_built = off._build_module()
        assert off_built.kwargs is None
        assert off_built.input_grads is False


def test_entropy_quantile_overrides_enable_verl_entropy_and_stay_off_by_default():
    # the shim reads model_output["entropy"], which verl only populates when calculate_entropy is
    # set. flash's recipe has entropy_coeff 0, so nothing else would turn it on.
    assert "actor_rollout_ref.actor.calculate_entropy=True" in rl_verl.build_verl_overrides(
        _overrides_cfg(entropy_quantile=0.2)
    )
    assert "actor_rollout_ref.actor.calculate_entropy=True" not in rl_verl.build_verl_overrides(
        _overrides_cfg()
    )


def test_resolve_grpo_inputs_no_longer_rejects_entropy_quantile():
    # the guard this replaces raised on any entropy_quantile < 1.0. the shim implements the masking,
    # so the resolver must pass the value through instead of failing the run.
    source = inspect.getsource(rl_inputs._resolve_grpo_inputs)
    assert "is not yet supported" not in source.split("entropy_quantile")[1].split("\n\n")[0]
    assert '"entropy_quantile": entropy_quantile' in source


def test_image_pad_ban_and_stop_shims_both_apply_to_the_same_method():
    # both wrap AgentLoopWorker._run_agent_loop. each guards itself with its own marker attribute,
    # and a shared marker would make whichever ran second silently no-op -- leaving either stop
    # strings or the image-pad ban missing with no error. executing both is the only way to catch
    # that: the sources look correct in isolation either way.
    import asyncio
    from types import ModuleType

    seen: dict = {}

    class _AgentLoopWorker:
        async def _run_agent_loop(self, sampling_params, *args, **kwargs):
            seen.update(sampling_params)
            return "ok"

    # both shims import the same verl module; hand them one stub so the two patches stack on the
    # very same function object, exactly as they do in the child.
    agent_loop_module = ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop_module.AgentLoopWorker = _AgentLoopWorker
    package = ModuleType("verl.experimental.agent_loop")
    package.agent_loop = agent_loop_module
    stubs = {
        "verl": ModuleType("verl"),
        "verl.experimental": ModuleType("verl.experimental"),
        "verl.experimental.agent_loop": package,
        "verl.experimental.agent_loop.agent_loop": agent_loop_module,
    }
    with _stubbed_modules(stubs):
        verl_patches.install_stop_sequences(("</answer>",))
        verl_patches.install_image_pad_ban(151655)
        asyncio.run(_AgentLoopWorker()._run_agent_loop({"temperature": 1.0}))

    assert seen["stop"] == ["</answer>"]
    assert seen["logit_bias"] == {151655: -100.0}
    # the untouched key proves each patch COPIED the dict rather than replacing it wholesale.
    assert seen["temperature"] == 1.0


def test_rollout_shims_survive_verls_real_run_agent_loop_signature():
    """the shims must tolerate the signature verl ACTUALLY calls, not a convenient stub.

    At freesolo-co/verl@32d6200d, ``trajectory`` is positional and required ``agent_name`` is
    keyword-only; the wrapper must also preserve ``**kwargs``.
    """
    import asyncio
    from types import ModuleType

    seen: dict = {}

    class _AgentLoopWorker:
        async def _run_agent_loop(
            self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs
        ):
            seen["params"] = dict(sampling_params)
            seen["trajectory"] = trajectory
            seen["agent_name"] = agent_name
            seen["trace"] = trace
            return "ok"

    agent_loop_module = ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop_module.AgentLoopWorker = _AgentLoopWorker
    package = ModuleType("verl.experimental.agent_loop")
    package.agent_loop = agent_loop_module

    # vllm is a worker-image dep too. stand in a recorder so the assertion below can prove the
    # constraint was WRAPPED: vllm accepts a raw dict, passes _verify_args(), then constrains
    # nothing -- silently, which is the whole reason render_structured_outputs_shim wraps it.
    class _StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    sampling_params_module = ModuleType("vllm.sampling_params")
    sampling_params_module.StructuredOutputsParams = _StructuredOutputsParams
    vllm_module = ModuleType("vllm")
    vllm_module.sampling_params = sampling_params_module

    stubs = {
        "verl": ModuleType("verl"),
        "verl.experimental": ModuleType("verl.experimental"),
        "verl.experimental.agent_loop": package,
        "verl.experimental.agent_loop.agent_loop": agent_loop_module,
        "vllm": vllm_module,
        "vllm.sampling_params": sampling_params_module,
    }
    with _stubbed_modules(stubs):
        verl_patches.install_stop_sequences(("</answer>",))
        verl_patches.install_image_pad_ban(151655)
        verl_patches.install_structured_outputs({"json": {"type": "object"}})
        # called exactly as verl calls it (agent_loop.py:583): two positionals plus keywords.
        asyncio.run(
            _AgentLoopWorker()._run_agent_loop(
                {"temperature": 1.0}, {"step": 0}, agent_name="tool_agent", trace=False
            )
        )

    # the arguments the shims do not own must arrive untouched.
    assert seen["trajectory"] == {"step": 0}
    assert seen["agent_name"] == "tool_agent"
    assert seen["trace"] is False
    assert seen["params"]["temperature"] == 1.0
    # and all three shims still applied.
    assert seen["params"]["stop"] == ["</answer>"]
    assert seen["params"]["logit_bias"] == {151655: -100.0}
    constraint = seen["params"]["structured_outputs"]
    assert isinstance(constraint, _StructuredOutputsParams)
    assert constraint.kwargs == {"json": {"type": "object"}}


def test_stop_sequences_gate_off_truncated_completion_masking():
    # main couples these: stop-string rollouts do not end on EOS, so masking truncated completions
    # would wrongly drop every one of them. the verl resolver must inherit that coupling, not
    # re-derive it.
    source = inspect.getsource(rl_inputs._resolve_grpo_inputs)
    assert "_worker_config.grpo_mask_truncated_completions(_t)" in source
    assert not worker_config.grpo_mask_truncated_completions(
        SimpleNamespace(stop_sequences=("</answer>",))
    )
    assert worker_config.grpo_mask_truncated_completions(SimpleNamespace(stop_sequences=()))


def _load_kl_ref_engine():
    """exec the kl-reference shim against a stub verl engine and hand back the patched class.

    the shim rebinds FSDPEngine._build_lora_module and .disable_adapter on import, so a stub that
    stands in for verl's real class is enough to exercise both halves without a gpu.
    """
    from types import ModuleType

    class _FSDPEngine:
        def __init__(self, module):
            self.module = module

        def _build_lora_module(self, module):
            return module

        def disable_adapter(self):
            raise AssertionError("the shim must replace disable_adapter, not defer to it")

    impl = ModuleType("verl.workers.engine.fsdp.transformer_impl")
    impl.FSDPEngine = _FSDPEngine
    fsdp_pkg = ModuleType("verl.workers.engine.fsdp")
    fsdp_pkg.transformer_impl = impl
    stubs = {
        "verl": ModuleType("verl"),
        "verl.workers": ModuleType("verl.workers"),
        "verl.workers.engine": ModuleType("verl.workers.engine"),
        "verl.workers.engine.fsdp": fsdp_pkg,
        "verl.workers.engine.fsdp.transformer_impl": impl,
    }
    with _stubbed_modules(stubs):
        verl_patches.install_kl_ref_adapter()
    return impl.FSDPEngine


def test_kl_ref_adapter_shim_anchors_the_reference_to_the_warm_start_adapter():
    # verl disables the adapter for the reference pass, yielding the bare base on warm starts.
    # execute all three forwards because rendered-source checks cannot catch this semantic mismatch.
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.q_proj(x)

    torch.manual_seed(0)
    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"])
    model = peft.get_peft_model(_Tiny(), config)
    # lora_B initializes to zeros, which makes a fresh adapter a no-op EQUAL to the base -- a
    # fixture left that way could not tell "anchored to sft" from "anchored to base" at all.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.1)

    x = torch.randn(2, 8)
    with torch.no_grad():
        sft_out = model(x).clone()
        with model.disable_adapter():
            base_out = model(x).clone()
    assert not torch.allclose(sft_out, base_out), "fixture cannot discriminate sft from base"

    params_before = set(dict(model.named_parameters()))
    state_before = set(model.state_dict())
    engine = _load_kl_ref_engine()(model)
    engine._build_lora_module(model)

    with torch.no_grad():
        # a training step moves only the trainable default adapter; the frozen snapshot must not
        # follow it, or the anchor drifts with the policy and constrains nothing.
        for name, param in model.named_parameters():
            if ".default." in name and "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.5)
        trained_out = model(x).clone()
        with engine.disable_adapter():
            ref_out = model(x).clone()
        after_out = model(x).clone()

    assert torch.allclose(ref_out, sft_out), "kl reference is not the warm-start adapter"
    assert not torch.allclose(ref_out, base_out), "kl reference fell back to the bare base"
    assert not torch.allclose(ref_out, trained_out), "kl reference drifted with the policy"
    # the policy forward has to come back bit-exact: the reference pass runs inside training.
    assert torch.equal(after_out, trained_out), "policy forward not restored after the reference"
    # non-persistent buffers, not a second adapter's parameters. new named_parameters would be
    # flattened by fsdp and trained by the optimizer; new state_dict keys would reach verl's merger,
    # which hand-builds the shipped adapter from every "lora_" key and derives target_modules from
    # key.split(".")[-3] -- a second adapter's keys resolve to lora_A/lora_B there.
    assert not set(dict(model.named_parameters())) - params_before
    assert not set(model.state_dict()) - state_before


def test_kl_ref_adapter_shim_refuses_to_run_without_a_snapshot():
    # both guards exist because the alternative is silent: an unpatched or half-applied snapshot
    # would leave the reference on the bare base, and the run would look completely healthy while
    # training against the wrong anchor. they must raise, never fall back.
    pytest.importorskip("torch")
    # `_load_kl_ref_engine` calls `install_kl_ref_adapter`, which imports peft. guarding only
    # torch makes this test error instead of skip in an environment that has torch but not
    # peft, so it must skip on both exactly like the sibling test above.
    pytest.importorskip("peft")
    import torch

    engine_cls = _load_kl_ref_engine()

    class _NoAdapterWeights(torch.nn.Module):
        # peft-shaped, but no ModuleDict holds the snapshot's leaves: nothing gets demoted.
        def __init__(self):
            super().__init__()
            self.peft_config = {"default": SimpleNamespace(r=4)}
            self.active_adapter = "default"

        def add_adapter(self, name, config):
            self.peft_config[name] = config

    module = _NoAdapterWeights()
    with pytest.raises(RuntimeError, match="no adapter weights to freeze"):
        engine_cls(module)._build_lora_module(module)

    class _NoSnapshot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.peft_config = {"default": SimpleNamespace(r=4)}

    with pytest.raises(RuntimeError, match="flash kl reference adapter missing"):
        engine_cls(_NoSnapshot()).disable_adapter()


def test_reasoning_parser_override_needs_both_thinking_and_a_constraint():
    # engine half. verl spreads engine_kwargs.vllm straight into AsyncEngineArgs, where
    # reasoning_parser is a real field, so this needs a plain hydra override and no shim.
    spec = {"json": {"type": "object"}}
    key = "+actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_r1"
    assert key in rl_verl.build_verl_overrides(
        _overrides_cfg(thinking=True, structured_outputs=spec)
    )
    # thinking off -> no reasoning phase to protect; no constraint -> the grammar gate never runs.
    for off in (
        {"thinking": False, "structured_outputs": spec},
        {"thinking": True, "structured_outputs": None},
    ):
        assert not [
            o
            for o in rl_verl.build_verl_overrides(_overrides_cfg(**off))
            if "reasoning_parser" in o
        ]


def test_build_verl_overrides_enable_fused_linear_ce():
    # 32k GRPO must not materialize [tokens, vocab] logits; fused torch-backend linear-CE
    # computes logprobs from hidden states in chunks (numerically exact).
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert "actor_rollout_ref.model.use_fused_kernels=True" in o
    assert "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch" in o


def test_model_revision_resolves_pinned_snapshot_for_verl():
    # model_revision no longer fails closed: prefetch pins the revision and verl gets the pinned
    # snapshot dir as model.path (a bare repo id would resolve the cached "main" ref offline).
    import inspect

    resolver_src = inspect.getsource(rl_inputs._resolve_grpo_inputs)
    assert "model_revision pinning is not yet supported" not in resolver_src
    # assert on the resolver being CALLED, not on snapshot_download's keywords appearing inline:
    # the resolution moved into _cached_model_path (shared with sft/opd), so pinning the argument
    # spelling here would only re-assert where the code happens to live today.
    run_src = inspect.getsource(rl_train.run_rl_train)
    assert '_cached_model_path(inp["model_id"], inp["model_revision"])' in run_src
    helper_src = inspect.getsource(sft_train._cached_model_path)
    assert "local_files_only=True" in helper_src


def test_unpinned_model_also_resolves_a_snapshot_dir_for_verl():
    """An EMPTY model_revision must resolve a real snapshot dir too, not pass the bare repo id.

    verl runs with ``HF_HUB_OFFLINE=1``; resolve every branch through ``_cached_model_path`` so a
    missing cache becomes retriable before a paid worker fails permanently.
    """
    import ast
    import inspect
    import textwrap

    fn = ast.parse(textwrap.dedent(inspect.getsource(rl_train.run_rl_train))).body[0]

    def _calls(node):
        return [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_cached_model_path"
        ]

    assert _calls(fn), "run_rl_train must resolve the model path through _cached_model_path"
    # every call sits at statement level in the function body, so none is guarded by a revision test
    guarded = [c for stmt in fn.body if isinstance(stmt, ast.If) for c in _calls(stmt)]
    assert not guarded, "the resolver must run for an unpinned revision too, not only a pinned one"


def test_pinned_snapshot_dir_is_what_reaches_verl_model_path():
    # resolving the pinned snapshot is only half the invariant: the RESOLVED path has to be the value
    # verl gets as actor_rollout_ref.model.path. passing inp["model_id"] here would resolve the cached
    # "main" ref offline and silently train the wrong commit, with every other assertion still green.
    # read the call with ast, not a substring: the argument list is multi-line and reformats.
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(rl_train._configure_rl_child)))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_build_verl_training_cfg"
    ]
    assert len(calls) == 1
    model_path = next(k for k in calls[0].keywords if k.arg == "model_path")
    assert isinstance(model_path.value, ast.Name)
    assert model_path.value.id == "model_path_for_verl"


# ------------------------------- resume (VERL-018) -------------------------------
def test_build_verl_overrides_enables_resume_mode():
    # without resume_mode=auto verl ignores a staged checkpoint and silently restarts at step 0.
    o = rl_verl.build_verl_overrides(_overrides_cfg())
    assert "trainer.resume_mode=auto" in o


def test_run_rl_train_wires_direct_gradient_evidence_into_the_publish_path():
    entry_source = inspect.getsource(rl_train.run_rl_train)
    verdict_source = inspect.getsource(rl_runner._validate_rl_child)
    metrics_source = inspect.getsource(rl_runner._ingest_step_metrics)
    export_source = inspect.getsource(rl_runner._export_final_adapter)
    assert "_validate_rl_child(" in entry_source
    assert "_check_grpo_had_a_gradient(" in verdict_source
    assert "state.grad_norms," in verdict_source
    assert "expected_steps=range(" in verdict_source
    assert entry_source.index("_validate_rl_child(") < entry_source.index("_export_final_adapter(")
    assert "export_peft_adapter(" in export_source
    assert 'step_metrics.get("grad_norm")' in metrics_source
    assert 'step_metrics.get("advantage_max")' in metrics_source
    assert 'step_metrics.get("advantage_min")' in metrics_source
    assert "_finalize_advantage_evidence(state, resume_step, expected_steps)" in verdict_source


def test_train_notes_report_whether_the_run_resumed():
    # without this a resumed run is indistinguishable from a fresh one in train_meta (trl reports it).
    inp = _notes_inp()
    common = _notes_common()
    fresh = rl_verl._build_verl_train_notes(inp, **common)
    assert fresh["resumed"] is False
    resumed = rl_verl._build_verl_train_notes(inp, **common, resumed=True)
    assert resumed["resumed"] is True


def _notes_inp():
    return {
        "epochs": 2,
        "group_size": 4,
        "kl_coef": 0.0,
        "entropy_quantile": None,
        "stop_sequences": (),
        "structured_outputs": None,
        "temperature": 1.0,
        "top_p": 1.0,
        "ppo_epochs": 1,
        "model_id": "Qwen/Qwen3-4B",
        "verl_total_epochs": 3,
        "seed": 7,
        "max_completion": 512,
        "prompts_per_step": 8,
        "engine_len": 4096,
    }


def _notes_common():
    return {"steps_run": 3, "retained_prompts": 8, "reward_history": [0.5], "loss_curve": [0.1]}


def _identity_summary(identities):
    digest = hashlib.sha256()
    for identity in sorted(
        identities,
        key=lambda value: (
            value["optimizer_step"],
            value["sample_index"],
            value["rollout_ordinal"],
            value["validate"],
        ),
    ):
        digest.update(
            (
                f"{identity['optimizer_step']}:{identity['sample_index']}:"
                f"{identity['rollout_ordinal']}:{int(identity['validate'])}\n"
            ).encode("ascii")
        )
    return {"count": len(identities), "sha256": digest.hexdigest()}


def test_successful_child_validation_publishes_exact_rollout_identity_evidence_in_notes():
    from flash.engine.worker.train.rl.rollout.identity import RolloutIdentityLedger

    ledger = RolloutIdentityLedger(1, 2)
    expected = [
        {
            "optimizer_step": 1,
            "sample_index": 0,
            "rollout_ordinal": ordinal,
            "validate": False,
        }
        for ordinal in range(2)
    ]
    ledger.register(expected)
    for identity in reversed(expected):
        ledger.record(identity, 0)
    ledger.seal(1)

    state = rl_runner._StepMetricState()
    state.reward_history.append(0.5)
    state.adv_spread_history.append(1.0)
    state.advantage_bounds[1] = (-0.25, 0.75)
    state.grad_norms[1] = 0.25
    runtime = SimpleNamespace(identity_ledger=ledger)
    rl_runner._validate_rl_child(0, state, 0, 1, None, reward_runtime=runtime)

    terminal_source = inspect.getsource(rl_train._write_terminal_metadata)
    assert "rollout_identity_evidence=state.rollout_identity_evidence" in terminal_source
    assert "advantage_spread_history=state.adv_spread_history" in terminal_source
    assert "advantage_bounds=state.advantage_bounds_evidence" in terminal_source
    assert "grad_norm_evidence=state.grad_norm_evidence" in terminal_source
    notes = rl_verl._build_verl_train_notes(
        _notes_inp(),
        **_notes_common(),
        rollout_identity_evidence=state.rollout_identity_evidence,
        advantage_spread_history=state.adv_spread_history,
        advantage_bounds=state.advantage_bounds_evidence,
        grad_norm_evidence=state.grad_norm_evidence,
    )
    assert notes["rollout_identity_evidence"] == {
        "steps": [
            {
                "optimizer_step": 1,
                "registered": _identity_summary(expected),
                "observed": _identity_summary(expected),
            }
        ],
        "validation": [],
    }
    assert notes["advantage_spread_history"] == [1.0]
    assert notes["advantage_bounds"] == [{"step": 1, "min": -0.25, "max": 0.75, "spread": 1.0}]
    assert notes["grad_norm_evidence"] == [{"step": 1, "grad_norm": 0.25}]


def test_already_complete_resume_finalizes_empty_rollout_identity_evidence():
    from flash.engine.worker.train.rl.rollout.identity import RolloutIdentityLedger

    state = rl_runner._StepMetricState(resume_step=5)
    state.set_prior_positive_step(3, checkpoint_step=5)
    runtime = SimpleNamespace(identity_ledger=RolloutIdentityLedger(1, 2))
    rl_runner._validate_rl_child(0, state, 5, 5, None, reward_runtime=runtime)
    assert state.rollout_identity_evidence == {"steps": [], "validation": []}
    assert state.adv_spread_history == []
    assert state.advantage_bounds_evidence == []
    assert state.grad_norm_evidence == []
    notes = rl_verl._build_verl_train_notes(
        _notes_inp(),
        **_notes_common(),
        advantage_spread_history=state.adv_spread_history,
        advantage_bounds=state.advantage_bounds_evidence,
        grad_norm_evidence=state.grad_norm_evidence,
    )
    assert notes["advantage_spread_history"] == []
    assert notes["advantage_bounds"] == []
    assert notes["grad_norm_evidence"] == []


def test_train_notes_carry_the_trl_observability_fields():
    # the console is uploaded only on FAILURE, so a successful run's train_meta is the sole record
    # of how it ran. the retired trl path reported these; without them a verl run cannot be compared to a
    # trl one, and the fp8-kv decision (resolved per-card at runtime) leaves no trace at all.
    notes = rl_verl._build_verl_train_notes(
        _notes_inp(),
        **_notes_common(),
        download_seconds=12.5,
        device_peak_gpu_gb=71.25,
        fp8_kv=True,
        wandb_project="acme",
        wandb_run_name="flash-rl-run123",
    )
    assert notes["download_seconds"] == 12.5
    assert notes["vllm_kv_cache_dtype"] == "fp8"
    assert notes["wandb_project"] == "acme"
    assert notes["wandb_run_name"] == "flash-rl-run123"
    # verl trains out-of-process, so nvidia-smi is the only reading that sees the trainer: both keys
    # carry the same device figure rather than a torch-allocated subset that would read ~0 here.
    assert notes["peak_gpu_gb"] == 71.25
    assert notes["device_peak_gpu_gb"] == 71.25
    # trl counts generated tokens from a padded upper bound; verl uses observed response lengths.
    # without the flag the two backends' token counts read as comparable when they are not.
    assert notes["gen_tokens_is_upper_bound"] is False


def test_train_notes_report_bf16_kv_when_fp8_did_not_engage():
    # fp8 is gated on cc>=8.9 AND a non-gdn model, so "requested" and "engaged" are not the same
    # thing. reporting fp8 unconditionally would claim a memory saving the run never got.
    notes = rl_verl._build_verl_train_notes(_notes_inp(), **_notes_common(), fp8_kv=False)
    assert notes["vllm_kv_cache_dtype"] is None


def test_train_notes_omit_wandb_identity_when_wandb_is_off():
    # verl logs from its own interpreter, so flash's in-process wandb.run is empty on this path and
    # the names come from the config. recording them when the logger is off would point a reader at
    # a dashboard run that was never created.
    notes = rl_verl._build_verl_train_notes(_notes_inp(), **_notes_common())
    assert notes["wandb_project"] is None
    assert notes["wandb_run_name"] is None
    # a sampler that never saw a card must not report a fabricated zero-gb peak.
    assert notes["peak_gpu_gb"] is None
    assert notes["device_peak_gpu_gb"] is None


def test_verl_grpo_logs_to_the_runs_own_wandb_project_and_name():
    # a hardcoded project/experiment pair lands every grpo run in one wandb experiment, so
    # concurrent runs overwrite each other's curves and an explicit [wandb] project is ignored. the
    # sft and opd verl backends already resolve both from the spec.
    o = rl_verl.build_verl_overrides(
        _overrides_cfg(project_name="acme", experiment_name="flash-rl-run123")
    )
    assert "trainer.project_name=acme" in o
    assert "trainer.experiment_name=flash-rl-run123" in o
    assert "trainer.project_name=flash_verl" not in o
    assert "trainer.experiment_name=grpo" not in o


def test_verl_grpo_wandb_names_survive_hydra_special_characters():
    # a run name is user-settable via [wandb] run_name; an unquoted '=' or ',' would split the
    # override and hydra would compose a different key entirely.
    o = rl_verl.build_verl_overrides(_overrides_cfg(experiment_name="run=a,b"))
    assert 'trainer.experiment_name="run=a,b"' in o


def test_train_notes_record_the_batch_shape_one_step_consumed():
    # the retired trl path reported the batch shape, so without it a verl run's reward curve cannot be read
    # against a trl one: the same step count at a different batch size is a different experiment.
    notes = rl_verl._build_verl_train_notes(_notes_inp(), **_notes_common())
    assert notes["max_completion_len"] == 512
    assert notes["prompts_per_step"] == 8
    # one optimizer step still sees the whole batch under data parallelism: use_dynamic_bsz
    # token-balances the same global batch across the dp ranks instead of dropping a remainder.
    assert notes["generations_per_step"] == 8 * 4


def test_train_notes_report_token_bounded_batching_as_unset_not_fabricated():
    # trl fixes a per-device SEQUENCE count; verl bounds the backward pass by tokens, so a
    # micro-batch holds however many sequences fit and varies step to step. reporting a number here
    # would read as directly comparable to trl's when nothing enforces it.
    notes = rl_verl._build_verl_train_notes(_notes_inp(), **_notes_common())
    assert notes["per_device_train_batch_size"] is None
    assert notes["gradient_accumulation_steps"] is None
    # the bound that IS enforced gets recorded in their place.
    assert notes["ppo_max_token_len_per_gpu"] == 4096
    # trl pins vllm's prefill batch because it hardcodes 4096; this path sets no such override.
    assert notes["vllm_max_num_batched_tokens"] is None


# ------------------- capability guards: the specs verl grpo refuses -------------------
# each test pins one rejection message so deleting a guard cannot pass via another raise.


def _capability_env(*, multi_turn=False, is_tool_env=False, image_uri=None, example_count=8):
    """a minimal single-turn text env, optionally flipped to a shape verl grpo handles differently.

    ``image_uri`` must be a source the normalizer accepts offline (a data uri): a remote https url
    is rejected outright unless the trusted-dataset opt-in is set, so it would fail the test for
    the wrong reason.
    """

    class _Env:
        package_root = None

        def __init__(self):
            self.multi_turn = multi_turn
            self.is_tool_env = is_tool_env
            self.max_turns = 3 if multi_turn else 0

        def dataset(self):
            return [{"index": i} for i in range(example_count)]

        # the four calls the multi-turn bridge drives an env through. defined unconditionally so a
        # test can delete one and assert the capability gate catches it.
        def new_rollout_state(self, ex, prepared_prompt):
            return {}

        def record_model_turn(self, state, text):
            return None

        def env_reply(self, state):
            return [{"role": "user", "content": "reply"}]

        def rollout_done(self, state):
            return True

        def prompt_messages(self, ex):
            if image_uri:
                # record_has_images matches an image content BLOCK, so build the real shape rather
                # than a sentinel key the resolver would not see.
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"question {ex['index']}"},
                            {"type": "image_url", "image_url": {"url": image_uri}},
                        ],
                    }
                ]
            return [{"role": "user", "content": f"question {ex['index']}"}]

    return _Env()


def _capability_image_uri():
    """a 2x2 png as a data uri, the smallest source normalize_image_source accepts offline."""
    import base64
    import io

    image_module = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    image_module.new("RGB", (2, 2), (255, 0, 0)).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


class _CapabilityTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        # renders each message's content, the way a real template does. a constant string would
        # make the multi-turn glue probe unfindable and fail the pre-rollout template gate -- which
        # is the gate working, not the code under test being wrong.
        rendered = "".join(f"<|{m['role']}|>{m['content']}<eos>" for m in messages)
        return rendered + ("<|assistant|>" if kwargs.get("add_generation_prompt") else "")

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[1])

    def convert_tokens_to_ids(self, token):
        return 151655 if token == "<|image_pad|>" else 0


class _CapabilityProcessor:
    """the shape rl_train asks of AutoProcessor on a multimodal job.

    it must expose ``tokenizer`` (the resolver reads pad/eos off it), render a chat template, and
    tokenize text+images together. the returned ids are what the prompt-budget filter measures, so
    a test that wants a row dropped controls it through the length here.
    """

    image_token_id = 151655

    def __init__(self, expanded_len=4):
        self.tokenizer = _CapabilityTokenizer()
        self.expanded_len = expanded_len
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, text=None, images=None, **kwargs):
        self.calls.append({"text": text, "images": images})
        return {"input_ids": [[1] * self.expanded_len]}


def _capability_resolve(
    monkeypatch,
    env,
    train=None,
    overrides=None,
    processor=None,
    model="Qwen/Qwen3.5-9B",
    gpu_count=1,
):
    """run the resolver against one env, with everything else on the supported path."""
    from flash.core.spec import JobSpec
    from flash.engine.worker.train.rl.launch import inputs as rl_inputs

    _Tokenizer = _CapabilityTokenizer

    # replace the lazy transformers module at the import boundary the resolver uses. assigning an
    # attribute on transformers' lazy module can be ignored by its custom from-import path.
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(
                from_pretrained=lambda *a, **k: processor or _CapabilityProcessor()
            )
        ),
    )

    spec = JobSpec.from_dict(
        {
            "model": model,
            "algorithm": "grpo",
            "train": {"prompts_per_step": 4, "epochs": 1, **(train or {})},
            "gpu": {"count": gpu_count},
        }
    )
    monkeypatch.setattr(rl_inputs._worker_state, "JOB_SPEC", spec)
    monkeypatch.setattr(rl_inputs._worker_state, "SEED", 42)
    monkeypatch.setattr(rl_inputs._worker_state, "THINKING", False)
    monkeypatch.setattr(rl_inputs._worker_state, "require_active_env", lambda: env)
    monkeypatch.setattr(rl_inputs._worker_config, "grpo_overrides", lambda: dict(overrides or {}))
    monkeypatch.setattr(
        rl_inputs._worker_config, "grpo_mask_truncated_completions", lambda _train: False
    )
    monkeypatch.setattr(rl_inputs._worker_hf, "load_tokenizer", lambda *a, **k: _Tokenizer())
    monkeypatch.setattr(rl_inputs, "seed_training_rngs", lambda seed: None)
    monkeypatch.setattr(rl_inputs, "model_max_position_embeddings", lambda *a, **k: 32768)
    # offline: the multi_turn branch reads the model's config + generation_config to build the
    # halting set. Those are local_files_only reads, but they still need the model in the hf cache,
    # so on a machine without one (a clean CI runner) they raise instead of resolving. The tests
    # using this helper assert on turn/agent-loop wiring, not on which ids halt a rollout.
    monkeypatch.setattr(
        rl_inputs, "generation_eos_from_cached_config", lambda *a, **k: frozenset({151645})
    )
    return rl_inputs._resolve_grpo_inputs()


def test_capability_guard_rejects_tool_env(monkeypatch):
    # trl grpo hands tool schemas AND callables to the trainer; verl gets neither, so a tool env
    # would train against completions that never call a tool.
    with pytest.raises(RuntimeError, match="function-calling tool environments"):
        _capability_resolve(monkeypatch, _capability_env(is_tool_env=True))


def test_multi_turn_env_resolves_and_selects_the_flash_agent_loop(monkeypatch):
    # the inverse of the guard this replaces: a multi-turn env used to be refused outright and fall
    # back to trl. it must now resolve, and the resolution must reach the ONE override that decides
    # which agent loop verl runs -- on the stock single_turn_agent the episode would end after the
    # first assistant turn and every environment reply would be dropped.
    inp = _capability_resolve(monkeypatch, _capability_env(multi_turn=True))
    assert inp["multi_turn"] is True
    assert inp["max_turns"] == 3
    cfg = rl_verl._build_verl_training_cfg(
        inp,
        ce_backend="torch",
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_path=inp["model_id"],
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    o = rl_verl.build_verl_overrides(cfg)
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=flash_grpo_multi_turn" in o


def test_single_turn_env_leaves_the_agent_loop_on_verl_default(monkeypatch):
    # the override must be GATED: emitting it on a single-turn job would route text rollouts
    # through the multi-turn bridge, which has no episode state for them.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["multi_turn"] is False
    assert not [
        o for o in rl_verl.build_verl_overrides(_overrides_cfg()) if "default_agent_loop" in o
    ]


def test_multi_turn_env_missing_a_rollout_method_is_refused(monkeypatch):
    # the bridge calls exactly four env methods; a missing one would otherwise surface mid-rollout
    # on the first episode, after the gpu time is already spent.
    env = _capability_env(multi_turn=True)
    del type(env).env_reply
    with pytest.raises(RuntimeError, match="missing required rollout methods"):
        _capability_resolve(monkeypatch, env)


def test_multi_turn_env_without_a_turn_limit_is_refused(monkeypatch):
    # an unbounded episode cannot be budgeted: the response tensor is sized from the turn limit and
    # a runaway env would loop until the engine context is exhausted every single rollout.
    env = _capability_env(multi_turn=True)
    env.max_turns = 0
    with pytest.raises(RuntimeError, match="bounded turn limit"):
        _capability_resolve(monkeypatch, env)


def test_resolver_admits_image_prompts_and_carries_the_processor(monkeypatch):
    # the inverse of the guard this replaces: an image env used to be refused outright and fall
    # back to trl. it must now resolve, and it must resolve through a PROCESSOR -- a bare tokenizer
    # would under-count the prompt by the whole placeholder expansion. asserting the processor is
    # carried out (not merely that resolve returned) is what pins the multimodal path: a resolver
    # that quietly took the text branch would still return a valid dict.
    processor = _CapabilityProcessor()
    inp = _capability_resolve(
        monkeypatch,
        _capability_env(image_uri=_capability_image_uri()),
        processor=processor,
    )
    assert inp["multimodal"] is True
    assert inp["processor"] is processor
    assert inp["image_pad_token_id"] == 151655
    # every prompt was measured through the processor, with its decoded image attached: a call
    # carrying images=None would mean the pixels never reached the token count.
    assert len(processor.calls) == len(inp["prompts"])
    assert all(len(call["images"] or []) == 1 for call in processor.calls)


def test_multimodal_prompts_carry_descriptors_and_rendered_text(monkeypatch):
    # the parquet writer needs each prompt's image DESCRIPTORS and the thinking probe needs the
    # RENDERED text. both are produced only on the multimodal branch, so a branch that returned
    # bare messages would break the writer downstream rather than here.
    inp = _capability_resolve(monkeypatch, _capability_env(image_uri=_capability_image_uri()))
    first = inp["prompts"][0]
    assert len(first["images"]) == 1
    assert first["rendered"] == "prompt"
    # the image block is normalized to a bare {"type": "image"} marker: the source moved into the
    # descriptor list, which is what _materialize_verl_images later writes to disk.
    blocks = first["prompt"][0]["content"]
    assert {"type": "image"} in blocks


def test_top_level_record_image_reaches_actor_and_environment_prompts():
    from flash.engine.worker.train.rl.launch import inputs as rl_inputs

    prompts = rl_inputs._build_grpo_prompts(
        [{"image": _capability_image_uri()}],
        [[{"role": "user", "content": "question"}]],
        True,
        _CapabilityProcessor(),
        _CapabilityTokenizer(),
        None,
        32,
    )

    assert any(block == {"type": "image"} for block in prompts[0]["prompt"][0]["content"])
    assert any(block == {"type": "image"} for block in prompts[0]["env_prompt"][0]["content"])


def test_text_prompts_freeze_qwen38_reasoning_fields_for_child_parity():
    from flash.engine.worker.train.rl.launch import inputs as rl_inputs

    prompts = rl_inputs._build_grpo_prompts(
        [{}],
        [
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "<think>old</think>answer"},
                {"role": "user", "content": "question"},
            ]
        ],
        False,
        None,
        _CapabilityTokenizer(),
        None,
        32,
    )

    expected = {
        "role": "assistant",
        "reasoning_content": "old",
        "content": "answer",
    }
    assert prompts[0]["prompt"][1] == expected
    assert prompts[0]["env_prompt"][1] == expected


def test_multimodal_prompts_preserve_reasoning_content_through_processor_and_child_transport(
    tmp_path,
):
    from flash.engine.worker.train.rl.launch import inputs as rl_inputs
    from flash.engine.worker.train.rl.launch.verl_config import (
        build_verl_dataset_rows,
        write_verl_grpo_parquet,
    )

    class _RecordingProcessor(_CapabilityProcessor):
        def __init__(self):
            super().__init__()
            self.template_messages = None

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["preserve_thinking"] is False
            self.template_messages = messages
            return "prompt"

    processor = _RecordingProcessor()
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "reasoning_content": "old", "content": "answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": _capability_image_uri()}},
            ],
        },
    ]

    prompts = rl_inputs._build_grpo_prompts(
        [{}], [messages], True, processor, processor.tokenizer, None, 32
    )
    row = build_verl_dataset_rows([prompts[0]["prompt"]], [0], [""], [prompts[0]["images"]])[0]

    expected = {"role": "assistant", "reasoning_content": "old", "content": "answer"}
    assert processor.template_messages[1]["reasoning_content"] == "old"
    assert processor.template_messages[1]["content"] == [{"type": "text", "text": "answer"}]
    assert prompts[0]["env_prompt"][1] == processor.template_messages[1]
    assert row["prompt"][1] == expected

    datasets = pytest.importorskip("datasets")
    path = tmp_path / "reasoning.parquet"
    write_verl_grpo_parquet([row], str(path))
    restored = datasets.Dataset.from_parquet(str(path))[0]["prompt"]
    assert restored[1] == expected


def test_multimodal_budget_filter_measures_the_expanded_prompt(monkeypatch):
    # verl RAISES on an over-budget multimodal prompt instead of truncating, so this filter is the
    # only thing between a long image prompt and a dead run. the tokenizer says 1 token; the
    # processor says the prompt is huge. the filter must believe the processor -- if it measured
    # with the tokenizer every row would be admitted and the run would die mid-rollout.
    processor = _CapabilityProcessor(expanded_len=10**6)
    with pytest.raises(ValueError, match="every training prompt exceeds"):
        _capability_resolve(
            monkeypatch,
            _capability_env(image_uri=_capability_image_uri()),
            processor=processor,
        )


def test_text_env_resolves_without_building_a_processor(monkeypatch):
    # the control for the three tests above: a text-only job must NOT pay for a processor, and must
    # not carry an image-pad ban into its rollouts. without this a resolver hardcoded to the
    # multimodal branch would pass every multimodal test above.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["multimodal"] is False
    assert inp["processor"] is None
    assert inp["image_pad_token_id"] is None


def test_kl_anchored_warm_start_is_accepted(monkeypatch, tmp_path):
    # verl's kl reference is the bare base whenever lora is active, so warm-start + kl used to be
    # refused: the penalty would drag the policy away from the sft adapter the run was told to
    # continue. the deferred kl-reference plugin patch anchors the reference to that adapter instead,
    # so the combination now resolves. the kl coefficient arrives through grpo_overrides, so it must go
    # through the helper rather than being patched separately.
    import flash.engine.worker.model.adapter as _adapter_mod

    adapter_dir = tmp_path / "warmstart"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "exclude_modules": r"^(?!model\.language_model(?:\.|$)).*$",
            }
        )
    )
    monkeypatch.setattr(_adapter_mod, "_download_adapter", lambda ref: str(adapter_dir))

    inp = _capability_resolve(
        monkeypatch,
        _capability_env(),
        train={"init_from_adapter": "org/some-sft-adapter"},
        overrides={"kl_penalty_coef": 0.1},
    )
    assert inp["warmstart_adapter"]
    assert inp["kl_coef"] == pytest.approx(0.1)


def test_grpo_checks_warmstart_modality_after_resolving_image_prompts(monkeypatch, tmp_path):
    import flash.engine.worker.model.adapter as adapter_mod

    adapter_dir = tmp_path / "warmstart"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "exclude_modules": r"^(?!model\.language_model(?:\.|$)).*$",
            }
        )
    )
    monkeypatch.setattr(adapter_mod, "_download_adapter", lambda ref: str(adapter_dir))

    with pytest.raises(ValueError, match="multimodal run cannot continue a text-only adapter"):
        _capability_resolve(
            monkeypatch,
            _capability_env(image_uri=_capability_image_uri()),
            train={"init_from_adapter": "org/text-only-adapter"},
        )


def test_35b_grpo_warm_start_requires_fused_expert_targets(monkeypatch, tmp_path):
    import flash.engine.worker.model.adapter as adapter_mod

    adapter_dir = tmp_path / "warmstart"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 32,
                "lora_alpha": 64,
                "exclude_modules": r"^(?!model\.language_model(?:\.|$)).*$",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter_mod, "_download_adapter", lambda ref: str(adapter_dir))

    with pytest.raises(ValueError, match="omits required expert targets"):
        _capability_resolve(
            monkeypatch,
            _capability_env(),
            train={"init_from_adapter": "org/pre-expert-adapter"},
            model="Qwen/Qwen3.6-35B-A3B",
            gpu_count=2,
        )


def test_per_turn_credit_assignment_is_accepted_on_single_turn_envs(monkeypatch, capsys):
    # per_turn only diverges from per_episode when there is more than one assistant turn to credit.
    # the multi-turn/tool guard above already rejects every env that could get there, so anything
    # reaching here is single-turn and the two modes are the same objective. rejecting a key that
    # is merely redundant would break configs that are asking for nothing wrong.
    inp = _capability_resolve(
        monkeypatch,
        _capability_env(),
        train={"credit_assignment": "per_turn"},
    )
    assert inp["max_prompt_len"] > 0
    assert "equivalent to per_episode" in capsys.readouterr().out


def test_default_credit_assignment_does_not_log_an_equivalence_note(monkeypatch, capsys):
    # the control: the note above must be tied to an explicitly non-default value, not printed on
    # every run. without this a hardcoded print would still satisfy the test above.
    _capability_resolve(
        monkeypatch,
        _capability_env(),
        train={"credit_assignment": "per_episode"},
    )
    assert "equivalent to per_episode" not in capsys.readouterr().out


def test_capability_guards_admit_the_supported_single_turn_text_env(monkeypatch):
    # the control: with none of the four shapes present the resolver must run to completion, so a
    # guard that fires on every env would fail here instead of passing the four tests above.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["max_prompt_len"] > 0


# ---------------------- multi-turn child wiring ----------------------
# the parent resolves the episode contract; the child enforces it. everything between the two
# crosses one of exactly three channels -- env vars, copied-in modules, and the bridge's http
# routes -- and none of them is type-checked at either end. a typo'd key or a dropped copy does not
# fail here: it fails on the first rollout of a paid run, after the engine is already up.


def _multi_turn_inp(**over):
    """the keys multi_turn_child_env reads, at their resolved types."""
    return {
        "max_turns": 4,
        "engine_len": 8192,
        "max_completion": 512,
        "stop_sequences": ("</answer>",),
        "eos_token_ids": frozenset({151645, 151643}),
        **over,
    }


def test_multi_turn_child_env_carries_every_variable_the_loop_reads():
    # the child reads these by name out of os.environ and has no defaults for the first three: a
    # missing FLASH_VERL_MULTITURN_URL/MAX_TURNS/MAX_MODEL_LEN raises KeyError inside the rollout.
    # asserted against the loop's OWN source rather than a hardcoded list, so renaming a key on one
    # side and not the other fails here instead of on the first episode.
    from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn

    emitted = rl_multi.multi_turn_child_env(
        _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=False
    )
    read_by_child = set(
        re.findall(
            r"os\.environ(?:\.get)?[\[(]\"(FLASH_VERL_[A-Z_]+)\"", inspect.getsource(grpo_multiturn)
        )
    )
    assert read_by_child, "the loop reads no FLASH_VERL_* variable; this test found nothing to pin"
    assert read_by_child <= set(emitted), (
        f"the child reads variables the parent never sets: {sorted(read_by_child - set(emitted))}"
    )


def test_grpo_child_environment_registers_exactly_one_plugin():
    source = inspect.getsource(rl_runner._build_rl_child_env)
    assert 'env_for_verl["VERL_USE_EXTERNAL_MODULES"] = "flash_grpo_plugin"' in source
    assert "flash_grpo_plugin.py" in {flat for _, flat in rl_multi.GRPO_CHILD_MODULES}


def test_multi_turn_child_env_serializes_values_the_child_can_parse_back():
    # every value crosses as a string. the child json-loads two of them and int()s two others, so a
    # repr() or a str(frozenset) here would raise mid-rollout rather than at launch.
    emitted = rl_multi.multi_turn_child_env(
        _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=True
    )
    assert all(isinstance(value, str) for value in emitted.values())
    assert int(emitted["FLASH_VERL_MAX_TURNS"]) == 4
    assert int(emitted["FLASH_VERL_MAX_MODEL_LEN"]) == 8192
    assert int(emitted["FLASH_VERL_MAX_COMPLETION_TOKENS"]) == 512
    assert json.loads(emitted["FLASH_VERL_STOP_SEQUENCES"]) == ["</answer>"]
    # sorted, not set-ordered: the child compares against this list every turn and an unstable order
    # would make halting depend on hash seed.
    assert json.loads(emitted["FLASH_VERL_EOS_TOKEN_IDS"]) == [151643, 151645]
    assert emitted["FLASH_VERL_THINKING"] == "1"
    assert (
        rl_multi.multi_turn_child_env(
            _multi_turn_inp(), reward_url="http://127.0.0.1:9/", thinking=False
        )["FLASH_VERL_THINKING"]
        == "0"
    )


def test_multi_turn_child_modules_are_copied_under_the_names_they_import_each_other_by(tmp_path):
    # each module falls back to a flat `flash_`-prefixed import of the next one. copying a file
    # under the wrong name leaves that fallback unresolvable, and the child's ImportError arrives
    # inside verl's plugin loader where it reads as a verl problem.
    written = rl_multi.copy_grpo_child_modules(str(tmp_path))
    names = {os.path.basename(path) for path in written}
    assert names == {name for _, name in rl_multi.GRPO_CHILD_MODULES}
    imported = set()
    for path in written:
        source = Path(path).read_text()
        assert source
        # every copy must parse standalone in the child interpreter.
        ast.parse(source)
        imported |= set(re.findall(r"from (flash_[a-z_]+) import", source))
    imported |= set(re.findall(r"from (flash_[a-z_]+) import", rl_reward.render_reward_module()))
    # every flat import target, including the generated reward module's transport, must be copied.
    assert imported <= {name.removesuffix(".py") for name in names}


def test_grpo_child_modules_do_not_fall_back_to_flash_in_the_isolated_child(tmp_path):
    for path in rl_multi.copy_grpo_child_modules(str(tmp_path)):
        source = Path(path).read_text()
        assert "except ImportError" not in source or "from flash." not in source
        ast.parse(source)


def test_the_shared_builder_is_the_sole_owner_of_the_grpo_child_path(monkeypatch, tmp_path):
    sentinel_path = "shared-builder-pythonpath"
    calls = []

    def build_child_env(*, shim_dir, wandb_enabled):
        calls.append((shim_dir, wandb_enabled))
        return {"PYTHONPATH": sentinel_path}

    monkeypatch.setattr(rl_runner, "_build_verl_child_env", build_child_env)
    files = {
        "shim_dir": str(tmp_path),
        "rank_device_claims": str(tmp_path / "rank_device_claims.txt"),
        "plugin_config_path": str(tmp_path / "flash_grpo_plugin_config.json"),
    }
    child = rl_runner._build_rl_child_env({"multi_turn": False}, files, [], "http://127.0.0.1:9/")

    assert calls == [(str(tmp_path), False)]
    assert child["PYTHONPATH"] == sentinel_path
    assert child["FLASH_GRPO_PLUGIN_CONFIG_PATH"] == files["plugin_config_path"]


# ---------------------- multi-turn per-turn generation cap ----------------------
def test_the_child_caps_each_turn_at_max_completion_tokens_not_the_whole_episode():
    # parity with the retired trl driver, which passed per_turn_max_tokens=max_completion into
    # _turn_budget. without the cap the first turn may spend the ENTIRE episode budget: the other
    # two bounds are transcript-wide, so a 4096-token engine window lets turn one generate 4096
    # tokens and leaves nothing for the rest of the episode. asserted on the child's own source
    # because the alternative is a full engine rollout to observe one min().
    from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn

    body = " ".join(inspect.getsource(grpo_multiturn).split())
    assert 'max_completion_tokens = int(os.environ["FLASH_VERL_MAX_COMPLETION_TOKENS"])' in body
    # the three budgets now live in _EpisodeTranscript.turn_budget, which the turn loop calls once
    # per turn. same min() over the same three quantities, spelled against the transcript's own
    # accumulators rather than loop locals.
    assert (
        "return min( max_completion_tokens, self.max_model_len - len(self.prefix_ids), "
        "self.response_capacity - len(self.response_ids), )" in body
    ), "the per-turn cap is not one of the three budgets bounding a turn"
    assert "max_tokens = episode.turn_budget(max_completion_tokens)" in body, (
        "the turn loop no longer applies the per-turn budget"
    )


def test_the_child_puts_no_deadline_on_a_bridge_call():
    # MultiTurnBridge serializes every env touch behind one lock, so with a whole generation in
    # flight a request spends most of its life QUEUED rather than being served slowly. a client
    # timeout there fails healthy episodes for arriving Nth -- a function of batch size, not of
    # the environment. a genuinely wedged env is caught by the stall watchdog instead, which
    # measures training progress rather than one request.
    from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn

    body = " ".join(inspect.getsource(grpo_multiturn.post_json).split())
    assert "urllib.request.urlopen(request) as response" in body
    assert "timeout=" not in body.split('"""')[-1], (
        "a client-side deadline is back on the bridge call"
    )


def test_the_child_puts_no_deadline_or_retry_on_a_generation_call():
    # keep TRAINING.md aligned: verl mints the request id internally, so callers cannot abort a
    # timed-out generation; retrying would leave the original request occupying kv cache.
    from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn

    source = inspect.getsource(grpo_multiturn)
    # checked before the anchor below, because the usual way to add a deadline is to wrap the call
    # -- which moves the anchor and would otherwise report as "the call vanished".
    assert "asyncio.wait_for" not in " ".join(source.split()), (
        "a deadline wrapper is back around generation; without an abort primitive the timed-out "
        "request keeps running, so TRAINING.md's request-policy paragraph must be rewritten"
    )
    # the kwargs come from the parsed call, not from slicing the text. the first argument is
    # `request_id=uuid4().hex`, so any scan that stops at the first ")" stops inside `uuid4(` and
    # never reaches the later kwargs -- which is exactly where a timeout would be added.
    calls = [
        node
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "server_manager"
    ]
    assert calls, (
        "the generation call moved; re-point this guard so a deadline cannot slip in unnoticed"
    )
    for call in calls:
        names = {kw.arg for kw in call.keywords if kw.arg}
        assert "timeout" not in names, (
            "a per-request timeout is back on the generation call; TRAINING.md's request-policy "
            "paragraph says there is none and has to be rewritten alongside it"
        )
        assert "deadline" not in names, (
            "a per-request deadline is back on the generation call; TRAINING.md's request-policy "
            "paragraph says there is none and has to be rewritten alongside it"
        )


def test_the_parent_sends_the_per_turn_cap_from_the_configured_completion_budget():
    # the cap is only real if the parent actually exports it; the child KeyErrors mid-rollout
    # otherwise, after the engine is already up and paid for.
    emitted = rl_multi.multi_turn_child_env(
        _multi_turn_inp(max_completion=321), reward_url="http://127.0.0.1:9/", thinking=False
    )
    assert emitted["FLASH_VERL_MAX_COMPLETION_TOKENS"] == "321"


# ---------------------- multi-turn bridge routes ----------------------
class _BridgeEnv:
    """the four calls MultiTurnBridge drives, recording what it was asked."""

    def __init__(
        self, *, replies=None, done_after=1, episode=1.0, max_episode_turns=None, prompt=None
    ):
        self.replies = replies if replies is not None else [{"role": "user", "content": "next"}]
        self.done_after = done_after
        self.episode = episode
        self.max_episode_turns = max_episode_turns
        self.prompt = list(prompt or ())
        self.recorded: list[str] = []
        self.scored: list[dict] = []

    def new_rollout_state(self, example, prepared_prompt):
        # `messages` starts as a copy of `prompt` and turns are appended onto it, matching
        # flash.envs.loading.adapter.new_rollout_state. anything reading the transcript has to account
        # for that seeding rather than treating `messages` as turns-only.
        prompt = [dict(message) for message in prepared_prompt]
        state: dict = {
            "example": example,
            "prompt": prompt,
            "messages": [dict(message) for message in prompt],
        }
        if self.max_episode_turns is not None:
            state["max_episode_turns"] = self.max_episode_turns
        return state

    def record_model_turn(self, state, text):
        self.recorded.append(text)
        state["messages"].append({"role": "assistant", "content": text})

    def env_reply(self, messages, state):
        state["messages"].extend(self.replies)
        return self.replies

    def rollout_done(self, state, max_turns):
        return len(self.recorded) >= self.done_after

    def rollout_rewards_many(self, items):
        from flash.envs.loading.base import RolloutReward

        self.scored.extend(state for _, state in items)
        return [RolloutReward(episode=self.episode, turns=None) for _ in items]


def _bridge_image(color="red"):
    from PIL import Image

    from flash.content.multimodal import normalize_image_source

    image = Image.new("RGB", (8, 8), color)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return image, normalize_image_source(encoded.getvalue(), None)


class _BridgeGlueTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}

    def apply_chat_template(self, messages, **kwargs):
        rendered = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                rendered.append(content)
                continue
            rendered.extend(
                "<image>" if block["type"] == "image" else block["text"] for block in content
            )
        return "".join(rendered)


class _BridgeGlueProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_counts = []
        self.template_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs.append(kwargs)
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def image_processor(self, *, images, return_tensors):
        assert return_tensors == "np"
        image = np.asarray(images[0].convert("RGB"), dtype=np.uint8)
        return {
            "pixel_values": image,
            "image_grid_thw": np.asarray([[1, image.shape[0], image.shape[1]]], dtype=np.int64),
        }

    def __call__(self, *, text, images, videos, return_tensors, **kwargs):
        self.image_counts.append(len(images))
        return {"input_ids": [[ord(character) for character in text[0]]]}


def _bridge(env, *, max_turns=4, examples=None, env_prompts=None, **kwargs):
    examples = examples if examples is not None else [{"index": 0}, {"index": 1}]
    if env_prompts is None:
        # what dataset preparation would have produced for these examples: the same opening the
        # env's own start_episode returns. tests that care about the two DISAGREEING pass their own.
        env_prompts = [[dict(message) for message in getattr(env, "prompt", ())] for _ in examples]
    return rl_multi.MultiTurnBridge(
        env, examples, env_prompts=env_prompts, max_turns=max_turns, **kwargs
    )


def test_bridge_exposes_exactly_the_routes_the_child_posts_to():
    # the child posts to four literal paths and the server 404s anything else, with the failure
    # surfacing as a transport error mid-episode. pinned against the child's own source so a rename
    # on either side fails here.
    from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn

    routes = set(_bridge(_BridgeEnv()).routes())
    posted = set(re.findall(r"\"(/multiturn/[a-z]+)\"", inspect.getsource(grpo_multiturn)))
    assert posted, "the child posts to no /multiturn path; this test found nothing to pin"
    assert posted <= routes, f"the child posts to unrouted paths: {sorted(posted - routes)}"


def test_bridge_start_mints_a_session_and_returns_the_turn_budget():
    env = _BridgeEnv()
    bridge = _bridge(env, max_turns=4)
    assert bridge.start({"index": 1, "session_id": "a"}) == {"max_turns": 4}
    assert bridge.open_sessions() == 1


def test_bridge_start_lets_a_per_example_budget_lower_the_cap_but_never_raise_it():
    # a per-example limit is the env asking for a SHORTER episode; honoring one that is longer would
    # let a single row overrun the response tensor the batch was sized for.
    assert _bridge(_BridgeEnv(max_episode_turns=2), max_turns=4).start(
        {"index": 0, "session_id": "a"}
    ) == {"max_turns": 2}
    assert _bridge(_BridgeEnv(max_episode_turns=99), max_turns=4).start(
        {"index": 0, "session_id": "a"}
    ) == {"max_turns": 4}
    # zero would make the loop skip generation entirely and score an empty transcript.
    assert _bridge(_BridgeEnv(max_episode_turns=0), max_turns=4).start(
        {"index": 0, "session_id": "a"}
    ) == {"max_turns": 1}


def test_bridge_start_authenticates_reasoning_content_and_rejects_malformed_metadata():
    from flash.engine.worker.train.rl.rollout.multi_turn import _BadRequest

    env = _BridgeEnv()
    prompt = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "reasoning_content": "old", "content": "answer"},
        {"role": "user", "content": "question"},
    ]
    bridge = _bridge(env, examples=[{"index": 0}], env_prompts=[prompt])

    assert bridge.start(
        {
            "index": 0,
            "session_id": "reasoning",
            "raw_prompt": prompt,
            "prompt_ids": [],
            "image_count": 0,
            "image_digests": [],
        }
    ) == {"max_turns": 4}
    assert bridge._sessions["reasoning"]["messages"] == prompt

    with pytest.raises(_BadRequest, match="does not match the frozen environment prompt"):
        _bridge(env, examples=[{"index": 0}], env_prompts=[prompt]).start(
            {
                "index": 0,
                "session_id": "wrong-reasoning",
                "raw_prompt": [
                    *prompt[:1],
                    {**prompt[1], "reasoning_content": "different"},
                    *prompt[2:],
                ],
                "prompt_ids": [],
                "image_count": 0,
                "image_digests": [],
            }
        )

    malformed = [*prompt[:1], {**prompt[1], "reasoning_content": ["old"]}, *prompt[2:]]
    with pytest.raises(ValueError, match="reasoning_content must be text"):
        _bridge(env, examples=[{"index": 0}], env_prompts=[prompt]).start(
            {
                "index": 0,
                "session_id": "bad-reasoning",
                "raw_prompt": malformed,
                "prompt_ids": [],
                "image_count": 0,
                "image_digests": [],
            }
        )
    with pytest.raises(ValueError, match="unsupported transcript metadata"):
        _bridge(env, examples=[{"index": 0}], env_prompts=[prompt]).start(
            {
                "index": 0,
                "session_id": "bad-metadata",
                "raw_prompt": [{**prompt[0], "name": None}, *prompt[1:]],
                "prompt_ids": [],
                "image_count": 0,
                "image_digests": [],
            }
        )


def test_bridge_start_passes_the_index_aligned_prepared_prompt_into_state_creation():
    class _RecordingEnv(_BridgeEnv):
        def __init__(self):
            super().__init__()
            self.starts = []

        def new_rollout_state(self, example, prepared_prompt):
            self.starts.append((example, prepared_prompt))
            return super().new_rollout_state(example, prepared_prompt)

    env = _RecordingEnv()
    example = {"index": 0}
    prepared_prompt = [{"role": "user", "content": "prepared"}]
    bridge = _bridge(env, examples=[example], env_prompts=[prepared_prompt])
    bridge.start({"index": 0, "session_id": "a"})

    assert env.starts == [(example, prepared_prompt)]
    state = bridge._sessions["a"]["state"]
    assert state["prompt"] == prepared_prompt
    assert state["messages"] == prepared_prompt


def test_bridge_accepts_child_string_for_frozen_singleton_text_block_without_replacing_it():
    class _RecordingEnv(_BridgeEnv):
        def __init__(self):
            super().__init__()
            self.starts = []

        def new_rollout_state(self, example, prepared_prompt):
            self.starts.append((example, prepared_prompt))
            return super().new_rollout_state(example, prepared_prompt)

    env = _RecordingEnv()
    frozen_prompt = [{"role": "user", "content": [{"type": "text", "text": "prepared"}]}]
    bridge = _bridge(env, examples=[{"index": 0}], env_prompts=[frozen_prompt])

    bridge.start(
        {
            "index": 0,
            "session_id": "a",
            "raw_prompt": [{"role": "user", "content": "prepared"}],
            "image_count": 0,
            "image_digests": [],
        }
    )

    expected_prompt = bridge._env_prompts[0]
    assert env.starts[0][1] is expected_prompt, "the child string replaced the frozen env prompt"
    assert expected_prompt[0]["content"] == [{"type": "text", "text": "prepared"}]
    assert bridge._sessions["a"]["state"]["prompt"] == expected_prompt
    assert bridge._sessions["a"]["messages"] == expected_prompt


@pytest.mark.parametrize(
    ("text_blocks", "child_text"),
    [
        ([{"type": "text", "text": "first"}, {"type": "text", "text": " second"}], "first second"),
        ([{"type": "text", "text": ""}], ""),
    ],
)
def test_bridge_authentication_joins_consecutive_text_blocks_like_the_chat_template(
    text_blocks, child_text
):
    frozen_prompt = [{"role": "user", "content": text_blocks}]
    child_prompt = [{"role": "user", "content": child_text}]
    tokenizer = _BridgeGlueTokenizer()
    assert tokenizer.apply_chat_template(frozen_prompt) == tokenizer.apply_chat_template(
        child_prompt
    )

    bridge = _bridge(
        _BridgeEnv(), examples=[{"index": 0}], env_prompts=[frozen_prompt], tokenizer=tokenizer
    )
    bridge.start(
        {
            "index": 0,
            "session_id": "a",
            "raw_prompt": child_prompt,
            "image_count": 0,
            "image_digests": [],
        }
    )
    assert bridge.open_sessions() == 1


def test_bridge_authentication_rejects_changed_text_after_text_block_concatenation():
    from flash.engine.worker.train.rl.rollout.multi_turn import _BadRequest

    bridge = _bridge(
        _BridgeEnv(),
        examples=[{"index": 0}],
        env_prompts=[
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": " second"},
                    ],
                }
            ]
        ],
    )
    with pytest.raises(_BadRequest, match="does not match the frozen environment prompt"):
        bridge.start(
            {
                "index": 0,
                "session_id": "a",
                "raw_prompt": [{"role": "user", "content": "first  second"}],
                "image_count": 0,
                "image_digests": [],
            }
        )
    assert bridge.open_sessions() == 0


def test_bridge_authentication_rejects_image_placement_and_media_digest_sabotage():
    from flash.engine.worker.train.rl.rollout.multi_turn import (
        _authentication_prompts_equal,
        _BadRequest,
    )

    exact_blocks = [
        {
            "role": "user",
            "metadata": {"source": "parent"},
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "image": "image-a"},
                {"type": "video", "video": "video-a"},
            ],
        }
    ]
    for sabotage in (
        [{**exact_blocks[0], "role": "assistant"}],
        [{**exact_blocks[0], "metadata": {"source": "child"}}],
        [
            {
                **exact_blocks[0],
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image", "image": "image-b"},
                    {"type": "video", "video": "video-a"},
                ],
            }
        ],
        [
            {
                **exact_blocks[0],
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image", "image": "image-a"},
                    {"type": "video", "video": "video-b"},
                ],
            }
        ],
    ):
        assert not _authentication_prompts_equal(exact_blocks, sabotage)

    image, descriptor = _bridge_image()
    tokenizer = _BridgeGlueTokenizer()
    processor = _BridgeGlueProcessor(tokenizer)
    frozen_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "after"},
            ],
        }
    ]
    child_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "image": "/tmp/image.png"},
                {"type": "text", "text": "after"},
            ],
        }
    ]
    bridge = _bridge(
        _BridgeEnv(),
        examples=[{"index": 0}],
        env_prompts=[frozen_prompt],
        prompt_descriptors=[[descriptor]],
        processor=processor,
        tokenizer=tokenizer,
    )
    correct_digests = list(bridge._prompt_digests[0])
    sabotages = [
        {
            "raw_prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/tmp/image.png"},
                        {"type": "text", "text": "beforeafter"},
                    ],
                }
            ],
            "image_count": 1,
            "image_digests": correct_digests,
        },
        {
            "raw_prompt": [{"role": "user", "content": "beforeafter"}],
            "image_count": 1,
            "image_digests": correct_digests,
        },
        {
            "raw_prompt": child_prompt,
            "image_count": 1,
            "image_digests": ["changed-image-content"],
        },
    ]
    try:
        for index, sabotage in enumerate(sabotages):
            with pytest.raises(_BadRequest):
                bridge.start({"index": 0, "session_id": f"s{index}", **sabotage})
        assert bridge.open_sessions() == 0
    finally:
        image.close()


def test_image_observation_prompt_without_initial_images_authenticates_through_verl_row():
    from flash.engine.worker.train.rl.launch import inputs as rl_inputs

    class _ImageObservationEnv(_BridgeEnv):
        image_observations = True

    env = _ImageObservationEnv()
    example = {"index": 0}
    source_prompt = [{"role": "user", "content": "question"}]
    processor = _CapabilityProcessor()
    prompts = rl_inputs._build_grpo_prompts(
        [example], [source_prompt], True, processor, processor.tokenizer, None, 32
    )
    prepared = prompts[0]
    rows = rl_verl.build_verl_dataset_rows(
        [prepared["prompt"]], [0], [""], image_uris=[prepared["images"]]
    )

    assert prepared["env_prompt"][0]["content"] == [{"type": "text", "text": "question"}]
    assert rows[0]["prompt"] == [{"role": "user", "content": "question"}]
    assert rows[0]["images"] == []

    bridge = _bridge(
        env,
        examples=[example],
        env_prompts=[prepared["env_prompt"]],
        prompt_ids=[prepared["prompt_ids"]],
        prompt_descriptors=[prepared["images"]],
        processor=processor,
        tokenizer=processor.tokenizer,
    )
    bridge.start(
        {
            "index": 0,
            "session_id": "a",
            "raw_prompt": rows[0]["prompt"],
            "prompt_ids": prepared["prompt_ids"],
            "image_count": 0,
            "image_digests": [],
        }
    )
    assert bridge._sessions["a"]["state"]["prompt"] == prepared["env_prompt"]


@pytest.mark.parametrize("shape", ["image", "image_url", "input_image", "mixed"])
def test_bridge_normalizes_and_authenticates_every_supported_image_reply_shape(shape):
    from flash.content.multimodal import image_descriptors_to_data_uris

    image, descriptor = _bridge_image()
    data_uri = image_descriptors_to_data_uris([descriptor], None)[0]
    blocks = {
        "image": [{"type": "image", "image": data_uri}],
        "image_url": [{"type": "image_url", "image_url": {"url": data_uri}}],
        "input_image": [{"type": "input_image", "input_image": data_uri}],
        "mixed": [
            {"type": "text", "text": "what changed?"},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    }[shape]
    env = _BridgeEnv(
        replies=[{"role": "user", "content": blocks}],
        done_after=99,
    )
    tokenizer = _BridgeGlueTokenizer()
    processor = _BridgeGlueProcessor(tokenizer)
    bridge = _bridge(
        env,
        examples=[{"index": 0}],
        env_prompts=[[{"role": "user", "content": "go"}]],
        prompt_ids=[[1]],
        processor=processor,
        tokenizer=tokenizer,
    )
    routes = bridge.routes()
    assert routes["/multiturn/start"](
        {
            "index": 0,
            "session_id": "a",
            "raw_prompt": [{"role": "user", "content": "go"}],
            "prompt_ids": [1],
            "image_count": 0,
            "image_digests": [],
        }
    ) == {"max_turns": 4}
    out = routes["/multiturn/step"](
        {
            "session_id": "a",
            "turn_ordinal": 0,
            "accepted_prefix": [1],
            "response_ids": [2],
            "completion_text": "answer",
            "image_count": 0,
            "image_digests": [],
        }
    )

    assert out["messages"][0]["content"][-1] == {"type": "image"}
    assert out["image_data_uris"] == [data_uri]
    assert out["image_count"] == 1
    assert out["image_digests"] == bridge._sessions["a"]["image_digests"]
    assert processor.image_counts == [1]
    assert processor.template_kwargs[-1]["preserve_thinking"] is False
    image.close()


def test_bridge_rejects_a_fifth_image_before_processor_glue_or_another_generation():
    from flash.content.multimodal import image_descriptors_to_data_uris
    from flash.engine.worker.train.core.child.glue import parent_image_digests

    source_images_and_descriptors = [
        _bridge_image(color) for color in ("red", "green", "blue", "yellow", "black")
    ]
    source_images = [item[0] for item in source_images_and_descriptors]
    descriptors = [item[1] for item in source_images_and_descriptors]
    data_uris = [
        image_descriptors_to_data_uris([descriptor], None)[0] for descriptor in descriptors
    ]
    prompt = [
        {
            "role": "user",
            "content": [{"type": "image", "image": uri} for uri in data_uris[:4]],
        }
    ]
    env = _BridgeEnv(
        replies=[{"role": "user", "content": [{"type": "image", "image": data_uris[4]}]}],
        done_after=99,
    )
    tokenizer = _BridgeGlueTokenizer()
    processor = _BridgeGlueProcessor(tokenizer)
    bridge = _bridge(
        env,
        examples=[{"index": 0}],
        env_prompts=[prompt],
        prompt_ids=[[1]],
        prompt_descriptors=[descriptors[:4]],
        processor=processor,
        tokenizer=tokenizer,
    )
    routes = bridge.routes()
    digests = parent_image_digests(processor, descriptors[:4], None)
    routes["/multiturn/start"](
        {
            "index": 0,
            "session_id": "a",
            "raw_prompt": prompt,
            "prompt_ids": [1],
            "image_count": 4,
            "image_digests": digests,
        }
    )
    with pytest.raises(ValueError, match="4-image limit"):
        routes["/multiturn/step"](
            {
                "session_id": "a",
                "turn_ordinal": 0,
                "accepted_prefix": [1],
                "response_ids": [2],
                "completion_text": "answer",
                "image_count": 4,
                "image_digests": digests,
            }
        )
    assert processor.image_counts == []
    for image in source_images:
        image.close()


def test_bridge_rejects_prefix_and_media_sabotage_before_recording_the_turn():
    from flash.engine.worker.train.rl.rollout.multi_turn import _BadRequest

    env = _BridgeEnv(done_after=99)
    tokenizer = _BridgeGlueTokenizer()
    bridge = _bridge(
        env,
        examples=[{"index": 0}],
        env_prompts=[[{"role": "user", "content": "go"}]],
        prompt_ids=[[1]],
        tokenizer=tokenizer,
    )
    routes = bridge.routes()
    routes["/multiturn/start"](
        {
            "index": 0,
            "session_id": "a",
            "raw_prompt": [{"role": "user", "content": "go"}],
            "prompt_ids": [1],
            "image_count": 0,
            "image_digests": [],
        }
    )
    base = {
        "session_id": "a",
        "turn_ordinal": 0,
        "accepted_prefix": [1],
        "response_ids": [2],
        "completion_text": "answer",
        "image_count": 0,
        "image_digests": [],
    }
    for changed, message in (
        ({"accepted_prefix": [9]}, "authenticated environment context"),
        ({"image_count": 1, "image_digests": ["wrong"]}, "authenticated context"),
    ):
        with pytest.raises(_BadRequest, match=message):
            routes["/multiturn/step"]({**base, **changed})
    assert env.recorded == []


@pytest.mark.parametrize("block_type", ["video", "audio", "tool_use", "input_file"])
def test_bridge_refuses_a_non_text_reply_block_instead_of_dropping_it(block_type):
    # the flattener keeps `type == "text"` and joins it, so any other block contributes NOTHING and
    # disappears. that is the stringified-image defect wearing a different type: the env meant the
    # model to see something, the model never saw it, and the run still looks healthy.
    env = _BridgeEnv(
        replies=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "look"}, {"type": block_type, "data": "x"}],
            }
        ],
        done_after=99,
    )
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    with pytest.raises(ValueError, match="unsupported content block type"):
        bridge.step({"session_id": "a", "completion_text": "answer"})


def test_bridge_refuses_a_malformed_text_block_rather_than_flattening_it_to_nothing():
    # a text block with no string text flattens to "", so the turn silently loses its content.
    env = _BridgeEnv(
        replies=[{"role": "user", "content": [{"type": "text", "text": None}]}],
        done_after=99,
    )
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    with pytest.raises(ValueError, match="missing text"):
        bridge.step({"session_id": "a", "completion_text": "answer"})


def test_bridge_refuses_a_non_object_reply_block():
    env = _BridgeEnv(replies=[{"role": "user", "content": ["plain string"]}], done_after=99)
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    with pytest.raises(ValueError, match="expected an object"):
        bridge.step({"session_id": "a", "completion_text": "answer"})


def test_bridge_flattens_a_text_only_block_reply_instead_of_stringifying_it():
    # text blocks ARE representable, so they must not become a repr either: an env that returns
    # openai-style text blocks would otherwise train the model on "[{'type': 'text', ...}]".
    env = _BridgeEnv(
        replies=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": " second"},
                ],
            }
        ],
        done_after=99,
    )
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    step = bridge.step({"session_id": "a", "completion_text": "answer"})
    assert step["messages"] == [{"role": "user", "content": "first second"}]


def test_bridge_still_passes_a_plain_string_reply_through_unchanged():
    # the control: the ordinary text path is what every existing multi-turn env uses, and it must be
    # untouched by the block handling above.
    env = _BridgeEnv(replies=[{"role": "user", "content": "next"}], done_after=99)
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    step = bridge.step({"session_id": "a", "completion_text": "answer"})
    assert step["messages"] == [{"role": "user", "content": "next"}]


def test_bridge_rejects_prompts_that_do_not_align_with_its_examples():
    # the two are indexed by the SAME integer the child sends. a length mismatch means some index
    # reads the wrong row's prompt, or IndexErrors mid-rollout; both are worth failing at
    # construction, before the engine is paid for.
    with pytest.raises(ValueError, match="one-to-one"):
        rl_multi.MultiTurnBridge(
            _BridgeEnv(), [{"index": 0}, {"index": 1}], env_prompts=[[]], max_turns=4
        )


def test_bridge_start_rejects_an_out_of_range_index_before_touching_the_env():
    # the index selects which dataset row the episode scores against. python's negative indexing
    # would silently score the wrong row, so the check has to run before the lookup.
    env = _BridgeEnv()
    bridge = _bridge(env, examples=[{"index": 0}])
    for index in (-1, 1, 99):
        with pytest.raises(IndexError, match="outside"):
            bridge.start({"index": index, "session_id": "a"})
    assert bridge.open_sessions() == 0


def test_bridge_start_refuses_to_reuse_a_live_session_id():
    # session ids key the episode state. silently replacing one would strand the first episode's
    # transcript and score the second one twice.
    bridge = _bridge(_BridgeEnv())
    bridge.start({"index": 0, "session_id": "a"})
    with pytest.raises(KeyError, match="duplicate"):
        bridge.start({"index": 1, "session_id": "a"})


def test_bridge_step_records_the_turn_and_returns_the_env_reply():
    env = _BridgeEnv(done_after=2, replies=[{"role": "user", "content": "again"}])
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    out = bridge.step({"session_id": "a", "completion_text": "first"})
    assert env.recorded == ["first"]
    assert out == {"terminal": False, "messages": [{"role": "user", "content": "again"}]}


def test_bridge_step_does_not_show_the_env_an_unusable_turn():
    # a truncated or skipped turn is terminal on the child side too. recording it would append a
    # cut-off assistant message to the transcript that then gets SCORED as if the model produced it.
    for payload in (
        {"session_id": "a", "completion_text": "cut", "truncated": True},
        {"session_id": "a", "completion_text": "cut", "skip_reason": "no room for a turn"},
    ):
        env = _BridgeEnv()
        bridge = _bridge(env)
        bridge.start({"index": 0, "session_id": "a"})
        assert bridge.step(payload) == {"terminal": True, "messages": []}
        assert env.recorded == [], "an unusable turn reached the env"


def test_bridge_step_stops_before_asking_a_finished_env_for_a_reply():
    # rollout_done is checked between recording and replying: an env that ended the episode must not
    # be asked to produce another user message, which its contract does not define past terminal.
    env = _BridgeEnv(done_after=1)
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    assert bridge.step({"session_id": "a", "completion_text": "done"}) == {
        "terminal": True,
        "messages": [],
    }


class _SignOffEnv(_BridgeEnv):
    """an env whose ``env_reply`` ENDS the episode, the way the adapter's does when ``step_episode``
    returns ``done=True``: ``rollout_done`` is false at the bridge's check BEFORE the reply and true
    at the one after. that ordering is what puts a final sign-off message on the terminal path;
    a plain ``done_after=1`` stops at the earlier check and never calls ``env_reply`` at all."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._signed_off = False
        self.last_state: dict = {}

    def env_reply(self, messages, state):
        self._signed_off = True
        self.last_state = state
        return super().env_reply(messages, state)

    def rollout_done(self, state, max_turns):
        return self._signed_off or super().rollout_done(state, max_turns)


def test_bridge_step_does_not_validate_a_terminal_reply_the_child_never_reads():
    # an env that signs off with a final image (or any non-text block) alongside done=True used to
    # take the reply-block guard and 400 the episode -- losing a completed rollout's reward over a
    # message that was never going to be shown to the model. the child breaks on `terminal` BEFORE
    # it reads `messages`, and env_reply has already recorded the reply into the scored state.
    env = _SignOffEnv(
        done_after=9,
        replies=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    )
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    out = bridge.step({"session_id": "a", "completion_text": "first"})
    assert out == {"terminal": True, "messages": []}
    # the sign-off still reached the state the reward path reads; it was skipped, not dropped.
    # _BridgeEnv.env_reply extends the same `state["messages"]` the adapter's does.
    assert env.replies[0] in env.last_state["messages"]


def test_bridge_step_still_refuses_a_remote_non_terminal_image_reply():
    # dynamic image support must not weaken the package-local source boundary.
    env = _BridgeEnv(
        done_after=9,
        replies=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}}
                ],
            }
        ],
    )
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    with pytest.raises(ValueError, match="remote image URLs are not supported"):
        bridge.step({"session_id": "a", "completion_text": "first"})


def test_bridge_step_on_an_unknown_session_raises_rather_than_scoring_a_blank_episode():
    with pytest.raises(KeyError, match="unknown multi-turn session"):
        _bridge(_BridgeEnv()).step({"session_id": "ghost", "completion_text": "x"})


def test_bridge_score_returns_the_episode_reward_for_that_session():
    env = _BridgeEnv(episode=0.75)
    bridge = _bridge(env)
    bridge.start({"index": 1, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    assert bridge.score({"session_id": "a", "turn_count": 1}) == {"score": 0.75}
    # scored against the state the turns accumulated into, not a fresh one.
    assert env.scored[0]["messages"][0]["content"] == "answer"


def test_bridge_score_converts_an_unscorable_episode_to_zero(capsys):
    # nan is score_rollouts' unscorable marker. verl has no equivalent: a nan advantage propagates
    # through the group baseline and poisons every OTHER rollout in the group, so one ungradable
    # episode would corrupt the whole step.
    env = _BridgeEnv(episode=float("nan"))
    bridge = _bridge(env)
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    assert bridge.score({"session_id": "a", "turn_count": 1}) == {"score": 0.0}
    assert "unscorable" in capsys.readouterr().out


def test_bridge_score_converts_a_non_finite_episode_to_zero():
    # inf reaches the group baseline the same way nan does, and is NOT caught by an isnan check.
    for episode in (float("inf"), float("-inf")):
        env = _BridgeEnv(episode=episode)
        bridge = _bridge(env)
        bridge.start({"index": 0, "session_id": "a"})
        bridge.step({"session_id": "a", "completion_text": "answer"})
        assert bridge.score({"session_id": "a", "turn_count": 1}) == {"score": 0.0}


def test_per_turn_credit_without_turn_rewards_warns_once_and_keeps_fallback(capsys):
    env = _BridgeEnv(episode=0.75)
    bridge = _bridge(env, per_turn_credit=True)
    for index, session_id in enumerate(("a", "b")):
        bridge.start({"index": index, "session_id": session_id})
        bridge.step({"session_id": session_id, "completion_text": "answer"})
        assert bridge.score({"session_id": session_id, "turn_count": 1}) == {
            "score": 0.75,
            "turns": None,
        }

    out = capsys.readouterr().out
    warning = "per-turn credit was requested"
    assert out.count(warning) == 1
    # the fallback the shim actually applies is per group, not per run
    # (test_per_turn_credit_shim_leaves_other_groups_on_episode_credit), so a warning claiming the
    # run switched schemes would misreport which credit assignment trained the model.
    assert "rollout group" in out
    assert "this run falls back" not in out


def test_bridge_hands_each_scored_episode_to_the_sample_recorder():
    # multi-turn has no per-completion breakdown -- the env scores a whole episode to a scalar --
    # so the transcript IS the only thing this path can publish for `flash runs log`.
    recorded: list[tuple] = []
    env = _BridgeEnv(episode=0.75)
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})

    assert len(recorded) == 1
    prompt, transcript, reward = recorded[0]
    assert reward == 0.75
    # the whole accumulated transcript, not just the last turn.
    assert [m["content"] for m in transcript] == ["answer"]
    assert prompt == []  # _BridgeEnv seeds no prompt; the shape is what matters here


def test_the_recorded_transcript_excludes_the_prompt_it_was_seeded_from():
    """`messages` starts as a copy of `prompt`, so it is not the transcript -- it CONTAINS it.

    Publishing the whole list repeats the prompt inside `completion` when it already rides the
    sample as `prompt_tail`: the reader sees it twice, and the doubled text eats the payload budget
    a long episode needs for its actual turns.
    """
    recorded: list[tuple] = []
    prompt = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is 3+4"},
    ]
    env = _BridgeEnv(episode=0.75, prompt=prompt)
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "7"})
    bridge.score({"session_id": "a", "turn_count": 1})

    published_prompt, transcript, _ = recorded[0]
    assert published_prompt == prompt
    assert [m["content"] for m in transcript] == ["7"]


def test_the_transcript_slice_keeps_a_turn_that_merely_looks_like_a_prompt_message():
    """Slice by LENGTH, not by equality.

    An env may legitimately produce a turn identical to a prompt message -- an echo env, a
    re-issued instruction, a two-token action space. Dropping matches instead of the seeded prefix
    would silently truncate exactly those episodes, and the sample would understate the turn count
    that the reward was computed over.
    """
    recorded: list[tuple] = []
    prompt = [{"role": "user", "content": "repeat after me: go"}]
    env = _BridgeEnv(episode=1.0, done_after=99, prompt=prompt)
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    # the env replies with the prompt message verbatim, then the model answers.
    env.replies = [dict(prompt[0])]
    bridge.step({"session_id": "a", "completion_text": "go"})
    bridge.score({"session_id": "a", "turn_count": 1})

    transcript = recorded[0][1]
    assert [m["content"] for m in transcript] == ["go", "repeat after me: go"]


def test_the_recorded_episode_is_the_zeroed_reward_not_the_raw_nan():
    # the sample carries the reward the rollout actually trained on. publishing nan here would show
    # a reward in the log that no advantage was ever computed from.
    recorded: list[tuple] = []
    bridge = _bridge(
        _BridgeEnv(episode=float("nan")),
        examples=[{"index": 0}],
        on_episode_scored=lambda *row: recorded.append(row),
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})

    assert [row[2] for row in recorded] == [0.0]


def test_the_recorded_transcript_is_a_snapshot_that_later_turns_cannot_mutate():
    # `step` appends to state["messages"] IN PLACE. handing the live list to the recorder would let
    # a concurrent episode's turn appear inside an already-published sample.
    recorded: list[tuple] = []
    env = _BridgeEnv(done_after=99)
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "first"})
    bridge.score({"session_id": "a", "turn_count": 1})
    snapshot = list(recorded[0][1])
    bridge.step({"session_id": "a", "completion_text": "second"})

    assert list(recorded[0][1]) == snapshot
    assert "second" not in [m.get("content") for m in recorded[0][1]]


def test_the_episode_recorder_runs_outside_the_session_lock():
    # the recorder is the caller's sample buffer, which has its own lock. taking it while holding
    # the session lock inverts the single-turn path's order (buffer lock only, never nested) and
    # any grading that touches both deadlocks.
    observed: list[bool] = []
    env = _BridgeEnv()
    bridge = _bridge(
        env,
        examples=[{"index": 0}],
        on_episode_scored=lambda *_: observed.append(bridge._lock.acquire(blocking=False)),
    )
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})

    assert observed == [True], "the session lock was still held when the recorder ran"
    bridge._lock.release()


def test_a_bridge_without_a_recorder_still_scores():
    # single-turn jobs build no bridge, but the recorder stays optional so the bridge is usable
    # (and testable) without a sample buffer behind it.
    bridge = _bridge(_BridgeEnv(episode=0.5))
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    assert bridge.score({"session_id": "a", "turn_count": 1}) == {"score": 0.5}


@pytest.mark.parametrize(
    "abort", [{"truncated": True}, {"skip_reason": "length"}], ids=["truncated", "skipped"]
)
def test_a_first_turn_abort_is_still_shown_in_the_sample(abort):
    """The turn the episode DIED on is the one worth reading, and it is trained on either way.

    `step` keeps an unusable turn out of `messages` so the env never scores it. Building the sample
    from that state alone therefore publishes an empty completion for a first-turn truncation --
    a model that generated right up to its token limit reads as a model that generated nothing
    ."""
    recorded: list[tuple] = []
    env = _BridgeEnv(done_after=99)
    bridge = _bridge(
        env, examples=[{"index": 0}], on_episode_scored=lambda *row: recorded.append(row)
    )
    bridge.start({"index": 0, "session_id": "a"})
    assert bridge.step({"session_id": "a", "completion_text": "ran out of ro", **abort}) == {
        "terminal": True,
        "messages": [],
    }
    bridge.score({"session_id": "a", "turn_count": 1})

    assert env.recorded == [], "the env was shown a turn it must never score"
    assert [m.get("content") for m in recorded[0][1]] == ["ran out of ro"]


def test_the_env_never_scores_the_aborted_turn_it_is_shown_in_the_sample():
    # the two sides are separate on purpose: the sample gains the turn, the scored state does not.
    # asserting only on the sample would pass an implementation that also appended it to `messages`,
    # which is the truncated-text-gets-graded bug the abort branch exists to prevent.
    env = _BridgeEnv(done_after=99)
    bridge = _bridge(env, examples=[{"index": 0}])
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "good turn"})
    bridge.step({"session_id": "a", "completion_text": "cut off", "truncated": True})
    bridge.score({"session_id": "a", "turn_count": 2})

    scored = [m.get("content") for m in env.scored[0]["messages"]]
    assert "cut off" not in scored, "the truncated turn reached the env's scoring state"
    assert "good turn" in scored


# ---------------------- multi-turn batched episode scoring ----------------------
def test_concurrently_finished_episodes_are_scored_in_one_env_call():
    # a generation is prompts_per_step * group_size episodes. scoring them one at a time turns one
    # judge round into hundreds of serial round-trips with the gpu idle. the env below records the
    # SIZE of every batch it is handed, so a regression to per-episode scoring shows up as many
    # calls of one rather than one call of many.
    class _BatchRecordingEnv(_BridgeEnv):
        def __init__(self):
            super().__init__()
            self.batch_sizes: list[int] = []

        def rollout_rewards_many(self, items):
            from flash.envs.loading.base import RolloutReward

            self.batch_sizes.append(len(items))
            return [RolloutReward(episode=1.0, turns=None) for _ in items]

    env = _BatchRecordingEnv()
    examples = [{"index": i} for i in range(8)]
    bridge = _bridge(env, examples=examples)
    for i in range(8):
        bridge.start({"index": i, "session_id": f"s{i}"})
        bridge.step({"session_id": f"s{i}", "completion_text": "answer"})

    scores: dict[int, float] = {}
    threads = [
        threading.Thread(
            target=lambda i=i: scores.__setitem__(
                i, bridge.score({"session_id": f"s{i}", "turn_count": 1})["score"]
            )
        )
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not [t for t in threads if t.is_alive()], "a scoring episode never completed"

    assert scores == dict.fromkeys(range(8), 1.0)
    assert max(env.batch_sizes) > 1, (
        f"every episode was scored alone: batch sizes {env.batch_sizes}"
    )
    assert sum(env.batch_sizes) == 8, "an episode was scored twice or not at all"


def test_a_batched_score_reaches_the_env_under_the_same_lock_every_other_call_takes():
    # scoring is batched to shorten how long the lock is held, NOT to drop it. `reward_thread_safe`
    # licenses racing the scorer against ITSELF; it says nothing about racing it against a
    # concurrent episode's `env_reply`, and no env contract permits that.
    class _LockObservingEnv(_BridgeEnv):
        def __init__(self):
            super().__init__()
            self.held_during_scoring: list[bool] = []

        def rollout_rewards_many(self, items):
            from flash.envs.loading.base import RolloutReward

            acquired = bridge._lock.acquire(blocking=False)
            self.held_during_scoring.append(not acquired)
            if acquired:
                bridge._lock.release()
            return [RolloutReward(episode=1.0, turns=None) for _ in items]

    env = _LockObservingEnv()
    bridge = _bridge(env, examples=[{"index": 0}])
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})
    assert env.held_during_scoring == [True], "the env was scored without the session lock held"


def test_a_failing_batch_fails_every_episode_in_it_rather_than_hanging_them():
    # a scoring raise must complete every waiter or the other episodes block until VERL-100 fires.
    # park the scorer until several episodes queue so the test exercises one real shared batch.
    entered = threading.Event()
    release = threading.Event()

    batch_sizes: list[int] = []

    class _FailingEnv(_BridgeEnv):
        def rollout_rewards_many(self, items):
            batch_sizes.append(len(items))
            entered.set()
            release.wait(timeout=30)
            raise RuntimeError("judge is down")

    bridge = _bridge(_FailingEnv(), examples=[{"index": i} for i in range(5)])
    for i in range(5):
        bridge.start({"index": i, "session_id": f"s{i}"})
        bridge.step({"session_id": f"s{i}", "completion_text": "answer"})

    outcomes: list[str] = []
    outcome_lock = threading.Lock()

    def _record(session_id):
        result = _score_capturing(bridge, session_id)
        with outcome_lock:
            outcomes.append(result)

    # s0 goes first and parks the scorer inside the env call. the other four queue behind it and
    # are taken as ONE batch on the next pass -- that is the batch whose failure must scatter.
    # daemon: a waiter this test proves is WEDGED never returns, and a non-daemon thread would
    # then block interpreter shutdown after the assertion had already fired -- turning a named
    # failure into a hang that reports nothing and burns the whole ci timeout.
    first = threading.Thread(target=lambda: _record("s0"), daemon=True)
    first.start()
    assert entered.wait(timeout=30), "the scorer never reached the env"

    rest = [threading.Thread(target=lambda i=i: _record(f"s{i}"), daemon=True) for i in range(1, 5)]
    for thread in rest:
        thread.start()

    # count waiters across pending and in-flight; the batcher drains pending immediately.
    # do not wait on pending while the scorer holds the bridge lock because callers cannot queue.
    time.sleep(0.5)
    release.set()

    threads = [first, *rest]
    for thread in threads:
        thread.join(timeout=30)
    alive = [t for t in threads if t.is_alive()]
    assert not alive, f"{len(alive)} episode(s) left hanging by a failed batch"
    assert sorted(outcomes) == ["judge is down"] * 5
    # the point of the test: a batch of MORE THAN ONE actually failed together. without this the
    # queue-depth wait above would still pass if each episode were scored on its own, which is the
    # shape under which the buggy `batch[0]`-only completion is indistinguishable from correct.
    assert max(batch_sizes) > 1, f"no shared batch ever formed; env saw batches {batch_sizes}"


def _score_capturing(bridge, session_id):
    """score one episode, returning the failure text instead of raising (for use off-thread)."""
    try:
        bridge.score({"session_id": session_id, "turn_count": 1})
    except Exception as error:
        return str(error)
    return "scored"


def test_the_scoring_thread_starts_on_first_use_rather_than_an_explicit_call():
    # the consumer must start on first use or score waits on an event nobody can set.
    # keep the call on a bounded thread so this regression fails by assertion instead of hanging ci.
    bridge = _bridge(_BridgeEnv(), examples=[{"index": 0}])
    assert bridge._scorer._thread is None, "the thread was started before any episode needed it"
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})

    scored: list = []
    worker = threading.Thread(
        target=lambda: scored.append(bridge.score({"session_id": "a", "turn_count": 1})),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "score blocked forever: nothing is draining the queue"
    assert scored == [{"score": 1.0}]
    assert bridge._scorer._thread is not None


def test_bridge_shutdown_stops_the_scoring_thread():
    # the run's finally calls this before the server goes down. a thread left running would keep
    # the worker process alive past the point flash considers the run finished.
    bridge = _bridge(_BridgeEnv(), examples=[{"index": 0}])
    bridge.start({"index": 0, "session_id": "a"})
    bridge.step({"session_id": "a", "completion_text": "answer"})
    bridge.score({"session_id": "a", "turn_count": 1})
    thread = bridge._scorer._thread
    assert thread is not None
    assert thread.is_alive()
    bridge.shutdown()
    thread.join(timeout=10)
    assert not thread.is_alive(), "the scoring thread outlived the bridge"


def test_the_run_shuts_the_bridge_down_before_the_server_it_is_mounted_on():
    # ordering matters: the server's routes block on the scoring thread, so stopping the server
    # first would strand a scoring episode on an event nothing will ever set.
    entry = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    src = " ".join(inspect.getsource(rl_train._shutdown_rl_runtime).split())
    assert "_shutdown_rl_runtime(resume_uploader, gpu_sampler, reward_runtime)" in entry
    assert "multi_turn_bridge.shutdown()" in src
    assert src.index("multi_turn_bridge.shutdown()") < src.index("server.shutdown()")


def test_bridge_close_releases_the_session():
    # every in-flight episode holds env state. a leak here grows for the whole run, and the child
    # closes in a finally precisely so a failed episode still frees it -- so close must tolerate a
    # session that was never started.
    bridge = _bridge(_BridgeEnv())
    bridge.start({"index": 0, "session_id": "a"})
    assert bridge.close({"session_id": "a"}) == {"closed": True}
    assert bridge.open_sessions() == 0
    assert bridge.close({"session_id": "a"}) == {"closed": True}


def _age_session(bridge, session_id, seconds):
    """Backdate a session's lease clock, so lease tests need no sleeps and cannot flake.

    White-box on purpose: the alternative is a tiny lease plus a real sleep, and the
    still-being-driven direction below would then race the lease it is trying to prove renews.
    """
    session = bridge._sessions[session_id]
    assert "touched_at" in session, "sessions carry no lease clock, so nothing can ever reap one"
    session["touched_at"] -= seconds
    return session


def test_bridge_reaps_a_session_whose_actor_died_before_it_could_close(capsys):
    # a ray rollout actor that dies between /multiturn/start and its `finally` /multiturn/close
    # leaves an entry nobody will ever remove. verl restarts the actor, the replacement starts a
    # fresh session, and the dead one's env state and transcript are retained for the rest of the
    # worker's life -- so the leak grows with every actor death, unbounded.
    env = _BridgeEnv()
    bridge = _bridge(env, session_lease_s=60.0)
    bridge.start({"index": 0, "session_id": "dead"})
    _age_session(bridge, "dead", 3600.0)

    bridge.start({"index": 1, "session_id": "live"})

    assert bridge.open_sessions() == 1, "the abandoned session outlived the actor that owned it"
    assert "dead" not in bridge._sessions
    # the leak it prevents is silent, so the reap must not be: repeated reaps mean actors are dying
    # mid-episode, which is a rollout problem worth seeing in the log.
    assert "reaped 1 abandoned multi-turn session(s)" in capsys.readouterr().out
    # reaping is a memory release, never a training event. scoring a dead episode would hand the
    # group baseline a reward for a rollout that never finished.
    assert env.scored == []


def test_bridge_never_reaps_a_session_that_is_still_being_driven():
    # the failure this guards is worse than the leak: reaping a live episode fails a working
    # rollout, because its next /multiturn/step gets `unknown multi-turn session`. an episode is
    # idle between turns for as long as a generate plus an env reply takes, so the lease has to be
    # renewed by activity rather than measured from the session's birth.
    bridge = _bridge(_BridgeEnv(done_after=99), session_lease_s=60.0)
    bridge.start({"index": 0, "session_id": "slow"})
    _age_session(bridge, "slow", 3600.0)

    bridge.step({"session_id": "slow", "completion_text": "a turn that took a very long time"})
    bridge.start({"index": 1, "session_id": "other"})

    assert bridge.open_sessions() == 2, "a live episode was reaped and its next turn will fail"
    assert bridge.step({"session_id": "slow", "completion_text": "next"})["messages"]


def test_bridge_lease_can_be_disabled_and_then_nothing_is_ever_reaped():
    # `0` is the off switch a caller reaches for when reaping is suspected of breaking a rollout.
    # without an explicit guard, a non-positive lease reads as "everything is stale" and reaps
    # every session on the next start -- the exact opposite of what disabling it should mean.
    bridge = _bridge(_BridgeEnv(), session_lease_s=0.0)
    bridge.start({"index": 0, "session_id": "dead"})
    _age_session(bridge, "dead", 10_000_000.0)
    bridge.start({"index": 1, "session_id": "live"})
    assert bridge.open_sessions() == 2


def test_bridge_routes_are_served_alongside_single_turn_scoring():
    # one server, one port: the child gets a single url and posts both /score and /multiturn/* to
    # it. mounting the bridge on its own server would leave the child's reward path pointing at a
    # port that only answers episodes.
    env = _BridgeEnv()
    bridge = _bridge(env, tokenizer=_BridgeGlueTokenizer())
    server, url = rl_multi.start_reward_server(
        lambda index, text: 1.0, example_count=2, multi_turn_bridge=bridge
    )
    try:

        def _post(path, payload):
            request = urllib.request.Request(
                url + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode())

        assert _post("/score", {"index": 0, "solution_str": "x"}) == {"score": 1.0}
        assert _post(
            "/multiturn/start",
            {
                "index": 0,
                "session_id": "a",
                "raw_prompt": [],
                "prompt_ids": [],
                "image_count": 0,
                "image_digests": [],
            },
        ) == {"max_turns": 4}
        _post(
            "/multiturn/step",
            {
                "session_id": "a",
                "turn_ordinal": 0,
                "accepted_prefix": [],
                "response_ids": [1],
                "completion_text": "answer",
                "image_count": 0,
                "image_digests": [],
            },
        )
        assert _post("/multiturn/score", {"session_id": "a", "turn_count": 1}) == {"score": 1.0}
        assert _post("/multiturn/close", {"session_id": "a"}) == {"closed": True}
    finally:
        server.shutdown()


def test_the_bridge_is_built_only_for_multi_turn_jobs():
    # a bridge on a single-turn job would expose episode routes with no episode state behind them,
    # and mounting it costs a lock the single-turn scoring path already has.
    # whitespace-normalized: the construction spans several lines, and what is under test is the
    # guard around it, not how the formatter wrapped the call.
    src = " ".join(inspect.getsource(rl_runner._start_reward_runtime).split())
    assert src.count("MultiTurnBridge(") == 1
    for fragment in (
        'env_prompts=[p["env_prompt"] for p in prompts]',
        'prompt_ids=[p["prompt_ids"] for p in prompts]',
        'prompt_descriptors=[p.get("images", ()) for p in prompts]',
        'processor=inp["processor"]',
        "tokenizer=tok",
        'per_turn_credit=bool(inp["per_turn_credit"])',
        "on_episode_scored=observability.record",
        "parent_work=observability.parent_work",
        "identity_ledger=identity_ledger",
    ):
        assert fragment in src
    assert 'if inp["multi_turn"] else None' in src


# ---------------------- multi-turn response tensor width ----------------------
def test_multi_turn_widens_the_response_tensor_to_hold_a_whole_episode(monkeypatch):
    # verl drops response_ids beyond data.max_response_length.
    # size for the longest transcript: engine length minus the shortest admitted prompt.
    inp = _capability_resolve(monkeypatch, _capability_env(multi_turn=True))
    assert inp["max_response_len"] > inp["max_completion"], "the episode tensor was not widened"
    assert inp["max_response_len"] == inp["engine_len"] - min(
        int(p["prompt_len"]) for p in inp["prompts"]
    )
    # max_completion stays the PER-TURN cap, exactly as the trl driver uses it.
    assert inp["max_completion"] < inp["engine_len"]


def test_single_turn_leaves_the_response_tensor_at_the_completion_width(monkeypatch):
    # the control: one completion IS the response, so widening it would inflate every rollout's
    # padded tensor and the token budget derived from it for no reason.
    inp = _capability_resolve(monkeypatch, _capability_env())
    assert inp["max_response_len"] == inp["max_completion"]


def test_the_response_width_reaches_verls_config_rather_than_max_completion(monkeypatch):
    # the derivation is worthless if the override still emits max_completion. this is the one line
    # that decides how wide the tensor verl allocates actually is.
    inp = _capability_resolve(monkeypatch, _capability_env(multi_turn=True))
    cfg = rl_verl._build_verl_training_cfg(
        inp,
        ce_backend="torch",
        train_files="/w/train.parquet",
        val_files="/w/val.parquet",
        model_path=inp["model_id"],
        thinking=False,
        loggers=["console"],
        fp8_kv=False,
        enforce_eager=False,
        attention_backend=None,
        mm_encoder_attn_backend=None,
        reward_path="/w/reward.py",
        local_dir="/w/ckpt",
        project_name="flash",
        experiment_name="flash-rl-run123",
    )
    assert f"data.max_response_length={inp['max_response_len']}" in rl_verl.build_verl_overrides(
        cfg
    )


# ---------------- reward observability: the buffer and the heartbeat drain ----------------
def _score_buffer(env, *, prompts=None, examples=None, generation_size=0):
    """`_score`'s grade-then-record pair, against a real buffer and fake env.

    The real local function requires a model, dataset, and verl child, so source tests separately
    pin
    this wiring; ``generation_size`` remains caller-driven.
    """
    buffer = RewardObservabilityBuffer(generation_size=generation_size)
    rollout_examples = examples if examples is not None else [{"gt": "7"}]
    message_prompts = prompts if prompts is not None else ["prompt-0"]

    def score(index: int, solution_str: str) -> float:
        breakdowns: list[dict[str, float] | None] = []
        value = rl_single.score_single_turn(
            env,
            solution_str,
            rollout_examples[int(index)],
            tok=None,
            thinking=False,
            prompt_opened_thinking=False,
            think_penalty=0.0,
            breakdowns=breakdowns,
        )
        buffer.record(message_prompts[int(index)], solution_str, value, breakdowns)
        return value

    return score, buffer


def test_score_batch_grades_before_it_records():
    """User grading must finish before the observability lock is taken per result."""
    src = " ".join(inspect.getsource(rl_runner._start_reward_runtime).split())
    body = src[src.index("def _score_batch(requests:") :]
    body = body[: body.index("def _score_for_profile")]

    assert body.count("observability.record(") == 1
    assert body.index("scored = score_single_turn_batch(") < body.index("observability.record(")


def test_the_recorded_prompt_is_the_one_the_batched_completion_was_graded_against():
    """Each scattered sample must use the same request index for its example and prompt."""
    src = " ".join(inspect.getsource(rl_runner._start_reward_runtime).split())
    body = src[src.index("def _score_batch(requests:") :]
    body = body[: body.index("def _score_for_profile")]

    assert (
        "(solution_str, rollout_examples[int(index)]) for index, solution_str in requests" in body
    )
    assert (
        "for (index, solution_str), (score, breakdowns) in zip(requests, scored, strict=True):"
        in body
    )
    assert (
        "observability.record(message_prompts[int(index)], solution_str, score, breakdowns)" in body
    )


@pytest.mark.usefixtures("_identity_graded")
def test_the_buffer_keeps_the_rollout_sample_and_its_named_breakdown():
    score, buffer = _score_buffer(_NamedBreakdownEnv())

    assert score(0, "7") == 1.0
    assert buffer.latest() == ("prompt-0", "7", 1.0)
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0, "quality": 0.5}


@pytest.mark.usefixtures("_identity_graded")
def test_the_buffers_are_bounded_when_the_generation_never_closes():
    # nothing bounds how many completions arrive before a boundary. the sample buffer evicts; the
    # metric accumulator must not GROW instead -- this process is already memory-tight.
    score, buffer = _score_buffer(_NamedBreakdownEnv())
    for _ in range(5000):
        score(0, "7")

    assert len(buffer._samples) == RewardObservabilityBuffer._SAMPLE_BUFFER_LIMIT
    # one float per NAME, whatever the completion count: the env reports two.
    assert set(buffer._pending_totals) == {"success", "quality"}
    assert buffer._pending_count == 5000


@pytest.mark.usefixtures("_identity_graded")
def test_every_completion_counts_toward_the_mean_however_large_the_generation():
    """A generation is ``batch_size * group_size`` completions, both unbounded, so no retention cap
    can hold one. Dropping the overflow biases the published mean toward whichever completions were
    graded last -- here, a run that succeeded early and failed late would report a flat 0
    ."""
    score, buffer = _score_buffer(
        _CountingBreakdownEnv(),
        prompts=["prompt-0"],
        examples=[{"gt": "7"}],
    )
    total = 5000
    for _ in range(total):
        score(0, "7")
    buffer.close_generation(1)

    # the env numbers each grading, so the true mean of `n` is the mean of 0..total-1.
    assert buffer.heartbeat_fields()["reward_metrics"]["n"] == (total - 1) / 2


@pytest.mark.usefixtures("_identity_graded")
def test_eviction_drops_the_oldest_rollouts_and_keeps_the_newest():
    """Over the limit, what SURVIVES has to be the recent end: samples answer "what is the model
    doing now", so evicting the newest would pin a stalled run's diagnostics to its oldest
    gradings. A length-only assertion passes either way -- this reads the retained values."""
    score, buffer = _score_buffer(
        _CountingBreakdownEnv(),
        prompts=["prompt-0"],
        examples=[{"gt": "7"}],
    )
    total = RewardObservabilityBuffer._SAMPLE_BUFFER_LIMIT + 50
    for _ in range(total):
        score(0, "7")

    assert buffer.latest()[2] == float(total - 1)
    assert [row[2] for row in buffer._samples] == [
        float(n) for n in range(total - RewardObservabilityBuffer._SAMPLE_BUFFER_LIMIT, total)
    ]


@pytest.mark.usefixtures("_identity_graded")
def test_a_run_with_no_rollouts_yet_omits_sampled_completions():
    # symmetric with the reward_metrics case below: an empty list on the wire reads as "this step
    # produced no rollouts", which is a different claim from "none have been scored yet".
    buffer = RewardObservabilityBuffer()

    assert "sampled_completions" not in buffer.heartbeat_fields()


@pytest.mark.usefixtures("_identity_graded")
def test_both_signals_pass_their_wire_bounds_before_publication():
    """The bounds are what make a payload safe to commit, and both are applied HERE -- the caller
    publishes whatever this returns. Asserted through the buffer rather than on the helpers (which
    have their own tests) so dropping either call from the publisher fails.

    The sample side asserts on neutralization and the cap together, because on this path both come
    from `select_rollout_samples` itself rather than from a second bounding pass.
    """
    buffer = RewardObservabilityBuffer()
    buffer.record("prompt-0", "done\x1b[2Jcleared", 1.0)
    for i in range(1, 5):  # distinct prompts, so dedup can't stand in for the cap
        buffer.record(f"prompt-{i}", f"completion-{i}", float(i))
    buffer.close_generation(1)
    with buffer._lock:  # 13 names, over the 12-metric cap
        buffer._latest_metrics.update({f"m{i}": 1.0 for i in range(13)})

    fields = buffer.heartbeat_fields()

    assert len(fields["reward_metrics"]) == 12
    assert len(fields["sampled_completions"]) == 3
    # a raw escape would let a rollout repaint the terminal of whoever runs `flash runs log`.
    assert "\x1b" not in fields["sampled_completions"][0]["completion"]


@pytest.mark.usefixtures("_identity_graded")
def test_a_reward_that_is_not_a_float_is_coerced_at_the_boundary():
    # rewards arrive from user grading code and go out as json. coercing on the way IN keeps a
    # numpy scalar or a bool from reaching the serializer a heartbeat away from the call site.
    buffer = RewardObservabilityBuffer()
    buffer.record("prompt-0", "completion-0", np.float32(0.25))

    reward = buffer.latest()[2]
    assert type(reward) is float
    assert reward == 0.25


@pytest.mark.usefixtures("_identity_graded")
def test_a_scalar_reward_run_publishes_no_named_metrics_at_all():
    # end to end for the empty case: a scores_breakdown-less env must reach the wire with the key
    # ABSENT, not with every name flattened to 0 by an empty-dict denominator.
    score, buffer = _score_buffer(_RewardOnlyEnv())
    score(0, "the answer is 7")
    buffer.close_generation(1)

    fields = buffer.heartbeat_fields()
    assert "reward_metrics" not in fields
    assert len(fields["sampled_completions"]) == 1


@pytest.mark.usefixtures("_identity_graded")
def test_the_heartbeat_publishes_averaged_metrics_and_bounded_samples():
    score, buffer = _score_buffer(
        _NamedBreakdownEnv(),
        prompts=["p0", "p1", "p2", "p3"],
        examples=[{"gt": "7"}, {"gt": "7"}, {"gt": "9"}, {"gt": "9"}],
    )
    for index, completion in enumerate(["7", "7", "7", "7"]):
        score(index, completion)

    buffer.close_generation(5)
    fields = buffer.heartbeat_fields()

    # two of four completions matched their gt, so success averages 0.5 across the generation.
    assert fields["reward_metrics"] == {"success": 0.5, "quality": 0.5}
    assert len(fields["sampled_completions"]) == 3  # hard cap, four rollouts buffered
    assert {s["generated_at_step"] for s in fields["sampled_completions"]} == {5}
    assert [s["reward"] for s in fields["sampled_completions"]] == [1.0, 1.0, 0.0]


@pytest.mark.usefixtures("_identity_graded")
def test_a_heartbeat_landing_mid_generation_republishes_the_last_complete_one():
    """A 30s liveness tick is not a generation boundary, and publishing on it is latency-biased.

    The completions that finish first are the fast ones -- short outputs, cache hits, envs that
    grade without i/o. A drain on the heartbeat cadence therefore reports THAT subset's mean as the
    step's reward, systematically over-representing whatever is cheap to produce. The reading has to
    stay pinned to the last whole generation until the next boundary seals a new one.
    """
    score, buffer = _score_buffer(
        _NamedBreakdownEnv(),
        prompts=["p0", "p1"],
        examples=[{"gt": "7"}, {"gt": "7"}],
    )
    score(0, "7")  # the fast completion: success 1.0
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 1.0

    # generation 2 is under way and only its fast half has been graded.
    score(1, "wrong")  # success 0.0

    mid = buffer.heartbeat_fields()
    assert mid["reward_metrics"]["success"] == 1.0, "published a partial generation's mean"
    assert [s["reward"] for s in mid["sampled_completions"]] == [1.0]

    # the slow half lands, then the boundary: now the whole generation publishes at once.
    score(0, "7")
    buffer.close_generation(2)
    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 0.5


@pytest.mark.usefixtures("_identity_graded")
def test_the_next_generation_cannot_be_sealed_into_the_step_line_that_is_still_in_flight():
    """The child's stdout is delivered asynchronously, so `step:N` can reach the parent AFTER
    generation N+1 has started scoring. A boundary taken at that moment seals both generations
    under step N and leaves N+1 with nothing of its own to publish.

    Counting closes the generation on the scoring thread that finishes it, so the in-flight line
    only names what was already sealed."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    score(0, "7")
    score(0, "7")  # generation 1 is complete here, whatever stdout is doing
    score(0, "wrong")  # generation 2 begins while `step:1` is still in the pipe

    buffer.close_generation(1)
    first = buffer.heartbeat_fields()
    assert first["reward_metrics"]["success"] == 1.0, "next generation leaked into this mean"
    assert [s["completion"] for s in first["sampled_completions"]] == ["7", "7"]
    assert {s["generated_at_step"] for s in first["sampled_completions"]} == {1}

    score(0, "wrong")
    buffer.close_generation(2)
    second = buffer.heartbeat_fields()
    assert second["reward_metrics"]["success"] == 0.0, "step 2 republished step 1"
    assert {s["generated_at_step"] for s in second["sampled_completions"]} == {2}


@pytest.mark.usefixtures("_identity_graded")
def test_a_generation_that_completes_before_the_previous_step_line_is_not_lost():
    """Stdout can fall a WHOLE generation behind, not just part of one.

    A single "already sealed" flag only remembers one unacknowledged generation, so the second seal
    overwrites the first: generation 1 is dropped and generation 2 publishes under step 1, leaving
    every later step misaligned. Small generations make that window ordinary."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    score(0, "7")
    score(0, "7")  # generation 1 complete: success 1.0
    score(0, "wrong")
    score(0, "wrong")  # generation 2 complete too, and NEITHER step line has arrived

    buffer.close_generation(1)
    first = buffer.heartbeat_fields()
    assert first["reward_metrics"]["success"] == 1.0, (
        "generation 1 was overwritten before it published"
    )
    assert [s["completion"] for s in first["sampled_completions"]] == ["7", "7"]
    assert {s["generated_at_step"] for s in first["sampled_completions"]} == {1}

    buffer.close_generation(2)
    second = buffer.heartbeat_fields()
    assert second["reward_metrics"]["success"] == 0.0
    assert [s["completion"] for s in second["sampled_completions"]] == ["wrong", "wrong"]
    assert {s["generated_at_step"] for s in second["sampled_completions"]} == {2}


@pytest.mark.usefixtures("_identity_graded")
def test_the_queue_of_unnamed_generations_is_bounded():
    # the queue holds whole generations, each retaining up to _SAMPLE_BUFFER_LIMIT completions. a
    # child that stops printing step lines never drains it, and this process is already memory-tight.
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    for _ in range(4 * RewardObservabilityBuffer._SEALED_QUEUE_LIMIT):
        score(0, "7")

    assert len(buffer._sealed_by_count) == RewardObservabilityBuffer._SEALED_QUEUE_LIMIT


@pytest.mark.usefixtures("_identity_graded")
def test_an_eviction_does_not_shift_every_later_step_onto_the_wrong_generation():
    """Dropping the oldest queued generation is not enough: its `step:N` line still arrives.

    Handing that line to the oldest SURVIVOR consumes a generation whose own line is still coming,
    so the offset never closes -- every step for the rest of the run publishes the next generation's
    output under the previous step's number. One eviction, permanently wrong diagnostics.
    """
    limit = RewardObservabilityBuffer._SEALED_QUEUE_LIMIT
    buffer = RewardObservabilityBuffer(generation_size=1)
    for i in range(limit + 1):  # one more than the queue holds: generation 0 is evicted
        buffer.record(f"prompt-{i}", f"gen-{i}", float(i))

    assert len(buffer._sealed_by_count) == limit

    # stdout catches up and names them in order. the line for the evicted generation is spent on it.
    published = []
    for step in range(1, limit + 2):
        buffer.close_generation(step)
        published.append(buffer._published[-1][1] if buffer._published else None)

    # step 1's generation is genuinely gone, so its reading is stale rather than another
    # generation's. every step after it is matched to the generation that actually produced it.
    assert published[0] is None
    assert published[1:] == [f"gen-{i}" for i in range(1, limit + 1)]


@pytest.mark.usefixtures("_identity_graded")
def test_the_step_preview_reads_the_generation_that_step_published():
    """The caller closes the generation and then previews it under that same step number.

    A late `step:N` line arrives with generation N+1 already scoring, so the newest recorded sample
    belongs to N+1. Previewing that labels N+1's completion as step N -- the mislabelling the queue
    exists to prevent, reintroduced one line later, and disagreeing with the heartbeat about the
    very same step.
    """
    buffer = RewardObservabilityBuffer(generation_size=2)
    buffer.record("p", "gen1-a", 1.0)
    buffer.record("p", "gen1-b", 1.0)  # generation 1 sealed by count
    buffer.record("p", "gen2-a", 9.0)  # generation 2 already scoring; `step:1` still in the pipe

    buffer.close_generation(1)

    assert buffer.latest()[1] == "gen1-b", "the preview labelled the next generation as this step"
    # and the heartbeat agrees with it, which is the point of reading the published generation.
    fields = buffer.heartbeat_fields()
    assert [s["completion"] for s in fields["sampled_completions"]] == ["gen1-a", "gen1-b"]


def test_a_preview_before_the_first_boundary_still_shows_a_rollout():
    # the fallback direction: with nothing published yet, the open generation is all there is, and
    # blanking the preview would read as "no rollouts" rather than "no boundary yet".
    buffer = RewardObservabilityBuffer()
    buffer.record("p", "first", 0.5)

    assert buffer.latest() == ("p", "first", 0.5)


@pytest.mark.usefixtures("_identity_graded")
def test_a_component_too_large_to_be_a_float_does_not_fail_the_reward_request():
    """`record` runs OUTSIDE `score_single_turn`'s error guard, so anything it raises 400s the
    reward request and aborts the run. An int larger than a float can hold raises OverflowError,
    which is neither TypeError nor ValueError."""
    score, buffer = _score_buffer(_OverflowingBreakdownEnv())

    assert score(0, "7") == 1.0  # the total graded fine; only the diagnostic component is unusable
    buffer.close_generation(1)
    metrics = buffer.heartbeat_fields()["reward_metrics"]
    assert metrics["success"] == 1.0, "a usable component was dropped with the unusable one"
    assert metrics["enormous"] == 0.0


def test_the_published_metric_bound_survives_a_value_too_large_to_be_a_float():
    # the same coercion runs again on the publish side, on the heartbeat thread, over a dict the
    # trl callback takes from its caller. escaping there kills liveness reporting for the whole run.
    from flash.engine.worker.io.heartbeat import _bounded_reward_metrics

    assert _bounded_reward_metrics({"huge": 10**400, "fine": 0.25}) == {"fine": 0.25}


@pytest.mark.usefixtures("_identity_graded")
def test_the_step_line_names_the_generation_the_count_already_sealed():
    """A counted seal publishes under the buffer's OWN ordinal, which is only a guess at what verl
    logged -- it counts from 1 and assumes no skipped or resumed steps. The arriving line carries
    the real number, so dropping the relabel would stamp every sample with a step the run never
    logged, and a reader correlating samples against the loss curve would line them up wrong."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    score(0, "7")
    score(0, "7")

    # verl resumed from a checkpoint: its first logged step is 41, not the buffer's internal 1.
    buffer.close_generation(41)

    assert {s["generated_at_step"] for s in buffer.heartbeat_fields()["sampled_completions"]} == {
        41
    }


@pytest.mark.usefixtures("_identity_graded")
def test_a_counted_seal_does_not_disarm_the_boundary_for_later_generations():
    """`_sealed_by_count` holds the generations the count sealed, waiting to be named. A step line
    that names one without taking it off the queue turns every later `close_generation` into a
    relabel of the same entry, so from the second generation on the buffer publishes nothing new --
    metrics frozen at generation 1 while the run continues."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    for _ in range(2):
        score(0, "7")
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 1.0

    for _ in range(2):
        score(0, "wrong")
    buffer.close_generation(2)

    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 0.0, "boundary went dead"


@pytest.mark.usefixtures("_identity_graded")
def test_a_generation_short_of_its_count_is_still_sealed_by_the_step_line():
    """The count seals a FULL generation; the step line is what seals a short one. If a named
    generation stays on `_sealed_by_count`, the next `close_generation` relabels it instead of
    sealing what is open -- so a generation that lost a completion (a `_score` that raised before
    reaching the bridge) republishes the PREVIOUS generation's numbers under the new step, and
    carries its samples forward. The run reads as healthy at exactly the step where grading broke."""
    score, buffer = _score_buffer(_NamedBreakdownEnv(), generation_size=2)
    for _ in range(2):
        score(0, "7")
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"]["success"] == 1.0

    # one of the two completions never reached the bridge, so the count cannot fire
    score(0, "wrong")
    buffer.close_generation(2)

    fields = buffer.heartbeat_fields()
    assert fields["reward_metrics"]["success"] == 0.0, "stale generation republished as step 2"
    assert len(fields["sampled_completions"]) == 1, "generation 1's samples leaked into 2"


@pytest.mark.usefixtures("_identity_graded")
def test_a_completion_that_failed_scoring_still_counts_toward_the_denominator():
    """A completion the env could not grade contributes no breakdown but is still part of the
    generation. Counting only the ones that produced a dict would divide by the scored subset --
    biasing every named metric HIGH exactly when scoring is degraded, so a half-broken env reports
    the same number as a healthy one."""
    buffer = RewardObservabilityBuffer()
    buffer.record("prompt-0", "a", 1.0, [{"success": 1.0, "total": 1.0}])
    buffer.record("prompt-0", "b", 0.0, [None])
    buffer.close_generation(1)

    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 0.5}


@pytest.mark.usefixtures("_identity_graded")
def test_a_component_whose_values_are_all_unusable_is_reported_as_zero_not_dropped():
    """Registering a name only once one of its values coerces would delete the component from the
    payload entirely -- and an env whose component is broken for the WHOLE generation is exactly
    when someone needs to see it. A flat 0 reads as "this scored nothing"; absence is
    indistinguishable from "this env has no such component"."""
    score, buffer = _score_buffer(_UnusableComponentEnv())
    score(0, "7")
    buffer.close_generation(1)

    assert buffer.heartbeat_fields()["reward_metrics"] == {
        "broken": 0.0,
        "diverged": 0.0,
        "quality": 0.5,
    }


@pytest.mark.usefixtures("_identity_graded")
def test_a_generation_that_reports_no_components_at_all_leaves_the_metrics_standing():
    """ "No breakdowns" and "breakdowns that all failed" look the same at the seal but mean opposite
    things. A multi-turn episode grades to a scalar and never reports components, so zeroing the
    known metrics for it would publish a scoring outage the env never had -- and the two record
    paths share one buffer, so a run that scores some rows per-completion and some per-episode
    would flip its metrics to 0 on every episode generation."""
    buffer = RewardObservabilityBuffer()
    buffer.record("p", "a", 1.0, [{"success": 1.0, "total": 1.0}])
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0}

    buffer.record("p", "b", 1.0)  # a scored episode, no per-completion breakdown to report
    buffer.close_generation(2)

    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0}, "read as an outage"


def test_a_generation_whose_every_grading_failed_publishes_the_known_names_as_zero():
    """The opposite case to the test above, and the two are only distinguishable by whether any
    breakdown was attempted. Here the env DOES report components and every completion failed to
    score, so the names are known and their true value this generation is 0. Leaving the previous
    generation's numbers standing would show a healthy reward through a total scoring outage --
    the metric would keep reading 1.0 while nothing was being graded at all."""
    buffer = RewardObservabilityBuffer()
    buffer.record("p", "a", 1.0, [{"success": 1.0, "total": 1.0}])
    buffer.close_generation(1)
    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0}

    # a breakdown slot per failed grading: attempted, produced nothing
    buffer.record("p", "b", 0.0, [None])
    buffer.close_generation(2)

    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 0.0}


@pytest.mark.usefixtures("_identity_graded")
def test_one_non_finite_score_cannot_poison_a_whole_components_mean():
    """Summing a NaN in makes the running total NaN forever: every later completion adds to it and
    the name publishes NaN for the rest of the generation. One diverged grading would take out a
    component that scored fine on every other completion -- and `json.dumps` writes bare `NaN`,
    which is not JSON, so a strict reader rejects the whole heartbeat over it."""
    buffer = RewardObservabilityBuffer()
    buffer.record("p", "a", 1.0, [{"quality": 1.0, "total": 1.0}])
    buffer.record("p", "b", 1.0, [{"quality": float("nan"), "total": 1.0}])
    buffer.record("p", "c", 1.0, [{"quality": 1.0, "total": 1.0}])
    buffer.close_generation(1)

    assert buffer.heartbeat_fields()["reward_metrics"] == {"quality": 2.0 / 3.0}


@pytest.mark.usefixtures("_identity_graded")
def test_the_published_payload_is_read_under_one_acquisition():
    """Both fields describe the same generation, so reading them under separate acquisitions lets a
    seal land between the two and the payload tears: metrics from generation N+1 shipped beside
    samples from N. Asserted on the source because reproducing the interleave needs a scoring
    thread to win a specific race -- a test that passes when it loses proves nothing."""
    body = inspect.getsource(RewardObservabilityBuffer.heartbeat_fields)
    snapshot = body[body.index("with self._lock:") : body.index("fields: dict = {}")]

    assert snapshot.count("with self._lock:") == 1, "the two reads can straddle a seal"
    assert "self._latest_metrics" in snapshot
    assert "self._published" in snapshot


def test_the_generation_size_is_the_configured_rollout_count():
    # the counted boundary is only correct if it counts a whole generation. verl runs with
    # test_freq=-1 and val_before_train=False, so every completion reaching the bridge is one of
    # these -- a validation pass would desynchronize the count from the step lines.
    src = " ".join(inspect.getsource(rl_runner._start_reward_runtime).split())
    construction = src[src.index("RewardObservabilityBuffer(") :]
    construction = construction[: construction.index("wandb_link")]
    assert 'generation_size=int(inp["prompts_per_step"]) * int(inp["group_size"])' in construction
    overrides = rl_verl.build_verl_overrides(_overrides_cfg())
    assert "trainer.test_freq=-1" in overrides
    assert "trainer.val_before_train=False" in overrides


@pytest.mark.usefixtures("_identity_graded")
def test_samples_carry_the_step_they_were_generated_at_not_the_current_one():
    """`generated_at_step` names the generation that PRODUCED the completion.

    The buffer is rolling, so a drain that stamps everything in it with the current step
    re-publishes older rollouts as if the model had just produced them -- a reader watching for
    behaviour change sees old text under a new step number.
    """
    score, buffer = _score_buffer(_NamedBreakdownEnv())
    score(0, "7")
    buffer.close_generation(4)

    # steps 5 and 6 generate nothing (the run is stalled, or verl logged without new rollouts).
    buffer.close_generation(5)
    buffer.close_generation(6)

    fields = buffer.heartbeat_fields()
    assert {s["generated_at_step"] for s in fields["sampled_completions"]} == {4}


@pytest.mark.usefixtures("_identity_graded")
def test_the_drain_clears_pending_breakdowns_and_then_repeats_the_last_reading():
    # the drain CLEARS the pending list. between generations there is nothing new to average, and
    # reporting {} there would blank the metric on every heartbeat that lands mid-generation rather
    # than holding the last real reading.
    score, buffer = _score_buffer(_NamedBreakdownEnv())
    score(0, "7")
    buffer.close_generation(1)

    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0, "quality": 0.5}
    assert buffer._pending_totals == {}
    assert buffer._pending_count == 0
    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 1.0, "quality": 0.5}

    score(0, "wrong")
    buffer.close_generation(2)
    assert buffer.heartbeat_fields()["reward_metrics"] == {"success": 0.0, "quality": 0.5}


class _ReleaseHookedLock:
    """The buffer's lock, with a callback fired the instant it is released.

    Landing a grading at that exact moment is what makes the atomicity test deterministic: a real
    thread race can't be, because lock handoff is barging-prone and a sleep long enough to make the
    interleave reliable is a sleep the correct code also passes.
    """

    def __init__(self, lock, on_release):
        self._lock = lock
        self._on_release = on_release

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *exc):
        released = self._lock.__exit__(*exc)
        self._on_release()
        return released


@pytest.mark.usefixtures("_identity_graded")
def test_metrics_and_samples_in_one_payload_describe_the_same_gradings():
    """The boundary's drain and sample seal must be ONE acquisition, or a payload tears.

    A grading landing between them is not lost -- it just belongs to the NEXT generation. What
    breaks is agreement: its sample would ride this generation's publication while its reward
    doesn't reach this generation's metrics, so `sampled_completions` and `reward_metrics` describe
    different gradings and a reader diagnosing a reward drop sees a sample the numbers next to it
    never scored.
    """
    score, buffer = _score_buffer(
        _NamedBreakdownEnv(),
        prompts=["prompt-0", "prompt-1"],  # distinct: both survive the per-prompt dedup
        examples=[{"gt": "7"}, {"gt": "7"}],
    )
    score(0, "7")  # success 1.0

    landed = []

    def _land_a_grading_the_instant_the_lock_drops():
        if landed:
            return  # record() takes the same lock; don't recurse into it
        landed.append(True)
        score(1, "wrong")  # success 0.0

    buffer._lock = _ReleaseHookedLock(buffer._lock, _land_a_grading_the_instant_the_lock_drops)
    buffer.close_generation(1)
    fields = buffer.heartbeat_fields()

    assert landed, "the hook never fired, so this asserts nothing about atomicity"
    # the second grading landed after the whole section, so NEITHER signal carries it.
    assert fields["reward_metrics"] == {"success": 1.0, "quality": 0.5}
    assert [sample["reward"] for sample in fields["sampled_completions"]] == [1.0]


def test_the_first_sample_bearing_heartbeat_is_forced():
    # the liveness daemon can claim a step before the stdout loop reaches it, and a step-gated stage
    # drops a second payload at an already-committed step. without force, the first heartbeat
    # carrying samples is exactly the one most likely to be suppressed.
    src = inspect.getsource(rl_runner._ingest_step_metrics)
    forced = src[src.index("if not state.sent_first_metrics or") :]
    forced = forced[: forced.index("gpu=gpu_diagnostics")]
    assert "heartbeat_fields = _reward_observability()" in src
    assert "force=True" in forced
    assert "**heartbeat_fields" in forced


def test_the_liveness_fields_hook_carries_reward_observability():
    # the rl_step liveness wrap is what publishes between stdout lines. without the fields hook
    # merging it, samples would only ever reach the wire on the one forced first-metrics heartbeat.
    src = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    assert 'fields=lambda: {"metrics_last": list(metrics_last), **_reward_observability()}' in src


def test_the_generation_boundary_is_the_step_line_and_the_heartbeat_never_drains():
    """The boundary is verl's step line, and it is the ONLY drain.

    Both halves are load-bearing and neither is reachable from a unit test -- `_reward_observability`
    is a local of a body that needs a model, a dataset and a verl interpreter. If the heartbeat
    drained as well, a 30s tick landing mid-generation would publish the subset of completions that
    had finished by then. If the step line did not close the generation, nothing ever would, and the
    buffer would report its first generation for the whole run.
    """
    entry_src = " ".join(inspect.getsource(rl_train.run_rl_train).split())

    hook = entry_src[entry_src.index("def _reward_observability()") :]
    hook = hook[: hook.index("with liveness_heartbeat(")]
    assert "reward_runtime.observability.heartbeat_fields()" in hook
    assert "close_generation" not in hook, "the heartbeat drains; the boundary would be bypassed"

    # sealed on the new-step branch, and BEFORE the preview reads the published rows so the logged
    # sample and the heartbeat describe the same generation.
    stdout_loop = " ".join(inspect.getsource(rl_runner._execute_rl_child).split())
    stdout_loop = stdout_loop[stdout_loop.index('progress["step"] = step_number') :]
    assert 'reward_runtime.identity_ledger.seal(progress["step"])' in stdout_loop
    assert 'reward_runtime.observability.close_generation(progress["step"])' in stdout_loop
    assert stdout_loop.index("identity_ledger.seal(") < stdout_loop.index(
        "observability.close_generation("
    )
    assert stdout_loop.index("observability.close_generation(") < stdout_loop.index(
        'samp = reward_runtime.observability.latest_for_step(progress["step"])'
    )
    # and the preview asks for THIS step's rows. the unchecked accessor answers with whatever was
    # published last, which on the drop-spend path belongs to an earlier step.
    assert "observability.latest()" not in stdout_loop, (
        "the preview would print older rows under this step number"
    )


def test_a_step_whose_generation_was_dropped_previews_nothing_rather_than_older_text():
    """The drop-spend path publishes nothing, so the newest rows belong to an EARLIER step.

    `close_generation` spends this step's line on a generation the queue already dropped and
    publishes nothing. `latest` keeps answering with the previous generation's rows, and the caller
    -- which cannot see that no publish happened -- prints them under the new step number, so the
    log claims this step generated text that a different step produced.
    """
    buffer = RewardObservabilityBuffer(generation_size=2)
    buffer.record("pA", "gen-A-a", 1.0)
    buffer.record("pA", "gen-A-b", 1.0)
    buffer.close_generation(10)  # generation A is published under step 10

    # overflow the sealed queue so a later generation is dropped before it was ever named.
    for gen in range(RewardObservabilityBuffer._SEALED_QUEUE_LIMIT + 2):
        buffer.record("pB", f"gen-B{gen}-a", 2.0)
        buffer.record("pB", f"gen-B{gen}-b", 2.0)
    assert buffer._dropped_unnamed, "the queue never dropped a generation, so the path is untested"

    buffer.close_generation(11)  # spent on the drop: nothing is published for step 11

    assert buffer.latest_for_step(11) is None, (
        "step 11 previewed rows that belong to an earlier generation"
    )
    # the stale reading is still reachable, deliberately: the heartbeat reports it as step 10's.
    assert buffer.latest()[1] == "gen-A-b"


def test_a_step_that_did_publish_still_previews_its_own_rows():
    # the control: the ordinary path must keep previewing, or the fix above silences every preview.
    buffer = RewardObservabilityBuffer(generation_size=2)
    buffer.record("p", "gen1-a", 1.0)
    buffer.record("p", "gen1-b", 1.0)

    buffer.close_generation(7)

    assert buffer.latest_for_step(7) == ("p", "gen1-b", 1.0)


def test_a_late_step_line_previews_the_generation_that_step_named():
    # the queue's whole purpose, asserted through the step-checked accessor: a late line names the
    # OLDEST unnamed generation, so that is the one this step may preview.
    buffer = RewardObservabilityBuffer(generation_size=2)
    buffer.record("p", "gen1-a", 1.0)
    buffer.record("p", "gen1-b", 1.0)
    buffer.record("p", "gen2-a", 9.0)  # generation 2 already scoring

    buffer.close_generation(1)

    assert buffer.latest_for_step(1)[1] == "gen1-b"
    assert buffer.latest_for_step(2) is None, "step 2 has not been named yet"


def test_nothing_is_previewed_for_a_step_before_the_first_boundary():
    # `latest` falls back to the open generation so an early preview is not blank. that fallback
    # must NOT leak into the step-checked accessor: those rows have not been named, so no step
    # number is correct for them.
    buffer = RewardObservabilityBuffer()
    buffer.record("p", "first", 0.5)

    assert buffer.latest() == ("p", "first", 0.5)
    assert buffer.latest_for_step(0) is None
    assert buffer.latest_for_step(1) is None


def _run_per_turn_shim(rows, uids, episode_advantages, response_mask=None):
    """execute the rendered shim against a stub verl and return the advantages it writes.

    executing the source is the only test that can fail for the right reason: the shim's whole job
    is to replace a module-global that verl calls by name, and a string assertion would pass just
    as happily on a shim that never installed itself.
    """
    from types import ModuleType, SimpleNamespace

    batch_size = len(uids)
    width = episode_advantages.shape[1]
    spans = np.empty(batch_size, dtype=object)
    turns = np.empty(batch_size, dtype=object)
    for row_index, row in enumerate(rows):
        spans[row_index] = None if row is None else list(row[0])
        turns[row_index] = None if row is None else list(row[1])

    batch = {"advantages": episode_advantages}
    if response_mask is not None:
        batch["response_mask"] = response_mask
    data = SimpleNamespace(
        batch=batch,
        non_tensor_batch={
            "uid": np.array(uids, dtype=object),
            "flash_turn_spans": spans,
            "flash_turn_rewards": turns,
        },
    )

    ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")
    # stock grpo's contribution: the shim must call through to it and build on what it returns.
    ray_trainer.compute_advantage = lambda payload, *args, **kwargs: payload
    ppo = ModuleType("verl.trainer.ppo")
    ppo.ray_trainer = ray_trainer
    stubs = {
        "verl": ModuleType("verl"),
        "verl.trainer": ModuleType("verl.trainer"),
        "verl.trainer.ppo": ppo,
        "verl.trainer.ppo.ray_trainer": ray_trainer,
    }
    with _stubbed_modules(stubs):
        verl_patches.install_per_turn_credit()
        # call the module global by name, exactly as ray_trainer.fit does at its call site.
        out = ray_trainer.compute_advantage(data, adv_estimator="grpo")
    return out.batch["advantages"]


def test_per_turn_credit_shim_centres_each_turn_against_its_group_sibling():
    pytest.importorskip("torch")
    import torch

    # two rollouts and two turns must invert credit on turn 1.
    # episode-level credit cannot produce that per-turn result.
    rows = [
        (((0, 2), (2, 4)), (1.0, 0.0)),
        (((0, 2), (2, 4)), (0.0, 1.0)),
    ]
    advantages = _run_per_turn_shim(rows, ["p0", "p0"], torch.zeros((2, 4), dtype=torch.float32))
    assert advantages[0].tolist() == [0.5, 0.5, -0.5, -0.5]
    assert advantages[1].tolist() == [-0.5, -0.5, 0.5, 0.5]


def test_per_turn_credit_shim_reproduces_the_reference_advantages():
    # pin literals produced by the removed reference builder; deriving them from this shim would
    # make the test assert on itself. the spans also verify broadcast and zero outside a span.
    pytest.importorskip("torch")
    import torch

    spans = [[(0, 3), (3, 5)], [(0, 2), (2, 6)]]
    turns = [[0.25, 0.75], [1.0, 0.5]]
    expected = torch.tensor(
        [
            [-0.375, -0.375, -0.375, 0.125, 0.125, 0.0],
            [0.375, 0.375, -0.125, -0.125, -0.125, -0.125],
        ],
        dtype=torch.float32,
    )
    got = _run_per_turn_shim(
        [
            (tuple(map(tuple, spans[0])), tuple(turns[0])),
            (tuple(map(tuple, spans[1])), tuple(turns[1])),
        ],
        ["p0", "p0"],
        # the episode tensor is what per-turn credit REPLACES, so its value cannot affect the
        # result; zeros make an accidental passthrough visible as a row of zeros.
        torch.zeros((2, 6), dtype=torch.float32),
    )
    assert torch.allclose(got, expected)


def test_per_turn_credit_shim_drops_a_whole_group_when_one_row_is_unusable():
    pytest.importorskip("torch")
    import torch

    # an unscorable row (bridge returned turns=None) must not leave its group centred on a smaller
    # sample: the surviving rows would be compared against a baseline built from a different
    # population than grpo's own. the whole group keeps stock grpo's tensor untouched.
    episode = torch.tensor([[0.3, 0.3, 0.3, 0.3], [-0.3, -0.3, -0.3, -0.3]], dtype=torch.float32)
    advantages = _run_per_turn_shim(
        [(((0, 2), (2, 4)), (1.0, 0.0)), None], ["p0", "p0"], episode.clone()
    )
    assert torch.equal(advantages, episode)


def test_per_turn_credit_shim_leaves_other_groups_on_episode_credit():
    pytest.importorskip("torch")
    import torch

    # the fallback is per group, not per batch: one broken group must not cost every other prompt
    # in the step its per-turn credit.
    episode = torch.zeros((4, 4), dtype=torch.float32)
    episode[2] = 0.7
    episode[3] = -0.7
    rows = [
        (((0, 2), (2, 4)), (1.0, 0.0)),
        (((0, 2), (2, 4)), (0.0, 1.0)),
        (((0, 2), (2, 4)), (1.0, 0.0)),
        None,
    ]
    advantages = _run_per_turn_shim(rows, ["p0", "p0", "p1", "p1"], episode.clone())
    assert advantages[0].tolist() == [0.5, 0.5, -0.5, -0.5]
    # the broken group kept exactly what stock grpo produced. compared against the input tensor
    # rather than a literal: 0.7 has no exact float32 representation, so a literal would compare
    # the shim's output against a value grpo never actually held.
    assert torch.equal(advantages[2], episode[2])
    assert torch.equal(advantages[3], episode[3])


def test_per_turn_credit_shim_ignores_a_turn_no_group_member_emitted():
    pytest.importorskip("torch")
    import torch

    # a zero-width span is a turn the model never produced. it must be excluded from the BASELINE,
    # not merely written nowhere: one sibling emits turn 1 and the other does not, so the emitting
    # row is the only member and its advantage is 0.0. counting the absent member would centre it
    # against a reward for tokens that do not exist -- here (2.0 - (2.0 + 8.0) / 2) = -3.0, a large
    # negative signal on a turn the model actually produced.
    rows = [
        (((0, 2), (2, 4)), (1.0, 2.0)),
        (((0, 2), (2, 2)), (0.0, 8.0)),
    ]
    advantages = _run_per_turn_shim(rows, ["p0", "p0"], torch.zeros((2, 4), dtype=torch.float32))
    assert advantages[0].tolist() == [0.5, 0.5, 0.0, 0.0]
    assert advantages[1].tolist() == [-0.5, -0.5, 0.0, 0.0]


def test_per_turn_credit_shim_keeps_glue_tokens_out_of_the_gradient():
    pytest.importorskip("torch")
    import torch

    # environment replies sit inside the transcript with response_mask 0. a turn span that reaches
    # over one must not hand it advantage -- the model did not generate those tokens.
    mask = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]], dtype=torch.float32)
    rows = [
        (((0, 2), (2, 4)), (1.0, 0.0)),
        (((0, 2), (2, 4)), (0.0, 1.0)),
    ]
    advantages = _run_per_turn_shim(
        rows, ["p0", "p0"], torch.zeros((2, 4), dtype=torch.float32), response_mask=mask
    )
    assert advantages[0].tolist() == [0.5, 0.5, 0.0, -0.5]
    assert advantages[1].tolist() == [-0.5, -0.5, 0.0, 0.5]


def test_per_turn_credit_shim_rejects_a_span_past_the_response_width():
    pytest.importorskip("torch")
    import torch

    # a span beyond the tensor would silently write nothing (python slicing clamps), training on
    # episode credit while the logs claim per-turn. fail loudly instead.
    rows = [(((0, 2), (2, 99)), (1.0, 0.0)), (((0, 2), (2, 4)), (0.0, 1.0))]
    with pytest.raises(ValueError, match="exceeds the response width"):
        _run_per_turn_shim(rows, ["p0", "p0"], torch.zeros((2, 4), dtype=torch.float32))


def test_per_turn_credit_shim_passes_through_a_batch_without_per_turn_metadata():
    # validation batches and any single-turn rollout carry no spans. the shim must return stock
    # grpo's tensor untouched rather than zeroing a batch it cannot credit.
    pytest.importorskip("torch")
    from types import ModuleType, SimpleNamespace

    import torch

    episode = torch.full((2, 3), 0.4, dtype=torch.float32)
    data = SimpleNamespace(
        batch={"advantages": episode.clone()},
        non_tensor_batch={"uid": np.array(["p0", "p0"], dtype=object)},
    )
    ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")
    ray_trainer.compute_advantage = lambda payload, *args, **kwargs: payload
    ppo = ModuleType("verl.trainer.ppo")
    ppo.ray_trainer = ray_trainer
    stubs = {
        "verl": ModuleType("verl"),
        "verl.trainer": ModuleType("verl.trainer"),
        "verl.trainer.ppo": ppo,
        "verl.trainer.ppo.ray_trainer": ray_trainer,
    }
    with _stubbed_modules(stubs):
        verl_patches.install_per_turn_credit()
        out = ray_trainer.compute_advantage(data, adv_estimator="grpo")
    assert torch.equal(out.batch["advantages"], episode)


def test_per_turn_credit_is_resolved_only_for_multi_turn_and_reaches_the_bridge():
    # single-turn envs cannot express per-turn credit (there is one turn), and trl says so while
    # accepting the key. the verl resolver must match that, and must no longer REJECT multi-turn.
    source = inspect.getsource(rl_inputs._resolve_grpo_inputs)
    assert "not supported for multi-turn environments" not in source
    assert '"per_turn_credit": per_turn_credit' in source
    plugin_source = inspect.getsource(rl_runner._write_rl_plugin_config)
    reward_source = inspect.getsource(rl_runner._start_reward_runtime)
    assert '"per_turn_credit": bool(inp["per_turn_credit"])' in plugin_source
    assert 'per_turn_credit=bool(inp["per_turn_credit"])' in reward_source


def test_multi_turn_bridge_returns_turns_only_under_per_turn_credit():
    # the loop keys off the presence of `turns`, so an episode-credit run must not send the key at
    # all: sending it would put every ordinary multi-turn run through the per-turn rewrite.
    class _Env:
        max_turns = 2

        def new_rollout_state(self, example, prepared_prompt):
            return {"prompt": list(prepared_prompt), "messages": list(prepared_prompt)}

        def rollout_rewards_many(self, items):
            from flash.envs.loading.base import RolloutReward

            return [RolloutReward(episode=1.0, turns=(0.25, 0.75)) for _ in items]

    examples = [{"question": "q"}]
    episode_only = _bridge(_Env(), examples=examples, max_turns=2)
    per_turn = _bridge(_Env(), examples=examples, max_turns=2, per_turn_credit=True)
    for bridge in (episode_only, per_turn):
        bridge.start({"index": 0, "session_id": "s", "prompt_ids": []})
    assert episode_only.score({"session_id": "s", "turn_count": 2}) == {"score": 1.0}
    assert per_turn.score({"session_id": "s", "turn_count": 2}) == {
        "score": 1.0,
        "turns": [0.25, 0.75],
    }


def test_multi_turn_bridge_sends_no_turns_when_the_env_vector_is_unusable():
    # score_rollouts canonicalises a bad vector to None. the bridge must forward that None rather
    # than a partial list, so the group falls back cleanly.
    class _Env:
        max_turns = 2

        def new_rollout_state(self, example, prepared_prompt):
            return {"prompt": list(prepared_prompt), "messages": list(prepared_prompt)}

        def rollout_rewards_many(self, items):
            from flash.envs.loading.base import RolloutReward

            # one reward for two turns: the validator rejects the count and drops to None.
            return [RolloutReward(episode=1.0, turns=(0.5,)) for _ in items]

    bridge = _bridge(_Env(), examples=[{"question": "q"}], max_turns=2, per_turn_credit=True)
    bridge.start({"index": 0, "session_id": "s", "prompt_ids": []})
    assert bridge.score({"session_id": "s", "turn_count": 2}) == {"score": 1.0, "turns": None}


def _drive_multi_turn_episode(
    *,
    stop_reasons,
    env,
    per_turn_credit=True,
    max_turns=4,
    monkeypatch=None,
    multi_modal_data=None,
    return_instance=False,
    raw_prompt=None,
    identity_ledger=None,
    rollout_ordinal=0,
    prompt_descriptors=None,
):
    """run the real child loop end to end against a real bridge, returning its agent loop output.

    the loop is the thing under test here: it is what appends turn_spans and what tells the bridge
    how many turns to score. a hand-built bridge conversation would restate that bookkeeping
    instead of exercising it.
    """
    from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn

    monkeypatch.setenv("FLASH_VERL_MULTITURN_URL", "http://bridge.invalid")
    monkeypatch.setenv("FLASH_VERL_MAX_TURNS", str(max_turns))
    monkeypatch.setenv("FLASH_VERL_MAX_MODEL_LEN", "4096")
    # generous enough that the per-turn cap never binds here: these tests exercise termination and
    # span accounting, and a cap that clipped a turn would change what they are measuring.
    monkeypatch.setenv("FLASH_VERL_MAX_COMPLETION_TOKENS", "4096")

    prepared_raw_prompt = (
        raw_prompt if raw_prompt is not None else [{"role": "user", "content": "go"}]
    )

    class _Tokenizer:
        """one codepoint per token, so spans are readable straight off response_ids."""

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(int(i)) for i in ids)

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def apply_chat_template(self, messages, **kwargs):
            rendered = []
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    rendered.append(content)
                    continue
                rendered.extend(
                    "<image>" if block["type"] == "image" else block["text"]
                    for block in content or ()
                )
            return "".join(rendered)

    tokenizer = _Tokenizer()
    processor = _BridgeGlueProcessor(tokenizer)
    bridge = _bridge(
        env,
        examples=[{"question": "q"}],
        env_prompts=[prepared_raw_prompt],
        max_turns=max_turns,
        prompt_ids=[[1, 2, 3]],
        prompt_descriptors=[list(prompt_descriptors or ())],
        processor=processor,
        tokenizer=tokenizer,
        per_turn_credit=per_turn_credit,
        identity_ledger=identity_ledger,
    )
    routes = bridge.routes()

    def bridge_post(url, path, payload):
        return routes[path](payload)

    class _Base:
        """mirrors the parts of verl's AgentLoopBase the loop actually calls."""

        def __init__(self):
            self.tokenizer = tokenizer
            self.processor = processor
            self.rollout_config = SimpleNamespace(response_length=256)
            self.server_manager = self
            self._sent = list(stop_reasons)
            # every generate call's media, so a test can assert the pixels ride along on turn 2+.
            self.generate_media = []

        def _get_mm_processor_kwargs(self, audio_data=None):
            return {}

        async def process_multi_modal_info(self, messages):
            # verl extracts media from the message CONTENT BLOCKS (rl_dataset.process_multi_modal_info
            # keys off `item["type"] in {"image", "video"}`), so record what shape this was handed:
            # a loop that flattened the prompt to text before extracting would arrive here with
            # nothing left to find, and the images would be silently gone.
            self.mm_info_contents = [message.get("content") for message in messages]
            data_uris = [
                block["image"]
                for message in messages
                if isinstance(message.get("content"), list)
                for block in message["content"]
                if block.get("type") == "image"
                and isinstance(block.get("image"), str)
                and block["image"].startswith("data:image/")
            ]
            if data_uris:
                from flash.content.multimodal import (
                    decode_image_descriptors,
                    normalize_image_source,
                )

                descriptors = [normalize_image_source(uri, None) for uri in data_uris]
                return {"images": decode_image_descriptors(descriptors, None)}
            return dict(multi_modal_data or {})

        async def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3]

        async def generate(
            self,
            *,
            request_id,
            prompt_ids,
            sampling_params,
            image_data=None,
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=None,
        ):
            self.generate_media.append(image_data)
            text, stop_reason = self._sent.pop(0)
            return SimpleNamespace(
                token_ids=[ord(c) for c in text],
                log_probs=[0.0] * len(text),
                num_preempted=0,
                stop_reason=stop_reason,
            )

    captured = {}

    def agent_loop_output(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    loop_class = grpo_multiturn.build_flash_grpo_multi_turn_agent_loop(
        register=lambda name: lambda cls: cls,
        agent_loop_base=_Base,
        agent_loop_output=agent_loop_output,
        bridge_post=bridge_post,
    )

    driven = {}

    async def _go():
        instance = loop_class()
        # the loop offloads the bridge's blocking posts onto this executor, so it has to be the
        # one actually running the coroutine.
        instance.loop = asyncio.get_running_loop()
        driven["instance"] = instance
        await instance.run(
            {},
            raw_prompt=(
                raw_prompt if raw_prompt is not None else [{"role": "user", "content": "go"}]
            ),
            index=0,
            flash_rollout_identity={
                "optimizer_step": 1,
                "sample_index": 0,
                "rollout_ordinal": rollout_ordinal,
                "validate": False,
            },
        )

    asyncio.run(_go())
    if return_instance:
        return captured, driven["instance"]
    return captured


class _SpanEnv:
    """an env that returns one reward per turn it was actually told about."""

    max_turns = 4

    def __init__(self):
        self.recorded: list[str] = []

    def new_rollout_state(self, example, prepared_prompt):
        return {"prompt": list(prepared_prompt), "messages": list(prepared_prompt)}

    def record_model_turn(self, state, text):
        self.recorded.append(text)
        state["messages"].append({"role": "assistant", "content": text})

    def rollout_done(self, state, max_turns=None):
        return False

    def env_reply(self, messages, state):
        return [{"role": "user", "content": "next"}]

    def rollout_rewards_many(self, items):
        from flash.envs.loading.base import RolloutReward

        return [RolloutReward(episode=1.0, turns=tuple(0.5 for _ in self.recorded)) for _ in items]


def test_multi_turn_child_preserves_exact_identity_through_start_and_score(monkeypatch):
    from flash.engine.worker.train.rl.rollout.identity import RolloutIdentityLedger

    ledger = RolloutIdentityLedger(1, 2)
    ledger.register(
        [
            {
                "optimizer_step": 1,
                "sample_index": 0,
                "rollout_ordinal": ordinal,
                "validate": False,
            }
            for ordinal in range(2)
        ]
    )
    for ordinal in range(2):
        _drive_multi_turn_episode(
            stop_reasons=[("answer", "completed")],
            env=_SpanEnv(),
            monkeypatch=monkeypatch,
            max_turns=1,
            identity_ledger=ledger,
            rollout_ordinal=ordinal,
        )
    ledger.seal(1)
    ledger.finalize({1})


def test_an_image_prompt_reaches_media_extraction_and_the_rollout(monkeypatch):
    # verl rewrites the parquet prompt into blocks before the child loop. media extraction must see
    # those blocks before the model-facing transcript is flattened, and the decoded pixels must stay
    # attached to the emitted episode for training.
    env = _SpanEnv()
    image, descriptor = _bridge_image()
    out, instance = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed")],
        env=env,
        monkeypatch=monkeypatch,
        max_turns=1,
        multi_modal_data={"images": [image]},
        prompt_descriptors=[descriptor],
        return_instance=True,
        raw_prompt=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/x.png"},
                    {"type": "text", "text": "\nDescribe this image."},
                ],
            }
        ],
    )

    assert out["multi_modal_data"] == {"images": [image]}
    assert env.recorded == ["ab"]
    contents = instance.mm_info_contents
    assert isinstance(contents[0], list), (
        "media extraction was handed flattened text; the images are already gone by this point"
    )
    assert {"type": "image", "image": "/tmp/x.png"} in contents[0]
    image.close()


def test_a_truncated_final_turn_still_earns_per_turn_credit_for_the_turns_before_it(monkeypatch):
    # the bridge does not record an aborted turn into env state (MultiTurnBridge.step returns before
    # record_model_turn), so the env returns no reward for it. the loop must not span it either, or
    # the vector is one short of the spans, score_rollouts rejects the count, and the row -- and via
    # the shim its whole group -- silently drops to episode credit.
    env = _SpanEnv()
    out = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed"), ("cd", "aborted")], env=env, monkeypatch=monkeypatch
    )
    assert env.recorded == ["ab"], "the aborted turn was recorded into env state"
    assert len(out["extra_fields"]["flash_turn_spans"]) == 1, "the aborted turn was spanned"
    assert out["extra_fields"]["flash_turn_rewards"] == [0.5], (
        "per-turn credit was dropped: the span count disagreed with the env's reward count"
    )
    # the truncated tokens stay in the transcript -- it is what the model actually generated and what
    # a later turn would have conditioned on -- but they are masked OUT of the loss, because the env
    # never saw or scored them. transcript: turn "ab", the env's "next" glue, then the aborted "cd".
    assert out["num_turns"] == 2
    assert out["response_ids"] == [ord(c) for c in "abnextcd"]
    assert out["response_mask"] == [1, 1, 0, 0, 0, 0, 0, 0]


def test_an_unspanned_truncated_turns_tokens_stay_in_the_transcript_but_out_of_the_loss(
    monkeypatch,
):
    # an aborted turn stays in the transcript but out of the loss: later turns condition on its
    # tokens, while response_mask=0 prevents training on credit the environment never assigned.
    env = _SpanEnv()
    out = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed"), ("cd", "aborted")], env=env, monkeypatch=monkeypatch
    )
    assert out["response_ids"][-2:] == [ord("c"), ord("d")], "the truncated turn's tokens were lost"
    assert out["response_mask"][-2:] == [0, 0], (
        "the truncated turn is still in the loss, so it trains on the previous turn's credit"
    )
    # and the masking must not have cost the episode its span/reward alignment: those are separate
    # fields, so excluding the turn from the loss leaves the surviving turn's credit intact.
    assert out["extra_fields"]["flash_turn_rewards"] == [0.5]


def test_multi_turn_rollout_carries_the_prompts_images_into_every_turn(monkeypatch):
    # an image-bearing prompt tokenizes to placeholder tokens that carry no pixels. the engine needs
    # the decoded media alongside them on EVERY generate call, because each turn re-sends the whole
    # prefix -- turn 2 conditioning on placeholders alone is the same failure as turn 1 doing it.
    # the training pass re-tokenizes the episode through the processor, so the output has to carry
    # the media too.
    env = _SpanEnv()
    image, descriptor = _bridge_image()
    sentinel = [image]
    raw_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/x.png"},
                {"type": "text", "text": "describe it"},
            ],
        }
    ]
    out, instance = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed")] * 4,
        env=env,
        monkeypatch=monkeypatch,
        multi_modal_data={"images": sentinel},
        raw_prompt=raw_prompt,
        prompt_descriptors=[descriptor],
        return_instance=True,
    )
    assert instance.generate_media == [sentinel] * 4, (
        "the prompt's images did not reach every turn's generate call"
    )
    assert out["multi_modal_data"] == {"images": sentinel}, (
        "the episode was emitted without the media the training pass re-tokenizes against"
    )
    image.close()


def test_dynamic_images_are_cumulative_ordered_masked_and_snapshotted(monkeypatch):
    from flash.content.multimodal import image_descriptors_to_data_uris
    from flash.engine.worker.train.core.child.glue import (
        parent_image_digests,
        processor_image_digests,
    )

    source_images_and_descriptors = [_bridge_image(color) for color in ("red", "green", "blue")]
    source_images = [item[0] for item in source_images_and_descriptors]
    descriptors = [item[1] for item in source_images_and_descriptors]
    data_uris = image_descriptors_to_data_uris(descriptors, None)

    class _DynamicImageEnv(_SpanEnv):
        def env_reply(self, messages, state):
            reply_index = len(self.recorded) - 1
            reply = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"observation {reply_index}"},
                        {"type": "image", "image": data_uris[reply_index]},
                    ],
                }
            ]
            state["messages"].extend(reply)
            return reply

        def rollout_done(self, state, max_turns=None):
            return len(self.recorded) >= 4

    env = _DynamicImageEnv()
    out, instance = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed")] * 4,
        env=env,
        monkeypatch=monkeypatch,
        return_instance=True,
    )

    assert [None if images is None else len(images) for images in instance.generate_media] == [
        None,
        1,
        2,
        3,
    ]
    output_images = out["multi_modal_data"]["images"]
    assert processor_image_digests(instance.processor, output_images) == parent_image_digests(
        instance.processor, descriptors, None
    )
    generated_text = "".join(
        chr(token_id)
        for token_id, included in zip(out["response_ids"], out["response_mask"], strict=True)
        if included
    )
    environment_text = "".join(
        chr(token_id)
        for token_id, included in zip(out["response_ids"], out["response_mask"], strict=True)
        if not included
    )
    assert generated_text == "abababab"
    assert environment_text.count("<image>") == 3
    assert all(
        len(snapshot) == index for index, snapshot in enumerate(instance.generate_media[1:], 1)
    )

    for image in output_images:
        image.close()
    for image in source_images:
        image.close()


def test_a_text_only_multi_turn_rollout_sends_no_media(monkeypatch):
    # the control: a text-only prompt must not start shipping an empty media payload. verl treats an
    # empty multi_modal_data dict as a multimodal row, so passing {} rather than None would push a
    # text-only episode down the processor path it has no pixels for.
    env = _SpanEnv()
    out, instance = _drive_multi_turn_episode(
        stop_reasons=[("ab", "completed")] * 4,
        env=env,
        monkeypatch=monkeypatch,
        return_instance=True,
    )
    assert instance.generate_media == [None] * 4
    assert out["multi_modal_data"] is None


def test_every_turn_is_spanned_when_none_of_them_abort(monkeypatch):
    # the negative control: without an abort, spans and rewards are one per turn and per-turn credit
    # is live. an over-eager skip would show up here as a missing span.
    env = _SpanEnv()
    out = _drive_multi_turn_episode(
        stop_reasons=[
            ("ab", "completed"),
            ("cd", "completed"),
            ("ef", "completed"),
            ("gh", "completed"),
        ],
        env=env,
        monkeypatch=monkeypatch,
    )
    assert env.recorded == ["ab", "cd", "ef", "gh"]
    assert len(out["extra_fields"]["flash_turn_spans"]) == 4
    assert out["extra_fields"]["flash_turn_rewards"] == [0.5, 0.5, 0.5, 0.5]


def test_the_rl_trainer_stores_the_frozen_base_in_bf16():
    """VERL-150: verl's fsdp.yaml default is fp32, which doubles the trainer's resident base.

    Keep the frozen base in bf16; FSDP computes in bf16 while LoRA weights remain fp32.
    The OPD half is pinned in test_opd_train.
    """
    overrides = rl_verl.build_verl_overrides(_overrides_cfg())
    want = "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
    # exact, not substring: "x=bfloat16" is a substring of "+x=bfloat16", so the obvious `in`
    # assertion passes against a spelling hydra REJECTS here ("Could not append to config. An item
    # is already at ..."). the key is declared in the yaml and takes a BARE override -- the inverse
    # of enable_sleep_mode, which is dataclass-only and requires `+`. neighbouring keys, opposite
    # prefixes. see ISSUES.md VERL-150.
    assert want in overrides
    assert f"+{want}" not in overrides, "must not be + prefixed"
    # ref is deliberately NOT set. it reads like a second resident copy and is not one:
    # ray_trainer.py:897 aliases ref_policy_wg to the actor worker whenever ref_in_actor holds, and
    # flash parses lora_rank with minimum=1 (schema/__init__.py:485) so it always holds. setting it
    # would free nothing. this asserts the absence so a future reader has to re-derive the above
    # rather than pattern-match it back in.
    assert not [o for o in overrides if "ref.fsdp_config.model_dtype" in o]


def test_grpo_builds_its_verl_child_env_from_the_allowlist():
    """Regression (rl_train.py): do not pass platform and user secrets to the verl child.

    Scoring remains behind the localhost bridge, so the child needs runtime settings and the bridge
    URL only. This is source-tested because the path requires a GPU and installed verl.
    """
    source = inspect.getsource(rl_runner._build_rl_child_env)
    assert "env_for_verl = _build_verl_child_env(" in source
    assert "env_for_verl = dict(os.environ)" not in source, (
        "grpo must not copy the whole parent environment into the verl child"
    )


def test_the_verl_child_allowlist_keeps_the_kernel_choice_but_drops_credentials(tmp_path):
    """The grpo child still needs the FLA_ kernel backend the parent picked (on sm100 FLA_TILELANG=0
    is a correctness floor, not a preference), so the allowlist must carry it while excluding the
    credentials. This pins both halves of that split for the path grpo now shares with sft/opd."""
    from flash.engine.worker.train.entry.sft_train import _build_verl_child_env

    keep = {"FLA_TILELANG": "0", "CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "/cache/hf"}
    drop = {"HF_TOKEN": "hub-secret", "GITHUB_TOKEN": "gh-secret", "RUNPOD_API_KEY": "prov-secret"}
    saved = {k: os.environ.get(k) for k in (*keep, *drop)}
    try:
        os.environ.update(keep)
        os.environ.update(drop)
        child = _build_verl_child_env(shim_dir=str(tmp_path), wandb_enabled=False)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    for key, value in keep.items():
        assert child[key] == value
    for key in drop:
        assert key not in child


def test_grpo_final_driver_env_scrubs_declared_prefixed_secrets_before_ray(monkeypatch, tmp_path):
    from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV

    declared = (
        "CUDA_SECRET",
        "FLA_CREDENTIAL",
        "PYTHONPATH",
        "WANDB_USER_SECRET",
        "WANDB_API_KEY",
    )
    for name in declared:
        monkeypatch.setenv(name, f"synthetic-{name.lower()}")
    monkeypatch.setenv(SECRET_ENV_KEYS_ENV, ",".join(declared))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("FLA_TILELANG", "0")
    monkeypatch.setenv("WANDB_MODE", "offline")
    files = {
        "shim_dir": str(tmp_path),
        "rank_device_claims": str(tmp_path / "rank-device-claims"),
        "plugin_config_path": str(tmp_path / "plugin-config.json"),
    }
    child = rl_runner._build_rl_child_env(
        {"multi_turn": False}, files, ["wandb"], "http://127.0.0.1:9/"
    )

    for name in declared:
        if name not in {"PYTHONPATH", "WANDB_API_KEY"}:
            assert name not in child
    assert child["PYTHONPATH"] == files["shim_dir"]
    assert SECRET_ENV_KEYS_ENV not in child
    assert "WANDB_API_KEY" in child
    assert child["WANDB_MODE"] == "offline"
    assert child["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert child["FLA_TILELANG"] == "0"
    assert child["FLASH_VERL_REWARD_URL"] == "http://127.0.0.1:9/"
    assert child["VERL_USE_EXTERNAL_MODULES"] == "flash_grpo_plugin"
    assert child["FLASH_GRPO_PLUGIN_CONFIG_PATH"] == files["plugin_config_path"]
    assert child["FLASH_RANK_DEVICE_CLAIMS"] == files["rank_device_claims"]

    captured = {}

    class _EmptyChildStream:
        def __init__(self, proc, **kwargs):
            captured["stream_proc"] = proc

        def __iter__(self):
            return iter(())

        def wait_and_classify(self):
            return 0

        def terminate(self):
            raise AssertionError("the successful child should not be terminated")

    process = SimpleNamespace()

    def popen(*args, **kwargs):
        captured["popen_env"] = kwargs["env"]
        return process

    monkeypatch.setattr(rl_runner.subprocess, "Popen", popen)
    monkeypatch.setitem(
        rl_runner._execute_rl_child.__globals__, "adopt_orphaned_descendants", lambda: None
    )
    monkeypatch.setattr(rl_runner, "ChildOutputTail", lambda: object())
    monkeypatch.setattr(rl_runner, "VerlChildSilenceWatchdog", lambda *args, **kwargs: object())
    monkeypatch.setattr(rl_runner, "_GrpoSubprocessStream", _EmptyChildStream)
    reward_runtime = SimpleNamespace(
        observability=SimpleNamespace(parent_work=object()), wandb_link={}
    )

    assert (
        rl_runner._execute_rl_child(
            python_bin="python",
            overrides=[],
            env_for_verl=child,
            # this test is about the child env, but the process census still needs the step count
            # it validates against; the production caller always supplies a fully resolved inp.
            inp={"steps": 1},
            state=rl_runner._StepMetricState(),
            reward_runtime=reward_runtime,
            _reward_observability=dict,
        )
        == 0
    )
    assert captured["popen_env"] is child
    assert captured["popen_env"]["PYTHONPATH"] == files["shim_dir"]
    assert captured["stream_proc"] is process


def test_grpo_finalization_carries_the_completed_step():
    """Regression (rl_train.py): write_train_meta emits `<phase>_train_done` and then the
    terminal `done`. Called without `step`, both land stepless and overwrite the stepped `rl_trained`
    heartbeat above them, and actual_steps_run() deliberately returns 0 for a non-training stage with
    no step -- so a cancel arriving between finalization and DONE reprices a fully trained run at
    zero steps and bills $0. The sft and opd finalizers already pass their final step; grpo was the
    only one that did not.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(rl_train._write_terminal_metadata)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_train_meta"
    ]
    assert len(calls) == 1, "expected exactly one grpo finalization call"
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "step" in keywords, "grpo finalization must pass the completed step"
    step_arg = keywords["step"]
    assert isinstance(step_arg, ast.Name)
    assert step_arg.id == "steps_run"

    # finalize only forwards a positive int, so a stepless spelling would silently no-op.
    from flash.engine.worker.train.core.lifecycle import finalize

    forwarding = inspect.getsource(finalize.write_train_meta)
    assert '"step": int(step)' in forwarding


# ---------------------- fail-closed shim markers and the fp8 probe ----------------------


def _shim_files(tmp_path):
    return {
        "shim_dir": str(tmp_path),
        "shim_py": str(tmp_path / "sitecustomize.py"),
        "shim_markers": str(tmp_path / "applied_shims.txt"),
        "rank_device_claims": str(tmp_path / "rank_device_claims.txt"),
        "plugin_config_path": str(tmp_path / "flash_grpo_plugin_config.json"),
        "multi_turn": False,
    }


def test_write_rl_shim_copies_plugin_bundle_and_serializes_expected_markers(tmp_path):
    files = _shim_files(tmp_path)
    inp = {
        "model_id": "Qwen/Qwen3.5-9B",
        # one card: the rank/device assertion renders empty, so it owes no marker here. the
        # multi-card case is pinned by the test below.
        "dp_cards": 1,
        "reentrant_checkpointing": True,
        "multimodal": False,
        "entropy_quantile": 0.2,
        "per_turn_credit": False,
        "stop_sequences": ("</answer>",),
        "image_pad_token_id": None,
        "structured_outputs": None,
        "save_at_steps": (7,),
        "steps": 20,
        "warmstart_adapter": "adapter",
        "kl_coef": 0.04,
        "multi_turn": False,
    }
    rl_runner._write_rl_shim(inp, files)
    rl_runner._write_rl_plugin_config(inp, files, gdn_reset_arch=None, loggers=[])

    assert files["expected_shims"] == [
        "nonempty-response-mask",
        "exact-rollout-identity",
        "reentrant-checkpointing",
        "text-lora-targeting",
        "entropy-quantile",
        "stop-sequences",
        "exact-save-steps",
        "kl-ref-adapter",
        "lora-rollout-guard",
    ]
    source = Path(files["shim_py"]).read_text()
    compile(source, "sitecustomize.py", "exec")
    assert "tilelang libcudart" in source
    assert "import verl" not in source
    assert (tmp_path / "flash_grpo_plugin.py").is_file()
    assert (tmp_path / "flash_grpo_patches.py").is_file()
    assert (tmp_path / "flash_verl_runtime.py").is_file()
    config = json.loads(Path(files["plugin_config_path"]).read_text())
    assert files["expected_shims"] == grpo_plugin.required_patch_names(config)
    assert config["gdn_model_type"] is None
    assert config["lora_language_prefix"] == "model.language_model"
    assert config["kl_ref_adapter"] is True


def test_plugin_config_puts_the_rank_device_assert_first_when_the_run_spans_cards(tmp_path):
    files = _shim_files(tmp_path)
    inp = {
        "model_id": "Qwen/Qwen3.5-9B",
        "dp_cards": 2,
        "reentrant_checkpointing": True,
        "multimodal": False,
        "entropy_quantile": None,
        "per_turn_credit": False,
        "stop_sequences": (),
        "image_pad_token_id": None,
        "structured_outputs": None,
        "save_at_steps": (),
        "steps": 20,
        "warmstart_adapter": None,
        "kl_coef": 0.0,
        "multi_turn": False,
    }
    rl_runner._write_rl_plugin_config(inp, files, gdn_reset_arch=None, loggers=[])

    assert files["expected_shims"] == [
        "rank-device-assert",
        "nonempty-response-mask",
        "exact-rollout-identity",
        "reentrant-checkpointing",
        "text-lora-targeting",
        "lora-rollout-guard",
    ]
    config = json.loads(Path(files["plugin_config_path"]).read_text())
    assert config["dp_cards"] == 2


def test_gdn_model_type_is_serialized_only_after_capability_resolution():
    configure_source = inspect.getsource(rl_train._configure_rl_child)
    config_source = inspect.getsource(rl_runner._write_rl_plugin_config)
    registry_source = inspect.getsource(grpo_plugin.required_patch_specs)
    assert "require_gdn_boundary_resets(caps, gdn_module)" in configure_source
    assert "_write_rl_plugin_config(" in configure_source
    assert '"gdn_model_type": gdn_reset_arch' in config_source
    assert '"gdn-varlen"' in registry_source
    assert "bool(model_type)" in registry_source


def test_the_stdout_loop_verifies_the_marker_set_at_the_first_step_line():
    """before/at training start: the first step line is the earliest point where sitecustomize is
    provably finished (fragments print while later ones are still applying, so the first OUTPUT
    line would race the file). a missing marker there means the child trains unpatched."""
    stdout_loop = " ".join(inspect.getsource(rl_runner._execute_rl_child).split())
    step_at = stdout_loop.index("step_number = verl_step_number(line)")
    verify_at = stdout_loop.index("verify_applied_shim_markers(shim_markers, expected_shims)")
    assert step_at < verify_at < stdout_loop.index("close_generation")
    # and the entry point wires the files dict (marker path + expected set) into both the loop
    # and the final verdict.
    entry = " ".join(inspect.getsource(rl_train.run_rl_train).split())
    assert "_reward_observability=_reward_observability, files=files," in entry
    assert 'rc, state, files["resume_step"], expected_steps, resume_uploader, files=files' in entry


def test_validate_rl_child_fails_a_run_whose_markers_are_missing(tmp_path):
    state = rl_runner._StepMetricState()
    state.reward_history.append(0.5)
    state.adv_spread_history.append(1.0)
    state.advantage_bounds[1] = (-0.5, 0.5)
    state.grad_norms[1] = 1.0
    marker = tmp_path / "applied_shims.txt"
    marker.write_text("entropy-quantile\n")
    # the complete set passes and falls through to the gradient verdict.
    rl_runner._validate_rl_child(
        0,
        state,
        0,
        1,
        None,
        files={"shim_markers": str(marker), "expected_shims": ["entropy-quantile"]},
    )
    with pytest.raises(RuntimeError, match="never proved"):
        rl_runner._validate_rl_child(
            0,
            state,
            0,
            1,
            None,
            files={
                "shim_markers": str(marker),
                "expected_shims": ["entropy-quantile", "kl-ref-adapter"],
            },
        )


def test_validate_rl_child_classifies_the_shim_exit_code_as_permanent():
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    with pytest.raises(RuntimeError, match="failed to apply") as err:
        rl_runner._validate_rl_child(
            backend_common.SHIM_FRAGMENT_FAILED_EXIT_CODE, rl_runner._StepMetricState(), 0, 1, None
        )
    # permanent by design: the same interpreter fails the same fragment on retry, so it must not
    # be classified as retriable infra.
    assert not isinstance(err.value, RetriableInfraError)
    assert "FLASH_VERL_PYTHON" in str(err.value)


def test_the_fp8_kv_probe_reads_the_child_capability_probe_not_parent_cuda(monkeypatch):
    """backend_common.fused_ce_backend's rule applies here too: opening cuda in this long-lived
    parent retains a context on the devices the verl child is about to own. a stub torch whose
    cuda attribute explodes stands in for 'somebody reintroduced the live probe'."""

    class _ExplodingCuda:
        def __getattr__(self, name):
            raise AssertionError("the fp8 probe touched torch.cuda in the parent")

    exploding_torch = types.ModuleType("torch")
    exploding_torch.cuda = _ExplodingCuda()
    monkeypatch.setitem(sys.modules, "torch", exploding_torch)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    inp = {"steps": 4}
    for caps, cc_ok in (
        ({"capability": [8, 9]}, True),
        ({"capability": [9, 0]}, True),
        ({"capability": [8, 6]}, False),
        ({"capability": None}, False),
        ({}, False),
    ):
        settings = rl_runner._resolve_training_settings(inp, caps)
        assert settings[0] == 4
        assert settings[-1] is cc_ok, caps


def test_multi_turn_bridge_counts_turns_it_actually_ran():
    """The bridge must report the turns it ran, so a collapse to single-turn is visible.

    Every existing terminal gate -- finite gradients, nonzero adapter delta, complete artifacts --
    stays green if a regression ends every episode after one turn, because a one-turn episode
    still trains. These counters are the only evidence that separates "multi-turn ran" from
    "multi-turn was configured". Driven through the real HTTP routes, not the bridge object, so
    the transport the child actually uses is covered.
    """
    env = _BridgeEnv(done_after=3)
    bridge = rl_multi.MultiTurnBridge(
        env,
        examples=[{"q": "a"}],
        env_prompts=[[{"role": "user", "content": "a"}]],
        max_turns=5,
        prompt_ids=[[1]],
        tokenizer=_BridgeGlueTokenizer(),
    )
    server, url = rl_multi.start_reward_server(
        lambda i, s: 1.0, example_count=1, multi_turn_bridge=bridge
    )

    def _post(path, payload):
        request = urllib.request.Request(
            url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=10)

    try:
        assert bridge.turn_accounting() == {
            "episodes_scored": 0,
            "turn_records": 0,
            "max_turns_observed": 0,
            "mean_turns_per_episode": None,
        }
        _post(
            "/multiturn/start",
            {
                "index": 0,
                "session_id": "s",
                "raw_prompt": [{"role": "user", "content": "a"}],
                "prompt_ids": [1],
                "image_count": 0,
                "image_digests": [],
            },
        ).close()
        prefix = [1]
        for turn in range(3):
            completion = str(turn)
            response_ids = [ord(completion)]
            response = json.load(
                _post(
                    "/multiturn/step",
                    {
                        "session_id": "s",
                        "turn_ordinal": turn,
                        "accepted_prefix": prefix,
                        "response_ids": response_ids,
                        "completion_text": completion,
                        "image_count": 0,
                        "image_digests": [],
                    },
                )
            )
            if response["terminal"]:
                break
            prefix.extend([*response_ids, *(ord(character) for character in "next")])
        _post("/multiturn/score", {"session_id": "s", "turn_count": 3}).close()
    finally:
        bridge.shutdown()
        server.shutdown()

    accounting = bridge.turn_accounting()
    assert accounting["episodes_scored"] == 1
    assert accounting["turn_records"] == 3
    assert accounting["max_turns_observed"] == 3
    assert accounting["mean_turns_per_episode"] == 3.0
    # the env really advanced: three assistant turns reached it, not one repeated.
    assert env.recorded == ["0", "1", "2"]


def test_single_turn_episode_is_reported_as_one_turn():
    """A genuinely single-turn episode reports 1, so the counter is not merely 'nonzero'.

    Without this, `turn_records >= 1` would pass for a run that collapsed to one turn per episode,
    which is exactly the regression the accounting exists to catch.
    """
    bridge = rl_multi.MultiTurnBridge(
        _BridgeEnv(done_after=1),
        examples=[{"q": "a"}],
        env_prompts=[[{"role": "user", "content": "a"}]],
        max_turns=5,
        prompt_ids=[[1]],
        tokenizer=_BridgeGlueTokenizer(),
    )
    server, url = rl_multi.start_reward_server(
        lambda i, s: 1.0, example_count=1, multi_turn_bridge=bridge
    )

    def _post(path, payload):
        request = urllib.request.Request(
            url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=10)

    try:
        _post(
            "/multiturn/start",
            {
                "index": 0,
                "session_id": "s",
                "raw_prompt": [{"role": "user", "content": "a"}],
                "prompt_ids": [1],
                "image_count": 0,
                "image_digests": [],
            },
        ).close()
        response = json.load(
            _post(
                "/multiturn/step",
                {
                    "session_id": "s",
                    "turn_ordinal": 0,
                    "accepted_prefix": [1],
                    "response_ids": [ord("x")],
                    "completion_text": "x",
                    "image_count": 0,
                    "image_digests": [],
                },
            )
        )
        assert response["terminal"] is True
        _post("/multiturn/score", {"session_id": "s", "turn_count": 1}).close()
    finally:
        bridge.shutdown()
        server.shutdown()

    accounting = bridge.turn_accounting()
    assert accounting["episodes_scored"] == 1
    assert accounting["turn_records"] == 1
    assert accounting["max_turns_observed"] == 1
    assert accounting["mean_turns_per_episode"] == 1.0


def test_turn_accounting_ignores_a_child_reported_turn_count():
    """The counters must come from the parent's ordinal, not from the child's self-report.

    These exist to prove the child's turn loop really iterated: a collapse to one turn per episode
    keeps every other gate green. Deriving them from `turn_count` in the child's own `/score`
    payload let the subject of the measurement supply the measurement -- a child that generated one
    turn could claim five and the accounting would agree. `step` validates every ordinal against
    `next_turn` before incrementing it, so the parent already knows the answer exactly.

    The payload here claims 5 turns for an episode the parent watched generate 2.
    """
    bridge = rl_multi.MultiTurnBridge(
        _BridgeEnv(done_after=2),
        examples=[{"q": "a"}],
        env_prompts=[[{"role": "user", "content": "a"}]],
        max_turns=5,
        prompt_ids=[[1]],
        tokenizer=_BridgeGlueTokenizer(),
    )
    server, url = rl_multi.start_reward_server(
        lambda i, s: 1.0, example_count=1, multi_turn_bridge=bridge
    )

    def _post(path, payload):
        request = urllib.request.Request(
            url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=10)

    try:
        _post(
            "/multiturn/start",
            {
                "index": 0,
                "session_id": "s",
                "raw_prompt": [{"role": "user", "content": "a"}],
                "prompt_ids": [1],
                "image_count": 0,
                "image_digests": [],
            },
        ).close()
        prefix = [1]
        generated = 0
        for turn in range(5):
            completion = str(turn)
            response_ids = [ord(completion)]
            response = json.load(
                _post(
                    "/multiturn/step",
                    {
                        "session_id": "s",
                        "turn_ordinal": turn,
                        "accepted_prefix": prefix,
                        "response_ids": response_ids,
                        "completion_text": completion,
                        "image_count": 0,
                        "image_digests": [],
                    },
                )
            )
            generated += 1
            if response["terminal"]:
                break
            prefix.extend([*response_ids, *(ord(character) for character in "next")])
        assert generated == 2, "the env was configured to end the episode after two turns"
        # the child claims more turns than it generated.
        _post("/multiturn/score", {"session_id": "s", "turn_count": 5}).close()
    finally:
        bridge.shutdown()
        server.shutdown()

    accounting = bridge.turn_accounting()
    assert accounting["episodes_scored"] == 1
    assert accounting["turn_records"] == 2, "the child's inflated turn_count reached the accounting"
    assert accounting["max_turns_observed"] == 2
    assert accounting["mean_turns_per_episode"] == 2.0


def test_a_failed_score_does_not_inflate_the_turn_accounting():
    """Accounting must describe episodes that were really scored, not requests that arrived.

    The counters were incremented on entry to `/multiturn/score`, before the scorer confirmed
    anything. `turn_accounting()` is published from the runner's `finally` path, so a run whose
    episodes failed to score still wrote turn totals into its durable notes -- inflated evidence
    for exactly the collapsed-multi-turn regression the counters exist to detect.
    """

    class _Unscorable(_BridgeEnv):
        def rollout_rewards_many(self, items):
            raise RuntimeError("env scoring failed")

    bridge = rl_multi.MultiTurnBridge(
        _Unscorable(done_after=1),
        examples=[{"q": "a"}],
        env_prompts=[[{"role": "user", "content": "a"}]],
        max_turns=5,
        prompt_ids=[[1]],
        tokenizer=_BridgeGlueTokenizer(),
    )
    server, url = rl_multi.start_reward_server(
        lambda i, s: 1.0, example_count=1, multi_turn_bridge=bridge
    )

    def _post(path, payload):
        request = urllib.request.Request(
            url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=10)

    try:
        _post(
            "/multiturn/start",
            {
                "index": 0,
                "session_id": "s",
                "raw_prompt": [{"role": "user", "content": "a"}],
                "prompt_ids": [1],
                "image_count": 0,
                "image_digests": [],
            },
        ).close()
        _post(
            "/multiturn/step",
            {
                "session_id": "s",
                "turn_ordinal": 0,
                "accepted_prefix": [1],
                "response_ids": [ord("x")],
                "completion_text": "x",
                "image_count": 0,
                "image_digests": [],
            },
        ).close()
        with pytest.raises(urllib.error.HTTPError):
            _post("/multiturn/score", {"session_id": "s", "turn_count": 4}).close()
    finally:
        bridge.shutdown()
        server.shutdown()

    accounting = bridge.turn_accounting()
    assert accounting["episodes_scored"] == 0
    assert accounting["turn_records"] == 0
    assert accounting["max_turns_observed"] == 0
    assert accounting["mean_turns_per_episode"] is None


def test_turn_accounting_reaches_the_durable_notes_not_only_the_heartbeat():
    """The counters have to land in `metrics.json` notes, which is what a terminal proof reads.

    The heartbeat's `metrics_last` row is a live view that a completed run's artifact bundle does
    not carry, so publishing there alone leaves the evidence unreadable after the fact. Assert both
    that the terminal writer forwards the bridge totals and that the notes builder keeps them.
    """
    terminal_source = inspect.getsource(rl_train._write_terminal_metadata)
    assert "multi_turn_accounting=(" in terminal_source
    assert "reward_runtime.multi_turn_bridge.turn_accounting()" in terminal_source

    accounting = {
        "episodes_scored": 128,
        "turn_records": 585,
        "max_turns_observed": 5,
        "mean_turns_per_episode": 4.5703125,
    }
    notes = rl_verl._build_verl_train_notes(
        _notes_inp(), **_notes_common(), multi_turn_accounting=accounting
    )
    assert notes["multi_turn_accounting"] == accounting
    # a copy, so a later bridge update cannot retroactively rewrite a published record.
    assert notes["multi_turn_accounting"] is not accounting


def test_single_turn_run_records_multi_turn_accounting_as_an_explicit_none():
    """Single-turn runs must record the key, so absence stays distinguishable from omission.

    A terminal gate cannot tell "this run had no episode loop" from "the counters were never
    wired" if the key is simply missing, and the second case is the one that hides a collapse.
    """
    notes = rl_verl._build_verl_train_notes(_notes_inp(), **_notes_common())
    assert "multi_turn_accounting" in notes
    assert notes["multi_turn_accounting"] is None


def _verl_step_line(step: int, *, adv_min: float, adv_max: float, grad_norm: float = 1.0) -> str:
    """one verl LocalLogger step line, in the exact shape the child prints."""
    return (
        f"step:{step} - critic/rewards/mean:1.0 - critic/rewards/max:1.0 - "
        f"critic/rewards/min:1.0 - critic/advantages/mean:0.0 - "
        f"critic/advantages/max:{adv_max} - critic/advantages/min:{adv_min} - "
        f"actor/pg_loss:0.0 - actor/grad_norm:{grad_norm}"
    )


def test_resumed_grpo_ignores_the_replayed_resume_step_bounds():
    """A resumed child replays its resume step; those bounds belong to the previous attempt.

    `child_io.append_step_metrics` documents the replay, and `_finalize_advantage_evidence`
    requires exactly `resume_step + 1 .. horizon`. Admitting the replayed line therefore reports
    the resume step as an `extra` step and fails a healthy resumed run at its terminal verdict.
    """
    from flash.engine.worker.train.entry import rl_train_runner

    state = rl_runner._StepMetricState()
    state.resume_step = 2
    observability = dict

    # the replayed line for the step the previous attempt already completed.
    rl_runner._ingest_step_metrics(
        _verl_step_line(2, adv_min=-0.5, adv_max=0.5),
        _notes_inp(),
        state,
        observability,
    )
    assert 2 not in state.advantage_bounds
    assert 2 not in state.grad_norms

    # the first genuinely new step is recorded.
    rl_runner._ingest_step_metrics(
        _verl_step_line(3, adv_min=-0.25, adv_max=0.75),
        _notes_inp(),
        state,
        observability,
    )
    assert sorted(state.advantage_bounds) == [3]
    assert state.grad_norms == {3: 1.0}

    # and the terminal verdict accepts the run instead of reporting step 2 as extra.
    rl_train_runner._finalize_advantage_evidence(state, 2, 3)
    assert state.advantage_bounds_evidence == [
        {"step": 3, "min": -0.25, "max": 0.75, "spread": 1.0}
    ]


def test_resumed_grpo_seeds_the_dump_watermark_at_the_resume_boundary():
    """The replayed step must not be sealed: only `resume_step + 1 ..` are ever registered.

    The stream loop dumps a sample and seals identities whenever the step differs from
    `last_dump_step`. Starting that watermark at -1 makes the replayed resume step look new, so
    `RolloutIdentityLedger.seal` raises "has no registered rollout identity set" and kills a
    resumed run at its first output line.
    """
    from flash.engine.worker.train.entry import rl_train_runner

    source = " ".join(inspect.getsource(rl_train_runner._execute_rl_child).split())
    assert "if resume_step: last_dump_step[0] = resume_step" in source

    from flash.engine.worker.train.rl.rollout.identity import RolloutIdentityLedger

    # the ledger a resumed run builds: registration starts after the resume boundary.
    ledger = RolloutIdentityLedger(1, 2)
    with pytest.raises(ValueError, match="no registered rollout identity set"):
        ledger.seal(2)


def test_build_verl_overrides_sizes_max_num_seqs_to_the_rollout_batch():
    """The rollout engine must be provisioned for the batch the step submits, not verl's default.

    verl leaves ``rollout.max_num_seqs`` at 1024 (rollout.yaml:79). vllm derives
    ``max_cudagraph_capture_size = min(max_num_seqs * 2, 512)`` over the ladder
    ``[1,2,4] + range(8,256,8) + range(256,513,16)`` -- 51 sizes at the default, captured twice
    when lora specialization is on -- and a gdn/mamba hybrid additionally reserves one recurrent
    state block per decode slot. Both are paid up front, inside the gpu_memory_utilization budget,
    so a 32-sequence run died at graph capture on 234, 358 and 460 GB alike.
    """
    o = rl_verl.build_verl_overrides(_overrides_cfg(prompts_per_step=8, group_size=4))
    assert "actor_rollout_ref.rollout.max_num_seqs=32" in o
    # and never verl's default, which is what made the reservation fixed rather than proportional.
    assert not any(override == "actor_rollout_ref.rollout.max_num_seqs=1024" for override in o)


def test_build_verl_overrides_floors_max_num_seqs_for_a_tiny_rollout_batch():
    # a batch of 1 would otherwise capture a single graph size and push every wider decode step
    # onto the eager path.
    o = rl_verl.build_verl_overrides(_overrides_cfg(prompts_per_step=1, group_size=1))
    assert "actor_rollout_ref.rollout.max_num_seqs=16" in o
