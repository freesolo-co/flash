"""CPU unit tests for the shared verl subprocess harness (verl_common)."""

from __future__ import annotations

import builtins
import contextlib
import ctypes
import errno
import inspect
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from flash.engine.worker import rl_verl
from flash.engine.worker import verl_common as vc


@pytest.mark.parametrize(
    ("rollout_batch", "expected"),
    [(1, 1), (8, 8), (16, 8), (128, 8), (4, 4), (12, 6), (9, 3), (10, 5), (11, 1), (91, 7)],
)
def test_agent_loop_workers_always_divides_the_rollout_batch(rollout_batch, expected):
    assert vc.agent_loop_workers(rollout_batch) == expected


def test_agent_loop_workers_invariant_holds_for_every_batch():
    # verl asserts the split is exact, so a non-divisor aborts the run before step 1.
    for batch in range(1, 600):
        workers = vc.agent_loop_workers(batch)
        assert batch % workers == 0
        assert 1 <= workers <= min(8, batch)


def test_agent_loop_workers_rejects_a_nonpositive_batch():
    with pytest.raises(ValueError, match="rollout_batch must be positive"):
        vc.agent_loop_workers(0)


def test_latest_global_step_dir_picks_highest(tmp_path):
    for step in (1, 5, 20, 3):
        os.makedirs(tmp_path / f"global_step_{step}" / "actor" / "huggingface", exist_ok=True)
    # a non-checkpoint dir must be ignored, not crash the scan.
    os.makedirs(tmp_path / "not_a_step", exist_ok=True)
    actor, step = vc.latest_global_step_dir(str(tmp_path))
    assert step == 20
    assert actor == os.path.join(str(tmp_path), "global_step_20", "actor")


def test_latest_global_step_dir_resolves_the_sft_layout(tmp_path):
    # verl's sft trainer saves straight into global_step_N with no actor/ level. hardcoding the rl
    # convention here pointed the model merger at a path that does not exist, and AutoConfig
    # reinterpreted it as a hub repo id -- so the run died on a bogus HFValidationError.
    for step in (1, 2):
        os.makedirs(tmp_path / f"global_step_{step}" / "huggingface", exist_ok=True)
    actor, step = vc.latest_global_step_dir(str(tmp_path))
    assert step == 2
    assert actor == os.path.join(str(tmp_path), "global_step_2")


def test_resolve_checkpoint_actor_dir_prefers_the_nested_rl_layout(tmp_path):
    # rl checkpoints carry huggingface/ at both levels (the merger wants the actor one).
    os.makedirs(tmp_path / "actor" / "huggingface")
    os.makedirs(tmp_path / "huggingface")
    assert vc.resolve_checkpoint_actor_dir(str(tmp_path)) == os.path.join(str(tmp_path), "actor")


def test_resolve_checkpoint_actor_dir_falls_back_when_no_marker_exists(tmp_path):
    # an interrupted save has no huggingface/ anywhere; name the rl path the caller likely wanted.
    os.makedirs(tmp_path / "actor")
    assert vc.resolve_checkpoint_actor_dir(str(tmp_path)) == os.path.join(str(tmp_path), "actor")
    empty = tmp_path / "empty"
    os.makedirs(empty)
    assert vc.resolve_checkpoint_actor_dir(str(empty)) == str(empty)


def test_latest_global_step_dir_raises_when_empty(tmp_path):
    with pytest.raises(RuntimeError, match="no global_step_N checkpoint"):
        vc.latest_global_step_dir(str(tmp_path))


def test_stamp_adapter_dir_provenance_sets_base_and_revision(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": None, "r": 16}))
    vc.stamp_adapter_dir_provenance(str(tmp_path), "org/model", "deadbeef")
    out = json.loads(cfg.read_text())
    assert out["base_model_name_or_path"] == "org/model"
    assert out["revision"] == "deadbeef"
    # untouched fields survive the stamp.
    assert out["r"] == 16


def test_stamp_adapter_dir_provenance_rejects_base_mismatch(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": "org/other"}))
    with pytest.raises(RuntimeError, match="does not match validated target"):
        vc.stamp_adapter_dir_provenance(str(tmp_path), "org/model")


def test_resolve_verl_python_prefers_preset(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/opt/verl/bin/python")
    assert vc.resolve_verl_python(str(tmp_path)) == "/opt/verl/bin/python"


@pytest.mark.parametrize("blank", ["", "   "])
def test_resolve_verl_python_treats_an_empty_preset_as_unset(monkeypatch, tmp_path, blank):
    # a worker IMAGE can export FLASH_VERL_PYTHON itself, and [worker_env] can only SET a key, never
    # delete one -- so omitting it from a spec leaves the image's interpreter in place. an empty
    # value is the only way a run can say "ignore the image's verl and provision the pinned fork",
    # and the error at rl_verl.py's mask_truncated_completions gate names exactly this remedy.
    calls = []
    monkeypatch.setenv("FLASH_VERL_PYTHON", blank)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    python_bin = vc.resolve_verl_python(str(tmp_path))

    assert python_bin.endswith("/verl-venv/bin/python")
    assert vc.VERL_REQUIREMENT in calls[1]


def test_worker_env_remedies_are_copy_pasteable_toml():
    # a [worker_env] snippet in an error gets pasted into a config verbatim, so it has to survive the
    # real parser: '[worker_env] KEY = "..."' reads fine in prose but is invalid TOML, because a
    # table header must end its line. a blocked run would just hit a second, more confusing error.
    import re
    import tomllib

    from flash.engine.worker import rl_verl

    # only assignment forms -- '[worker_env] can set a key but never delete one' is prose, not a
    # snippet, and carries no '=' to paste.
    pattern = re.compile(r"\[worker_env\][^\n]*?[A-Z_]+\s*=\s*(\"[^\"]*\"|'[^']*')")
    snippets = [m.group(0) for m in pattern.finditer(inspect.getsource(rl_verl))]
    assert snippets, "expected rl_verl to advertise at least one [worker_env] remedy"

    for snippet in snippets:
        # a valid snippet is the header, a newline, then the assignment -- exactly what we tell users.
        header, _, assignment = snippet.partition("]")
        parsed = tomllib.loads(f"{header}]\n{assignment.split('as ')[-1].strip()}")
        assert parsed == {"worker_env": {"FLASH_VERL_PYTHON": ""}}
        # and the prose must not run the header into the assignment on one line.
        assert not re.match(r"\[worker_env\]\s+[A-Z_]+\s*=", snippet), snippet


def _fake_verl_venv(tmp_path, *, stamp: str | None):
    """materialize a verl-venv as an earlier attempt would have left it on the pod."""
    venv = tmp_path / "verl-venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python").write_text("")
    if stamp is not None:
        (venv / "flash-verl-requirement").write_text(stamp)
    return venv


def _record_run(calls, *, keep_check: bool = False):
    """stand in for subprocess.run, creating the venv dir `uv venv` would have created."""

    def fake_run(command, check):
        calls.append((command, check) if keep_check else command)
        if command[:2] == ["uv", "venv"]:
            os.makedirs(os.path.join(command[2], "bin"), exist_ok=True)

    return fake_run


def test_resolve_verl_python_installs_pinned_gpu_dependencies(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    python_bin = vc.resolve_verl_python(str(tmp_path))

    assert python_bin.endswith("/verl-venv/bin/python")
    assert calls[0][:2] == ["uv", "venv"]
    install = calls[1]
    assert vc.VERL_REQUIREMENT == (
        "verl @ git+https://github.com/freesolo-co/verl@b7492fa3b7ab843294d06dbf754e887950f559c7"
    )
    assert vc.VERL_REQUIREMENT in install
    assert "liger-kernel" in install
    assert "bitsandbytes>=0.49" in install
    assert "qwen-vl-utils" in install
    assert "torchvision" in install
    assert "xgrammar==0.1.25" in install
    assert "tqdm" in install
    assert "pyarrow" in install
    assert len(calls) == 2
    # the stamp is written only after a successful install, so a crashed install is never reused.
    stamp = tmp_path / "verl-venv" / "flash-verl-requirement"
    assert stamp.read_text() == vc.VERL_REQUIREMENT


def test_resolve_verl_python_reuses_a_venv_built_from_the_current_pin(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    _fake_verl_venv(tmp_path, stamp=vc.VERL_REQUIREMENT)

    vc.resolve_verl_python(str(tmp_path))

    # reinstalling verl's torch/vllm on every retry would cost many minutes of paid gpu time.
    assert calls == []


@pytest.mark.parametrize(
    "stale", ["verl @ git+https://github.com/freesolo-co/verl@" + "0" * 40, None]
)
def test_resolve_verl_python_rebuilds_a_venv_that_is_not_the_current_pin(
    monkeypatch, tmp_path, stale
):
    # a retry reuses the pod workdir. a venv from an earlier pin (or a partial install, stamp=None)
    # must be rebuilt rather than silently training on the wrong verl.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    venv = _fake_verl_venv(tmp_path, stamp=stale)
    (venv / "marker").write_text("from the previous attempt")

    vc.resolve_verl_python(str(tmp_path))

    assert [c[:2] for c in calls] == [["uv", "venv"], ["uv", "pip"]]
    assert not (venv / "marker").exists()


def test_resolve_verl_python_clears_a_venv_whose_creation_was_interrupted(monkeypatch, tmp_path):
    # a pod killed during `uv venv` leaves a pyvenv.cfg but no bin/python. uv then refuses to write
    # into that directory ("a virtual environment already exists"), so leaving it in place would wedge
    # every later retry. it must be removed before uv runs.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    venv = tmp_path / "verl-venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("half-written")

    vc.resolve_verl_python(str(tmp_path))

    assert not (venv / "pyvenv.cfg").exists()
    assert [c[:2] for c in calls] == [["uv", "venv"], ["uv", "pip"]]


def test_verl_pin_is_an_immutable_commit_on_the_freesolo_fork():
    # a branch or tag would let the runtime move under a pinned flash release.
    _, _, ref = vc.VERL_REQUIREMENT.partition("git+")
    url, _, commit = ref.rpartition("@")
    assert url == "https://github.com/freesolo-co/verl"
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_verl_pin_matches_the_version_opd_requires_exactly():
    # the pin MUST stay on the verl 0.8.0 base. opd_plugin patches 0.8.0 internals and imports
    # verl.trainer.main_ppo_sync, which verl deleted after 0.8.0, so a pin built on a newer base
    # installs a verl that fails opd's exact-version gate and cannot import its own entrypoint.
    from flash.engine.worker import opd_plugin as plugin

    assert plugin._STRUCTURED_RUNTIME_EXACT_VERSIONS["verl"] == "0.8.0"
    # asserting the constant alone would let a newer-base commit land silently, so bind the pinned
    # commit itself to that version. this is the sha of the truncation-mask and 3d position id
    # commits cherry-picked onto the v0.8.0 tag; moving the pin must be a deliberate edit here, with
    # the base re-verified.
    _, _, ref = vc.VERL_REQUIREMENT.partition("git+")
    _, _, commit = ref.rpartition("@")
    assert commit == "b7492fa3b7ab843294d06dbf754e887950f559c7"


def test_resolve_verl_python_installs_wandb_best_effort_when_requested(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls, keep_check=True))

    vc.resolve_verl_python(str(tmp_path), install_wandb=True)

    assert vc.VERL_REQUIREMENT in calls[1][0]
    assert calls[2] == (
        ["uv", "pip", "install", "--python", str(tmp_path / "verl-venv/bin/python"), "wandb"],
        False,
    )


def _probe_interpreter(tmp_path, name, body):
    """write a stub interpreter that answers the RolloutConfig probe like a real python would."""
    stub = tmp_path / name
    stub.write_text("#!/bin/sh\n" + body + "\n")
    stub.chmod(0o755)
    return str(stub)


def test_verl_supports_rollout_field_true_when_field_declared(tmp_path):
    fork = _probe_interpreter(tmp_path, "fork-python", "echo 1")
    assert vc.verl_supports_rollout_field(fork, "mask_truncated_completions") is True


def test_verl_supports_rollout_field_false_when_field_absent(tmp_path):
    # stock verl 0.8.0: RolloutConfig has no such field, so the probe prints 0.
    stock = _probe_interpreter(tmp_path, "stock-python", "echo 0")
    assert vc.verl_supports_rollout_field(stock, "mask_truncated_completions") is False


def test_verl_supports_rollout_field_false_when_verl_missing(tmp_path):
    # import error inside the probe: nonzero exit must read as unsupported, not crash the caller.
    broken = _probe_interpreter(tmp_path, "broken-python", "exit 1")
    assert vc.verl_supports_rollout_field(broken, "mask_truncated_completions") is False


def test_verl_supports_rollout_field_false_when_interpreter_missing(tmp_path):
    # a bogus FLASH_VERL_PYTHON must not raise OSError out of the capability check.
    missing = str(tmp_path / "does-not-exist")
    assert vc.verl_supports_rollout_field(missing, "mask_truncated_completions") is False


def test_resolve_verl_python_returns_preset_unmodified(monkeypatch, tmp_path):
    # flash does not own a preset interpreter and must never mutate it; capability is checked
    # separately by verl_supports_rollout_field.
    calls = []
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/opt/verl/bin/python")
    monkeypatch.setattr(vc.subprocess, "run", lambda *a, **k: calls.append(a))

    assert vc.resolve_verl_python(str(tmp_path)) == "/opt/verl/bin/python"
    assert calls == []


def test_run_verl_training_streams_steps_and_returns_code():
    seen: list[int] = []
    lines: list[str] = []
    beats: list[int] = []
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'foo step: 1 bar'; echo 'step: 2'; echo done"],
        env=dict(os.environ),
        on_step=seen.append,
        on_line=lines.append,
        heartbeat=lambda: beats.append(1),
        heartbeat_interval_s=0.0,
    )
    assert code == 0
    assert seen == [1, 2]
    assert lines[-1] == "done\n"
    # heartbeat_interval_s=0 => fires on every scanned line (3 lines here).
    assert len(beats) >= 1


def test_run_verl_training_propagates_nonzero_exit():
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'step: 1'; exit 7"],
        env=dict(os.environ),
        on_step=lambda _s: None,
    )
    assert code == 7


def test_run_verl_training_terminates_child_when_callback_fails():
    def fail(_line):
        raise RuntimeError("checkpoint upload failed")

    with pytest.raises(RuntimeError, match="checkpoint upload failed"):
        vc.run_verl_training(
            ["bash", "-c", "echo ready; sleep 30"],
            env=dict(os.environ),
            on_line=fail,
        )


def test_run_verl_training_kills_the_grandchild_not_just_the_direct_child(tmp_path):
    # verl spawns vllm's EngineCore as a grandchild. killing only the direct child reparents it to
    # init with its cuda context intact, stranding the gpu for every later run. the child here traps
    # SIGTERM and exits without touching its own child, so only a process-group signal reaches the
    # grandchild -- exactly the shape of the real leak.
    marker = tmp_path / "grandchild.pid"
    # "ready" is withheld until the grandchild has recorded its pid, so teardown -- which on_line
    # triggers -- can never race ahead of the marker write.
    script = (
        "trap 'exit 0' TERM; "
        f"bash -c 'echo $$ > {marker}; sleep 30' & "
        f"for _ in $(seq 1 500); do [ -s {marker} ] && break; sleep 0.02; done; "
        "echo ready; "
        "sleep 30"
    )

    def fail(_line):
        raise RuntimeError("stream died")

    with pytest.raises(RuntimeError, match="stream died"):
        vc.run_verl_training(["bash", "-c", script], env=dict(os.environ), on_line=fail)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.exists(), "grandchild never recorded its pid"
    grandchild = int(marker.read_text().strip())

    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"grandchild {grandchild} survived teardown and is still holding the gpu")


def test_child_output_tail_retains_bounded_recent_lines():
    tail = vc.ChildOutputTail(limit=3)
    for i in range(6):
        tail.record(f"line{i}\n")
    # bounded: the oldest are evicted, the most recent survive in order.
    assert tail.tail() == ["line3", "line4", "line5"]
    assert tail.tail(limit=2) == ["line4", "line5"]
    # a limit wider than the buffer is not an error and does not pad.
    assert tail.tail(limit=99) == ["line3", "line4", "line5"]


def test_child_output_tail_drops_blank_lines_and_caps_line_width():
    tail = vc.ChildOutputTail()
    tail.record("\n")
    tail.record("   \n")  # whitespace-only survives rstrip("\n") and is intentionally kept
    tail.record("x" * 5000 + "\n")
    kept = tail.tail()
    assert "" not in kept, "a bare newline must not occupy a slot in the window"
    widest = max(len(line) for line in kept)
    assert widest <= vc._CHILD_TAIL_LINE_CHARS, (
        f"retained a {widest}-char line; verl prints multi-KB config blocks and the tail "
        "has to ride inside an uploaded heartbeat payload"
    )


def test_run_verl_training_records_child_output_into_the_tail():
    """the child's words must reach the tail, because the parent's stdout reaches no log stream."""
    tail = vc.ChildOutputTail()
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'ray placement group pending'; echo 'step: 1'; echo 'wedged here'"],
        env=dict(os.environ),
        tail=tail,
    )
    assert code == 0
    captured = tail.tail()
    # this is the diagnostic that was unretrievable in production: the child's last line before it
    # stopped producing output (ISSUES VERL-061).
    assert captured[-1] == "wedged here"
    assert "ray placement group pending" in captured


def test_run_verl_training_without_a_tail_is_unchanged():
    """tail is opt-in; omitting it must not alter streaming or the exit code."""
    lines: list[str] = []
    code = vc.run_verl_training(
        ["bash", "-c", "echo 'step: 4'; exit 3"],
        env=dict(os.environ),
        on_line=lines.append,
    )
    assert code == 3
    assert lines == ["step: 4\n"]


def test_stall_tail_fields_reports_only_before_the_first_step():
    tail = vc.ChildOutputTail()
    tail.record("ray: placement group pending\n")

    # pre-first-step: this is the blind window, so the child's words must be carried out.
    fields = vc.stall_tail_fields(0, tail)
    assert fields == {"child_tail": ["ray: placement group pending"]}

    # once training progresses the step/loss stream is the diagnostic; the tail would be pure
    # payload bloat on every tick.
    assert vc.stall_tail_fields(1, tail) == {}
    assert vc.stall_tail_fields(500, tail) == {}


def test_stall_tail_fields_is_empty_when_the_child_has_said_nothing():
    # an empty key would claim the child spoke and said nothing, which is a different fact from
    # "the child has produced no output at all".
    assert vc.stall_tail_fields(0, vc.ChildOutputTail()) == {}


def test_stall_tail_fields_reports_how_long_the_child_has_been_silent():
    # the whole point: a child still loading shards and a child wedged forever both present a fully
    # populated tail whose newest line looks plausible. only whether the tail CHANGED separates them,
    # and without this the comparison has to be reconstructed by hand from consecutive heartbeats.
    tail = vc.ChildOutputTail()
    staleness = vc.ChildTailStaleness()
    tail.record("Started a local Ray instance\n")

    first = vc.stall_tail_fields(0, tail, staleness=staleness)
    assert first["child_tail_silent_ticks"] == 0

    # two more ticks with the child saying nothing new: same tail, rising silence.
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 1
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 2

    # the child speaks again, so it is slow rather than stuck and the counter resets.
    tail.record("loading checkpoint shards 1/4\n")
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 0


def test_child_tail_silence_survives_the_retention_limit():
    # staleness is counted from lines WRITTEN, not from the retained window: once the ring buffer is
    # full its contents can keep changing while its length does not, and a length-based comparison
    # would then report a talking child as silent.
    tail = vc.ChildOutputTail(limit=3)
    staleness = vc.ChildTailStaleness()
    for i in range(3):
        tail.record(f"line{i}\n")
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 0
    tail.record("line3\n")  # evicts line0; the deque stays length 3
    assert len(tail.tail()) == 3
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 0


def test_child_tail_silence_is_measured_from_the_childs_first_line():
    # a child silent for the first ticks then talking must not be credited with the silence that
    # preceded its first line -- otherwise a slow starter reports as long-wedged the moment it speaks.
    tail = vc.ChildOutputTail()
    staleness = vc.ChildTailStaleness()
    for _ in range(4):
        assert vc.stall_tail_fields(0, tail, staleness=staleness) == {}
    tail.record("first words\n")
    assert vc.stall_tail_fields(0, tail, staleness=staleness)["child_tail_silent_ticks"] == 0


def test_stall_tail_fields_omits_silence_when_no_tracker_is_supplied():
    # the field must not appear as a fabricated 0 for callers that do not track staleness: absent and
    # "zero ticks silent" are different claims.
    tail = vc.ChildOutputTail()
    tail.record("only line\n")
    assert vc.stall_tail_fields(0, tail) == {"child_tail": ["only line"]}


def test_stall_tail_fields_narrows_to_the_most_recent_lines():
    tail = vc.ChildOutputTail()
    for i in range(vc.STALL_TAIL_LINES + 25):
        tail.record(f"line{i}\n")
    carried = vc.stall_tail_fields(0, tail)["child_tail"]
    assert len(carried) == vc.STALL_TAIL_LINES
    # the most recent lines are the ones that matter: they are what the child said last.
    assert carried[-1] == f"line{vc.STALL_TAIL_LINES + 24}"


# ---------------------- child teardown escalates to SIGKILL ----------------------
# these tests drive real processes through the real escalation, which needs three kernel features
# this repo only ever runs teardown on: fork, /proc for group membership, and libc for the subreaper
# probe. the rest of the suite is os-independent, so guard rather than fail on a platform that
# cannot answer (codex[bot]). checked at runtime like the /proc/self/maps test in test_worker_stack,
# instead of on the platform name, so the condition is the capability actually required.
_needs_process_teardown = pytest.mark.skipif(
    not hasattr(os, "fork") or not os.path.isdir("/proc"),
    reason="teardown tests drive real process groups: needs os.fork and /proc",
)


# PR_SET_CHILD_SUBREAPER / PR_GET_CHILD_SUBREAPER, from linux/prctl.h.
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37


def _child_subreaper_setting() -> int:
    """This process's current subreaper flag, skipping the test where prctl cannot answer."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    current = ctypes.c_int(0)
    if libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(current), 0, 0, 0) != 0:
        pytest.skip("PR_GET_CHILD_SUBREAPER unavailable")  # pragma: no cover - linux 3.4+
    return current.value


@contextlib.contextmanager
def _child_subreaper_enabled():
    """Adopt orphaned grandchildren for the duration, then restore the previous setting.

    pid 1 in `Dockerfile.worker` adopts them for free; a shared pytest process does not, so these
    tests ask for the same behaviour explicitly. Leaving it set would change subprocess semantics for
    every LATER test in the process -- they would adopt orphans they never reap, accumulating
    zombies and making failures order-dependent (codex[bot]).
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    previous = _child_subreaper_setting()
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        pytest.skip("PR_SET_CHILD_SUBREAPER unavailable")  # pragma: no cover - linux 3.4+
    try:
        yield
    finally:
        libc.prctl(_PR_SET_CHILD_SUBREAPER, previous, 0, 0, 0)


@pytest.fixture
def subreaper():
    """`_child_subreaper_enabled` as a fixture, for the tests that need adopted grandchildren."""
    with _child_subreaper_enabled():
        yield


@pytest.fixture
def quick_teardown_grace(monkeypatch):
    """shorten the escalation grace so a test that deliberately ignores SIGTERM stays fast.

    every child below refuses the term on purpose, so each one waits out the full production grace
    before escalating -- 30s of a 30.6s file for three tests. the constant is read from the module
    at call time, so patching it needs no production seam (codex[bot]).

    it is only ever shortened. what these tests assert is that SIGKILL is what performs the
    termination and that nothing is left unreaped, and neither depends on how long the grace was.
    """
    monkeypatch.setattr(vc, "_TEARDOWN_GRACE_S", 0.5)
    return 0.5


@_needs_process_teardown
def test_kill_process_group_escalates_to_sigkill_when_sigterm_is_ignored(quick_teardown_grace):
    # the whole point of this helper is the escalation, so the child has to actually ignore SIGTERM.
    # a child that dies on the term would pass just as well against a bare killpg, which is the bug.
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('ready', flush=True)\n"
            "time.sleep(300)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        # wait until the handler is installed, otherwise the term could land before SIG_IGN and the
        # child would die to the term -- passing without ever exercising the escalation.
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        started = time.monotonic()
        vc.kill_process_group(child)
        assert child.poll() is not None, "child ignoring SIGTERM survived kill_process_group"
        # negative-signal returncode identifies which signal actually reaped it.
        assert child.returncode == -signal.SIGKILL
        assert time.monotonic() - started < 60
    finally:
        if child.poll() is None:  # pragma: no cover - only on an unexpected failure
            child.kill()
            child.wait(timeout=10)
        if child.stdout is not None:
            child.stdout.close()


@_needs_process_teardown
def test_kill_process_group_reaps_a_grandchild_that_outlives_the_leader(quick_teardown_grace):
    # the real shape of this failure: the verl trainer dies on the term but its vllm EngineCore
    # grandchild ignores it. escalating off proc.wait() alone returns the instant the leader is
    # reaped, leaving the grandchild holding the cuda context -- so drive the escalation off the
    # group, and prove it by having the leader exit cleanly on SIGTERM.
    leader_src = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c',\n"
        '    "import signal,time\\n"\n'
        '    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"\n'
        "    \"print('gready', flush=True)\\n\"\n"
        '    "time.sleep(300)\\n"], stdout=subprocess.PIPE, text=True)\n'
        "assert g.stdout.readline().strip() == 'gready'\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_src],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    grandchild_pid = None
    try:
        assert leader.stdout is not None
        grandchild_pid = int(leader.stdout.readline().strip())
        started = time.monotonic()
        vc.kill_process_group(leader)
        assert leader.poll() is not None
        # the leader dying is not the property under test -- the grandchild being gone is. poll
        # rather than assert instantly: sigkill delivery and reaping are asynchronous, so a bare
        # check here races the kernel and flakes.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:  # pragma: no cover - only when the escalation genuinely fails
            pytest.fail("grandchild ignoring SIGTERM survived kill_process_group")
        assert time.monotonic() - started < 60
    finally:
        if grandchild_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(grandchild_pid, signal.SIGKILL)
        if leader.poll() is None:  # pragma: no cover - only on an unexpected failure
            leader.kill()
            leader.wait(timeout=10)
        if leader.stdout is not None:
            leader.stdout.close()


def test_a_group_whose_only_member_is_a_zombie_is_not_read_as_alive():
    # Dockerfile.worker runs python directly as pid 1 with no init, so an orphaned EngineCore is
    # never reaped and sits as a zombie indefinitely. killpg(pgid, 0) succeeds for that group, so
    # driving the escalation off addressability alone burned the whole drain deadline on every
    # teardown even though the cuda context was already released -- and sigkill cannot clear a
    # zombie, only reaping can (codex[bot]). fork rather than Popen: subprocess reaps for you, which
    # is the very behaviour the worker image lacks.
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns to pytest
        os.setsid()
        time.sleep(300)
        os._exit(0)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and os.getpgid(pid) != pid:
            time.sleep(0.02)
        pgid = os.getpgid(pid)
        assert vc._process_group_alive(pgid), "a running group must read as alive"

        os.kill(pid, signal.SIGKILL)
        # deliberately unreaped: waitpid here would clear the zombie and destroy the case.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not vc._process_is_zombie(pid):
            time.sleep(0.02)
        assert vc._process_is_zombie(pid), "child should be an unreaped zombie"
        assert os.killpg(pgid, 0) is None, "the zombie group is still addressable"

        assert not vc._process_group_alive(pgid), (
            "a zombie-only group must not read as alive, or teardown waits out the full deadline"
        )
    finally:
        with contextlib.suppress(ProcessLookupError, ChildProcessError):
            os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


def test_teardown_returns_promptly_when_the_survivor_is_only_a_zombie():
    # the cost of the bug, measured. with a zombie-only group reading as alive, kill_process_group
    # spins out the escalation deadline AND then the drain deadline -- two full grace periods --
    # before handing the worker back, though the cuda context went with the exit. the stub stands in
    # for the direct child: subprocess.Popen would reap, and reaping is what the worker image's
    # missing init never does.
    class _ExitedChild:
        def __init__(self, pid):
            self.pid = pid

        def wait(self, timeout=None):
            return 0

    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns to pytest
        os.setsid()
        os._exit(0)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not vc._process_is_zombie(pid):
            time.sleep(0.02)
        assert vc._process_is_zombie(pid), "child should be an unreaped zombie"

        started = time.monotonic()
        vc.kill_process_group(_ExitedChild(pid))
        elapsed = time.monotonic() - started

        assert elapsed < vc._TEARDOWN_GRACE_S, (
            f"teardown took {elapsed:.1f}s waiting on a zombie that no signal can clear"
        )
    finally:
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


def test_an_unreadable_process_status_does_not_read_as_exited(monkeypatch):
    # teardown can run with the worker out of file descriptors, where opening /proc/<pid>/stat
    # raises EMFILE for every live member at once. reading that as a zombie made the whole group
    # report drained, so SIGKILL was never sent and an EngineCore kept its cuda context precisely on
    # a resource-failure path (codex[bot]). only disappearance proves an exit.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    try:
        pgid = os.getpgid(child.pid)
        assert not vc._process_is_zombie(child.pid), "a running pid is not a zombie"

        real_open = builtins.open

        def out_of_descriptors(path, *args, **kwargs):
            if str(path).startswith("/proc/") and str(path).endswith("/stat"):
                raise OSError(errno.EMFILE, "Too many open files")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", out_of_descriptors)
        assert not vc._process_is_zombie(child.pid), "EMFILE is not evidence that the pid exited"
        assert vc._process_group_alive(pgid), (
            "a group whose members are merely unreadable must still be signalled"
        )
    finally:
        child.kill()
        child.wait(timeout=10)


def test_a_missing_process_still_reads_as_exited():
    # the other half of the same split: narrowing to disappearance is only correct if disappearance
    # is still caught, or a group that really has drained would spin out both deadlines.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)  # reaped by Popen, so the pid is fully gone rather than a zombie
    assert vc._process_is_zombie(child.pid), "a reaped pid must not read as alive"


def test_an_empty_group_snapshot_is_rechecked_against_the_kernel(monkeypatch):
    # /proc can be listed just before a fork publishes a new child while the member that WAS there
    # exits inside the same window, so one snapshot can show nobody in a group that still holds the
    # gpu. teardown then returned before SIGKILL and the newly forked child -- which inherited the
    # group and never received the earlier signal -- kept its cuda context (codex[bot]).
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    try:
        pgid = os.getpgid(child.pid)
        monkeypatch.setattr(vc, "_process_group_members", lambda _pgid: [])
        assert vc._process_group_alive(pgid), (
            "an empty walk of a group the kernel still knows is a missed member, not a drain"
        )
    finally:
        child.kill()
        child.wait(timeout=10)


@_needs_process_teardown
def test_a_zombie_only_snapshot_taken_before_a_fork_is_not_a_drain(subreaper):
    """The same race the empty snapshot has, one member later: /proc is listed with the leader
    present, and the leader forks and exits before its status is inspected. The walk is then
    nonempty and zombie-only while the child that inherited the group is alive and unlisted, so a
    verdict taken from it skips SIGKILL and leaves the gpu held (codex[bot])."""
    leader = os.fork()
    if leader == 0:  # pragma: no cover - runs only in the forked child
        os.setpgid(0, 0)
        if os.fork() == 0:
            time.sleep(300)
            os._exit(0)
        os._exit(0)  # the leader exits at once; its child keeps the group
    child = None
    try:
        os.setpgid(leader, leader)
        deadline = time.monotonic() + 10
        while not vc._process_is_zombie(leader):
            assert time.monotonic() < deadline, "the leader never exited"
            time.sleep(0.01)
        child = next(p for p in vc._process_group_members(leader) or () if p != leader)

        # the stale walk: taken before the fork was published, so it sees only the leader.
        scans = [[leader]]
        real = vc._process_group_members
        vc._process_group_members = lambda pgid: scans.pop(0) if scans else real(pgid)
        try:
            assert vc._process_group_alive(leader), (
                "a zombie-only walk taken mid-fork was read as a drain, so SIGKILL is skipped"
            )
        finally:
            vc._process_group_members = real
    finally:
        if child is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(child, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(child, 0)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(leader, 0)


@_needs_process_teardown
def test_a_group_that_really_is_zombie_only_still_drains(subreaper):
    # and the recheck must not make the drain unreachable. a zombie holds the group id, so signal 0
    # keeps answering for one -- the escalation would burn both deadlines on every teardown, which
    # is the defect `_process_group_alive` exists to prevent.
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs only in the forked child
        os.setpgid(0, 0)
        os._exit(0)
    try:
        os.setpgid(pid, pid)
        deadline = time.monotonic() + 10
        while not vc._process_is_zombie(pid):
            assert time.monotonic() < deadline, "the member never exited"
            time.sleep(0.01)
        assert vc._process_group_addressable(pid), "a zombie must still hold the group id"
        assert not vc._process_group_alive(pid), (
            "a settled zombie-only group is drained; reporting it alive burns both deadlines"
        )
    finally:
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


def test_the_subreaper_setting_does_not_outlive_the_test_that_enabled_it():
    """Both halves in one test rather than across two: a later test asserting the flag is back to 0
    also passes when it happens to run FIRST, so it would prove nothing under random ordering.

    The restore is exercised from a 0 setting AND from a 1 setting. Only the first is what the suite
    actually runs, but a `finally` that never fires is indistinguishable from a correct one there --
    both end at 1 while the block is open and 0 is where the process already was. Entering with the
    flag set is the case that separates them.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    entered = _child_subreaper_setting()
    try:
        for previous in (0, 1):
            libc.prctl(_PR_SET_CHILD_SUBREAPER, previous, 0, 0, 0)
            with _child_subreaper_enabled():
                assert _child_subreaper_setting() == 1, (
                    "adoption was never enabled, so the teardown tests above prove nothing"
                )
            assert _child_subreaper_setting() == previous, (
                "PR_SET_CHILD_SUBREAPER leaked out, so later tests adopt orphans they never reap"
            )
    finally:
        libc.prctl(_PR_SET_CHILD_SUBREAPER, entered, 0, 0, 0)


def test_an_empty_snapshot_of_a_gone_group_is_still_drained():
    # and the recheck must not make teardown unfailable: when the group is genuinely gone, the walk
    # being empty has to stay an answer or every teardown burns both deadlines.
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    pgid = os.getpgid(child.pid)
    child.wait(timeout=10)
    assert vc._process_group_members(pgid) == []
    assert not vc._process_group_alive(pgid), (
        "a group with no members and no kernel entry is drained"
    )


@_needs_process_teardown
def test_teardown_reaps_an_adopted_grandchild_rather_than_leaving_a_zombie(
    quick_teardown_grace, subreaper
):
    # pid 1 has no init in Dockerfile.worker, so an EngineCore orphaned when the trainer exits is
    # reparented onto the worker. SIGKILL turns it into a zombie that no signal can clear -- only a
    # wait can -- so leaving it costs one permanent process-table entry per failed or cancelled run
    # (codex[bot]). the `subreaper` fixture reproduces that adoption inside this test process.
    leader_src = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c',\n"
        '    "import signal,time\\n"\n'
        '    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"\n'
        "    \"print('gready', flush=True)\\n\"\n"
        '    "time.sleep(300)\\n"], stdout=subprocess.PIPE, text=True)\n'
        "assert g.stdout.readline().strip() == 'gready'\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_src],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    grandchild_pid = None
    try:
        assert leader.stdout is not None
        grandchild_pid = int(leader.stdout.readline().strip())
        vc.kill_process_group(leader)
        # the escalation already proves it was killed; what is under test is that nothing is left in
        # the process table afterwards.
        assert not os.path.exists(f"/proc/{grandchild_pid}"), (
            "an adopted grandchild was killed but never reaped, leaking a pid per run"
        )
        # the direct child is skipped so Popen keeps the status it is owed.
        assert leader.returncode is not None
    finally:
        if grandchild_pid is not None:  # pragma: no cover - only on an unexpected failure
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(grandchild_pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(grandchild_pid, 0)
        if leader.poll() is None:  # pragma: no cover - only on an unexpected failure
            leader.kill()
            leader.wait(timeout=10)
        if leader.stdout is not None:
            leader.stdout.close()


@_needs_process_teardown
def test_a_straggler_that_dies_after_the_deadline_is_reaped_by_the_next_teardown(
    monkeypatch, subreaper
):
    # SIGKILL cannot be refused, but it also cannot be DELIVERED to a process in uninterruptible
    # sleep. such a member can outlast the drain deadline and turn into a zombie afterwards -- past
    # the last wait its own teardown performs -- so with no record no future wait is ever scheduled
    # and the entry is permanent on a pid-1 worker (codex[bot]).
    monkeypatch.setattr(vc, "_UNREAPED_STRAGGLERS", set())

    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs only in the forked child
        os.setpgid(0, 0)
        time.sleep(300)
        os._exit(0)
    try:
        os.setpgid(pid, pid)
        # the state at the drain deadline: the member is ours and STILL RUNNING, so the nonblocking
        # wait cannot take a status. skip a pid that is not in this group so nothing is excluded.
        vc._reap_group_zombies(pid, skip=-1)
        assert pid in vc._UNREAPED_STRAGGLERS, (
            "a live adopted member was forgotten at the deadline, so no later wait can reap it"
        )

        # teardown has returned. the member dies only now, exactly as one leaving uninterruptible
        # sleep with a pending SIGKILL would.
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while os.path.exists(f"/proc/{pid}") and not vc._process_is_zombie(pid):
            assert time.monotonic() < deadline, "the straggler never exited"
            time.sleep(0.01)

        # the NEXT teardown is what must collect it, so drive a real one rather than calling the
        # sweep directly -- otherwise this passes even with the sweep never wired into the path.
        later = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(0)"], start_new_session=True
        )
        later.wait(timeout=10)
        vc.kill_process_group(later)

        assert not os.path.exists(f"/proc/{pid}"), (
            "a straggler that died after its deadline was never reaped, leaking a pid per run"
        )
        assert pid not in vc._UNREAPED_STRAGGLERS
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


@_needs_process_teardown
def test_a_straggler_that_was_never_ours_is_not_tracked_forever(monkeypatch):
    # the other direction: `waitpid` raising ChildProcessError means the pid was never ours, which
    # is the ordinary case anywhere but pid 1. remembering those would grow the set without bound
    # on every teardown, and no wait here could ever clear them.
    monkeypatch.setattr(vc, "_UNREAPED_STRAGGLERS", set())
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    try:
        # a real, live process in its own group -- but reached through a wait that reports it is
        # not this process's child to reap.
        def _not_ours(pid, options):
            raise ChildProcessError()

        monkeypatch.setattr(vc.os, "waitpid", _not_ours)
        vc._reap_group_zombies(os.getpgid(child.pid), skip=-1)
        assert not vc._UNREAPED_STRAGGLERS, (
            "a process this worker cannot reap was tracked anyway, so the set grows unboundedly"
        )
    finally:
        monkeypatch.undo()
        child.kill()
        child.wait(timeout=10)


def test_grpo_teardown_uses_the_shared_escalating_kill():
    # the grpo path used to hand-roll killpg(pid, 15) and swallow the wait timeout, so a vllm
    # EngineCore that ignored the term kept its cuda context and stranded the gpu for later jobs.
    # pin the call site: a bare killpg here would reintroduce exactly that.
    source = inspect.getsource(rl_verl)
    assert "kill_process_group(proc)" in source
    assert "os.killpg" not in source, "grpo teardown must not hand-roll a non-escalating killpg"
