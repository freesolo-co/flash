"""CPU unit tests for the shared verl subprocess harness (verl_common)."""

from __future__ import annotations

import inspect
import json
import os
import time
from types import SimpleNamespace

import pytest

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
    assert any(vc.VERL_REQUIREMENT_URL in arg for arg in calls[1])


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
    assert any(vc.VERL_REQUIREMENT_URL in arg for arg in install)
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


def test_provisioned_venv_can_import_the_entrypoints_flash_launches(monkeypatch, tmp_path):
    # grpo used to stay in flash's own interpreter, so this fallback only ever had to satisfy sft and
    # opd. now every backend routes through it, and Dockerfile.worker records what a verl-only install
    # yields: `main_ppo` dies on ModuleNotFoundError cachetools, `main_ppo_sync` on uvicorn. verl
    # declares none of vllm/cachetools/uvicorn/fastapi yet imports them at module level on the launch
    # path, so an install of VERL_REQUIREMENT alone provisions an interpreter that cannot launch.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    vc.resolve_verl_python(str(tmp_path))

    install = calls[1]
    # the [vllm] extra, as Dockerfile.worker's VERL_SPEC asks for it, on the same commit.
    assert f"verl[vllm] @ {vc.VERL_REQUIREMENT_URL}" in install
    # the extra's own ceiling is vllm<=0.12.0, which registers neither Qwen3.5 arch. pin past it.
    assert "vllm==0.19.1" in install
    for module in ("cachetools", "uvicorn", "fastapi"):
        assert module in install, f"verl imports {module} at module level but never declares it"
    # opd's entrypoint calls tq.init()/tq.close() and verl's setup.py omits TransferQueue.
    assert "TransferQueue==0.1.7" in install


def test_the_venv_stamp_records_the_pin_not_the_install_extras(monkeypatch, tmp_path):
    # the stamp gates rebuilds. if it recorded the extra-bearing spec while VERL_REQUIREMENT stayed
    # bare, every later call would see a mismatch and rebuild the venv from scratch on a paid pod.
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))

    vc.resolve_verl_python(str(tmp_path))
    assert (tmp_path / "verl-venv" / "flash-verl-requirement").read_text() == vc.VERL_REQUIREMENT

    (tmp_path / "verl-venv" / "bin" / "python").write_text("")
    vc.resolve_verl_python(str(tmp_path))
    assert len(calls) == 2, "a venv built from the current pin must not be rebuilt"


def test_resolve_verl_python_reuses_a_venv_built_from_the_current_pin(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls))
    _fake_verl_venv(tmp_path, stamp=vc.VERL_REQUIREMENT)

    vc.resolve_verl_python(str(tmp_path))

    # reinstalling verl's torch/vllm on every retry would cost many minutes of paid gpu time.
    assert calls == []


@pytest.mark.parametrize("stale", ["verl @ git+https://github.com/freesolo-co/verl@" + "0" * 40, None])
def test_resolve_verl_python_rebuilds_a_venv_that_is_not_the_current_pin(monkeypatch, tmp_path, stale):
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


def test_resolve_verl_python_installs_wandb_best_effort_when_requested(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.delenv("FLASH_VERL_PYTHON", raising=False)
    monkeypatch.setattr(vc.subprocess, "run", _record_run(calls, keep_check=True))

    vc.resolve_verl_python(str(tmp_path), install_wandb=True)

    assert any(vc.VERL_REQUIREMENT_URL in arg for arg in calls[1][0])
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


def _blackwell_probe(monkeypatch, *, major, flashinfer_ok=True):
    """stub the two subprocess probes: capability first, then `import flashinfer`."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "import flashinfer" in cmd[-1]:
            return SimpleNamespace(returncode=0 if flashinfer_ok else 1)
        return SimpleNamespace(returncode=0, stdout=f"{major}\n", stderr="")

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    return calls


def _cc_probe(monkeypatch, cc):
    """stub the capability probe; `cc` is what torch reports, or () for no cuda."""

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=f"{cc}\n", stderr="")

    monkeypatch.setattr(vc.subprocess, "run", fake_run)


@pytest.mark.parametrize("cc", [(8, 6), (8, 9), (7, 5), (11, 0)])
def test_unvalidated_arches_force_the_rollout_eager(monkeypatch, cc):
    # vllm 0.19.1's graph capture dies in aot_compile (sm86) / triton slot-mapping on the arches the
    # retired trl driver never validated. verl defaults enforce_eager False AND asks for
    # FULL_AND_PIECEWISE, so without this an sm89 rtx 4090 -- the catalog's recommended_gpu for the
    # small models, i.e. the DEFAULT grpo route -- captures more graphs than the config known to fail.
    _cc_probe(monkeypatch, cc)
    assert vc.resolve_rollout_enforce_eager("/verl/bin/python") is True


@pytest.mark.parametrize("cc", [(8, 0), (9, 0), (10, 0), (12, 0)])
def test_validated_arches_keep_verl_graph_capture(monkeypatch, cc):
    # a100/h100 were validated with graphs, and blackwell (incl. the b200 rollout work) depends on
    # them. forcing eager here would be a silent throughput regression, not a safety net.
    _cc_probe(monkeypatch, cc)
    assert vc.resolve_rollout_enforce_eager("/verl/bin/python") is False


def test_capability_probe_failure_leaves_verl_default_alone(monkeypatch):
    # a probe that cannot answer must not guess eager onto an unknown card.
    def boom(*a, **k):
        raise OSError("no interpreter")

    monkeypatch.setattr(vc.subprocess, "run", boom)
    assert vc.resolve_rollout_enforce_eager("/verl/bin/python") is False


def test_no_cuda_leaves_verl_default_alone(monkeypatch):
    # torch reports () with no visible card; literal_eval yields an empty tuple, not a crash.
    _cc_probe(monkeypatch, ())
    assert vc.resolve_rollout_enforce_eager("/verl/bin/python") is False


def test_capability_probe_runs_against_the_verl_interpreter(monkeypatch):
    # verl owns the rollout engine and pins its own vllm, so flash's own torch would answer for the
    # wrong environment -- and on a heterogeneous host, potentially the wrong card.
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="(8, 9)\n", stderr="")

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    vc.resolve_rollout_enforce_eager("/verl/bin/python")
    assert calls
    assert all(cmd[0] == "/verl/bin/python" for cmd in calls)


@pytest.mark.parametrize("major", [10, 12])
def test_blackwell_pins_flashinfer_and_sdpa_vit(monkeypatch, major):
    _blackwell_probe(monkeypatch, major=major)
    assert vc.resolve_blackwell_attention_backends("/verl/bin/python") == (
        "FLASHINFER",
        "TORCH_SDPA",
    )


def test_blackwell_falls_back_to_triton_when_flashinfer_is_abi_broken(monkeypatch):
    # flashinfer can install yet fail to import against this torch. an unconditional FLASHINFER would
    # ship fine and only die at engine init on a paid gpu, so degrade to a registered PTX-independent
    # decoder backend. the ViT pin is unaffected -- it is a separate selection.
    _blackwell_probe(monkeypatch, major=12, flashinfer_ok=False)
    assert vc.resolve_blackwell_attention_backends("/verl/bin/python") == (
        "TRITON_ATTN",
        "TORCH_SDPA",
    )


@pytest.mark.parametrize("major", [8, 9, -1])
def test_non_blackwell_leaves_both_backends_to_vllm(monkeypatch, major):
    # vllm's capability-ordered defaults are correct off blackwell (flash-attn is the right decoder
    # choice on ampere/hopper), so pinning anything there would override a working selection.
    _blackwell_probe(monkeypatch, major=major)
    assert vc.resolve_blackwell_attention_backends("/verl/bin/python") == (None, None)


def test_blackwell_probe_failure_leaves_vllm_defaults_alone(monkeypatch):
    # a probe that cannot answer must not guess an override onto an unknown card.
    def boom(*a, **k):
        raise OSError("no interpreter")

    monkeypatch.setattr(vc.subprocess, "run", boom)
    assert vc.resolve_blackwell_attention_backends("/verl/bin/python") == (None, None)


def test_blackwell_probe_runs_against_the_verl_interpreter(monkeypatch):
    # verl owns the rollout engine and pins its own vllm stack, so a flash-side `import flashinfer`
    # would answer for the wrong environment. both probes must target the passed interpreter.
    calls = _blackwell_probe(monkeypatch, major=12)
    vc.resolve_blackwell_attention_backends("/verl/bin/python")
    assert len(calls) == 2
    assert all(cmd[0] == "/verl/bin/python" for cmd in calls)
