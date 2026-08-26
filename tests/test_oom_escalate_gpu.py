"""CUDA-OOM escalation coverage (CPU-only; no GPU/network)."""

from __future__ import annotations

import types

import pytest

import flash.engine.worker.entry.worker as worker_entry
import flash.engine.worker.perf as worker_perf
import flash.engine.worker.runtime.state as worker_state


@pytest.fixture(autouse=True)
def _cuda_untouched(monkeypatch):
    """Assert the boot condition unless a test says otherwise: no CUDA context in this process.

    `preflight_gpu_occupancy` declines to run once one exists, so without this every occupancy assertion
    below would pass for the wrong reason -- silently, and identically on a CI box with no torch and
    a developer box with a live one.
    """
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "cuda_is_initialized", lambda: False)


def test_is_cuda_oom_is_structured(monkeypatch):
    # torch-free (the offline CI image has no torch): is_cuda_oom imports torch internally under a
    # try/except, so the counter + MemoryError paths classify without it.
    from flash.engine.worker.perf import lifecycle as lc

    # A generic RuntimeError that SAYS "out of memory" is not an OOM without a real signal.
    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 0)
    assert lc.is_cuda_oom(RuntimeError("Triton Error [CUDA]: out of memory")) is False
    assert lc.is_cuda_oom(ValueError("bad config")) is False
    # vLLM can reject startup before torch records an allocator OOM; those deterministic memory
    # preflight messages must still trigger the larger-GPU walk.
    assert (
        lc.is_cuda_oom(
            RuntimeError(
                "Free memory on device cuda:0 (2.42/31.36 GiB) on startup is less than "
                "desired GPU memory utilization (0.37421194528246804, 11.73 GiB)"
            )
        )
        is True
    )
    assert lc.is_cuda_oom(RuntimeError("No available memory for the cache blocks")) is True
    # torch's allocator counter advancing is the structured "error code"
    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 2)
    assert lc.is_cuda_oom(RuntimeError("whatever")) is True
    # a host-RAM OOM is never a GPU OOM (a bigger card can't fix it), even with the counter pinned >0
    assert lc.is_cuda_oom(MemoryError("host ram: out of memory")) is False


def test_a_child_process_oom_is_classified_from_its_output():
    """A verl child's OOM must reach the lifecycle as an OOM, not a permanent job_failed.

    The parent classifies an in-process OOM from `torch.cuda.OutOfMemoryError` and the allocator
    counter. Neither crosses a process boundary: the verl child is a separate interpreter, its
    `num_ooms` is its own, and the terminal raise carries only "subprocess exited with status N".
    So the child's own output is the only evidence, and without it the one OOM shape that happens
    DURING training (rather than at vllm startup) is never retried on a larger card.
    """
    from flash.engine.worker.perf.lifecycle import is_cuda_oom
    from flash.engine.worker.train.entry.backend_common import (
        ChildOutputTail,
        raise_for_classified_verl_exit,
    )

    tail = ChildOutputTail()
    tail.record(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB "
        "(GPU 0; 79.15 GiB total capacity)\n"
    )
    assert tail.cuda_oom_evidence is not None, "the child's torch OOM left no evidence on the tail"
    with pytest.raises(RuntimeError, match="outofmemoryerror") as raised:
        raise_for_classified_verl_exit(1, tail)
    # the raised error is what the lifecycle classifies, so the evidence has to survive into it
    assert is_cuda_oom(raised.value) is True, (
        "the classified error does not read back as an oom, so the run would not be retried "
        "on a larger card"
    )


def test_child_output_that_merely_mentions_memory_is_not_an_oom():
    """The widened matcher must not escalate the GPU for a failure a bigger card cannot fix.

    Pairs with the test above: matching a bare "out of memory" would classify a host-RAM OOM, an
    environment's own error text, or a Triton message as a CUDA OOM and burn a retry on a larger
    card for a run that would fail there identically.
    """
    from flash.engine.worker.train.entry.backend_common import (
        ChildOutputTail,
        raise_for_classified_verl_exit,
    )

    for line in (
        "MemoryError: host ram: out of memory",
        "Triton Error [CUDA]: out of memory",
        "ValueError: the grader ran out of memory budget",
    ):
        tail = ChildOutputTail()
        tail.record(line + "\n")
        assert tail.cuda_oom_evidence is None, f"{line!r} was wrongly read as a cuda oom"
        raise_for_classified_verl_exit(1, tail)  # must not raise


RAY_HOST_RAM_KILL = (
    "ray.exceptions.OutOfMemoryError: 6 worker(s) were killed due to the node running low on "
    "memory. Memory on the node was 42.45GB / 42.84GB (0.991)"
)


def test_ray_host_ram_kill_is_not_a_cuda_oom(monkeypatch):
    """A ray host-RAM kill must never escalate the GPU.

    `OutOfMemoryError` is ALSO ray's class name, so a matcher whose `torch.`/`cuda.` prefixes were
    both optional degenerated to bare `outofmemoryerror` and matched `ray.exceptions.OutOfMemoryError`
    verbatim. That classified a node whose SYSTEM memory was exhausted as a CUDA OOM and retried on a
    larger-VRAM card ("retrying on a larger GPU (> 24 GB)") while the GPU sat idle at 0.9/24.0 GB --
    the wrong resource, at a higher hourly rate, failing identically on the bigger card.
    """
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 0)
    assert lc.cuda_oom_message_evidence(RAY_HOST_RAM_KILL) is None
    assert lc.is_cuda_oom(RuntimeError(RAY_HOST_RAM_KILL)) is False
    # the qualified torch spellings must keep classifying, or training OOMs stop escalating at all
    assert lc.cuda_oom_message_evidence("torch.OutOfMemoryError: CUDA out of memory") is not None
    assert lc.cuda_oom_message_evidence("torch.cuda.OutOfMemoryError: boom") is not None


def test_ray_host_ram_kill_is_classified_and_named(monkeypatch):
    """The run's reported cause must name host RAM, and must not read back as a CUDA OOM.

    The ray kill line sits ~988 lines deep in the log, behind the thread-pool and bridge errors it
    CAUSED, and the operator-visible message ("train phase 'rl' produced no /tmp/metrics.json")
    names neither. Classifying at the child boundary is what puts the scarce resource in the failure
    the supervisor prints.
    """
    from flash.engine.worker.perf import lifecycle as lc
    from flash.engine.worker.verl.diagnostics import ChildOutputTail, raise_for_classified_verl_exit

    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 0)
    tail = ChildOutputTail()
    tail.record(RAY_HOST_RAM_KILL + "\n")
    # the cascade the kill caused prints after it; the kill is still what gets reported
    tail.record("RuntimeError: can't start new thread\n")
    assert tail.host_ram_kill_evidence is not None
    assert tail.cuda_oom_evidence is None

    with pytest.raises(RuntimeError, match="HOST RAM") as raised:
        raise_for_classified_verl_exit(1, tail)
    assert "system RAM" in str(raised.value), "the remedy does not name the resource to select on"
    assert lc.is_cuda_oom(raised.value) is False, (
        "the host-RAM kill reads back as a cuda oom, so the run would still escalate VRAM"
    )


def test_cuda_oom_still_wins_when_no_host_ram_kill_is_present(monkeypatch):
    """Ordering guard: adding the host-RAM branch must not shadow genuine CUDA OOM escalation."""
    from flash.engine.worker.perf import lifecycle as lc
    from flash.engine.worker.verl.diagnostics import ChildOutputTail, raise_for_classified_verl_exit

    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 0)
    tail = ChildOutputTail()
    tail.record("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
    assert tail.host_ram_kill_evidence is None
    with pytest.raises(RuntimeError, match="outofmemoryerror") as raised:
        raise_for_classified_verl_exit(1, tail)
    assert lc.is_cuda_oom(raised.value) is True


def test_host_ram_evidence_outranks_a_stale_allocator_counter(monkeypatch):
    """A prior allocator OOM must not re-classify a later host-RAM kill.

    `cuda_oom_count()` is CUMULATIVE and process-wide, so one recovered allocator OOM earlier in the
    run leaves it above zero for every later failure. Without this precedence the explicit host-RAM
    error falls through to `return cuda_oom_count() > 0` and reads back as a CUDA OOM again --
    re-arming the exact VRAM escalation this fix removes, on the runs most likely to have touched
    the allocator. A failure that named its own cause must not be diagnosed by a stale counter.
    """
    from flash.engine.worker.entry.worker import _worker_failure_flags
    from flash.engine.worker.perf import lifecycle as lc
    from flash.engine.worker.verl.diagnostics import ChildOutputTail, raise_for_classified_verl_exit

    tail = ChildOutputTail()
    tail.record(RAY_HOST_RAM_KILL + "\n")
    with pytest.raises(RuntimeError) as raised:
        raise_for_classified_verl_exit(1, tail)

    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 1)  # poisoned by an earlier allocator OOM
    assert lc.is_cuda_oom(raised.value) is False
    assert _worker_failure_flags(raised.value) == {"retriable": False, "oom": False}
    # the counter must still classify a failure that carries NO host-RAM evidence
    assert lc.is_cuda_oom(RuntimeError("some unrelated worker failure")) is True


def test_mixed_evidence_reports_both_without_quoting_the_cuda_token(monkeypatch):
    """With both signals the message must name both, yet still not read back as a CUDA OOM.

    `is_cuda_oom` classifies this very string, so echoing the matched token (`torch.OutOfMemoryError`)
    would make the message re-match and re-enable the VRAM escalation the branch exists to prevent.
    The CUDA signal is therefore described, never quoted -- and the text must not overclaim that GPU
    memory was fine, because the tail keeps only the first of each category and cannot say which
    resource ended the run.
    """
    from flash.engine.worker.perf import lifecycle as lc
    from flash.engine.worker.verl.diagnostics import ChildOutputTail, raise_for_classified_verl_exit

    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 0)
    tail = ChildOutputTail()
    tail.record(RAY_HOST_RAM_KILL + "\n")
    tail.record("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")

    with pytest.raises(RuntimeError, match="HOST RAM") as raised:
        raise_for_classified_verl_exit(1, tail)
    message = str(raised.value)
    assert "ALSO reported" in message, "the mixed case hides the cuda signal from the operator"
    assert "gpu memory was not the scarce resource" not in message, "overclaims with both signals"
    assert lc.cuda_oom_message_evidence(message) is None, (
        "the message quotes a matchable cuda token"
    )
    assert lc.is_cuda_oom(raised.value) is False


@pytest.mark.parametrize("ray_first", [True, False])
def test_host_ram_wins_when_both_signals_are_present(monkeypatch, ray_first):
    """When BOTH signals appear, the host-RAM kill is the cause and must win either print order.

    A node dying of system memory kills the workers holding the GPU, so a torch OOM can be printed
    by a worker on its way down -- BEFORE or AFTER ray's kill line, depending on which process
    flushed first. Escalating VRAM on that torch line buys a bigger card for a node whose RAM is
    what ran out, so the verdict must not depend on interleaving.
    """
    from flash.engine.worker.perf import lifecycle as lc
    from flash.engine.worker.verl.diagnostics import ChildOutputTail, raise_for_classified_verl_exit

    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 0)
    torch_line = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
    lines = [RAY_HOST_RAM_KILL + "\n", torch_line]
    tail = ChildOutputTail()
    for line in lines if ray_first else reversed(lines):
        tail.record(line)

    with pytest.raises(RuntimeError, match="HOST RAM") as raised:
        raise_for_classified_verl_exit(1, tail)
    assert lc.is_cuda_oom(raised.value) is False


def test_is_cuda_oom_typed_torch_error():
    torch = pytest.importorskip("torch")  # skipped on the torch-less offline CI image
    from flash.engine.worker.perf.lifecycle import is_cuda_oom

    assert is_cuda_oom(torch.cuda.OutOfMemoryError("x")) is True  # typed allocator signal


def test_is_cuda_oom_none_is_never_oom(monkeypatch):
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 5)
    assert lc.is_cuda_oom(None) is False


def _card(gpu, vram):
    return types.SimpleNamespace(gpu=gpu, vram_gb=vram, provider="runpod", hourly_usd=1.0)


def test_retry_filter_keeps_only_strictly_larger_cards():
    from flash.runner.supervise.retry_decision import _strictly_larger_candidates

    candidates = [_card("A100", 80), _card("A100b", 80), _card("Pro6000", 96), _card("B200", 180)]

    assert [candidate.gpu for candidate in _strictly_larger_candidates(candidates, 0)] == [
        candidate.gpu for candidate in candidates
    ]
    assert [candidate.gpu for candidate in _strictly_larger_candidates(candidates, 80)] == [
        "Pro6000",
        "B200",
    ]
    assert _strictly_larger_candidates(candidates, 180) == ()


def _shape(gpu, vram, count):
    return types.SimpleNamespace(
        gpu=gpu, vram_gb=vram, gpu_count=count, provider="runpod", hourly_usd=1.0
    )


def test_sft_oom_escalation_uses_the_ranks_that_joined():
    from flash.providers.core.allocator import _executed_width, _fitting_candidates
    from flash.providers.core.base import Candidate
    from flash.providers.core.sharding import combined_vram_gb
    from flash.runner.supervise.retry_decision import (
        _candidate_usable_vram_gb,
        _strictly_larger_candidates,
    )

    failed = Candidate("runpod", "RTX 4090", 0.69, 24, 4)
    fallback = Candidate("runpod", "A100 SXM 40GB", 1.0, 40, 1)
    candidates = _fitting_candidates(
        [failed, fallback],
        35,
        _executed_width("sft", {"batch_size": 8}, {"sft_retained_examples": 10}),
    )
    failed, fallback = candidates

    assert failed.gpu_count == 4
    assert failed.executed_gpu_count == 2
    assert _candidate_usable_vram_gb(failed) == pytest.approx(combined_vram_gb(24, 2))
    assert _candidate_usable_vram_gb(failed) == pytest.approx(35.2)
    assert _candidate_usable_vram_gb(failed) != pytest.approx(combined_vram_gb(24, 4))
    assert _strictly_larger_candidates([fallback], _candidate_usable_vram_gb(failed)) == (fallback,)


def test_non_sft_oom_escalation_still_uses_every_rented_card():
    from flash.providers.core.allocator import _executed_width, _fitting_candidates
    from flash.providers.core.base import Candidate
    from flash.providers.core.sharding import combined_vram_gb
    from flash.runner.supervise.retry_decision import (
        _candidate_usable_vram_gb,
        _strictly_larger_candidates,
    )

    failed = Candidate("runpod", "RTX 4090", 0.69, 24, 4)
    fallback = Candidate("runpod", "A100 SXM 40GB", 1.0, 40, 1)
    failed = _fitting_candidates(
        [failed],
        35,
        _executed_width("grpo", {"batch_size": 8}, {"sft_retained_examples": 10}),
    )[0]

    assert failed.executed_gpu_count == failed.gpu_count == 4
    assert _candidate_usable_vram_gb(failed) == pytest.approx(combined_vram_gb(24, 4))
    assert _candidate_usable_vram_gb(failed) == pytest.approx(62.4)
    assert _strictly_larger_candidates([fallback], _candidate_usable_vram_gb(failed)) == ()


def test_an_oom_retry_never_moves_to_a_shape_the_fit_model_calls_smaller():
    """The escalation filter must measure candidates the way the allocator sized them.

    the raw `gpu_count * vram_gb` product is not the allocator's fit model: `combined_vram_gb`
    subtracts a replicated-per-card floor and applies a shard efficiency, so a sharded pair is worth
    materially less than its card count suggests. mixing the two scales let a retry go BACKWARDS --
    2x80 raw-counts as 160 GB against a 141 GB card that just OOM'd, but the same pair is modelled as
    130.4 GB usable, so the "larger" retry is smaller than the shape that already failed and burns a
    paid attempt to reach the same OOM.
    """
    from flash.providers.core.sharding import combined_vram_gb
    from flash.runner.supervise.retry_decision import (
        _candidate_usable_vram_gb,
        _strictly_larger_candidates,
    )

    single_h200 = _shape("H200", 141, 1)
    pair_h100 = _shape("H100x2", 80, 2)
    assert combined_vram_gb(80, 2) < 141  # the premise: the pair is the smaller shape

    floor = _candidate_usable_vram_gb(single_h200)
    assert pair_h100 not in _strictly_larger_candidates([pair_h100], floor)

    # and the floor is recorded on the same scale: a sharded shape that OOMs must not write a floor
    # so inflated that genuinely larger single cards get filtered out. 3x40 raw-counts as 120 GB,
    # which would wrongly exclude a 96 GB card that the fit model rates higher (89.6 GB usable).
    triple_40 = _shape("L40Sx3", 40, 3)
    single_96 = _shape("Pro6000", 96, 1)
    assert single_96 in _strictly_larger_candidates(
        [single_96], _candidate_usable_vram_gb(triple_40)
    )


def test_surfaced_worker_flags_reads_both_flags_in_one_pass():
    from flash.providers.runpod.execution.jobs import surfaced_worker_flags

    say = lambda _m: None  # noqa: E731
    reads = {"n": 0}

    def reader(force=False):
        reads["n"] += 1
        return {"oom": True, "attempt": 0, "retriable": False, "stage": "rl_train"}

    _key, retriable, oom = surfaced_worker_flags(reader, None, say, 0, launch_ts=1.0)
    assert (retriable, oom) == (False, True)
    assert reads["n"] == 1
    assert surfaced_worker_flags(
        lambda force=False: {"retriable": True, "attempt": 0, "ts": 10_500.0},
        None,
        say,
        0,
        launch_ts=10_000.0,
    )[1:] == (True, False)
    assert surfaced_worker_flags(
        lambda force=False: {"retriable": True, "attempt": 1, "ts": 10_500.0},
        None,
        say,
        0,
        launch_ts=10_000.0,
    )[1:] == (False, False)
    assert surfaced_worker_flags(None, None, say)[1:] == (False, False)


def test_heartbeat_oom_for_attempt_gates_stale_flag():
    from flash.providers._lifecycle.instances.poll import heartbeat_oom_for_attempt

    assert heartbeat_oom_for_attempt({"oom": True, "attempt": 0}, 0) is True
    assert heartbeat_oom_for_attempt({"oom": True, "attempt": 0}, 1) is False
    assert heartbeat_oom_for_attempt({"oom": True, "attempt": 1}, 1) is True
    assert heartbeat_oom_for_attempt({"oom": True}, 1) is False
    assert heartbeat_oom_for_attempt({"oom": True, "attempt": 0}, None) is False
    assert heartbeat_oom_for_attempt(None, 0) is False
    assert heartbeat_oom_for_attempt({"retriable": True}, 0) is False


@pytest.mark.parametrize(
    ("heartbeat_attempt", "current_attempt"),
    [(0, 0), (7, 7)],
)
def test_heartbeat_oom_accepts_only_canonical_attempt_identities(
    heartbeat_attempt, current_attempt
):
    from flash.providers._lifecycle.instances.poll import heartbeat_oom_for_attempt

    assert heartbeat_oom_for_attempt({"oom": True, "attempt": heartbeat_attempt}, current_attempt)


@pytest.mark.parametrize(
    "malformed_attempt",
    [
        True,
        False,
        1.0,
        -1,
        "0",
        "7",
        "007",
        "-1",
        "+1",
        " 1",
        "1 ",
        "",
        chr(0x661),
        chr(0xFF11),
        object(),
    ],
)
def test_heartbeat_oom_rejects_malformed_heartbeat_attempt(malformed_attempt):
    from flash.providers._lifecycle.instances.poll import heartbeat_oom_for_attempt

    assert heartbeat_oom_for_attempt({"oom": True, "attempt": malformed_attempt}, 1) is False


@pytest.mark.parametrize(
    "malformed_attempt",
    [
        True,
        False,
        1.0,
        -1,
        "0",
        "7",
        "007",
        "-1",
        "+1",
        " 1",
        "1 ",
        "",
        chr(0x661),
        chr(0xFF11),
        object(),
    ],
)
def test_heartbeat_oom_rejects_malformed_current_attempt(malformed_attempt):
    from flash.providers._lifecycle.instances.poll import heartbeat_oom_for_attempt

    assert heartbeat_oom_for_attempt({"oom": True, "attempt": 1}, malformed_attempt) is False


def test_poll_job_maps_only_matching_oom_attempt(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(polling.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda _eid, _jid, **_kw: {"status": "FAILED", "error": "x"},
    )
    handle = jobs.JobHandle("ep", "name", "rpk-" + "0" * 64, "job", 2, 1.0)

    res = polling.poll_job(
        handle,
        interval_s=0,
        heartbeat_reader=lambda force=False: {"oom": True, "attempt": 2},
        current_attempt=2,
    )
    assert res.failure == "oom"

    res = polling.poll_job(
        handle,
        interval_s=0,
        heartbeat_reader=lambda force=False: {"oom": True, "attempt": 1},
        current_attempt=2,
    )
    assert res.failure == "job_failed"


def test_worker_failure_flags_prioritize_retriable_over_oom(monkeypatch):

    monkeypatch.setattr(worker_perf, "is_cuda_oom", lambda _exc: True)
    assert worker_entry._worker_failure_flags(RuntimeError("cuda oom")) == {
        "retriable": False,
        "oom": True,
    }
    assert worker_entry._worker_failure_flags(worker_perf.RetriableInfraError("bad host")) == {
        "retriable": True,
        "oom": False,
    }


def test_a_dirty_card_is_infra_retriable_and_never_an_oom(monkeypatch):
    """A co-tenanted card must retry on a FRESH instance, not escalate to a bigger one.

    The observed failure: Flash sized the run at >=19 GB, RunPod handed over a 4090 reporting
    total=22.5 used=18.6 free=3.4, with only 0.486 GB owned by any process in our container. The
    sizing was right and the card was dirty, but the run trained anyway and died on an OOM ~80s of
    paid GPU later -- classified `oom`, which escalates to a larger card and spends the small OOM
    retry budget. Both are the wrong recovery for a card that was merely occupied.
    """
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: (3.4, 22.5))
    with pytest.raises(lc.DirtyGpuError) as excinfo:
        lc.preflight_gpu_occupancy(1)

    assert "19.1 GB of 22.5 GB (85%) already in use before this run has touched it" in str(
        excinfo.value
    )
    # the whole point: infra retry, NOT an oom escalation onto a bigger (equally dirty) card.
    monkeypatch.setattr(worker_perf, "is_cuda_oom", lambda _exc: False)
    assert worker_entry._worker_failure_flags(excinfo.value) == {"retriable": True, "oom": False}


def _install_nvml_memory_stub(monkeypatch, readings):
    import sys

    queried = []
    pynvml = types.ModuleType("pynvml")
    pynvml.nvmlInit = lambda: None
    pynvml.nvmlShutdown = lambda: None

    def _handle(device_index):
        queried.append(device_index)
        if readings[device_index] is None:
            raise RuntimeError("unreadable device")
        return device_index

    def _memory(device_index):
        free_gb, total_gb = readings[device_index]
        return types.SimpleNamespace(free=int(free_gb * 1024**3), total=int(total_gb * 1024**3))

    pynvml.nvmlDeviceGetHandleByIndex = _handle
    pynvml.nvmlDeviceGetMemoryInfo = _memory
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    return queried


def test_clean_gpu0_dirty_gpu1_is_rejected(monkeypatch):
    from flash.engine.worker.perf import lifecycle as lc

    queried = _install_nvml_memory_stub(monkeypatch, {0: (22.1, 22.5), 1: (3.4, 22.5)})
    with pytest.raises(lc.DirtyGpuError, match="GPU device 1") as excinfo:
        lc.preflight_gpu_occupancy(2)

    assert queried == [0, 1]
    assert worker_entry._worker_failure_flags(excinfo.value) == {"retriable": True, "oom": False}


def test_unreadable_gpu0_does_not_hide_dirty_gpu1(monkeypatch):
    from flash.engine.worker.perf import lifecycle as lc

    queried = _install_nvml_memory_stub(monkeypatch, {0: None, 1: (3.4, 22.5)})
    with pytest.raises(lc.DirtyGpuError, match="GPU device 1"):
        lc.preflight_gpu_occupancy(2)

    assert queried == [0, 1]


def test_invalid_gpu0_does_not_hide_dirty_gpu1(monkeypatch):
    from flash.engine.worker.perf import lifecycle as lc

    queried = _install_nvml_memory_stub(monkeypatch, {0: (0.0, 0.0), 1: (3.4, 22.5)})
    with pytest.raises(lc.DirtyGpuError, match="GPU device 1"):
        lc.preflight_gpu_occupancy(2)

    assert queried == [0, 1]


def test_resolved_count_bounds_gpu_enumeration(monkeypatch):
    from flash.engine.worker.perf import lifecycle as lc

    queried = _install_nvml_memory_stub(
        monkeypatch,
        {
            0: (3.4, 22.5),
            1: (22.1, 22.5),
            2: (22.1, 22.5),
            3: (22.1, 22.5),
        },
    )
    monkeypatch.setattr(
        worker_state, "JOB_SPEC", types.SimpleNamespace(gpu=types.SimpleNamespace(count=2))
    )
    with pytest.raises(lc.DirtyGpuError, match="GPU device 0"):
        worker_entry._preflight_gpu_occupancy_for_spec()

    assert queried == [0, 1]


def test_sub_five_percent_occupancy_is_an_explicit_limit(monkeypatch):
    """This boundary is best-effort occupancy screening, not proof that the run fits."""
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: (21.5, 22.5))
    lc.preflight_gpu_occupancy(1)


def test_five_percent_boundary_is_accepted_and_above_is_rejected(monkeypatch):
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: (19.0, 20.0))
    lc.preflight_gpu_occupancy(1)

    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: (18.999, 20.0))
    with pytest.raises(lc.DirtyGpuError, match="GPU device 0"):
        lc.preflight_gpu_occupancy(1)


@pytest.mark.parametrize(
    ("free_gb", "total_gb"),
    [
        (22.5, 22.5),  # nothing on it at all
        (22.1, 22.5),  # a clean card: only the driver's own reserve is gone (1.8%)
        (174.0, 180.0),  # same 3.3% reserve on a big card, which is more absolute GB
    ],
)
def test_preflight_accepts_a_card_nobody_else_is_using(monkeypatch, free_gb, total_gb):
    """The occupancy screen accepts empty cards and expected driver reserve.

    It has no opinion about whether the run fits. Reconstructing fit here would reject clean cards
    when profile-measured execution differs from the authored spec or driver-reported usable memory
    differs from the catalog's nominal tier.
    """
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: (free_gb, total_gb))
    lc.preflight_gpu_occupancy(1)


def test_preflight_is_inert_when_the_driver_will_not_answer(monkeypatch):
    """No CUDA, or a driver that will not answer, is not evidence of a dirty card."""
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: None)
    lc.preflight_gpu_occupancy(1)

    # a total of 0 is a nonsense reading, not a 100%-occupied card. dividing by it would raise.
    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: (0.0, 0.0))
    lc.preflight_gpu_occupancy(1)


def test_the_check_declines_once_this_process_holds_a_cuda_context(monkeypatch):
    """Our own context is indistinguishable from a co-tenant's, so a late reading is not usable.

    Attribution cannot rescue it. `nvidia-smi --query-compute-apps` reports HOST pids while the
    worker container has a PRIVATE pid namespace (`docker run` in providers/_lifecycle/instance
    passes no `--pid=host`), so a `/proc/<pid>` test inside the container fails in both directions:
    our own rows do not resolve and get counted as a stranger's (false reject on a clean card), and
    under `--pid=host` a real co-tenant's row DOES resolve and gets credited to us (silently waving
    through the dirty card this exists to refuse).

    So the guarantee is temporal, not analytical: read the card before we have touched it, and
    decline afterwards. 18.6 GB of somebody else's work on the card is not enough to raise here.
    """
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: (3.4, 22.5))
    monkeypatch.setattr(lc, "cuda_is_initialized", lambda: True)
    lc.preflight_gpu_occupancy(1)  # no raise: the reading would include our own context


def test_boot_reads_the_card_before_anything_initializes_cuda():
    """The check must run before `_force_fla_triton_gdn_on_sm100`, which creates a context.

    That function calls `torch.cuda.get_device_capability`, and from that moment `preflight_gpu_occupancy`
    correctly declines to judge the card -- so ordering it after would not weaken the check, it would
    silently disable it. Asserted on the source because the alternative is booting a worker.
    """
    import inspect

    # the boot function specifically, not the module: at module scope the definition of
    # `_preflight_gpu_occupancy_for_spec` precedes everything, so the ordering assertion would hold no
    # matter how the calls were arranged and the test would pass while the check was disabled.
    body = inspect.getsource(worker_entry._run_worker_mode)
    preflight = body.index("_preflight_gpu_occupancy_for_spec()")
    forcer = body.index("_force_fla_triton_gdn_on_sm100()")
    assert preflight < forcer, "the occupancy read must precede the first CUDA context"


def test_a_small_co_tenant_that_still_breaks_a_close_fitting_run_is_refused(monkeypatch):
    """A 5 GB co-tenant is above the fixed occupancy boundary and must be rejected.

    The case demonstrates useful protection for a close-fitting run without claiming that the
    screen detects every co-tenant or proves exact fit. Occupancy at or below 5% remains accepted.
    """
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: (17.5, 22.5))
    with pytest.raises(lc.DirtyGpuError):
        lc.preflight_gpu_occupancy(1)


def test_cuda_is_initialized_is_false_before_any_cuda_call(monkeypatch):
    """The gate that decides whether the reading is usable must not itself import or start torch.

    Reading `sys.modules` rather than importing means a boot where torch has not loaded yet answers
    "untouched" instead of loading torch to find out -- and an unimportable torch does not silently
    flip the answer to "clean card".
    """
    import sys
    import types

    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.undo()  # this test is about `cuda_is_initialized` itself, not its stub

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert lc.cuda_is_initialized() is False  # torch not even imported: nothing of ours on the card

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert lc.cuda_is_initialized() is False  # imported, but no context yet

    torch.cuda = types.SimpleNamespace(is_initialized=lambda: True)
    assert lc.cuda_is_initialized() is True

    # unanswerable -> assume touched, so the check declines rather than judging a reading it cannot
    # trust. the conservative direction here is silence, not a DirtyGpuError.
    def _boom():
        raise RuntimeError("no")

    torch.cuda = types.SimpleNamespace(is_initialized=_boom)
    assert lc.cuda_is_initialized() is True


def test_preflight_never_re_derives_what_the_run_needs(monkeypatch):
    """The worker-side gate must not reach for the allocator's sizing model.

    Two sizing models that disagree is the failure this guards. ``allocate()`` sizes SFT from
    profile-measured overrides (``_overridden_train`` turns an authored batch 8 into the executed
    batch 1), so recomputing from ``JOB_SPEC.train`` here demands more VRAM than the card was
    rented for and rejects the instance the allocator correctly picked.
    """
    import flash.providers.core.allocator as allocator
    from flash.engine.worker.entry.worker import _preflight_gpu_occupancy_for_spec
    from flash.engine.worker.perf import lifecycle as lc

    def _fail(*_a, **_k):
        raise AssertionError("the occupancy preflight must not re-size the run")

    monkeypatch.setattr(allocator, "required_vram_gb", _fail)
    monkeypatch.setattr(lc, "_nvml_memory_gb", lambda _device_index: (22.1, 22.5))
    _preflight_gpu_occupancy_for_spec()


def test_free_vram_reads_nvml_and_never_starts_cuda(monkeypatch):
    """The source must be the driver, and reading it must not create the context it would then count.

    Two failures in one: `memory_allocated`/`memory_reserved` only count what THIS process reserved,
    so a card holding 18 GB of another container's work reads as entirely free; and
    `torch.cuda.mem_get_info` needs a context and CREATES one, so measuring with it adds our own
    memory to the number being measured and trips `cuda_is_initialized` for everything after.
    """
    import sys
    import types

    from flash.engine.worker.perf import lifecycle as lc

    torch = types.ModuleType("torch")

    def _forbidden(*_a, **_k):
        raise AssertionError("the vram reading must not go through torch")

    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=_forbidden,
        memory_allocated=_forbidden,
        memory_reserved=_forbidden,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    pynvml = types.ModuleType("pynvml")
    pynvml.nvmlInit = lambda: None
    pynvml.nvmlShutdown = lambda: None
    pynvml.nvmlDeviceGetHandleByIndex = lambda _i: object()
    # driver: 3.4 GB free of 22.5 GB, which the torch allocator would have reported as 0 reserved.
    pynvml.nvmlDeviceGetMemoryInfo = lambda _h: types.SimpleNamespace(
        free=int(3.4 * 1024**3), total=int(22.5 * 1024**3)
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    assert lc.free_vram_gb() == pytest.approx(3.4, abs=0.05)
    assert lc.total_vram_gb() == pytest.approx(22.5, abs=0.05)


def test_cuda_oom_allocation_detail_recovers_the_numbers_the_token_drops():
    """The classification token alone left an operator with no sizing information.

    ``cuda_oom_message_evidence`` deliberately returns only the matched class, because its result
    is fed back through ``is_cuda_oom``. That kept retry classification honest but discarded the
    only figures that say whether the request was hopeless or missed by a gigabyte.
    """
    from flash.engine.worker.perf.lifecycle import (
        cuda_oom_allocation_detail,
        cuda_oom_message_evidence,
    )

    line = (
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. "
        "GPU 0 has a total capacity of 179.06 GiB of which 1.31 GiB is free."
    )
    # the classification token stays narrow -- widening it would re-trigger VRAM escalation
    assert cuda_oom_message_evidence(line) == "torch.outofmemoryerror"
    detail = cuda_oom_allocation_detail(line)
    assert detail is not None
    assert "2.00 gib" in detail
    assert "179.06 gib" in detail
    assert "1.31 gib" in detail


def test_allocation_detail_keeps_the_figures_in_torchs_classic_parenthetical_spelling():
    """Torch prints these figures two ways, and the caller returns the whole matched span.

    A spelling the pattern cannot reach is therefore not merely unparsed -- it is dropped from the
    operator's message, which is the exact loss this detail exists to prevent. The classic form puts
    the unit BEFORE the phrase (``79.15 GiB total capacity``, ``18.69 MiB free``), so a pattern
    anchored only on ``capacity of``/``is free`` stopped after the allocation size and threw away
    both numbers on a line that carried them.
    """
    from flash.engine.worker.perf.lifecycle import (
        cuda_oom_allocation_detail,
        host_ram_kill_evidence,
        is_cuda_oom,
    )

    classic = (
        "RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB "
        "(GPU 0; 79.15 GiB total capacity; 18.69 MiB free; 2.00 GiB reserved in total by PyTorch)"
    )
    detail = cuda_oom_allocation_detail(classic)
    assert detail is not None
    assert "20.00 mib" in detail
    assert "79.15 gib" in detail, f"capacity dropped from the classic spelling: {detail}"
    assert "18.69 mib" in detail, f"free figure dropped from the classic spelling: {detail}"

    # the capacity-only variant already in this tree must keep its one figure too.
    capacity_only = (
        "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 79.15 GiB total capacity)"
    )
    capacity_detail = cuda_oom_allocation_detail(capacity_only)
    assert capacity_detail is not None
    assert "79.15 gib" in capacity_detail

    # widening must not reach a host-RAM kill, whose remedy is the OPPOSITE of a bigger card.
    host_kill = (
        "ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory"
    )
    assert cuda_oom_allocation_detail(host_kill) is None
    assert host_ram_kill_evidence(host_kill) is not None
    assert not is_cuda_oom(RuntimeError(host_kill))


def test_cuda_oom_allocation_detail_is_absent_when_the_line_carries_no_numbers():
    from flash.engine.worker.perf.lifecycle import cuda_oom_allocation_detail

    assert cuda_oom_allocation_detail("RuntimeError: CUDA error: out of memory") is None


def test_bare_cuda_error_out_of_memory_is_classified():
    """``RuntimeError: CUDA error: out of memory`` is a real driver-level OOM spelling.

    It was previously unrecognised, so a run that died this way reported no cause at all and did
    not escalate VRAM.
    """
    from flash.engine.worker.perf.lifecycle import cuda_oom_message_evidence, is_cuda_oom

    line = "RuntimeError: CUDA error: out of memory"
    assert cuda_oom_message_evidence(line) == "cuda error: out of memory"
    assert is_cuda_oom(RuntimeError(line))


def test_host_ram_kill_is_still_not_a_cuda_oom_after_widening():
    """The widened pattern must not swallow ray's host-RAM kill, whose remedy is the OPPOSITE.

    A bigger card cannot fix system-memory exhaustion, so misclassifying this would escalate VRAM
    while the gpu sat idle.
    """
    from flash.engine.worker.perf.lifecycle import host_ram_kill_evidence, is_cuda_oom

    line = "ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory"
    assert host_ram_kill_evidence(line) is not None
    assert not is_cuda_oom(RuntimeError(line))


def test_allocation_detail_survives_a_flood_that_evicts_the_oom_line():
    """The reporter's case: one OOM line, then ~190 PunicaWrapper warnings.

    The bounded child tail evicts the OOM line entirely, so both the cause and its figures have to
    be latched at record time rather than recovered from the ring buffer.
    """
    from flash.engine.worker.verl.diagnostics import (
        ChildOutputTail,
        raise_for_classified_verl_exit,
    )

    tail = ChildOutputTail()
    tail.record(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. "
        "GPU 0 has a total capacity of 179.06 GiB of which 1.31 GiB is free."
    )
    for index in range(190):
        tail.record(f"WARNING PunicaWrapper padding {index}")

    assert tail.cuda_oom_evidence == "torch.outofmemoryerror"
    assert tail.cuda_oom_allocation_detail is not None

    with pytest.raises(RuntimeError) as excinfo:
        raise_for_classified_verl_exit(1, tail)
    message = str(excinfo.value)
    assert "2.00 gib" in message
    assert "179.06 gib" in message
