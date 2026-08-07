"""CUDA-OOM escalation coverage (CPU-only; no GPU/network)."""

from __future__ import annotations

import types

import pytest


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
    from flash.engine.worker.backend_common import ChildOutputTail, raise_for_classified_verl_exit
    from flash.engine.worker.perf.lifecycle import is_cuda_oom

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
    from flash.engine.worker.backend_common import ChildOutputTail, raise_for_classified_verl_exit

    for line in (
        "MemoryError: host ram: out of memory",
        "Triton Error [CUDA]: out of memory",
        "ValueError: the grader ran out of memory budget",
    ):
        tail = ChildOutputTail()
        tail.record(line + "\n")
        assert tail.cuda_oom_evidence is None, f"{line!r} was wrongly read as a cuda oom"
        raise_for_classified_verl_exit(1, tail)  # must not raise


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


def test_oom_escalated_keeps_only_strictly_larger_cards():
    from flash.runner.lifecycle import _oom_escalated

    cands = [_card("A100", 80), _card("A100b", 80), _card("Pro6000", 96), _card("B200", 180)]
    assert [c.gpu for c in _oom_escalated(cands, 0)] == [
        c.gpu for c in cands
    ]  # no OOM -> unchanged
    assert {c.gpu for c in _oom_escalated(cands, 80)} == {"Pro6000", "B200"}  # >80 only
    assert _oom_escalated(cands, 180) == []  # OOM'd the biggest -> nowhere larger


def _shape(gpu, vram, count):
    return types.SimpleNamespace(
        gpu=gpu, vram_gb=vram, gpu_count=count, provider="runpod", hourly_usd=1.0
    )


def test_an_oom_retry_never_moves_to_a_shape_the_fit_model_calls_smaller():
    """The escalation filter must measure candidates the way the allocator sized them.

    the raw `gpu_count * vram_gb` product is not the allocator's fit model: `combined_vram_gb`
    subtracts a replicated-per-card floor and applies a shard efficiency, so a sharded pair is worth
    materially less than its card count suggests. mixing the two scales let a retry go BACKWARDS --
    2x80 raw-counts as 160 GB against a 141 GB card that just OOM'd, but the same pair is modelled as
    130.4 GB usable, so the "larger" retry is smaller than the shape that already failed and burns a
    paid attempt to reach the same OOM.
    """
    from flash.providers.base import combined_vram_gb
    from flash.runner.lifecycle import _candidate_usable_vram_gb, _oom_escalated

    single_h200 = _shape("H200", 141, 1)
    pair_h100 = _shape("H100x2", 80, 2)
    assert combined_vram_gb(80, 2) < 141  # the premise: the pair is the smaller shape

    floor = _candidate_usable_vram_gb(single_h200)
    assert pair_h100 not in _oom_escalated([pair_h100], floor)

    # and the floor is recorded on the same scale: a sharded shape that OOMs must not write a floor
    # so inflated that genuinely larger single cards get filtered out. 3x40 raw-counts as 120 GB,
    # which would wrongly exclude a 96 GB card that the fit model rates higher (89.6 GB usable).
    triple_40 = _shape("L40Sx3", 40, 3)
    single_96 = _shape("Pro6000", 96, 1)
    assert single_96 in _oom_escalated([single_96], _candidate_usable_vram_gb(triple_40))


def test_surfaced_worker_flags_reads_both_flags_in_one_pass():
    from flash.providers.runpod.jobs import surfaced_worker_flags

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
    from flash.providers._poll import heartbeat_oom_for_attempt

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
    from flash.providers._poll import heartbeat_oom_for_attempt

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
    from flash.providers._poll import heartbeat_oom_for_attempt

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
    from flash.providers._poll import heartbeat_oom_for_attempt

    assert heartbeat_oom_for_attempt({"oom": True, "attempt": 1}, malformed_attempt) is False


def test_poll_job_maps_only_matching_oom_attempt(monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(jobs.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda _eid, _jid, **_kw: {"status": "FAILED", "error": "x"},
    )
    handle = jobs.JobHandle("ep", "name", "rpk-0123456789ab", "job", 2, 1.0)

    res = jobs.poll_job(
        handle,
        interval_s=0,
        heartbeat_reader=lambda force=False: {"oom": True, "attempt": 2},
        current_attempt=2,
    )
    assert res.failure == "oom"

    res = jobs.poll_job(
        handle,
        interval_s=0,
        heartbeat_reader=lambda force=False: {"oom": True, "attempt": 1},
        current_attempt=2,
    )
    assert res.failure == "job_failed"


def test_worker_failure_flags_prioritize_retriable_over_oom(monkeypatch):
    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "is_cuda_oom", lambda _exc: True)
    assert worker._worker_failure_flags(RuntimeError("cuda oom")) == {
        "retriable": False,
        "oom": True,
    }
    assert worker._worker_failure_flags(worker.RetriableInfraError("bad host")) == {
        "retriable": True,
        "oom": False,
    }
