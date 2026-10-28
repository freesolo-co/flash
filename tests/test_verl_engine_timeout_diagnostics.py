"""Keep the initiating rollout timeout visible after a cascade of worker errors."""

import pytest

from flash.engine.worker.verl.diagnostics import ChildOutputTail, raise_for_classified_verl_exit


@pytest.mark.parametrize(
    "evidence",
    [
        (
            "[rank0]:[E1028 ProcessGroupNCCL.cpp:688] [Rank 0] Watchdog caught collective "
            "operation timeout: WorkNCCL(SeqNum=303389, OpType=_ALLGATHER_BASE, "
            "NumelIn=9808640, NumelOut=19617280, Timeout(ms)=600000)"
        ),
        (
            "(EngineCore pid=3394) ERROR [core.py:1110] "
            "TimeoutError: RPC call to sample_tokens timed out."
        ),
    ],
)
def test_timeout_survives_downstream_error_storm(evidence):
    tail = ChildOutputTail(limit=3)
    tail.record(evidence)
    for _ in range(200):
        tail.record("EngineDeadError: EngineCore encountered an issue")
    assert evidence not in tail.tail()
    with pytest.raises(RuntimeError) as caught:
        raise_for_classified_verl_exit(1, tail)
    assert evidence in str(caught.value)
    assert "rollout engine timeout" in str(caught.value)
    # Recovery or a caught error must not override an otherwise successful child exit.
    raise_for_classified_verl_exit(0, tail)


def test_generic_engine_death_does_not_invent_a_timeout():
    tail = ChildOutputTail()
    tail.record("EngineDeadError: EngineCore encountered an issue")
    assert tail.engine_timeout_evidence is None
    raise_for_classified_verl_exit(1, tail)


def test_first_timeout_is_bounded_and_redacted(monkeypatch):
    secret = "hf_" + "s" * 40
    monkeypatch.setenv("HF_TOKEN", secret)
    tail = ChildOutputTail()
    tail.record("TimeoutError: RPC call to sample_tokens timed out. " + secret + "x" * 2000)
    first = tail.engine_timeout_evidence
    tail.record("Watchdog caught collective operation timeout: WorkNCCL(SeqNum=10)")
    assert tail.engine_timeout_evidence == first
    assert secret not in first
    assert len(first) <= 1000


def test_oom_evidence_retains_precedence():
    tail = ChildOutputTail()
    tail.record("torch.OutOfMemoryError: CUDA out of memory.")
    tail.record("TimeoutError: RPC call to sample_tokens timed out.")
    with pytest.raises(RuntimeError, match=r"torch\.outofmemoryerror"):
        raise_for_classified_verl_exit(1, tail)
