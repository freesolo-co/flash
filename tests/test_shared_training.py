"""Admission, authorization, fair execution, replay, and attribution invariants."""

import concurrent.futures

import pytest

from flash.training.shared.contracts import AdmissionError, ConflictError, Limits, Principal
from flash.training.shared.journal import Journal

A = Principal("org-a", "key-a")
B = Principal("org-b", "key-b")
CONFIG = {"algorithm": "sft", "seed": 3, "learning_rate": 0.001}
TRAIN = {"version": 0, "rows": [{"tokens": [1, 2, 3], "mask": [False, True]}]}


@pytest.fixture
def journal(tmp_path):
    value = Journal(tmp_path / "pool.sqlite")
    yield value
    value.db.close()


def finish(journal, command=None):
    command = command or journal.claim()
    journal.finish(command, {"version": 0}, execution_ns=100)
    return command


def test_two_tenants_share_one_execution_lane_and_reconcile_exactly(journal):
    a = journal.create(A, "a", CONFIG)
    b = journal.create(B, "b", CONFIG)
    first = journal.claim()
    assert journal.claim() is None
    finish(journal, first)
    finish(journal)
    for n in range(4):
        journal.submit(A, a, str(n), "train", TRAIN)
    journal.submit(B, b, "b0", "train", TRAIN)
    tenants = [finish(journal)["tenant"] for _ in range(5)]
    assert tenants[:2] == [A.tenant, B.tenant]
    assert journal.usage(A)["execution_ns"] == 500
    assert journal.usage(B)["execution_ns"] == 200
    journal.record_overhead("startup", 50)
    assert journal.pool_usage() == {
        "attributed_execution_ns": 700,
        "unattributed_overhead_ns": 50,
        "unknown_executions": 0,
        "measured_pool_ns": 750,
    }


def test_fairness_groups_api_keys_and_sessions_by_tenant(journal):
    a = journal.create(A, "a", CONFIG)
    a2 = journal.create(Principal(A.tenant, "another-key"), "a2", CONFIG)
    b = journal.create(B, "b", CONFIG)
    assert [finish(journal)["tenant"] for _ in range(3)] == [A.tenant, B.tenant, A.tenant]
    journal.submit(A, a, "a", "train", TRAIN)
    journal.submit(Principal(A.tenant, "another-key"), a2, "a2", "train", TRAIN)
    journal.submit(B, b, "b", "train", TRAIN)
    assert [finish(journal)["tenant"] for _ in range(3)] == [B.tenant, A.tenant, A.tenant]


def test_foreign_keys_cannot_read_submit_cancel_or_restore(journal):
    a = journal.create(A, "a", CONFIG)
    finish(journal)
    command = journal.submit(A, a, "train", "train", TRAIN)
    for principal in (B, Principal(A.tenant, "another-key")):
        with pytest.raises(KeyError):
            journal.status(principal, a)
        with pytest.raises(KeyError):
            journal.submit(principal, a, "foreign", "train", TRAIN)
        with pytest.raises(KeyError):
            journal.cancel(principal, a, command)


def test_admission_atomic_under_concurrent_creations(tmp_path):
    journal = Journal(tmp_path / "pool.sqlite", limits=Limits(adapters=2, adapters_per_tenant=2))

    def create(index):
        try:
            return journal.create(A, str(index), CONFIG)
        except AdmissionError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(create, range(20)))
    assert sum(result is not None for result in results) == 2


def test_idempotency_does_not_duplicate_optimizer_work_or_usage(journal):
    a = journal.create(A, "a", CONFIG)
    assert journal.create(A, "a", CONFIG) == a
    with pytest.raises(ConflictError):
        journal.create(A, "a", {**CONFIG, "seed": 4})
    finish(journal)
    command = journal.submit(A, a, "update", "train", TRAIN)
    assert journal.submit(A, a, "update", "train", TRAIN) == command
    claimed = finish(journal)
    assert journal.submit(A, a, "update", "train", TRAIN) == command
    with pytest.raises(ConflictError):
        journal.submit(A, a, "update", "train", {**TRAIN, "version": 1})
    with pytest.raises(ConflictError):
        finish(journal, claimed)
    assert journal.claim() is None
    assert journal.usage(A)["execution_ns"] == 200


def test_cancel_frees_capacity_but_cannot_interrupt_other_tenants(journal):
    a = journal.create(A, "a", CONFIG)
    finish(journal)
    command = journal.submit(A, a, "update", "train", TRAIN)
    journal.cancel(A, a, command)
    assert journal.claim() is None
    journal.submit(A, a, "update2", "train", TRAIN)
    running = journal.claim()
    with pytest.raises(ConflictError):
        journal.cancel(A, a, running["id"])
    finish(journal, running)


def test_restart_never_replays_uncertain_optimizer_update(tmp_path):
    path = tmp_path / "pool.sqlite"
    journal = Journal(path)
    a = journal.create(A, "a", CONFIG)
    finish(journal)
    journal.submit(A, a, "update", "train", TRAIN)
    journal.claim()
    journal.submit(A, a, "later", "train", TRAIN)
    journal.db.close()
    restarted = Journal(path)
    restarted.recover()
    state = restarted.status(A, a)
    assert state["state"] == "lost"
    assert [c["state"] for c in state["commands"]] == ["completed", "unknown", "cancelled"]
    assert restarted.claim() is None
    assert restarted.usage(A)["unknown_executions"] == 1
    assert restarted.usage(A)["execution_ns"] == 100


def test_limits_reject_one_tenant_without_blocking_another(tmp_path):
    journal = Journal(tmp_path / "pool.sqlite", limits=Limits(queued_tokens_per_tenant=4))
    a, b = journal.create(A, "a", CONFIG), journal.create(B, "b", CONFIG)
    finish(journal)
    finish(journal)
    journal.submit(A, a, "first", "train", TRAIN)
    with pytest.raises(AdmissionError):
        journal.submit(A, a, "second", "train", TRAIN)
    journal.submit(B, b, "first", "train", TRAIN)
    journal.submit(A, a, "close", "close", {})
    with pytest.raises(ConflictError):
        journal.submit(A, a, "after-close", "train", TRAIN)


@pytest.mark.parametrize(
    "payload",
    [
        {"version": True, "rows": TRAIN["rows"]},
        {"version": 0, "rows": [{"tokens": [1, 2], "mask": [1]}]},
        {"version": 0, "rows": [{"tokens": [1, 128], "mask": [True]}]},
        {"version": 0, "rows": [{"tokens": [1, 2], "mask": [False]}]},
        {"version": 0, "rows": TRAIN["rows"], "tenant": "other"},
    ],
)
def test_invalid_training_data_is_rejected_before_queueing(journal, payload):
    a = journal.create(A, "a", CONFIG)
    finish(journal)
    with pytest.raises(ValueError, match=r"integer|boolean|supervised|unknown"):
        journal.submit(A, a, "bad", "train", payload)
    assert journal.claim() is None


def test_restore_only_accepts_owned_completed_checkpoint(journal):
    a = journal.create(A, "a", CONFIG)
    finish(journal)
    journal.submit(A, a, "save", "checkpoint", {})
    command = journal.claim()
    journal.finish(command, {"version": 1, "checkpoint": f"{a}/1"}, execution_ns=2)
    a2 = journal.create(A, "a2", CONFIG)
    b = journal.create(B, "b", CONFIG)
    with pytest.raises(KeyError):
        journal.submit(B, b, "restore", "restore", {"source_session": a, "version": 1})
    with pytest.raises(KeyError):
        journal.submit(A, a2, "missing", "restore", {"source_session": a, "version": 2})
    payload = {"source_session": a, "version": 1}
    command = journal.submit(A, a2, "restore", "restore", payload)
    assert journal.submit(A, a2, "restore", "restore", payload) == command


def test_resource_costs_reconcile_without_rounding_or_customer_charges(tmp_path):
    journal = Journal(tmp_path / "pool.sqlite", rate_microusd_per_hour=4_000_000)
    journal.create(A, "a", CONFIG)
    journal.create(B, "b", CONFIG)
    finish(journal)
    finish(journal)
    journal.record_overhead("idle", 123)
    a = journal.usage(A)["resource_cost_estimate"]
    b = journal.usage(B)["resource_cost_estimate"]
    pool = journal.pool_usage()
    assert (
        a["numerator"] + b["numerator"] + pool["overhead_cost_estimate"]["numerator"]
        == pool["resource_cost_estimate"]["numerator"]
    )
    assert "not_invoice" in a["basis"]
    with pytest.raises(ValueError, match="immutable pricing"):
        Journal(tmp_path / "pool.sqlite", rate_microusd_per_hour=5_000_000)


def test_independent_connections_cannot_overbook_adapter_capacity(tmp_path):
    path = tmp_path / "pool.sqlite"
    limits = Limits(adapters=2, adapters_per_tenant=2)
    journals = [Journal(path, limits=limits) for _ in range(4)]

    def create(index):
        try:
            return journals[index % 4].create(A, str(index), CONFIG)
        except AdmissionError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(create, range(20)))
    assert sum(value is not None for value in results) == 2


def test_grpo_normalizer_uses_pool_context_and_preserves_idempotency(tmp_path):
    journal = Journal(tmp_path / "pool.sqlite", limits=Limits(context_tokens=32))
    session = journal.create(A, "a", {**CONFIG, "algorithm": "grpo"})
    finish(journal)
    payload = {
        "version": 0,
        "rows": [{**TRAIN["rows"][0], "old_logprobs": [0.0, -2.0], "advantages": [0, 1]}],
    }
    command = journal.submit(A, session, "train", "train", payload)
    assert journal.submit(A, session, "train", "train", payload) == command
    assert journal.claim()["payload"]["normalizer"] == 32


def test_residency_reports_clean_release_and_uncertain_restart(journal):
    session = journal.create(A, "a", CONFIG)
    finish(journal)
    resident = journal.usage(A)["adapters"][0]
    assert resident["loaded_ns"] > 0
    assert resident["released_ns"] is None
    journal.submit(A, session, "close", "close", {})
    finish(journal)
    released = journal.usage(A)["adapters"][0]
    assert released["released_ns"] >= resident["loaded_ns"]
    assert released["uncertain"] == 0
    journal.create(B, "b", CONFIG)
    finish(journal)
    journal.recover()
    lost = journal.usage(B)["adapters"][0]
    assert lost["uncertain"] == 1
    assert lost["released_ns"] is None
