"""Shared GRPO queue initialization and stall diagnostics."""

import sys
import threading
import types

import pytest

from flash.engine.worker.train.core.child.queue import _init_transfer_queue


def test_transfer_queue_init_passes_the_config_through_and_returns():
    seen = []
    _init_transfer_queue(lambda conf: seen.append(conf), {"backend": "SimpleStorage"}, timeout_s=30)
    assert seen == [{"backend": "SimpleStorage"}]


def test_transfer_queue_init_that_never_returns_fails_with_the_resource_state(monkeypatch):
    # the wedge this bounds: tq.init reserves its controller and storage units through ray and waits
    # with no timeout, so an unsatisfiable reservation stops the run before a single gpu is touched --
    # silently, because tq's logger defaults to WARNING. without a deadline the run burns the whole
    # setup grace period at 0% gpu and is reported as a generic stall.
    import flash.engine.worker.train.core.child.queue as plugin

    monkeypatch.setattr(
        plugin,
        "_describe_ray_resources",
        lambda: "cluster CPU=1.0 GPU=1.0, free CPU=0.0 GPU=0.0",
    )
    release = threading.Event()
    with pytest.raises(RuntimeError, match="transfer_queue init did not finish") as excinfo:
        _init_transfer_queue(lambda conf: release.wait(60), None, timeout_s=0.2)
    # the message must carry the resource state: "it timed out" alone does not distinguish an
    # unschedulable reservation from a slow start, which is the whole diagnosis.
    assert "free CPU=0.0" in str(excinfo.value)
    # and it must name the frame the thread is parked in. resource counts separate the placement-group
    # wait from the other two, but the controller ray.get and the get_config spin look identical from
    # outside -- only the stack tells them apart. here the init is parked in Event.wait.
    assert "stalled at wait (threading.py:" in str(excinfo.value)
    # the message must NOT claim the cause is capacity: two of the three waits stall with ample free
    # resources, and asserting a scheduling failure would send an operator to the wrong remediation.
    assert "only the first is a capacity problem" in str(excinfo.value)
    assert "so this is a scheduling problem" not in str(excinfo.value)
    release.set()


def test_transfer_queue_init_failure_propagates_unwrapped():
    # a real tq.init error must not be reported as a timeout, and must not be swallowed by the worker
    # thread it now runs on.
    class Boom(RuntimeError):
        pass

    def _raise(conf):
        raise Boom("controller refused the sampler")

    with pytest.raises(Boom, match="controller refused the sampler"):
        _init_transfer_queue(_raise, None, timeout_s=30)


def test_ray_resource_probe_reports_the_cluster_and_free_counts(monkeypatch):
    # ray is only present inside the verl child image, so the probe is exercised against a stub. the
    # message must name both totals and free counts: a wedged reservation is diagnosed by the gap.
    import flash.engine.worker.train.core.child.queue as plugin

    stub = types.ModuleType("ray")
    stub.cluster_resources = lambda: {"CPU": 8.0, "GPU": 1.0}
    stub.available_resources = lambda: {"CPU": 2.0, "GPU": 1.0}
    monkeypatch.setitem(sys.modules, "ray", stub)
    described = plugin._describe_ray_resources()
    assert "cluster CPU=8.0 GPU=1.0" in described
    assert "free CPU=2.0 GPU=1.0" in described


def test_ray_resource_probe_reports_a_fully_consumed_resource_as_zero(monkeypatch):
    # ray DROPS a resource key from available_resources() once it is fully allocated rather than
    # reporting 0.0 -- verified against ray 2.56.1: consuming every cpu removes "CPU" from the mapping.
    # that is the exhaustion case this probe exists to name, so rendering it as "free CPU=None" would
    # read as a broken probe at the exact moment the diagnosis matters. an earlier version of this test
    # hardcoded zero-valued keys in the stub and so could not have caught it.
    import flash.engine.worker.train.core.child.queue as plugin

    stub = types.ModuleType("ray")
    stub.cluster_resources = lambda: {"CPU": 8.0}
    stub.available_resources = lambda: {"memory": 1.0}
    monkeypatch.setitem(sys.modules, "ray", stub)
    described = plugin._describe_ray_resources()
    assert "free CPU=0.0" in described
    # a node with no gpu omits the key from cluster_resources() too, and 0.0 describes that correctly.
    assert "cluster CPU=8.0 GPU=0.0" in described
    assert "None" not in described


def test_ray_resource_probe_never_raises_on_the_timeout_path(monkeypatch):
    # _describe_ray_resources only runs while building a failure message, so a probe error there would
    # replace the diagnosis it exists to produce.
    import flash.engine.worker.train.core.child.queue as plugin

    stub = types.ModuleType("ray")
    stub.cluster_resources = lambda: (_ for _ in ()).throw(RuntimeError("ray is gone"))
    stub.available_resources = dict
    monkeypatch.setitem(sys.modules, "ray", stub)
    assert "ray resources unreadable" in plugin._describe_ray_resources()


def test_stalled_thread_probe_names_the_frame_the_thread_is_parked_in():
    # the point of the probe: identify WHICH of tq.init's three unbounded waits is stuck. a placement
    # group that never fills, a ray.get on the controller, and the `while conf is None` get_config spin
    # are indistinguishable from ray's resource counts, so the parked frame is the only discriminator.
    import flash.engine.worker.train.core.child.queue as plugin

    entered = threading.Event()
    release = threading.Event()

    def _park_in_a_named_frame():
        entered.set()
        release.wait(60)

    thread = threading.Thread(target=_park_in_a_named_frame, daemon=True)
    thread.start()
    try:
        assert entered.wait(10)
        described = plugin._describe_stalled_thread(thread.ident)
    finally:
        release.set()
        thread.join(10)
    # innermost frame first, so the wait itself leads and its caller follows -- that ordering is what
    # makes the message readable in a log line.
    assert described.startswith("wait (threading.py:")
    assert "_park_in_a_named_frame" in described


def test_stalled_thread_probe_reports_a_thread_that_is_gone_instead_of_raising():
    # a thread can finish between the is_alive check and the probe. that must degrade to a note in the
    # failure message, not an exception that replaces the timeout diagnosis.
    import flash.engine.worker.train.core.child.queue as plugin

    thread = threading.Thread(target=lambda: None, daemon=True)
    thread.start()
    thread.join(10)
    assert plugin._describe_stalled_thread(thread.ident) == "stack unavailable"
    assert plugin._describe_stalled_thread(None) == "stack unavailable"
