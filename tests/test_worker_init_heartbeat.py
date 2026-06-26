"""The init-phase heartbeat must never block on CUDA telemetry.

Regression for the consumer-GPU warm-start "hang": ``run_rl``/``run_sft`` start a daemon thread
that heartbeats ``rl_initializing``/``sft_initializing`` every 30s while the MAIN thread is blocked
inside ``GRPOTrainer.__init__`` / ``SFTTrainer.__init__`` (a long, CUDA- and allocator-busy section
— vLLM colocate engine build + weight load + cold kernel JIT). The old code called
``gpu_diagnostics()`` (``include_torch=True``) on that side thread, which issues ``torch.cuda``
queries (``mem_get_info`` / ``memory_allocated`` / ``memory_reserved`` / ``get_device_name``). Those
serialize on the CUDA driver lock and PyTorch's caching-allocator mutex — both held by the init
thread — so the heartbeat thread could FREEZE for the whole init. The control plane then saw no
heartbeat and false-flagged a HANG on a run that was merely doing a slow (but live) cold init. The
fix: the init heartbeats use ``gpu_diagnostics(include_torch=False)`` (nvidia-smi only, out of
process, 8s timeout, GIL released during the wait).

These tests reproduce the freeze deterministically on CPU (no GPU needed) by injecting a fake
``torch`` whose CUDA query blocks, then prove the ``include_torch=False`` path stays responsive.
A source/AST wiring check pins the call sites so the regression can't silently return.
"""

from __future__ import annotations

import ast
import inspect
import sys
import threading
import types

import pytest

from flash.engine.worker.perf import diagnostics


@pytest.fixture
def fast_nvidia(monkeypatch):
    """A fast, GPU-free nvidia-smi stand-in so the ``include_torch=False`` path returns instantly
    regardless of whether nvidia-smi exists on the test host."""
    monkeypatch.setattr(
        diagnostics, "_query_nvidia_gpu", lambda: {"gpu_util_pct": 0, "device_name": "FAKE-GPU"}
    )
    monkeypatch.setattr(diagnostics, "_query_nvidia_processes", lambda: [])


def _install_blocking_torch(monkeypatch, gate: threading.Event) -> None:
    """Inject a fake ``torch`` whose CUDA memory queries block until ``gate`` is set — standing in
    for the CUDA driver / allocator lock held by the main thread inside ``*Trainer.__init__``."""

    def _blocking_mem_get_info():
        gate.wait(timeout=30.0)  # held "by the init thread" until the test releases it
        return (1 << 30, 1 << 34)

    torch = types.ModuleType("torch")
    torch.__version__ = "2.10.0-fake"
    torch.version = types.SimpleNamespace(cuda="12.8")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda i=0: "FAKE-GPU",
        mem_get_info=_blocking_mem_get_info,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def _run_async(fn):
    state: dict = {}

    def _target():
        state["value"] = fn()
        state["done"] = True

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return th, state


def test_include_torch_true_freezes_when_cuda_is_locked(monkeypatch, fast_nvidia):
    """Repro: with the torch path on, a side-thread diag wedges while CUDA is "locked"."""
    gate = threading.Event()
    _install_blocking_torch(monkeypatch, gate)

    th, state = _run_async(lambda: diagnostics.gpu_diagnostics(include_torch=True))
    th.join(timeout=1.5)
    assert not state.get("done"), (
        "expected gpu_diagnostics(include_torch=True) to block on the CUDA query "
        "(this is the heartbeat freeze the fix removes)"
    )

    gate.set()  # main thread released the CUDA lock
    th.join(timeout=5.0)
    assert state.get("done"), "diag should complete once the CUDA lock is released"


def test_include_torch_false_stays_responsive_while_cuda_locked(monkeypatch, fast_nvidia):
    """Fix: with the torch path off, the diag returns promptly even while CUDA is wedged."""
    gate = threading.Event()  # NEVER set -> CUDA stays "locked" for the whole test
    _install_blocking_torch(monkeypatch, gate)

    th, state = _run_async(lambda: diagnostics.gpu_diagnostics(include_torch=False))
    th.join(timeout=2.0)

    assert state.get("done"), (
        "gpu_diagnostics(include_torch=False) must NOT touch torch.cuda and must return even while "
        "the init thread holds the CUDA lock"
    )
    diag = state["value"]
    # The nvidia-smi-only payload — and crucially NO torch keys that would have required a CUDA call.
    assert diag.get("gpu_util_pct") == 0
    assert "torch" not in diag
    assert "torch_memory_free_gb" not in diag


# --------------------------------------------------------------------------------------------
# Wiring guards: the two init-heartbeat closures must keep calling gpu_diagnostics(include_torch=
# False). A future edit that drops the kwarg (or flips it) reintroduces the freeze, so assert it at
# the source/AST level (the closures are locals, so parse the enclosing function's source).


def _inner_func_node(outer_func, inner_name: str) -> ast.FunctionDef | None:
    tree = ast.parse(inspect.getsource(outer_func))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == inner_name:
            return node
    return None


def _assert_gpu_diag_disables_torch(node: ast.FunctionDef, where: str) -> None:
    calls = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "gpu_diagnostics"
    ]
    assert calls, f"{where}: no longer calls gpu_diagnostics — check this guard is still valid"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        val = kwargs.get("include_torch")
        msg = f"{where}: init heartbeat must call gpu_diagnostics(include_torch=False) (got {ast.dump(call)})"
        assert isinstance(val, ast.Constant), msg
        assert val.value is False, msg


def test_rl_init_heartbeat_disables_torch_telemetry():
    from flash.engine.worker import rl

    node = _inner_func_node(rl.run_rl, "_rl_init_heartbeat")
    assert node is not None, "run_rl no longer defines _rl_init_heartbeat"
    _assert_gpu_diag_disables_torch(node, "run_rl._rl_init_heartbeat")


def test_sft_init_heartbeat_disables_torch_telemetry():
    from flash.engine.worker import sft

    node = _inner_func_node(sft.run_sft, "_sft_init_heartbeat")
    assert node is not None, "run_sft no longer defines _sft_init_heartbeat"
    _assert_gpu_diag_disables_torch(node, "run_sft._sft_init_heartbeat")


# --------------------------------------------------------------------------------------------
# prefetch_model: snapshot_download blocks with NO heartbeat until it returns. A cold cache can pull
# tens of GB for many minutes — longer than the stall watchdog AND the provider setup grace — so the
# download must ping a progress heartbeat (re-arming the watchdog) or a HEALTHY long fetch self-kills.
def test_prefetch_model_heartbeats_during_download():
    from flash.engine.worker import hf

    node = _inner_func_node(hf.prefetch_model, "_prefetch_heartbeat")
    assert node is not None, (
        "prefetch_model no longer pings a heartbeat during snapshot_download — a long cold-cache "
        "download would look like a hang and trip the stall watchdog / provider setup grace"
    )
    # nvidia-smi only (the GPU isn't in use yet, and torch telemetry could block).
    _assert_gpu_diag_disables_torch(node, "prefetch_model._prefetch_heartbeat")
    # ...and it must emit the progress stage that proves liveness + re-arms the watchdog.
    stages = [
        n.args[0].value
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "heartbeat"
        and n.args
        and isinstance(n.args[0], ast.Constant)
    ]
    assert "model_prefetching" in stages, f"expected a model_prefetching heartbeat, got {stages}"


def test_prefetch_model_joins_heartbeat_with_a_bounded_timeout():
    """The prefetch heartbeat thread must be joined with a BOUNDED timeout, never unbounded.

    An unbounded ``join()`` would wedge the worker indefinitely if the side thread were stuck inside
    ``heartbeat()``'s HF upload (``hf_upload_file`` -> ``huggingface_hub.upload_file`` has no hard
    timeout): the model download would have completed, yet the main thread would hang here forever.
    Ordering of ``model_prefetched`` after the last ``model_prefetching`` does NOT depend on this
    join — it is guaranteed by the ``is_set()`` guard in the loop (asserted below) plus heartbeat()'s
    ``_HB_UPLOAD_LOCK`` (which serializes any in-flight upload before ``model_prefetched``'s) — so the
    join only reaps the daemon thread and must stay bounded."""
    from flash.engine.worker import hf

    tree = ast.parse(inspect.getsource(hf.prefetch_model))
    joins = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "join"
    ]
    assert joins, "prefetch_model no longer joins its heartbeat thread"
    for call in joins:
        # A bounded join is expressed as a positional timeout (join(10.0)) or join(timeout=...).
        positional = call.args and isinstance(call.args[0], ast.Constant)
        keyworded = any(kw.arg == "timeout" for kw in call.keywords)
        assert positional or keyworded, (
            "prefetch_model must join() its heartbeat thread with a BOUNDED timeout — an unbounded "
            "join can wedge the worker if the side thread is stuck inside an HF upload"
        )


def test_prefetch_heartbeat_skips_emit_once_done_is_set():
    """The ordering guarantee that replaces the (removed) unbounded join: the side thread must
    re-check ``_prefetch_done`` after its wait and RETURN without emitting once the download is done,
    so no ``model_prefetching`` can START after ``model_prefetched`` and overtake it on HF."""
    from flash.engine.worker import hf

    node = _inner_func_node(hf.prefetch_model, "_prefetch_heartbeat")
    assert node is not None, "prefetch_model no longer defines _prefetch_heartbeat"

    # Find the loop body and assert it guards on _prefetch_done.is_set() BEFORE the heartbeat call.
    guards = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "is_set"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "_prefetch_done"
    ]
    assert guards, (
        "_prefetch_heartbeat must re-check _prefetch_done.is_set() after waking, then return without "
        "emitting model_prefetching — otherwise a stale stage can race past model_prefetched"
    )


# --------------------------------------------------------------------------------------------
# Stall watchdog default. A true hang (no heartbeat for the window) must self-dump every thread's
# stack and fail the run instead of wedging silently until the control-plane kill (the
# "process wedged, no console upload" gap). This is safe-by-default because every heartbeat re-arms
# the timer and the init heartbeats above now keep ticking through a slow-but-live cold init.


@pytest.mark.skipif(
    bool(
        {"FLASH_STALL_FAULTHANDLER_S", "FLASH_STALL_FAULTHANDLER_STARTUP_S"}
        & set(__import__("os").environ)
    ),
    # Both knobs are asserted below; either one set in the env overrides the defaults under test.
    reason="env overrides the watchdog default(s) under test",
)
def test_stall_watchdog_enabled_by_default():
    import importlib

    # Note: ``from flash.engine.worker import heartbeat`` resolves to the re-exported heartbeat()
    # FUNCTION, not the submodule — import the module path explicitly. Reload so the module-level
    # defaults reflect the env AT TEST TIME, not whatever env was present when it was first imported
    # (another test may have reloaded it under a patched env).
    hb = importlib.reload(importlib.import_module("flash.engine.worker.heartbeat"))

    assert hb._STALL_FAULTHANDLER_S >= 600, "stall watchdog must be ON by default so hangs self-dump"
    # The first-arm startup grace must be at least as wide as the steady-state window.
    assert hb._STALL_STARTUP_GRACE_S >= hb._STALL_FAULTHANDLER_S


def test_stall_watchdog_disabled_via_worker_env(monkeypatch):
    import importlib

    hb = importlib.import_module("flash.engine.worker.heartbeat")

    monkeypatch.setenv("FLASH_STALL_FAULTHANDLER_S", "0")
    try:
        importlib.reload(hb)
        assert hb._STALL_FAULTHANDLER_S == 0, "FLASH_STALL_FAULTHANDLER_S=0 must disable the watchdog"
    finally:
        monkeypatch.delenv("FLASH_STALL_FAULTHANDLER_S", raising=False)
        importlib.reload(hb)  # restore to env-current state for the rest of the suite
