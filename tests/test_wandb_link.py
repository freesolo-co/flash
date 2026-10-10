"""the w&b run link must survive the verl subprocess boundary, on all three verl backends.

the sdk's link_wandb reads notes["wandb_url"]. verl calls wandb.init inside the training
subprocess, so the parent process cannot inspect the live run directly. these tests pin both halves
of the replacement channel: the
child-side shim that reports the url, and the parent-side parser that reads it back.

note on scope: tests/test_metrics_schema_agent_contract.py pins what CONSUMERS read, but it
asserts against a hand-written notes mirror declared in that file, so it passes no matter what the
backends actually emit. the producer-side keys are pinned here instead, against real source.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import pathlib
import sys
import types

import pytest

from flash.engine.worker.train.core.child import runtime as child_runtime
from flash.engine.worker.train.core.child.runtime import install_wandb_link_reporting
from flash.engine.worker.train.entry.backend_common import parse_wandb_link

_URL = "https://wandb.ai/acme/flash/runs/abc123"
_ID = "abc123"
_WORKER = pathlib.Path(__file__).resolve().parents[1] / "flash" / "engine" / "worker"


class _Run:
    url = _URL
    id = _ID


class _ExitStack:
    def __init__(self):
        self.callbacks: list[tuple[object, tuple]] = []

    def register(self, callback, *args):
        self.callbacks.append((callback, args))
        return callback

    def close(self) -> None:
        for callback, args in reversed(self.callbacks):
            callback(*args)


class _FinishingRun(_Run):
    def __init__(
        self,
        service: dict[str, bool],
        *,
        active: dict[str, object] | None = None,
        fail: bool = False,
        transient_failures: int = 0,
    ):
        self.service = service
        self.active = active
        self.fail = fail
        self.transient_failures = transient_failures
        self.finish_calls = 0
        self.finished = False

    def finish(self, *, exit_code: int):
        self.finish_calls += 1
        assert exit_code == 0
        if self.finished:
            return
        if self.transient_failures:
            self.transient_failures -= 1
            raise RuntimeError("transient finish failure")
        if self.fail:
            raise RuntimeError("finish failed")
        if not self.service["open"]:
            raise RuntimeError("wandb service is already closed")
        self.finished = True
        if self.active is not None and self.active.get("run") is self:
            self.active["run"] = None


def _exec_shim(
    init_impl,
    *,
    exit_stack: _ExitStack | None = None,
    finish_impl=None,
    init_calls: int = 1,
    module_out: dict[str, object] | None = None,
):
    """run the shim the way the CHILD interpreter would, and return what it printed.

    the shim is a source fragment destined for another interpreter, so reading the rendered string
    proves nothing about behavior; exec it against a stand-in wandb (the real one is not a flash
    dependency) and capture stdout, which is the actual parent-facing channel.
    """
    fake = types.ModuleType("wandb")
    fake.init = init_impl
    if finish_impl is not None:
        fake.finish = finish_impl
    if module_out is not None:
        module_out["wandb"] = fake
    saved_wandb = sys.modules.get("wandb")
    saved_atexit = child_runtime.atexit
    sys.modules["wandb"] = fake
    if exit_stack is not None:
        child_runtime.atexit = types.SimpleNamespace(register=exit_stack.register)
    try:
        install_wandb_link_reporting()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            for _ in range(init_calls):
                returned = sys.modules["wandb"].init(project="flash")
        return buffer.getvalue(), returned
    finally:
        child_runtime.atexit = saved_atexit
        if saved_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = saved_wandb


def test_child_shim_reports_a_link_the_parent_can_parse():
    # the whole contract in one hop: what the child prints must be what the parent reads back.
    printed, _ = _exec_shim(lambda *a, **k: _Run())
    assert parse_wandb_link(printed.strip()) == {"wandb_url": _URL, "wandb_id": _ID}


def test_child_shim_returns_the_real_run_and_forwards_arguments():
    # the shim wraps wandb.init; swallowing the run object or its arguments would break verl's own
    # logging, turning an observability fix into a training-path regression.
    run = _Run()
    seen: dict = {}

    def _init(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return run

    _, returned = _exec_shim(_init)
    assert returned is run
    assert seen["kwargs"] == {"project": "flash"}


def test_child_shim_finishes_before_wandb_service_teardown():
    stack = _ExitStack()
    service = {"open": True}
    active: dict[str, object] = {"run": None}
    run = _FinishingRun(service, active=active)

    def _teardown():
        service["open"] = False

    def _init(*_args, **_kwargs):
        stack.register(_teardown)
        active["run"] = run
        return run

    def _module_finish():
        active_run = active["run"]
        if active_run is not None:
            active_run.finish(exit_code=0)

    module_out: dict[str, object] = {}
    _exec_shim(
        _init,
        exit_stack=stack,
        finish_impl=_module_finish,
        module_out=module_out,
    )
    stack.close()

    assert run.finished is True
    assert run.finish_calls == 1
    assert active["run"] is None
    assert service["open"] is False
    # verl tracking.__del__ calls module-level wandb.finish after the service callback stack.
    fake_wandb = module_out["wandb"]
    assert isinstance(fake_wandb, types.ModuleType)
    fake_wandb.finish()
    assert run.finish_calls == 1


def test_child_shim_registers_one_effective_finish_for_a_reused_run():
    stack = _ExitStack()
    service = {"open": True}
    run = _FinishingRun(service)
    stack.register(lambda: service.update(open=False))

    _exec_shim(lambda *_a, **_k: run, exit_stack=stack, init_calls=2)
    assert len(stack.callbacks) == 3
    stack.close()

    assert run.finished is True
    assert run.finish_calls == 1


def test_child_shim_retries_a_transient_finish_for_a_reused_run():
    stack = _ExitStack()
    service = {"open": True}
    run = _FinishingRun(service, transient_failures=1)
    stack.register(lambda: service.update(open=False))

    _exec_shim(lambda *_a, **_k: run, exit_stack=stack, init_calls=2)
    stack.close()

    assert run.finished is True
    assert run.finish_calls == 2
    assert service["open"] is False


def test_child_shim_finishes_each_distinct_run():
    stack = _ExitStack()
    service = {"open": True}
    runs = [_FinishingRun(service), _FinishingRun(service)]
    pending = iter(runs)
    stack.register(lambda: service.update(open=False))

    _exec_shim(lambda *_a, **_k: next(pending), exit_stack=stack, init_calls=2)
    stack.close()

    assert [run.finished for run in runs] == [True, True]
    assert [run.finish_calls for run in runs] == [1, 1]


def test_child_shim_treats_finish_failure_as_nonfatal():
    stack = _ExitStack()
    service = {"open": True}
    run = _FinishingRun(service, fail=True)
    stack.register(lambda: service.update(open=False))

    _exec_shim(lambda *_a, **_k: run, exit_stack=stack)
    stack.close()

    assert run.finish_calls == 1
    assert service["open"] is False


def test_child_shim_stays_silent_when_the_run_has_no_url():
    # better an unlinked run than a marker carrying a null url the parent would have to special-case.
    class _NoUrl:
        url = None
        id = _ID

    printed, _ = _exec_shim(lambda *a, **k: _NoUrl())
    assert printed.strip() == ""


def test_child_shim_survives_a_run_object_that_raises():
    # a logging link must never be able to abort paid training, so an exploding attribute is
    # swallowed rather than propagated into verl's init call.
    class _Exploding:
        @property
        def url(self):
            raise RuntimeError("boom")

    printed, _ = _exec_shim(lambda *a, **k: _Exploding())
    assert printed.strip() == ""


def test_child_shim_is_inert_when_wandb_is_not_installed():
    # sitecustomize runs for EVERY child, including console-only runs on interpreters without wandb.
    real_import = builtins.__import__

    def _no_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no module named wandb")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _no_wandb
    try:
        install_wandb_link_reporting()
    finally:
        builtins.__import__ = real_import


def test_parser_reads_a_marker_that_verl_prefixed():
    # verl decorates child output with rank and timestamp, so the marker is rarely line-initial.
    line = f'[rank0] 2026-07-29 FLASH_WANDB_LINK {{"wandb_url": "{_URL}", "wandb_id": "{_ID}"}}'
    assert parse_wandb_link(line) == {"wandb_url": _URL, "wandb_id": _ID}


@pytest.mark.parametrize(
    "line",
    [
        "an ordinary training log line",
        "FLASH_WANDB_LINK {truncated",
        'FLASH_WANDB_LINK ["not", "a", "dict"]',
        'FLASH_WANDB_LINK {"wandb_url": null}',
        'FLASH_WANDB_LINK {"wandb_url": ""}',
        'FLASH_WANDB_LINK {"wandb_id": "orphan"}',
        "",
    ],
)
def test_parser_rejects_malformed_lines_without_raising(line):
    # this parses child stdout under multi-rank interleaving: a torn line must yield None, because
    # raising here would kill a healthy run over a logging detail.
    assert parse_wandb_link(line) is None


def test_parser_tolerates_a_missing_id():
    # wandb_url is the consumed field; a run id that never arrived normalises to None rather than
    # discarding the url the sdk actually needs.
    assert parse_wandb_link(f'FLASH_WANDB_LINK {{"wandb_url": "{_URL}"}}') == {
        "wandb_url": _URL,
        "wandb_id": None,
    }


# extra files a trainer's write_train_meta notes builder can live in, after the module was split
# to stay under the file-size limit. paths are relative to _WORKER.
_NOTES_BUILDER_MODULES = {
    "rl_train.py": ("train/rl/launch/verl_config.py",),
    "opd_train.py": ("train/opd/orchestration/overrides.py",),
    "sft_train.py": ("train/sft/setup/config.py",),
}


def _dict_literal_keys(node: ast.Dict) -> set[str]:
    """top-level string keys of a dict literal; nested dicts (grpo_recipe) are not notes keys."""
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _emitted_notes_keys(module: str) -> set[str]:
    """the notes keys write_train_meta actually receives in one worker module.

    read from real source rather than a mirror: a mirror can only fail when someone edits the
    mirror, which is exactly how this gap survived.

    grpo passes notes=_build_verl_train_notes(...), so the keys live in that function's RETURNED
    dict, not in the call kwargs. counting kwarg names instead would leave the probe vacuous --
    dropping "wandb_url" from the returned dict would keep the suite green while link_wandb got
    nothing, which is the exact defect this module exists to catch.
    """
    # the trainer entry modules moved under worker/train/entry/ in the layout regrouping.
    tree = ast.parse((_WORKER / "train" / "entry" / module).read_text())
    # the notes builder need not live in the same file as the write_train_meta call: grpo calls it
    # from rl_train.py but defines it in train/rl/verl_config.py. collect builders from the module
    # AND from wherever it was split to, or the sentinel below reports every key as unresolvable.
    builder_sources = [tree]
    builder_sources.extend(
        ast.parse((_WORKER / extra).read_text()) for extra in _NOTES_BUILDER_MODULES.get(module, ())
    )
    builders = {
        n.name: n
        for src in builder_sources
        for n in ast.walk(src)
        if isinstance(n, ast.FunctionDef)
    }

    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        )
        if name != "write_train_meta":
            continue
        for kw in node.keywords:
            if kw.arg != "notes":
                continue
            if isinstance(kw.value, ast.Dict):
                keys.update(_dict_literal_keys(kw.value))
            elif isinstance(kw.value, ast.Call):
                fn = getattr(kw.value.func, "id", None) or getattr(kw.value.func, "attr", "")
                # an unresolvable builder means the probe cannot see the real keys. surface that as a
                # sentinel so the assert fails loudly instead of silently checking nothing.
                target = builders.get(fn)
                if target is None:
                    keys.add(f"<unresolved builder {fn}>")
                    continue
                for ret in ast.walk(target):
                    if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Dict):
                        keys.update(_dict_literal_keys(ret.value))
    return keys


@pytest.mark.parametrize("module", ["sft_train.py", "opd_train.py", "rl_train.py"])
def test_every_verl_backend_emits_the_wandb_link_keys(module):
    # all three run wandb.init out of process, so all three need the marker channel. one backend
    # quietly missing it is invisible to the consumer-side contract test.
    keys = _emitted_notes_keys(module)
    assert keys, f"{module}: found no write_train_meta notes -- broken probe, not a passing test"
    assert "wandb_url" in keys, f"{module} drops wandb_url; sdk link_wandb reads it"
    assert "wandb_id" in keys, f"{module} drops wandb_id"


@pytest.mark.parametrize("url", ["javascript:alert(1)", "http://wandb.ai/a/b/runs/c", "not-a-url"])
def test_parser_accepts_https_urls_only(url):
    # the sdk renders this as a clickable link, and child stdout also carries rollout text, so a
    # non-https scheme is rejected rather than forwarded to the consumer.
    assert parse_wandb_link(f'FLASH_WANDB_LINK {{"wandb_url": "{url}"}}') is None
