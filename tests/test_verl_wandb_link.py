"""the w&b run link must survive the verl subprocess boundary, on all three verl backends.

the sdk's link_wandb reads notes["wandb_url"]. trl gets it for free by spreading
wandb_run_info() -- it calls wandb.init in the flash process, so wandb.run is live there. verl
calls wandb.init inside the TRAINING SUBPROCESS, so the parent's wandb.run is None and that same
spread would silently yield nothing. these tests pin both halves of the replacement channel: the
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

from flash.engine.worker.verl_common import parse_wandb_link, render_wandb_link_shim

_URL = "https://wandb.ai/acme/flash/runs/abc123"
_ID = "abc123"
_WORKER = pathlib.Path(__file__).resolve().parents[1] / "flash" / "engine" / "worker"


class _Run:
    url = _URL
    id = _ID


def _exec_shim(init_impl):
    """run the shim the way the CHILD interpreter would, and return what it printed.

    the shim is a source fragment destined for another interpreter, so reading the rendered string
    proves nothing about behavior; exec it against a stand-in wandb (the real one is not a flash
    dependency) and capture stdout, which is the actual parent-facing channel.
    """
    fake = types.ModuleType("wandb")
    fake.init = init_impl
    saved = sys.modules.get("wandb")
    sys.modules["wandb"] = fake
    try:
        exec(compile(render_wandb_link_shim(), "<shim>", "exec"), {})
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            returned = sys.modules["wandb"].init(project="flash")
        return buffer.getvalue(), returned
    finally:
        if saved is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = saved


def test_child_shim_reports_a_link_the_parent_can_parse():
    # the whole contract in one hop: what the child prints must be what the parent reads back.
    printed, _ = _exec_shim(lambda *a, **k: _Run())
    assert parse_wandb_link(printed.strip()) == {"wandb_url": _URL, "wandb_id": _ID}


def test_child_shim_returns_the_real_run_and_forwards_arguments():
    # the shim WRAPS wandb.init; swallowing the run object or its arguments would break verl's own
    # logging, turning an observability fix into a training-path regression.
    run = _Run()
    seen: dict = {}

    def _init(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return run

    _, returned = _exec_shim(_init)
    assert returned is run
    assert seen["kwargs"] == {"project": "flash"}


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
        exec(compile(render_wandb_link_shim(), "<shim>", "exec"), {})
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
    tree = ast.parse((_WORKER / module).read_text())
    builders = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

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


@pytest.mark.parametrize("module", ["sft_verl.py", "opd_verl.py", "rl_verl.py"])
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
