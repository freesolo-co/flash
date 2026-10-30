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
import pathlib

import pytest

from flash.engine.worker.train.entry.backend_common import parse_wandb_link

_URL = "https://wandb.ai/acme/flash/runs/abc123"
_ID = "abc123"
_WORKER = pathlib.Path(__file__).resolve().parents[1] / "flash" / "engine" / "worker"


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


@pytest.mark.parametrize("module", ["sft_train.py", "rl_train.py"])
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
