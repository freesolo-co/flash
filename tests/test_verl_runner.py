"""CPU tests for the verl runner's reverse-text benchmark task (dataset + reward).

The verl GPU path itself needs the verl stack on a GPU; these cover the torch-free pieces that
define WHAT verl trains on when ``FLASH_VERL_ENV=reverse-text`` — used for the flash(verl)-vs-tinker
training-cost benchmark on the same task.
"""

from __future__ import annotations

import importlib.util

from flash.engine import verl_runner as V


def test_reverse_text_rows_ground_truth_is_reversed():
    rows = V._reverse_text_rows(6)
    assert len(rows) == 6
    for r in rows:
        text = r["prompt"][1]["content"]
        assert r["prompt"][0]["content"] == V._REVERSE_SYS
        assert r["reward_model"]["ground_truth"] == text[::-1]
        assert r["data_source"] == "reverse_text"
    # reproducible (fixed seed) -> stable dataset across runs
    assert [r["prompt"][1]["content"] for r in V._reverse_text_rows(6)] == [
        r["prompt"][1]["content"] for r in rows
    ]


def _load_reward(tmp_path, env):
    V.WORKDIR = str(tmp_path)
    V._write_reward(env=env)
    spec = importlib.util.spec_from_file_location("verl_reward_t", f"{tmp_path}/verl_reward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_score


def test_reverse_text_reward_scores_correctness(tmp_path):
    score = _load_reward(tmp_path, "reverse-text")
    gt = "hello"[::-1]  # "olleh"
    # perfect answer in tags -> ~1.0
    assert score("reverse_text", "<reversed_text>olleh</reversed_text>", gt) == 1.0
    # correct content, no tags -> high but below the tagged max (no format bonus)
    assert 0.85 <= score("reverse_text", "olleh", gt) < 1.0
    # wrong answer in tags -> low (only the format bonus)
    assert score("reverse_text", "<reversed_text>zzzzz</reversed_text>", gt) < 0.2
    # empty ground truth -> 0
    assert score("reverse_text", "anything", "") == 0.0


def test_default_reward_is_length_gate(tmp_path):
    score = _load_reward(tmp_path, "")
    assert score("x", "non-empty", "gt") == 1.0
    assert score("x", "", "gt") == 0.0
