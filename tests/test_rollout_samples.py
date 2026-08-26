from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from flash.engine.result.rollout_samples import build_rollout_sample, select_rollout_samples


def test_build_rollout_sample_shows_full_text_and_sanitizes_without_redacting_placeholder(
    monkeypatch,
) -> None:
    secret = "sk-test-secret-1234567890"
    monkeypatch.setenv("ROLLOUT_SAMPLE_API_KEY", secret)
    prompt = "p" * 520 + f" api_key={secret}"
    completion = "ANSWER: <value> " + secret + " " + "c" * 1100

    sample = build_rollout_sample(prompt, completion, reward=0.75, generated_at_step=2)

    # Full prompt + completion are preserved (no length truncation), only the secret is redacted.
    assert len(sample["prompt_tail"]) > 500
    assert "p" * 520 in sample["prompt_tail"]
    assert secret not in sample["prompt_tail"]
    assert secret not in sample["completion"]
    assert sample["completion"].startswith("ANSWER: <value> <redacted>")
    assert sample["completion"].endswith("c" * 1100)
    assert "[truncated]" not in sample["completion"]
    assert sample["reward"] == 0.75
    assert "loss" not in sample
    assert sample["generated_at_step"] == 2


def test_build_rollout_sample_redacts_prompt_secret_without_truncating(monkeypatch) -> None:
    secret = "prompt-boundary-secret-xyz"
    monkeypatch.setenv("ROLLOUT_SAMPLE_API_KEY", secret)
    prompt = f"prefix-{secret}{'p' * 490}"

    sample = build_rollout_sample(prompt, "completion", reward=1.0, generated_at_step=1)

    assert "secret-xyz" not in sample["prompt_tail"]
    assert secret not in sample["prompt_tail"]
    assert "<redacted>" in sample["prompt_tail"]
    assert sample["prompt_tail"].startswith("prefix-<redacted>")
    assert sample["prompt_tail"].endswith("p" * 490)


def test_build_rollout_sample_redacts_completion_secret_without_truncating(monkeypatch) -> None:
    secret = "completion-boundary-secret-xyz"
    monkeypatch.setenv("ROLLOUT_SAMPLE_API_KEY", secret)
    completion = f"{'c' * 988}{secret}-suffix"

    sample = build_rollout_sample("prompt", completion, reward=1.0, generated_at_step=1)

    assert secret not in sample["completion"]
    assert "<redacted>" in sample["completion"]
    assert sample["completion"].endswith("-suffix")
    assert "[truncated]" not in sample["completion"]


def test_build_rollout_sample_neutralizes_terminal_control_characters() -> None:
    sample = build_rollout_sample(
        "line one\rrewrite\nline two\tindented",
        "answer\x1b[2J\x9b2J\nnext\x00done\x7f",
        reward=1.0,
        generated_at_step=1,
    )

    assert sample["prompt_tail"] == "line one\\x0drewrite\nline two\\x09indented"
    assert sample["completion"] == "answer\\x1b[2J\\x9b2J\nnext\\x00done\\x7f"
    for field in ("prompt_tail", "completion"):
        assert all(
            char == "\n" or 0x20 <= ord(char) < 0x7F or ord(char) >= 0xA0 for char in sample[field]
        )


def test_build_rollout_sample_replaces_non_text_multimodal_parts() -> None:
    image_data = "image-base64-payload" * 100
    audio_data = "audio-base64-payload" * 100
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe the media"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                {"type": "input_audio", "input_audio": {"data": audio_data}},
                {"type": "tool_result", "payload": {"opaque": "do-not-log"}},
            ],
        }
    ]

    sample = build_rollout_sample(prompt, "answer", reward=1.0, generated_at_step=1)

    assert sample["prompt_tail"] == ("user: describe the media\n<image>\n<audio>\n<non-text>")
    assert image_data not in sample["prompt_tail"]
    assert audio_data not in sample["prompt_tail"]
    assert "do-not-log" not in sample["prompt_tail"]


def test_build_rollout_sample_carries_loss_scalar_for_opd() -> None:
    sample = build_rollout_sample("prompt", "student answer", loss=0.4213, generated_at_step=3)

    assert sample["loss"] == 0.4213
    assert "reward" not in sample
    assert sample["completion"] == "student answer"
    assert sample["generated_at_step"] == 3


@pytest.mark.parametrize("scalar", ["reward", "loss"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_scalar_is_omitted_so_progress_stays_parseable(scalar, value) -> None:
    """A diverged step must not publish a scalar json cannot represent.

    ``json.dumps`` writes bare ``NaN``/``Infinity`` by default -- not json. A strict reader rejects
    the whole progress record over one such field, so the step's other diagnostics die with it.
    Assert on
    ``allow_nan=False``, which is exactly the strict reader's behaviour: a plain ``json.dumps`` here
    would serialize the defect happily and the test could never fail."""
    sample = build_rollout_sample("prompt", "completion", generated_at_step=2, **{scalar: value})

    assert scalar not in sample, f"non-finite {scalar} reached the wire"
    assert sample["completion"] == "completion"
    json.dumps(sample, allow_nan=False)


def test_select_rollout_samples_prefers_distinct_prompts_then_fills_repeats() -> None:
    triples = [
        ("prompt-a", "a-first", 0.1),
        ("prompt-a", "a-second", 0.2),
        ("prompt-b", "b-first", 0.3),
        ("prompt-c", "c-first", 0.4),
        ("prompt-d", "d-first", 0.5),
    ]

    selected = select_rollout_samples(triples, generated_at_step=7)

    assert [sample["completion"] for sample in selected] == ["a-first", "b-first", "c-first"]
    assert [sample["reward"] for sample in selected] == [0.1, 0.3, 0.4]
    assert all(sample["generated_at_step"] == 7 for sample in selected)


def test_select_rollout_samples_hard_caps_at_three() -> None:
    triples = [(f"prompt-{index}", f"completion-{index}", index) for index in range(8)]

    selected = select_rollout_samples(triples, generated_at_step=1)

    assert len(selected) == 3


def test_select_rollout_samples_loss_scalar_for_opd() -> None:
    triples = [
        ([{"role": "user", "content": "q-a"}], "answer-a", 0.11),
        ([{"role": "user", "content": "q-b"}], "answer-b", 0.22),
    ]

    selected = select_rollout_samples(triples, generated_at_step=5, scalar="loss")

    assert [sample["loss"] for sample in selected] == [0.11, 0.22]
    assert all("reward" not in sample for sample in selected)
    assert [sample["prompt_tail"] for sample in selected] == ["user: q-a", "user: q-b"]
    assert all(sample["generated_at_step"] == 5 for sample in selected)


def test_select_rollout_samples_rejects_unknown_scalar() -> None:
    with pytest.raises(ValueError, match="scalar"):
        select_rollout_samples([("p", "c", 1.0)], generated_at_step=1, scalar="entropy")


@pytest.mark.parametrize(
    "module",
    ["flash.providers._lifecycle.instances.poll", "flash.runner", "flash.cli.commands"],
)
def test_control_plane_modules_import_without_running_worker_init(module: str) -> None:
    """The control plane must not execute worker startup just by being imported.

    ``flash.engine.worker.__init__`` parses ``ATTEMPT`` at module scope and RAISES on a malformed
    value, which is correct inside a managed worker and wrong everywhere else: the CLI and the
    provider pollers run in processes that never set it. Importing any submodule of that package
    runs the package ``__init__`` on the way in, so a single ``from flash.engine.worker.x import y``
    in shared code is enough to make every one of these modules fail to import.

    A malformed ``ATTEMPT`` is the cheapest way to arm that: it turns an invisible side effect into
    a loud one. The subprocess is the point -- an in-process import would find the module already in
    ``sys.modules`` from collection and prove nothing.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        env={**os.environ, "ATTEMPT": "not-a-number"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"importing {module} ran worker init: {result.stderr.strip().splitlines()[-1:]}"
    )
