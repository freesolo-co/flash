"""`flash train --cost`: map a training config to a pre-flight cost."""

from __future__ import annotations

import copy
import types

import pytest

from flash.cli.commands import cmd_train
from flash.cost.spec import count_sft_train_tokens
from flash.cost.spec import runconfig_from_spec as _runconfig_from_spec
from flash.cost.spec import spec_steps as _spec_steps
from flash.cost.types import RunConfig
from flash.engine.recipe import RECIPE
from flash.envs.registry import _FLASH_TRAIN_MAX_EXAMPLES, training_env_params
from flash.schema import spec_from_dict

GRPO_RAW = {
    "model": "Qwen/Qwen3.5-9B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {
        "steps": 50,
        "group_size": 8,
        "batch_size": 16,
        "max_tokens": 512,
        "max_length": 2048,
        "hf_repo": "owner/runs",
    },
    "gpu": {"type": "RTX 5090"},
}


def _spec(**overrides):
    raw = copy.deepcopy(GRPO_RAW)
    for key, value in overrides.items():
        section, _, leaf = key.partition(".")
        if leaf:
            raw.setdefault(section, {})[leaf] = value
        else:
            raw[section] = value
    return spec_from_dict(raw)


def test_runconfig_from_grpo_spec_maps_fields():
    spec = _spec()
    cfg = _runconfig_from_spec(spec)
    assert cfg.model_id == "Qwen/Qwen3.5-9B"
    assert cfg.method == "grpo"
    assert cfg.steps == 50
    assert cfg.batch_size == 16
    assert cfg.group_size == 8
    assert cfg.completion_len == 512  # GRPO max_tokens
    assert cfg.seq_len == 2048
    assert cfg.environment == "github:freesolo-co/envs@main:gsm8k/environment.py"


def test_grpo_uses_recipe_steps_when_omitted():
    spec = _spec()
    object.__setattr__(spec.train, "steps", None)
    assert _spec_steps(spec) == RECIPE.rl.num_steps


def test_sft_steps_derived_from_examples():
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "environment": {"id": "github:acme/envs@main:sft-data/environment.py"},
            "train": {
                "max_examples": 320,
                "batch_size": 16,
                "epochs": 2,
                "hf_repo": "owner/runs",
            },
            "gpu": {"type": "RTX 4090"},
        }
    )
    # batch_size 16 is a multiple of the per-device micro-batch (4) so realized == requested:
    # epochs(2) x ceil(320 / 16) = 40.
    assert _spec_steps(spec) == 40
    cfg = _runconfig_from_spec(spec)
    assert cfg.method == "sft"
    assert cfg.group_size is None  # SFT carries no completions-per-prompt
    assert cfg.completion_len is None


def _sft_spec(**train):
    raw = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "sft",
        "environment": {"id": "github:acme/envs@main:sft-data/environment.py"},
        "train": {"hf_repo": "owner/runs", **train},
        "gpu": {"type": "RTX 4090"},
    }
    return spec_from_dict(raw)


def _worker_sft_steps(*, examples, requested_batch, epochs, max_steps=0):
    """Independent re-derivation of the worker's per-seed SFT optimizer-step count (engine.worker:
    fixed per-device micro-batch 4 + ceil grad-accum -> realized global batch), to pin _spec_steps
    against what actually runs."""
    from flash.engine.vram import _sft_per_device_bs

    per_device = max(1, min(_sft_per_device_bs(), requested_batch))
    grad_accum = max(1, -(-requested_batch // per_device))
    realized = per_device * grad_accum
    n = max(1, -(-examples // realized) * epochs)
    return min(n, max_steps) if max_steps > 0 else n


def test_sft_steps_default_epochs_mirror_the_worker():
    # No [train].epochs -> the worker uses RECIPE.sft.num_epochs (2), NOT 1. The estimate must too.
    spec = _sft_spec(max_examples=320, batch_size=16)  # epochs omitted
    assert spec.train.epochs is None
    assert RECIPE.sft.num_epochs == 2
    assert _spec_steps(spec) == _worker_sft_steps(examples=320, requested_batch=16, epochs=2) == 40


def test_sft_steps_use_worker_realized_grad_accum_batch():
    # batch_size 6 is NOT a multiple of the micro-batch (4): the worker realizes per_device(4) x
    # grad_accum(ceil(6/4)=2) = 8, so steps = epochs(2) x ceil(320/8) = 80 -- NOT the raw-batch
    # ceil(320/6)*2 = 108 the old derivation produced.
    spec = _sft_spec(max_examples=320, batch_size=6, epochs=2)
    assert _spec_steps(spec) == _worker_sft_steps(examples=320, requested_batch=6, epochs=2) == 80


def test_sft_ignores_train_steps_for_step_count():
    # train.steps is a GRPO concept; SFT must derive from epochs/examples/realized-batch and NOT
    # honor a stray train.steps.
    spec = _sft_spec(max_examples=320, batch_size=16, epochs=2, steps=9999)
    assert _spec_steps(spec) == 40


def test_sft_steps_unpinned_counts_the_env_dataset(monkeypatch):
    # No max_examples -> the worker trains the FULL dataset, so the step count must use the env's
    # REAL train-split size (count_env_examples), not a magic constant.
    monkeypatch.setattr("flash.cost.spec.count_env_examples", lambda env_id, params=None: 320)
    spec = _sft_spec(batch_size=16, epochs=2)  # max_examples omitted
    assert spec.train.max_examples is None
    assert _spec_steps(spec) == _worker_sft_steps(examples=320, requested_batch=16, epochs=2) == 40


def test_sft_steps_max_examples_zero_means_no_cap(monkeypatch):
    # max_examples = 0 is "no cap" (the worker trains the FULL dataset: `max_examples or 0` then
    # truncate only `if > 0`), NOT "0 examples". It must fall through to counting the env dataset,
    # else a `max_examples = 0` run would price 1 step and be grossly under-charged..
    monkeypatch.setattr("flash.cost.spec.count_env_examples", lambda env_id, params=None: 320)
    spec = _sft_spec(max_examples=0, batch_size=16, epochs=2)
    assert spec.train.max_examples == 0
    assert _spec_steps(spec) == _worker_sft_steps(examples=320, requested_batch=16, epochs=2) == 40


def test_sft_uncapped_count_forwards_loader_no_cap(monkeypatch):
    captured = []

    def fake_count(env_id, params=None):
        captured.append(dict(params or {}))
        return 10_178

    monkeypatch.setattr("flash.cost.spec.count_env_examples", fake_count)

    specs = (
        _sft_spec(batch_size=32, epochs=2),
        _sft_spec(max_examples=0, batch_size=32, epochs=2),
    )
    for spec in specs:
        assert _spec_steps(spec) == _worker_sft_steps(
            examples=10_178, requested_batch=32, epochs=2
        ) == 638

    assert captured == [
        {_FLASH_TRAIN_MAX_EXAMPLES: None},
        {_FLASH_TRAIN_MAX_EXAMPLES: None},
    ]


def test_grpo_train_max_examples_does_not_forward_loader_cap():
    spec = _spec(**{"train.max_examples": 10_178})

    assert training_env_params(spec) == {}


def test_sft_train_token_count_forwards_loader_no_cap(monkeypatch):
    captured = []

    def fake_load_env(env_id, params=None):
        captured.append(dict(params or {}))

    monkeypatch.setattr("flash.cost.spec._load_env", fake_load_env)
    spec = _sft_spec(max_examples=10_178, batch_size=32, epochs=2)

    assert count_sft_train_tokens(spec) is None
    assert captured == [{_FLASH_TRAIN_MAX_EXAMPLES: None}]


def test_sft_steps_unpinned_requires_countable_env(monkeypatch):
    # No max_examples means the worker trains the full dataset. If the local cost path cannot
    # count that dataset, it must fail clearly instead of guessing a magic example count.
    monkeypatch.setattr("flash.cost.spec.count_env_examples", lambda env_id, params=None: None)
    spec = _sft_spec(batch_size=16, epochs=2)
    with pytest.raises(ValueError, match="environment dataset could not be counted"):
        _spec_steps(spec)


def test_sft_steps_pinned_examples_skips_the_uncounted_fallback(monkeypatch, capsys):
    # An explicit [train].max_examples must price exactly that, never touching the env-count
    # fallback (no warning, no default).
    monkeypatch.setattr(
        "flash.cost.spec.count_env_examples",
        lambda env_id, params=None: (_ for _ in ()).throw(AssertionError("should not count")),
    )
    spec = _sft_spec(max_examples=320, batch_size=16, epochs=2)
    assert _spec_steps(spec) == _worker_sft_steps(examples=320, requested_batch=16, epochs=2) == 40
    assert "could not count" not in capsys.readouterr().err


def test_sft_runconfig_carries_actual_train_tokens(monkeypatch):
    monkeypatch.setattr("flash.cost.spec.count_sft_train_tokens", lambda spec: 190_679)
    spec = _sft_spec(max_examples=320, batch_size=16, epochs=2)
    cfg = _runconfig_from_spec(spec)
    assert cfg.steps == 40
    assert cfg.train_tokens == 190_679


def test_runconfig_preserves_positional_seq_len_compatibility():
    cfg = RunConfig("Qwen/Qwen3.5-4B", "sft", 10, 2048)
    assert cfg.seq_len == 2048
    assert cfg.train_tokens is None


def test_sft_train_token_count_renders_and_tokenizes_dataset(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from flash.cost.spec import count_sft_train_tokens

    class FakeEnv:
        def dataset(self):
            return [{"prompt": "aa", "answer": "bbb"}, {"prompt": "c", "answer": "dddd"}]

        def prompt_messages(self, ex):
            return [{"role": "user", "content": ex["prompt"]}]

        def sft_completion(self, ex):
            return [{"role": "assistant", "content": ex["answer"]}]

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

        def apply_chat_template(self, msgs, **kwargs):
            return "|".join(m["content"] for m in msgs)

        def __call__(self, rows, truncation=False, max_length=None):
            cap = max_length if truncation and max_length else 10**9
            return {"input_ids": [list(range(min(len(row), cap))) for row in rows]}

    monkeypatch.setattr("flash.cost.spec._load_env", lambda env_id, params=None: FakeEnv())
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer()
    )

    spec = _sft_spec(max_examples=2, batch_size=16, epochs=3, max_length=100)
    # Rendered rows: "aa|bbb" and "c|dddd"; tokenize_for_packing appends "<eos>" to each.
    assert count_sft_train_tokens(spec) == (len("aa|bbb<eos>") + len("c|dddd<eos>")) * 3


def test_sft_train_token_count_shuffles_before_max_examples(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from flash.cost.spec import count_sft_train_tokens

    class FakeEnv:
        def dataset(self):
            return [
                {"prompt": "a", "answer": "bb"},
                {"prompt": "ccc", "answer": "dddd"},
                {"prompt": "eeeee", "answer": "ffffff"},
            ]

        def prompt_messages(self, ex):
            return [{"role": "user", "content": ex["prompt"]}]

        def sft_completion(self, ex):
            return [{"role": "assistant", "content": ex["answer"]}]

    class FakeTokenizer:
        eos_token = ""
        pad_token = None
        all_special_ids = ()

        def apply_chat_template(self, msgs, **kwargs):
            return "".join(m["content"] for m in msgs)

        def __call__(self, rows, truncation=False, max_length=None):
            cap = max_length if truncation and max_length else 10**9
            return {"input_ids": [[ord(ch) for ch in row[:cap]] for row in rows]}

    monkeypatch.setattr("flash.cost.spec._load_env", lambda env_id, params=None: FakeEnv())
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer()
    )

    spec = _sft_spec(max_examples=1, batch_size=16, epochs=1, max_length=100)
    # random.Random(FIXED_SEED=42).shuffle([0, 1, 2]) selects row 1 first, not row 0.
    assert count_sft_train_tokens(spec) == len("cccdddd")


def test_sft_train_token_count_drops_completion_free_rows(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from flash.cost.spec import count_sft_train_tokens

    class FakeEnv:
        def dataset(self):
            return [{"prompt": "p", "answer": "xy"}, {"prompt": "drop", "answer": ""}]

        def prompt_messages(self, ex):
            return [{"role": "user", "content": ex["prompt"]}]

        def sft_completion(self, ex):
            return [{"role": "assistant", "content": ex["answer"]}]

    class FakeTokenizer:
        eos_token = ""
        pad_token = None
        all_special_ids = ()

        def apply_chat_template(self, msgs, **kwargs):
            return "".join(m["content"] for m in msgs)

        def __call__(self, rows, truncation=False, max_length=None):
            cap = max_length if truncation and max_length else 10**9
            return {"input_ids": [[ord(ch) for ch in row[:cap]] for row in rows]}

    monkeypatch.setattr("flash.cost.spec._load_env", lambda env_id, params=None: FakeEnv())
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer()
    )

    spec = _sft_spec(max_examples=2, batch_size=16, epochs=2, max_length=100)
    assert count_sft_train_tokens(spec) == len("pxy") * 2


def test_sft_max_steps_caps_the_derived_count():
    spec = _sft_spec(max_examples=10_000, batch_size=16, epochs=2, max_steps=5)
    assert _spec_steps(spec) == 5


def test_sft_steps_honor_big_vocab_per_device_cap():
    # For a sub-3B short-ctx SFT the worker vocab-sizes the per-device micro-batch (the big-vocab
    # logits cap), which with CEIL'd grad-accum changes the REALIZED global batch -- so the priced
    # step count must mirror the capped batch, not the fixed pd=4 one. Qwen3.5-0.8B (0.9B, ~248k
    # vocab) at a 1024 ctx leaves CE un-fused -> per_device caps 4->1, so batch 6 realizes 1x6=6
    # (not 4x2=8): steps = epochs(2) x ceil(320/6) = 108, NOT the uncapped ceil(320/8)*2 = 80.
    import math

    from flash.catalog import vocab_size_for
    from flash.engine.vram import sft_logits_fused, sft_per_device, sft_realized_batch

    raw = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "environment": {"id": "github:acme/envs@main:sft-data/environment.py"},
        "train": {
            "hf_repo": "owner/runs",
            "max_examples": 320, "batch_size": 6, "epochs": 2, "max_length": 1024,
        },
        "gpu": {"type": "RTX 4090"},
    }
    spec = spec_from_dict(raw)
    v = vocab_size_for("Qwen/Qwen3.5-0.8B")
    fused = sft_logits_fused(0.9, 1024)
    assert fused is False
    assert sft_per_device(6, seq_len=1024, vocab=v, fused=fused) == 1
    assert sft_realized_batch(6, seq_len=1024, vocab=v, fused=fused) == 6
    assert _spec_steps(spec) == math.ceil(320 / 6) * 2 == 108
    assert _spec_steps(spec) != 80  # the pre-fix uncapped (pd=4 -> realized 8) step count


def test_cmd_train_cost_prints_breakdown_without_submitting(tmp_path, capsys):
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\n"
        "steps = 50\n"
        'hf_repo = "owner/runs"\n'
        "[gpu]\n"
        'type = "RTX 5090"\n'
    )
    args = types.SimpleNamespace(
        config=str(cfg),
        overrides=[],
        extra_configs=[],
        cost=True,
        dry_run=False,
        background=False,
    )
    # --cost is local: it must NOT touch the control-plane client. GRPO needs no env load, and
    # estimate_cost sizes VRAM offline, so no network is required for a listed model.
    rc = cmd_train(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "TOTAL" in out
    assert "$" in out
    assert "GPU" in out  # the breakdown names the chosen (provisional cheapest-fit) class


def test_cmd_train_cost_rejects_unlisted_model(tmp_path):
    """Cost is catalog-only: ``--cost`` on a non-catalog model errors cleanly (no open-model
    sizing)."""
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'model = "some-org/unlisted-7b"\n'
        'model_policy = "allow"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
        "[train]\n"
        "steps = 10\n"
        'hf_repo = "owner/runs"\n'
    )
    args = types.SimpleNamespace(
        config=str(cfg), overrides=[], extra_configs=[], cost=True, dry_run=False, background=False
    )
    with pytest.raises((KeyError, ValueError)):
        cmd_train(args)
