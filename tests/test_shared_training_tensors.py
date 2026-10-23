"""Real PEFT/tensor qualification; CUDA runs use this same suite on the worker."""

import copy
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from flash.training.shared.backend import PeftBackend
from flash.training.shared.contracts import ConflictError
from flash.training.shared.losses import objective


def base():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(123)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        eos_token_id=31,
        attention_dropout=0.0,
    )
    return LlamaForCausalLM(config).to(os.environ.get("FLASH_SHARED_TEST_DEVICE", "cpu"))


def adapter(backend, name):
    from peft import get_peft_model_state_dict

    return {
        k: v.detach().clone()
        for k, v in get_peft_model_state_dict(backend.model, adapter_name=name).items()
    }


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def payload(algorithm, version):
    row = {"tokens": [1, 2, 3, 4], "mask": [False, True, True]}
    if algorithm == "grpo":
        row.update(old_logprobs=[-3.5] * 3, advantages=[0, 1, -1])
    if algorithm == "opd":
        row.update(teacher_logprobs=[0, -2, -2], teacher_groups=[-1, 0, 1])
    return {"version": version, "rows": [row]}


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_interleaved_adapters_match_isolated_training_and_restore(tmp_path, algorithm):
    backend = PeftBackend(base(), root=tmp_path / "shared", rank=2, alpha=4)
    initial_base = {k: v.clone() for k, v in backend.model.state_dict().items()}
    config = {"algorithm": algorithm, "seed": 7, "learning_rate": 0.01}
    backend.create("alice", config)
    backend.create("bob", {**config, "seed": 8})
    bob_before = adapter(backend, "bob")
    initial_a = adapter(backend, "alice")
    first = backend.train("alice", payload(algorithm, 0))
    equal(adapter(backend, "bob"), bob_before)
    after_a = adapter(backend, "alice")
    assert any(not torch.equal(initial_a[k], after_a[k]) for k in initial_a)
    optimizer_a = copy.deepcopy(backend.adapters["alice"]["optimizer"].state_dict())
    backend.train("bob", payload(algorithm, 0))
    equal(adapter(backend, "alice"), after_a)
    equal(backend.adapters["alice"]["optimizer"].state_dict(), optimizer_a)
    saved = backend.checkpoint("alice", {})
    second = backend.train("alice", payload(algorithm, 1))
    final = adapter(backend, "alice")
    isolated = PeftBackend(base(), root=tmp_path / "isolated", rank=2, alpha=4)
    isolated.create("solo", config)
    assert isolated.train("solo", payload(algorithm, 0)) == first
    assert isolated.train("solo", payload(algorithm, 1)) == second
    equal(adapter(isolated, "solo"), final)
    backend.create("restored", config)
    backend.restore("restored", {"checkpoint": saved["checkpoint"]})
    assert backend.train("restored", payload(algorithm, 1)) == second
    equal(adapter(backend, "restored"), final)
    for key, value in initial_base.items():
        adjusted = "base_model.model." + key
        adjusted = adjusted.replace("q_proj.weight", "q_proj.base_layer.weight").replace(
            "v_proj.weight", "v_proj.base_layer.weight"
        )
        assert torch.equal(backend.model.state_dict()[adjusted], value)


def test_sampling_does_not_change_training_state_or_other_adapter(tmp_path):
    backend = PeftBackend(base(), root=tmp_path)
    config = {"algorithm": "grpo", "seed": 7, "learning_rate": 0.01}
    backend.create("a", config)
    backend.create("b", config)
    before = copy.deepcopy(backend.adapters["a"]["rng"])
    request = {"version": 0, "prompt": [1, 2], "max_tokens": 3, "seed": 0}
    first = backend.sample("a", request)
    backend.sample("b", {**request, "seed": 9})
    assert backend.sample("a", request) == first
    equal(backend.adapters["a"]["rng"], before)
    assert len(first["tokens"]) == len(first["logprobs"])


def test_losses_against_independent_formulas():
    row = {"mask": [False, True, True]}
    logs = torch.tensor([-2.0, -3.0, -4.0], requires_grad=True)
    loss = objective([logs], [row], "sft")
    assert float(loss) == 3.5
    loss.backward()
    assert logs.grad.tolist() == [0, -0.5, -0.5]
    row.update(teacher_groups=[-1, 0, 0], teacher_logprobs=[0, -6.0, -6.0])
    logs.grad = None
    loss = objective([logs], [row], "opd")
    # one aligned two-token group: ((-7)-(-6))/2 * (-7) / 2.
    assert float(loss) == 1.75
    loss.backward()
    assert logs.grad.tolist() == [0, -0.25, -0.25]


def test_grpo_clipping_gradient_and_masked_overflow():
    # positive/negative clipped cases contribute no gradient; unmasked
    # exponentiation of the prompt's deliberately extreme old logprob would overflow.
    logs = torch.tensor([-3.0, -2.0, -2.0, -2.0], requires_grad=True)
    row = {
        "mask": [False, True, True, True],
        "old_logprobs": [-1e6, -3.0, -1.0, -2.0],
        "advantages": [0.0, 1.0, -1.0, 2.0],
    }
    loss = objective([logs], [row], "grpo", normalizer=4)
    assert float(loss.detach()) == pytest.approx((-1.2 + 0.8 - 2) / 4)
    loss.backward()
    assert logs.grad.tolist() == [0.0, 0.0, 0.0, -0.5]
    row["old_logprobs"][1] = -1e6
    with pytest.raises(ConflictError, match="incompatible"):
        objective([logs], [row], "grpo")


def test_stale_version_rejection_preserves_all_adapter_state(tmp_path):
    backend = PeftBackend(base(), root=tmp_path)
    config = {"algorithm": "sft", "seed": 7, "learning_rate": 0.01}
    backend.create("a", config)
    backend.train("a", payload("sft", 0))
    before = adapter(backend, "a")
    optimizer = copy.deepcopy(backend.adapters["a"]["optimizer"].state_dict())
    with pytest.raises(ConflictError, match="version"):
        backend.train("a", payload("sft", 0))
    equal(before, adapter(backend, "a"))
    equal(optimizer, backend.adapters["a"]["optimizer"].state_dict())
