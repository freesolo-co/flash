"""Numerical custom-loss and checkpoint tests on real torch, without model downloads."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flash.custom import Datum, TrainingClient
from flash.custom.backend import TorchBackend
from flash.custom.session import TrainingSession


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Embedding(7, 7)
        self.config = SimpleNamespace(_name_or_path="test/tiny")

    def get_output_embeddings(self):
        return self.projection

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.projection(input_ids))

    def save_pretrained(self, path):
        path.mkdir(parents=True)
        torch.save(self.state_dict(), path / "model.pt")


class TinyTokenizer:
    def save_pretrained(self, path):
        pass


class InProcessHTTP:
    def __init__(self, session):
        self.session = session

    def _request(self, method, path, body=None, **kwargs):
        if method == "GET":
            return self.session.status()
        return self.session.execute(body)


@pytest.fixture
def client_and_backend(tmp_path, monkeypatch):
    torch.manual_seed(9)
    backend = TorchBackend(TinyLM(), TinyTokenizer(), checkpoint_dir=tmp_path)
    session = TrainingSession(backend)
    monkeypatch.setattr(
        "flash.custom.client.ApiClient", lambda *args, **kwargs: InProcessHTTP(session)
    )
    return TrainingClient("http://localhost", api_key="test"), backend


def test_two_pass_cross_sequence_loss_has_exact_direct_autograd_gradient(client_and_backend):
    client, backend = client_and_backend
    direct = copy.deepcopy(backend.model)
    data = [Datum([1, 2, 3, 4]), Datum([1, 5, 6])]

    def loss_fn(rows, probs):
        margin = probs[0][1:].sum() - probs[1][1:].sum()
        loss = -torch.nn.functional.logsigmoid(0.3 * margin)
        return loss, {"margin": margin.item()}

    logits = [direct(torch.tensor([row.tokens[:-1]])).logits[0] for row in data]
    logprobs = [
        values.log_softmax(-1).gather(-1, torch.tensor(row.tokens[1:])[:, None]).squeeze(-1)
        for row, values in zip(data, logits, strict=True)
    ]
    expected, _ = loss_fn(data, logprobs)
    expected.backward()
    actual = client.forward_backward_custom(data, loss_fn)
    assert actual["metrics"]["custom_loss"] == pytest.approx(expected.item())
    assert torch.allclose(
        backend.model.projection.weight.grad, direct.projection.weight.grad, atol=1e-6
    )
    # prompt-only input position receives no gradient from completion-only DPO scoring.
    assert torch.count_nonzero(backend.model.projection.weight.grad[1]) == 0


def test_gradient_accumulation_step_and_checkpoint_restore_optimizer(client_and_backend):
    client, backend = client_and_backend
    data = [Datum([1, 2, 3], [0, 1])]
    client.save_checkpoint("initial")
    initial = backend.model.projection.weight.detach().clone()
    client.forward_backward(data)
    gradient = backend.model.projection.weight.grad.clone()
    client.forward_backward(data)
    assert torch.allclose(backend.model.projection.weight.grad, gradient * 2)
    assert torch.equal(initial, backend.model.projection.weight)
    result = client.optim_step(learning_rate=0.1)
    assert result["metrics"]["accumulated_backward_calls"] == 2
    assert not torch.equal(initial, backend.model.projection.weight)
    updated = backend.model.projection.weight.detach().clone()
    client.load_checkpoint("initial")
    assert torch.equal(initial, backend.model.projection.weight)
    client.forward_backward(data)
    client.forward_backward(data)
    client.optim_step(learning_rate=0.1)
    assert torch.equal(updated, backend.model.projection.weight)
    assert client.model_version == 3


def test_bad_custom_loss_never_accumulates_gradients(client_and_backend):
    client, backend = client_and_backend
    data = [Datum([1, 2])]

    def nonfinite(rows, probs):
        return probs[0].sum() * float("nan"), {}

    with pytest.raises(ValueError, match="finite scalar"):
        client.forward_backward_custom(data, nonfinite)
    assert backend.pending_backward == 0
    assert all(p.grad is None for p in backend.parameters)


def test_checkpoint_rejects_pending_gradients_and_name_traversal(client_and_backend):
    client, backend = client_and_backend
    client.forward_backward([Datum([1, 2], [1])])
    with pytest.raises(ValueError, match="accumulated gradients"):
        backend.validate("save_checkpoint", {"name": "pending"})
    with pytest.raises(ValueError, match="checkpoint name"):
        backend.validate("load_checkpoint", {"name": "../../untrusted"})


def test_new_backend_restores_optimizer_rng_and_algorithm_state(client_and_backend, tmp_path):
    client, original = client_and_backend
    data = [Datum([1, 2, 3], [0, 1])]
    client.forward_backward(data)
    client.optim_step(learning_rate=0.1, algorithm_state={"cursor": 17, "reference": [-2.0]})
    client.save_checkpoint("durable")
    expected_random = torch.rand(4)
    expected_scores = original.forward(data)["outputs"]
    client.forward_backward(data)
    client.optim_step(learning_rate=0.1)
    expected_weights = original.model.projection.weight.detach().clone()

    # a new model, optimizer and checkpoint directory: nothing relies on the old session.
    fresh = TorchBackend(TinyLM(), TinyTokenizer(), checkpoint_dir=tmp_path / "fresh")
    restored = fresh.restore_checkpoint(tmp_path / "durable")
    assert restored["algorithm_state"] == {"cursor": 17, "reference": [-2.0]}
    assert torch.equal(torch.rand(4), expected_random)
    assert fresh.forward(data)["outputs"] == expected_scores
    fresh.forward_backward(data)
    fresh.optim_step(learning_rate=0.1, max_grad_norm=1.0)
    assert torch.equal(fresh.model.projection.weight, expected_weights)


def test_mismatched_base_revision_rejected_before_weight_mutation(client_and_backend, tmp_path):
    client, _original = client_and_backend
    client.save_checkpoint("base")
    fresh = TorchBackend(TinyLM(), TinyTokenizer(), checkpoint_dir=tmp_path / "fresh")
    fresh.model.config._commit_hash = "a" * 40
    before = fresh.model.projection.weight.detach().clone()
    with pytest.raises(ValueError, match="revision differs"):
        fresh.restore_checkpoint(tmp_path / "base")
    assert torch.equal(before, fresh.model.projection.weight)
