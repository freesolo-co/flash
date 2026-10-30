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


def _direct_log_softmax(model, tokens):
    return model(torch.tensor([tokens[:-1]])).logits[0].log_softmax(-1)


def test_soft_targets_give_exact_forward_kl_gradient(client_and_backend):
    client, backend = client_and_backend
    direct = copy.deepcopy(backend.model)
    tokens = [1, 2, 3, 4]
    target_tokens = [[2, 5, 0], [], [4, 6]]
    target_weights = [[0.7, 0.2, 0.1], [], [0.9, 0.1]]
    # the observed next token keeps its own weight alongside the soft targets.
    datum = Datum(tokens, [0.0, 0.5, 0.0], target_tokens, target_weights)
    log_probs = _direct_log_softmax(direct, tokens)
    expected = -0.5 * log_probs[1, tokens[2]]
    for position, (ids, weights) in enumerate(zip(target_tokens, target_weights, strict=True)):
        for token, weight in zip(ids, weights, strict=True):
            expected = expected - weight * log_probs[position, token]
    expected.backward()

    result = client.forward_backward([datum])
    assert result["metrics"]["surrogate_loss"] == pytest.approx(expected.item(), abs=1e-6)
    assert torch.allclose(
        backend.model.projection.weight.grad, direct.projection.weight.grad, atol=1e-6
    )
    returned = result["outputs"][0]["target_logprobs"]
    assert [len(values) for values in returned] == [3, 0, 2]
    assert returned[2][0] == pytest.approx(log_probs[2, 4].item(), abs=1e-6)


def test_forward_scores_listed_candidates(client_and_backend):
    client, backend = client_and_backend
    tokens = [1, 2, 3]
    scored = client.forward([Datum(tokens, target_tokens=[[6, 2], [3]])])["outputs"][0]
    log_probs = _direct_log_softmax(backend.model, tokens)
    assert scored["target_logprobs"][0] == pytest.approx(
        [log_probs[0, 6].item(), log_probs[0, 2].item()], abs=1e-6
    )
    assert scored["target_logprobs"][1] == pytest.approx([log_probs[1, 3].item()], abs=1e-6)
    assert backend.pending_backward == 0


def test_custom_soft_target_loss_matches_direct_autograd(client_and_backend):
    client, backend = client_and_backend
    direct = copy.deepcopy(backend.model)
    teacher = {0: ([2, 5], [-0.2, -1.8]), 1: ([3, 1, 4], [-0.1, -2.5, -3.0])}
    tokens = [1, 2, 3]
    target_tokens = [teacher[0][0], teacher[1][0]]

    def kl_loss(rows, probs, candidates):
        # forward KL to the teacher's renormalized top-k, plus a small hard-label term.
        loss = -0.1 * probs[0].sum()
        for position, student in enumerate(candidates[0]):
            reference = torch.tensor(teacher[position][1]).softmax(-1)
            loss = loss + (reference * (reference.log() - student)).sum()
        return loss, {"positions": len(candidates[0])}

    log_probs = _direct_log_softmax(direct, tokens)
    direct_next = log_probs.gather(-1, torch.tensor(tokens[1:])[:, None]).squeeze(-1)
    direct_candidates = [log_probs[i, ids] for i, ids in enumerate(target_tokens)]
    expected, _ = kl_loss(None, [direct_next], [direct_candidates])
    expected.backward()

    result = client.forward_backward_custom([Datum(tokens, target_tokens=target_tokens)], kl_loss)
    assert result["metrics"]["custom_loss"] == pytest.approx(expected.item(), abs=1e-6)
    assert result["metrics"]["positions"] == 2
    assert torch.allclose(
        backend.model.projection.weight.grad, direct.projection.weight.grad, atol=1e-6
    )


def test_two_argument_custom_losses_are_unchanged_without_targets(client_and_backend):
    client, _ = client_and_backend
    result = client.forward_backward_custom(
        [Datum([1, 2, 3])], lambda rows, probs: (-probs[0].sum(), {})
    )
    assert result["outputs"][0].get("target_logprobs") is None


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
    assert not (tmp_path / "durable.zip").exists()
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


def test_managed_rollback_cannot_diverge_from_published_step(client_and_backend):
    client, backend = client_and_backend
    client.save_checkpoint("earlier")
    client.forward_backward([Datum([1, 2], [1])])
    client.optim_step(learning_rate=0.1)
    backend.checkpoints = SimpleNamespace(published={1: backend.checkpoint_dir / "step-1"})
    before = backend.model.projection.weight.detach().clone()
    with pytest.raises(ValueError, match=r"custom\.resume_from"):
        backend.validate("load_checkpoint", {"name": "earlier"})
    assert torch.equal(before, backend.model.projection.weight)


def test_mismatched_base_revision_rejected_before_weight_mutation(client_and_backend, tmp_path):
    client, _original = client_and_backend
    client.save_checkpoint("base")
    fresh = TorchBackend(TinyLM(), TinyTokenizer(), checkpoint_dir=tmp_path / "fresh")
    fresh.model.config._commit_hash = "a" * 40
    before = fresh.model.projection.weight.detach().clone()
    with pytest.raises(ValueError, match="revision differs"):
        fresh.restore_checkpoint(tmp_path / "base")
    assert torch.equal(before, fresh.model.projection.weight)
