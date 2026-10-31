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


def test_target_matrix_gives_exact_forward_kl_gradient(client_and_backend):
    client, backend = client_and_backend
    direct = copy.deepcopy(backend.model)
    tokens = [1, 2, 3, 4]
    # [T, K]: each row lists K tokens scored where tokens[i+1] is predicted; zero weights pad.
    target_tokens = [[2, 5, 0], [3, 0, 0], [4, 6, 1]]
    weights = [[0.7, 0.2, 0.1], [0.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
    log_probs = _direct_log_softmax(direct, tokens)
    selected = log_probs.gather(-1, torch.tensor(target_tokens))
    expected = -(torch.tensor(weights) * selected).sum()
    expected.backward()

    result = client.forward_backward([Datum(tokens, weights, target_tokens)])
    assert result["metrics"]["surrogate_loss"] == pytest.approx(expected.item(), abs=1e-6)
    assert torch.allclose(
        backend.model.projection.weight.grad, direct.projection.weight.grad, atol=1e-6
    )
    assert torch.allclose(torch.tensor(result["outputs"][0]["logprobs"]), selected, atol=1e-6)


def test_forward_returns_logprobs_with_the_target_shape(client_and_backend):
    client, backend = client_and_backend
    tokens = [1, 2, 3]
    log_probs = _direct_log_softmax(backend.model, tokens)
    shaped = client.forward([Datum(tokens, target_tokens=[[6, 2], [3, 1]])])["outputs"][0]
    assert torch.allclose(
        torch.tensor(shaped["logprobs"]),
        log_probs.gather(-1, torch.tensor([[6, 2], [3, 1]])),
        atol=1e-6,
    )
    # without target_tokens the targets remain the next tokens, one per position.
    plain = client.forward([Datum(tokens)])["outputs"][0]
    assert plain["logprobs"] == pytest.approx(
        log_probs.gather(-1, torch.tensor([[2], [3]])).squeeze(-1).tolist(), abs=1e-6
    )
    assert backend.pending_backward == 0


def test_custom_loss_reads_target_shaped_logprobs(client_and_backend):
    client, backend = client_and_backend
    direct = copy.deepcopy(backend.model)
    tokens = [1, 2, 3]
    target_tokens = [[2, 5, 6], [3, 1, 4]]
    teacher = torch.tensor([[-0.2, -1.8, -9.0], [-0.1, -2.5, -3.0]]).softmax(-1)

    def kl_loss(rows, probs):
        # forward KL to the teacher's renormalized top-k: the loss sees [T, K] logprobs.
        return (teacher * (teacher.log() - probs[0])).sum(), {"positions": probs[0].shape[0]}

    expected, _ = kl_loss(
        None, [_direct_log_softmax(direct, tokens).gather(-1, torch.tensor(target_tokens))]
    )
    expected.backward()

    data = [Datum(tokens, target_tokens=target_tokens)]
    result = client.forward_backward_custom(data, kl_loss)
    assert result["metrics"]["custom_loss"] == pytest.approx(expected.item(), abs=1e-6)
    assert result["metrics"]["positions"] == 2
    assert torch.allclose(
        backend.model.projection.weight.grad, direct.projection.weight.grad, atol=1e-6
    )


def test_custom_loss_snapshots_nested_targets_before_callback(client_and_backend):
    client, backend = client_and_backend
    direct = copy.deepcopy(backend.model)
    data = [Datum([1, 2, 3], target_tokens=[[2, 5], [3, 4]])]
    selected = _direct_log_softmax(direct, data[0].tokens).gather(
        -1, torch.tensor(data[0].target_tokens)
    )
    expected = -selected.square().sum()
    expected.backward()

    def loss_fn(rows, logprobs):
        rows[0].tokens[1] = 6
        rows[0].target_tokens[0][0] = 6  # mutate an inner list
        rows[0].target_tokens[1] = [0, 0]  # replace a row in the outer list
        return -logprobs[0].square().sum(), {}

    result = client.forward_backward_custom(data, loss_fn)
    assert result["metrics"]["custom_loss"] == pytest.approx(expected.item(), abs=1e-5)
    assert torch.allclose(torch.tensor(result["outputs"][0]["logprobs"]), selected, atol=1e-6)
    assert torch.allclose(
        backend.model.projection.weight.grad, direct.projection.weight.grad, atol=1e-5
    )


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


class CheckpointedLM(TinyLM):
    supports_gradient_checkpointing = True

    def __init__(self, dropout=0.0):
        super().__init__()
        self.config.model_type = "qwen3_5_text"
        self.config.attention_dropout = 0.0
        self.attention_dropout = 0.0
        self.dropout = torch.nn.Dropout(dropout)
        self.checkpointing_kwargs = None
        self.modes = []

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpointing_kwargs = gradient_checkpointing_kwargs

    def forward(self, input_ids, use_cache=False):
        self.modes.append(self.training)
        output = super().forward(input_ids, use_cache)
        output.logits = torch.nn.functional.dropout(
            output.logits, p=self.attention_dropout, training=self.training
        )
        return output


def test_checkpointed_backward_runs_in_train_mode_and_returns_to_eval(tmp_path):
    model = CheckpointedLM()
    backend = TorchBackend(model, TinyTokenizer(), checkpoint_dir=tmp_path)
    assert backend.checkpointing
    assert model.checkpointing_kwargs == {"use_reentrant": False}
    backend.forward([Datum([1, 2, 3])])
    backend.forward_backward([Datum([1, 2, 3], [1.0, 1.0])])
    with pytest.raises(RuntimeError, match="non-finite"):
        backend.forward_backward([Datum([1, 2, 3], [float("inf"), 0.0])])
    # scoring stays in eval; only backward passes recompute, and eval returns after a failure.
    assert model.modes == [False, True, True]
    assert not model.training


def test_models_with_active_dropout_never_checkpoint(tmp_path):
    model = CheckpointedLM(dropout=0.1)
    backend = TorchBackend(model, TinyTokenizer(), checkpoint_dir=tmp_path)
    assert not backend.checkpointing
    assert model.checkpointing_kwargs is None
    backend.forward_backward([Datum([1, 2, 3], [1.0, 1.0])])
    assert model.modes == [False]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("setting", ["config", "module", "both"])
def test_functional_attention_dropout_keeps_both_passes_in_eval(tmp_path, nested, setting):
    model = CheckpointedLM()
    text_config = model.config
    if nested:
        model.config = SimpleNamespace(model_type="qwen3_5", text_config=text_config)
    if setting in {"config", "both"}:
        text_config.attention_dropout = 0.5
    if setting in {"module", "both"}:
        model.attention_dropout = 0.5
    backend = TorchBackend(model, TinyTokenizer(), checkpoint_dir=tmp_path)
    assert not backend.checkpointing
    assert model.checkpointing_kwargs is None
    targets = [[2, 5], [3, 4]]
    scored = backend.forward([Datum([1, 2, 3], target_tokens=targets)])
    trained = backend.forward_backward([Datum([1, 2, 3], [[1, 0], [0, 1]], targets)])
    assert scored["outputs"] == trained["outputs"]
    assert model.modes == [False, False]


@pytest.mark.parametrize("model_type", ["qwen3_5_text", "qwen3_5_moe_text"])
@pytest.mark.parametrize("nested", [False, True])
def test_checkpointing_accepts_audited_dense_and_moe_configs(tmp_path, model_type, nested):
    model = CheckpointedLM()
    model.config.model_type = model_type
    if nested:
        model.config = SimpleNamespace(
            model_type=model_type.removesuffix("_text"), text_config=model.config
        )
    backend = TorchBackend(model, TinyTokenizer(), checkpoint_dir=tmp_path)
    assert backend.checkpointing


@pytest.mark.parametrize(
    "behavior", ["unknown", "unknown_text", "missing_dropout", "batchnorm", "dropout2d"]
)
def test_unaudited_or_stateful_training_behavior_never_checkpoints(tmp_path, behavior):
    model = CheckpointedLM()
    if behavior == "unknown":
        model.config.model_type = "unqualified_architecture"
    elif behavior == "unknown_text":
        model.config.model_type = "qwen3_5"
        model.config.text_config = SimpleNamespace(model_type="unknown", attention_dropout=0.0)
    elif behavior == "missing_dropout":
        del model.config.attention_dropout
    elif behavior == "batchnorm":
        model.normalization = torch.nn.BatchNorm1d(7)
    else:
        model.dropout = torch.nn.Dropout2d(0.5)
    backend = TorchBackend(model, TinyTokenizer(), checkpoint_dir=tmp_path)
    assert not backend.checkpointing
    assert model.checkpointing_kwargs is None
    backend.forward_backward([Datum([1, 2], [1])])
    assert model.modes == [False]


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
