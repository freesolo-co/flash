"""Answer-position scoring (Datum.target_positions) and the option CE+Brier loss helper."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

from flash.custom import Datum
from flash.custom.types import parse_batch
from flash.custom.validation import validate_operation

torch = pytest.importorskip("torch")

from flash.custom.losses import option_ce_brier_loss, option_ce_brier_loss_fn
from tests.test_custom_gradients import client_and_backend  # noqa: F401


def _direct_log_softmax(model, tokens):
    return model(torch.tensor([tokens[:-1]])).logits[0].log_softmax(-1)


def _parse(row, **kwargs):
    return parse_batch(
        [row],
        vocab_size=kwargs.get("vocab_size", 100),
        max_tokens=kwargs.get("max_tokens", 10),
        max_batch=2,
        weighted=kwargs.get("weighted", False),
    )


def test_target_positions_round_trip_through_the_wire_format() -> None:
    datum = Datum([1, 2, 3, 4], [[0.5, 0.5]], [[2, 5]], target_positions=[2])
    assert datum.to_dict()["target_positions"] == [2]
    assert "target_positions" not in Datum([1, 2]).to_dict()
    [parsed] = _parse(datum.to_dict(), weighted=True)
    assert parsed == datum
    assert parsed.scored_rows == 1
    assert Datum([1, 2, 3]).scored_rows == 2


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        ([], "strictly increasing"),
        ([2, 1], "strictly increasing"),
        ([1, 1], "strictly increasing"),
        ([3], "strictly increasing"),
        ([-1], "strictly increasing"),
        ([True], "strictly increasing"),
        ([1.0], "strictly increasing"),
        (1, "strictly increasing"),
    ],
)
def test_invalid_target_positions_rejected(positions, message) -> None:
    with pytest.raises(ValueError, match=message):
        _parse({"tokens": [1, 2, 3, 4], "target_positions": positions})


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"weights": [1.0, 0.0]}, "weights must have len\\(target_positions\\) entries"),
        (
            {"target_tokens": [[2], [3]], "weights": [[1.0], [1.0]]},
            "target_tokens must have len\\(target_positions\\) rows",
        ),
        ({"target_tokens": [[2, 3]], "weights": [[1.0]]}, "shape of target_tokens"),
    ],
)
def test_rows_follow_target_positions(row, message) -> None:
    with pytest.raises(ValueError, match=message):
        _parse({"tokens": [1, 2, 3, 4], "target_positions": [2], **row}, weighted=True)


def test_answer_position_datum_costs_k_entries_not_every_position() -> None:
    tokens = [1] * 1601
    columns = [list(range(64))]
    # every position would be 1600 * 64 > 100000 entries; one answer position is 64.
    with pytest.raises(ValueError, match="split the batch"):
        _parse({"tokens": tokens, "target_tokens": columns * 1600}, max_tokens=2048)
    [row] = _parse(
        {"tokens": tokens, "target_tokens": columns, "target_positions": [1599]},
        max_tokens=2048,
    )
    assert row.target_tokens == columns


def test_forward_top_k_budget_counts_scored_positions() -> None:
    backend = SimpleNamespace(vocab_size=100, max_tokens=6000, max_batch=2)
    long_row = {"tokens": [1] * 5002}
    with pytest.raises(ValueError, match="top-k output exceeds"):
        validate_operation(backend, "forward", {"data": [long_row], "top_k": 20})
    answer_only = {**long_row, "target_positions": [5000]}
    args = validate_operation(backend, "forward", {"data": [answer_only], "top_k": 20})
    assert args["data"][0].target_positions == [5000]


def test_forward_scores_only_listed_positions(client_and_backend) -> None:  # noqa: F811
    client, backend = client_and_backend
    tokens = [1, 2, 3, 4, 5]
    log_probs = _direct_log_softmax(backend.model, tokens)
    shaped = client.forward(
        [Datum(tokens, target_tokens=[[6, 2], [3, 1]], target_positions=[0, 3])], top_k=2
    )["outputs"][0]
    expected = log_probs[[0, 3]].gather(-1, torch.tensor([[6, 2], [3, 1]]))
    assert torch.allclose(torch.tensor(shaped["logprobs"]), expected, atol=1e-6)
    assert len(shaped["top_logprobs"]) == 2
    # implicit next-token targets at the listed positions.
    plain = client.forward([Datum(tokens, target_positions=[1, 2])])["outputs"][0]
    assert plain["logprobs"] == pytest.approx(
        [log_probs[1, 3].item(), log_probs[2, 4].item()], abs=1e-6
    )


def test_target_positions_gradient_equals_full_rows_with_zero_weights(
    client_and_backend,  # noqa: F811
) -> None:
    client, backend = client_and_backend
    full_backend = copy.deepcopy(backend)
    tokens = [1, 2, 3, 4]
    option_ids = [2, 5, 6]
    weights = [[0.7, -0.2, 0.1]]
    client.forward_backward([Datum(tokens, weights, [option_ids], target_positions=[1])])

    full = Datum(
        tokens,
        [[0.0, 0.0, 0.0], weights[0], [0.0, 0.0, 0.0]],
        [option_ids, option_ids, option_ids],
    )
    full_backend.forward_backward([full])
    assert torch.allclose(
        backend.model.projection.weight.grad, full_backend.model.projection.weight.grad, atol=1e-6
    )


def test_option_ce_brier_custom_loss_has_exact_direct_gradient(client_and_backend) -> None:  # noqa: F811
    client, backend = client_and_backend
    direct = copy.deepcopy(backend.model)
    option_ids = [2, 5, 6]
    data = [
        Datum([1, 2, 3], target_tokens=[option_ids], target_positions=[1]),
        Datum([4, 1, 5, 3], target_tokens=[option_ids], target_positions=[2]),
    ]
    targets = [0, [0.1, 0.3, 0.6]]
    rows = [
        _direct_log_softmax(direct, row.tokens)[row.target_positions].gather(
            -1, torch.tensor(row.target_tokens)
        )
        for row in data
    ]
    expected, _ = option_ce_brier_loss(rows, targets, brier_weight=0.5)
    expected.backward()

    result = client.forward_backward_custom(
        data, option_ce_brier_loss_fn(targets, brier_weight=0.5)
    )
    assert result["metrics"]["custom_loss"] == pytest.approx(expected.item(), abs=1e-6)
    assert set(result["metrics"]) >= {"option_ce", "option_brier", "option_accuracy"}
    assert torch.allclose(
        backend.model.projection.weight.grad, direct.projection.weight.grad, atol=1e-6
    )


def test_custom_loss_snapshots_target_positions_before_callback(client_and_backend) -> None:  # noqa: F811
    client, backend = client_and_backend
    data = [Datum([1, 2, 3, 4], target_tokens=[[2, 5]], target_positions=[1])]

    def mutating_loss(rows, probs):
        rows[0].target_positions[0] = 0
        return -probs[0].sum(), {}

    client.forward_backward_custom(data, mutating_loss)
    # the backward pass scored position 1, matching the forward pass.
    assert backend.model.projection.weight.grad is not None
    assert torch.count_nonzero(backend.model.projection.weight.grad[1]) == 0
    assert torch.count_nonzero(backend.model.projection.weight.grad[2]) > 0


def test_option_loss_values_match_the_formula() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    q = torch.softmax(logits[0], dim=-1)
    # shifting raw logprobs leaves the option renormalization unchanged.
    shifted = logits - 7.0
    loss, metrics = option_ce_brier_loss([shifted], [0])
    ce = -math.log(q[0].item())
    brier = (q[0].item() - 1) ** 2 + q[1].item() ** 2 + q[2].item() ** 2
    assert loss.item() == pytest.approx(ce + brier, abs=1e-6)
    assert metrics == pytest.approx(
        {"option_ce": ce, "option_brier": brier, "option_accuracy": 1.0}, abs=1e-6
    )

    soft = [0.2, 0.3, 0.5]
    loss, metrics = option_ce_brier_loss([logits[0], logits[0]], [soft, 2], brier_weight=0.0)
    soft_ce = -sum(t * math.log(p) for t, p in zip(soft, q.tolist(), strict=True))
    assert loss.item() == pytest.approx((soft_ce - math.log(q[2].item())) / 2, abs=1e-6)
    assert metrics["option_accuracy"] == 0.0


def test_option_loss_gradient_matches_finite_differences() -> None:
    targets = [1, [0.25, 0.25, 0.5, 0.0]]

    def objective(first, second):
        return option_ce_brier_loss([first, second], targets, brier_weight=1.3)[0]

    first = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
    second = torch.randn(4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(objective, (first, second))


@pytest.mark.parametrize(
    ("rows", "targets", "kwargs", "message"),
    [
        ([torch.zeros(3)], [3], {}, "outside"),
        ([torch.zeros(3)], [True], {}, "index or a probability"),
        ([torch.zeros(3)], [[0.5, 0.6, 0.0]], {}, "non-negative probabilities"),
        ([torch.zeros(3)], [[1.0, 0.0]], {}, "non-negative probabilities"),
        ([torch.zeros(2, 3)], [0], {}, "one row"),
        ([torch.zeros(1)], [0], {}, "at least 2"),
        ([torch.zeros(3)], [0, 1], {}, "one nonempty entry"),
        ([], [], {}, "one nonempty entry"),
        ([torch.zeros(3)], [0], {"brier_weight": -1}, "brier_weight"),
        ([torch.zeros(3)], [0], {"brier_weight": float("nan")}, "brier_weight"),
    ],
)
def test_option_loss_rejects_malformed_inputs(rows, targets, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        option_ce_brier_loss(rows, targets, **kwargs)
