"""Client recovery, private streaming exports, and editable CLI scaffolding."""

from __future__ import annotations

import subprocess
import sys

import pytest

from flash.client.http import ApiError, RequestTimeoutError
from flash.custom import TrainingClient
from flash.custom.types import Datum


class LostReplyHTTP:
    def __init__(self):
        self.value = 0
        self.envelopes = []
        self.drop = True

    def _request(self, method, path, body=None, **kwargs):
        if method == "GET":
            return {"session_id": "session", "sequence": 0, "model_version": 0}
        self.envelopes.append(body)
        if len(self.envelopes) == 1:
            self.value += 1
        if self.drop:
            self.drop = False
            raise RequestTimeoutError("lost receipt")
        return {"session_id": "session", "sequence": body["sequence"], "model_version": self.value}


def test_pending_optimizer_request_freezes_intent_and_blocks_new_work(monkeypatch):
    http = LostReplyHTTP()
    monkeypatch.setattr("flash.custom.client.ApiClient", lambda *args, **kwargs: http)
    client = TrainingClient("https://gpu.example", api_key="session-token")
    with pytest.raises(RequestTimeoutError):
        client.optim_step(learning_rate=0.001)
    pending = client.pending_request
    pending["arguments"]["learning_rate"] = 1.0
    with pytest.raises(RuntimeError, match=r"retry_pending\(\)"):
        client.optim_step(learning_rate=0.5)
    client.retry_pending()
    assert http.envelopes[0] == http.envelopes[1]
    assert http.envelopes[1]["arguments"]["learning_rate"] == 0.001
    assert http.value == 1
    assert client.pending_request is None


@pytest.mark.parametrize(
    ("status", "detail", "remains_pending"),
    [
        (400, {"code": "custom_rejected"}, False),
        (409, {"code": "custom_rejected"}, False),
        (413, {"code": "custom_rejected"}, False),
        (413, "unrecognized proxy response", True),
        (502, "proxy failure", True),
        (400, "unrecognized proxy response", True),
    ],
)
def test_only_known_preexecution_rejections_clear_pending(
    monkeypatch, status, detail, remains_pending
):
    http = LostReplyHTTP()
    original = http._request

    def reject(method, path, body=None, **kwargs):
        if method == "GET":
            return original(method, path, body, **kwargs)
        raise ApiError(status, "rejected", detail=detail)

    http._request = reject
    monkeypatch.setattr("flash.custom.client.ApiClient", lambda *args, **kwargs: http)
    client = TrainingClient("https://gpu.example", api_key="session-token")
    with pytest.raises(ApiError):
        client.optim_step(learning_rate=0.001)
    assert (client.pending_request is not None) is remains_pending


def test_oversized_batch_can_be_split_without_losing_session(monkeypatch):
    http = LostReplyHTTP()
    http.drop = False
    monkeypatch.setattr("flash.custom.client.ApiClient", lambda *args, **kwargs: http)
    client = TrainingClient("https://gpu.example", api_key="session-token")
    row = Datum([1] * 32768, [0.123456789] * 32767)
    # ~0.46 MB per row; 80 rows pass the 32 MiB body cap before any request is sent.
    with pytest.raises(ValueError, match="split the batch"):
        client.forward_backward([row] * 80)
    assert client.pending_request is None
    assert http.envelopes == []
    client.forward_backward([row])
    assert http.envelopes[0]["sequence"] == 1
    assert client.pending_request is None


def test_scaffold_is_editable_and_never_overwrites(tmp_path):
    import ast

    from flash.cli.parsing.main import _build_parser

    destination = tmp_path / "algorithm.py"
    args = _build_parser().parse_args(["custom", "init", str(destination)])
    args.func(args)
    assert "make_dpo_loss" in destination.read_text()
    compile(destination.read_text(), str(destination), "exec")
    assert any(
        isinstance(node, ast.AsyncFunctionDef) and node.name == "main"
        for node in ast.parse(destination.read_text()).body
    )
    with pytest.raises(FileExistsError):
        args.func(args)


def test_custom_api_imports_without_any_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "-c", "from flash.custom import Datum, TrainingClient"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


class ScoringHTTP:
    """Answers forward with fixed logprobs and records the forward_backward weights."""

    def __init__(self, logprobs):
        self.logprobs = logprobs
        self.sent = []
        self.sequence = 0

    def _request(self, method, path, body=None, **kwargs):
        state = {"session_id": "session", "sequence": self.sequence, "model_version": 0}
        if method == "GET":
            return state
        self.sequence = state["sequence"] = body["sequence"]
        self.sent.append(body)
        if body["operation"] == "forward":
            return {**state, "outputs": [{"logprobs": row} for row in self.logprobs]}
        return {**state, "metrics": body["arguments"]["metrics"]}


# a pairwise margin loss over completion logprobs, with dL/dlogprob written out by hand.
_MANUAL_GRADIENT_SCRIPT = """
import sys
from flash.custom import Datum, TrainingClient
import flash.custom.client as module

class HTTP:
    sequence = 0
    sent = []
    def _request(self, method, path, body=None, **kwargs):
        state = {"session_id": "s", "sequence": self.sequence, "model_version": 0}
        if method == "GET":
            return state
        self.sequence = state["sequence"] = body["sequence"]
        self.sent.append(body)
        if body["operation"] == "forward":
            return {**state, "outputs": [{"logprobs": [-0.5, -1.0]}, {"logprobs": [-2.0, -0.25]}]}
        return {**state, "metrics": body["arguments"]["metrics"]}

http = HTTP()
module.ApiClient = lambda *args, **kwargs: http
client = TrainingClient("http://localhost", api_key="k")

def loss_fn(data, logprobs):
    margin = sum(logprobs[0]) - sum(logprobs[1])
    return -margin, {"margin": margin}, [[-1.0] * 2, [1.0] * 2]

result = client.forward_backward_custom(
    [Datum([1, 2, 3]), Datum([1, 4, 5])], loss_fn, gradients=True
)
assert "torch" not in sys.modules, "torch was imported"
weights = [row["weights"] for row in http.sent[-1]["arguments"]["data"]]
assert weights == [[1.0, 1.0], [-1.0, -1.0]], weights
assert result["metrics"]["custom_loss"] == -0.75, result
"""


def test_manual_gradients_train_without_torch_installed():
    result = subprocess.run(
        [sys.executable, "-S", "-c", _MANUAL_GRADIENT_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_manual_gradients_keep_target_matrix_shape_and_none_rows(monkeypatch):
    http = ScoringHTTP([[[-0.1, -2.0], [-0.3, -1.5]], [-0.2, -0.4]])
    monkeypatch.setattr("flash.custom.client.ApiClient", lambda *args, **kwargs: http)
    client = TrainingClient("http://localhost", api_key="k")
    data = [Datum([1, 2, 3], target_tokens=[[2, 4], [3, 5]]), Datum([1, 4, 5])]

    def loss_fn(rows, logprobs):
        logprobs[0][0][0] = 99.0  # mutating the copy never changes the shape reference.
        return 0.5, {}, [[[0.25, -0.5], [0.0, 1.0]], None]

    client.forward_backward_custom(data, loss_fn, gradients=True)
    sent = http.sent[-1]["arguments"]["data"]
    assert sent[0]["weights"] == [[-0.25, 0.5], [-0.0, -1.0]]
    assert sent[1]["weights"] == [0.0, 0.0]
    assert sent[0]["target_tokens"] == [[2, 4], [3, 5]]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ((0.5, {}), "must return"),
        ((float("nan"), {}, [[0.0, 0.0]]), "finite number"),
        ((0.5, [], [[0.0, 0.0]]), "JSON dictionary"),
        ((0.5, {}, []), "one entry per datum"),
        ((0.5, {}, [[0.0]]), "must have 2 entries"),
        ((0.5, {}, [[0.0, float("inf")]]), "non-finite"),
    ],
)
def test_bad_manual_gradients_never_reach_the_gpu(monkeypatch, result, message):
    http = ScoringHTTP([[-0.1, -0.2]])
    monkeypatch.setattr("flash.custom.client.ApiClient", lambda *args, **kwargs: http)
    client = TrainingClient("http://localhost", api_key="k")
    with pytest.raises(ValueError, match=message):
        client.forward_backward_custom([Datum([1, 2, 3])], lambda *_: result, gradients=True)
    assert [body["operation"] for body in http.sent] == ["forward"]


def test_missing_torch_points_at_the_torch_free_mode(monkeypatch):
    http = ScoringHTTP([[-0.1, -0.2]])
    monkeypatch.setattr("flash.custom.client.ApiClient", lambda *args, **kwargs: http)
    monkeypatch.setitem(sys.modules, "torch", None)
    client = TrainingClient("http://localhost", api_key="k")
    with pytest.raises(ImportError, match="gradients=True"):
        client.forward_backward_custom([Datum([1, 2, 3])], lambda *_: None)
    assert http.sent == []  # no GPU pass is spent before the missing dependency is reported.


def test_scaffold_project_creates_normal_train_config_without_overwriting(tmp_path):
    from flash.cli.parsing.main import _build_parser
    from flash.schema import spec_and_train_keys_from_file

    destination = tmp_path / "algorithm.py"
    args = _build_parser().parse_args(
        [
            "custom",
            "init",
            str(destination),
            "--project",
            "11111111-1111-4111-8111-111111111111",
        ]
    )
    args.func(args)
    config = destination.with_suffix(".toml")
    spec, _ = spec_and_train_keys_from_file(str(config), project_required=True)
    assert spec.algorithm == "custom"
    assert spec.custom["file"] == destination.name
    assert spec.project == args.project
    destination.unlink()
    with pytest.raises(FileExistsError):
        args.func(args)
    assert not destination.exists()
    assert spec_and_train_keys_from_file(str(config))[0].project == args.project


def test_clients_have_no_user_owned_provisioning_api():
    from flash.custom import AsyncTrainingClient

    assert not hasattr(TrainingClient, "create")
    assert not hasattr(AsyncTrainingClient, "create")


def test_text_prompts_and_oversized_encodes_are_explained_before_sending(monkeypatch):
    # researchers passed text to sample and >100k characters to encode; both used to fail
    # only on the GPU worker with a terse rejection.
    http = LostReplyHTTP()
    http.drop = False
    receipt = http._request

    def encoded(method, path, body=None, **kwargs):
        return {**receipt(method, path, body, **kwargs), "tokens": [1]}

    http._request = encoded
    monkeypatch.setattr("flash.custom.client.ApiClient", lambda *args, **kwargs: http)
    client = TrainingClient("https://gpu.example", api_key="session-token")
    with pytest.raises(TypeError, match=r"client\.encode\(prompt_text\)"):
        client.sample("<|im_start|>user\nhi<|im_end|>\n")
    with pytest.raises(ValueError, match="at most 100,000 characters"):
        client.encode("x" * 100_001)
    assert http.envelopes == []
    assert client.pending_request is None
    assert client.encode("x" * 100_000) == [1]
    assert http.envelopes[0]["operation"] == "encode"


def test_documented_limits_match_the_worker_validation():
    from types import SimpleNamespace

    from flash.custom.validation import validate_operation

    backend = SimpleNamespace(vocab_size=10, max_tokens=8, max_batch=2, pending_backward=0)
    validate_operation(backend, "encode", {"text": "x" * 100_000})
    with pytest.raises(ValueError, match="100000 characters"):
        validate_operation(backend, "encode", {"text": "x" * 100_001})
    with pytest.raises(ValueError, match="nonempty token list"):
        validate_operation(backend, "sample", {"prompt": "text"})
