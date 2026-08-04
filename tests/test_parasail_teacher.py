from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.engine.worker.kimi_k3_encoding import KimiK3Encoding
from flash.engine.worker.parasail_teacher import ParasailTeacherClient
from flash.engine.worker.teacher import TeacherError, _TeacherScoreRequest


class _ByteEncoding:
    def encode(self, text, **_kwargs):
        return [byte + 1 for byte in text.encode("utf-8")]

    def decode_single_token_bytes(self, token_id):
        return bytes([token_id - 1])


class _BoundaryEncoding:
    def __init__(self):
        self._next = 1000
        self._bytes = {}
        self._encoded = {}

    def encode(self, text, **_kwargs):
        if text in self._encoded:
            return list(self._encoded[text])
        if text == "xé":
            values = [text.encode("utf-8")]
        else:
            values = [bytes([byte]) for byte in text.encode("utf-8")]
        ids = []
        for value in values:
            token_id = self._next
            self._next += 1
            self._bytes[token_id] = value
            ids.append(token_id)
        self._encoded[text] = tuple(ids)
        return ids

    def decode_single_token_bytes(self, token_id):
        return self._bytes[token_id]


def _request(*, completion="Hà Nội", prefill=""):
    return _TeacherScoreRequest.from_messages(
        [
            {"role": "system", "content": "answer exactly"},
            {"role": "user", "content": "Thủ đô của Việt Nam?"},
        ],
        fireworks_prompt="System: answer exactly\nUser: Thủ đô của Việt Nam?\nAssistant: "
        + prefill,
        assistant_prefill=prefill,
        completion_text=completion,
    )


def _client(encoding):
    return ParasailTeacherClient(
        "secret",
        "https://api.parasail.io/v1",
        "parasail-kimi-k3",
        encoding_repo="moonshotai/Kimi-K3",
        encoding_revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        tokenizer_config_sha256="config-hash",
        tokenizer_model_sha256="model-hash",
        encoding=KimiK3Encoding(encoding),
    )


def _response(client, request, *, score=-0.25):
    encoded = client.encoding.encode_request(request)
    prompt_logprobs = [None]
    prompt_logprobs.extend(
        {
            str(token_id): {
                "logprob": score,
                "decoded_token": "provider text is intentionally ignored",
            }
        }
        for token_id in encoded.token_ids[1:]
    )
    return {
        "prompt_token_ids": list(encoded.token_ids),
        "prompt_logprobs": prompt_logprobs,
        "choices": [{"index": 0, "token_ids": [123]}],
        "usage": {
            "prompt_tokens": len(encoded.token_ids),
            "completion_tokens": 1,
        },
    }


def test_parasail_request_shape_and_exact_usage():
    client = _client(_ByteEncoding())
    request = _request()
    captured = {}
    response = _response(client, request)

    def post(path, body):
        captured.update({"path": path, "body": body})
        return response

    client._post = post
    tokens = client.score(request)

    assert captured == {
        "path": "/chat/completions",
        "body": {
            "model": "parasail-kimi-k3",
            "messages": [
                {"role": "system", "content": "answer exactly"},
                {"role": "user", "content": "Thủ đô của Việt Nam?"},
                {"role": "assistant", "content": "Hà Nội"},
            ],
            "prompt_logprobs": 1,
            "continue_final_message": True,
            "return_token_ids": True,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
        },
    }
    assert tokens.input_tokens == response["usage"]["prompt_tokens"]
    assert tokens.output_tokens == 1
    assert "".join(token.text for token in tokens) == "Hà Nội"
    assert all(token.logprob == -0.25 for token in tokens)
    assert tokens[-1].end == len("Hà Nội")


def test_parasail_uses_numeric_ids_and_local_unicode_spans():
    client = _client(_ByteEncoding())
    request = _request(completion="თბილისი")
    response = _response(client, request, score=-0.5)
    for entry in response["prompt_logprobs"][1:]:
        next(iter(entry.values()))["decoded_token"] = "corrupted"
    client._post = lambda _path, _body: response

    tokens = client.score(request)

    assert "".join(token.text for token in tokens) == "თბილისი"
    assert any(token.start == token.end for token in tokens)
    assert tokens[-1].end == len("თბილისი")
    assert all(token.logprob == -0.5 for token in tokens)


def test_parasail_supports_token_crossing_prefill_completion_boundary():
    client = _client(_BoundaryEncoding())
    request = _request(completion="é", prefill="x")
    response = _response(client, request)
    client._post = lambda _path, _body: response

    tokens = client.score(request)

    assert len(tokens) == 1
    assert tokens[0].text == "é"
    assert (tokens[0].start, tokens[0].end) == (0, 1)


def test_parasail_rejects_full_prompt_id_mismatch():
    client = _client(_ByteEncoding())
    request = _request()
    response = _response(client, request)
    old_id = response["prompt_token_ids"][5]
    response["prompt_token_ids"][5] += 1
    record = response["prompt_logprobs"][5].pop(str(old_id))
    response["prompt_logprobs"][5][str(response["prompt_token_ids"][5])] = record
    client._post = lambda _path, _body: response

    with pytest.raises(TeacherError, match="pinned local encoding") as error:
        client.score(request)
    assert error.value.permanent is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda response: response.update(prompt_logprobs=[]), "prompt logprobs"),
        (
            lambda response: response["prompt_logprobs"].__setitem__(1, {"other": {"logprob": -1}}),
            "realized token-id",
        ),
        (
            lambda response: response["prompt_logprobs"].__setitem__(
                1,
                {
                    str(response["prompt_token_ids"][1]): {
                        "logprob": 1e-12,
                    }
                },
            ),
            "realized logprob",
        ),
        (lambda response: response["choices"][0].update(token_ids=[]), "exactly one"),
        (lambda response: response["usage"].update(completion_tokens=2), "output usage"),
    ],
)
def test_parasail_rejects_malformed_success_responses(mutation, message):
    client = _client(_ByteEncoding())
    request = _request()
    response = _response(client, request)
    mutation(response)
    client._post = lambda _path, _body: response

    with pytest.raises(TeacherError, match=message) as error:
        client.score(request)
    assert error.value.permanent is True


def test_parasail_score_many_is_bounded_and_ordered():
    client = _client(_ByteEncoding())
    active = 0
    peak = 0
    lock = threading.Lock()

    def score_request(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return request.completion_text

    client._score_request = score_request
    requests = [replace(_request(), completion_text=str(index)) for index in range(20)]

    assert client.score_many(requests) == [str(index) for index in range(20)]
    assert peak <= 8
    assert peak > 1


def test_parasail_requires_structured_messages_without_leaking_key():
    client = _client(_ByteEncoding())

    with pytest.raises(TeacherError, match="structured") as error:
        client.score_many([("prompt", "completion")])

    assert error.value.permanent is True
    assert "secret" not in str(error.value)


_GOLDEN = json.loads(
    (Path(__file__).with_name("fixtures") / "kimi_k3_remote_prompt_ids.json").read_text(
        encoding="utf-8"
    )
)


class _GoldenSegmentEncoding:
    def __init__(self):
        self._segment_ids = {}
        for key, token_ids in _GOLDEN["segment_token_ids"].items():
            text, allow_special = json.loads(key)
            self._segment_ids[(text, allow_special)] = tuple(token_ids)
        self._token_bytes = {
            int(token_id): bytes.fromhex(value)
            for token_id, value in _GOLDEN["token_bytes_hex"].items()
        }

    def encode(self, text, **kwargs):
        allow_special = kwargs.get("allowed_special") == "all"
        return list(self._segment_ids[(text, allow_special)])

    def decode_single_token_bytes(self, token_id):
        return self._token_bytes[token_id]


@pytest.mark.parametrize(
    "case_name",
    [
        "ascii",
        "vietnamese",
        "georgian",
        "assistant_prefill",
        "assistant_history",
        "multi_turn",
    ],
)
def test_kimi_k3_encoding_matches_independent_remote_prompt_vectors(case_name):
    case = _GOLDEN["cases"][case_name]
    request = _TeacherScoreRequest.from_messages(
        case["messages"],
        fireworks_prompt="unused",
        assistant_prefill=case["assistant_prefill"],
        completion_text=case["completion_text"],
    )

    encoded = KimiK3Encoding(_GoldenSegmentEncoding()).encode_request(request)

    assert _GOLDEN["source"] == {
        "captured_from": "approved Gate 0 /chat/completions prompt_token_ids",
        "model": "parasail-kimi-k3",
        "provider": "parasail",
        "revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
    }
    assert encoded.token_ids == tuple(case["prompt_token_ids"])
    assert "".join(token.text for token in encoded.completion_tokens) == case["completion_text"]


def test_kimi_k3_local_rendering_preserves_nonstandard_roles():
    request = _TeacherScoreRequest.from_messages(
        [{"role": "tool-result", "content": "result text"}],
        fireworks_prompt="Tool: result text\nAssistant: ",
        assistant_prefill="",
        completion_text="done",
    )

    encoded = KimiK3Encoding(_ByteEncoding()).encode_request(request)
    rendered = bytes(token_id - 1 for token_id in encoded.token_ids).decode()

    assert 'role="tool-result"' in rendered
    assert "result text" in rendered


def test_parasail_preflight_forces_pinned_encoding_load(monkeypatch):
    import flash.engine.worker.parasail_teacher as parasail_mod

    loaded = []
    encoding = KimiK3Encoding(_ByteEncoding())

    def load_encoding(**kwargs):
        loaded.append(kwargs)
        return encoding

    monkeypatch.setattr(parasail_mod, "load_kimi_k3_encoding", load_encoding)
    client = ParasailTeacherClient(
        "secret",
        "https://api.parasail.io/v1",
        "parasail-kimi-k3",
        encoding_repo="moonshotai/Kimi-K3",
        encoding_revision="revision",
        tokenizer_config_sha256="config-hash",
        tokenizer_model_sha256="model-hash",
    )
    client._post = lambda *_args, **_kwargs: pytest.fail(
        "provider scoring must not run during preflight"
    )

    client.preflight()

    assert loaded == [
        {
            "repo_id": "moonshotai/Kimi-K3",
            "revision": "revision",
            "tokenizer_config_sha256": "config-hash",
            "tokenizer_model_sha256": "model-hash",
        }
    ]
    assert client.encoding is encoding


def _write_tiny_pinned_files(tmp_path):
    import flash.engine.worker.kimi_k3_encoding as encoding_mod

    config = {
        "tokenizer_class": "TikTokenTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]},
        "added_tokens_decoder": {
            str(token_id): {"content": token}
            for token, token_id in encoding_mod._EXPECTED_SPECIAL_TOKENS.items()
        },
    }
    config_path = tmp_path / "tokenizer_config.json"
    model_path = tmp_path / "tiktoken.model"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    model_path.write_bytes(b"tiny pinned ranks")
    return config_path, model_path


def test_download_and_build_validates_exact_pinned_inputs_without_network(monkeypatch, tmp_path):
    import huggingface_hub
    import tiktoken
    import tiktoken.load

    import flash.engine.worker.kimi_k3_encoding as encoding_mod

    config_path, model_path = _write_tiny_pinned_files(tmp_path)
    calls = []

    def download(*, repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        return str(config_path if filename == "tokenizer_config.json" else model_path)

    class TinyRanks:
        def __len__(self):
            return 163584

    built = {}

    def build_encoding(**kwargs):
        built.update(kwargs)
        return SimpleNamespace(n_vocab=163840)

    monkeypatch.setattr(encoding_mod.importlib.metadata, "version", lambda _name: "0.13.0")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(tiktoken.load, "load_tiktoken_bpe", lambda path: TinyRanks())
    monkeypatch.setattr(tiktoken, "Encoding", build_encoding)

    result = encoding_mod._download_and_build(
        repo_id="moonshotai/Kimi-K3",
        revision="pinned-revision",
        tokenizer_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        tokenizer_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )

    assert calls == [
        ("moonshotai/Kimi-K3", "tokenizer_config.json", "pinned-revision"),
        ("moonshotai/Kimi-K3", "tiktoken.model", "pinned-revision"),
    ]
    assert built["name"] == "flash-kimi-k3"
    assert built["pat_str"] == encoding_mod._PAT_STR
    assert len(built["mergeable_ranks"]) == 163584
    assert built["special_tokens"] == dict(encoding_mod._EXPECTED_SPECIAL_TOKENS) | {
        f"<|reserved_token_{token_id}|>": token_id for token_id in range(163590, 163838)
    }
    assert result._encoding.n_vocab == 163840


def test_download_and_build_rejects_wrong_tiktoken_version_before_download(monkeypatch):
    import huggingface_hub

    import flash.engine.worker.kimi_k3_encoding as encoding_mod

    monkeypatch.setattr(encoding_mod.importlib.metadata, "version", lambda _name: "0.12.0")
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: pytest.fail("download must not run for a wrong tiktoken version"),
    )

    with pytest.raises(TeacherError, match=r"tiktoken==0\.13\.0") as error:
        encoding_mod._download_and_build(
            repo_id="moonshotai/Kimi-K3",
            revision="pinned-revision",
            tokenizer_config_sha256="unused",
            tokenizer_model_sha256="unused",
        )

    assert error.value.permanent is True


@pytest.mark.parametrize("bad_file", ["config", "model"])
def test_download_and_build_rejects_hash_mismatch_before_parsing(monkeypatch, tmp_path, bad_file):
    import huggingface_hub

    import flash.engine.worker.kimi_k3_encoding as encoding_mod

    config_path, model_path = _write_tiny_pinned_files(tmp_path)

    def download(*, filename, **_kwargs):
        return str(config_path if filename == "tokenizer_config.json" else model_path)

    monkeypatch.setattr(encoding_mod.importlib.metadata, "version", lambda _name: "0.13.0")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if bad_file == "config":
        config_hash = "0" * 64
    else:
        model_hash = "0" * 64

    with pytest.raises(TeacherError, match="hash mismatch") as error:
        encoding_mod._download_and_build(
            repo_id="moonshotai/Kimi-K3",
            revision="pinned-revision",
            tokenizer_config_sha256=config_hash,
            tokenizer_model_sha256=model_hash,
        )

    assert error.value.permanent is True
