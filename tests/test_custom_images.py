"""Custom training with image rows, on a tiny real Qwen3.5 vision-language model (no downloads)."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
Image = pytest.importorskip("PIL.Image")

from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorPil,
)

from flash.client.http import ApiError
from flash.custom import Datum, TrainingClient
from flash.custom.backend import TorchBackend
from flash.custom.session import TrainingSession
from flash.custom.types import MAX_IMAGES_PER_DATUM, parse_batch

PAD, START, END, EOS = 60, 62, 63, 5
PLACEHOLDER = "<|image_pad|>"


class TinyTokenizer:
    """Single-character tokens over 'a'-'z' (ids 6-31), plus the image placeholder."""

    eos_token_id = EOS
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return PAD if token == PLACEHOLDER else 0

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in text.split(PLACEHOLDER):
            ids.extend(6 + (ord(c) - 97) % 26 for c in part)
            ids.append(PAD)
        return ids[:-1]

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(str(t) for t in tokens)

    def save_pretrained(self, path):
        pass


class TinyProcessor:
    """The Qwen processor contract: each placeholder becomes its image's pad run."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_processor = Qwen2VLImageProcessorPil(
            min_pixels=56 * 56, max_pixels=112 * 112, patch_size=14, merge_size=2
        )

    def __call__(self, text, images, add_special_tokens=False):
        grids = self.image_processor(images=images, return_tensors="pt")["image_grid_thw"]
        runs = iter(int(g.prod()) // 4 for g in grids)
        ids = []
        for index, part in enumerate(text[0].split(PLACEHOLDER)):
            if index:
                ids.extend([PAD] * next(runs))
            ids.extend(self.tokenizer.encode(part))
        return {"input_ids": [ids]}


def _tiny_model():
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 512,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        image_token_id=PAD,
        video_token_id=61,
        vision_start_token_id=START,
        vision_end_token_id=END,
    )
    model = Qwen3_5ForConditionalGeneration(config)
    model.config._name_or_path = "test/tiny-vl"
    # custom sessions adapt language modules only; the vision tower stays frozen.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "visual" not in name
    model.save_pretrained = lambda path: path.mkdir(parents=True)
    return model


def _uri(color, size=(100, 80)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class InProcessHTTP:
    def __init__(self, session):
        self.session = session

    def _request(self, method, path, body=None, **kwargs):
        if method == "GET":
            return self.session.status()
        try:
            return self.session.execute(body)
        except ValueError as error:
            # the worker answers a validation rejection with 400 custom_rejected.
            raise ApiError(400, str(error), detail={"code": "custom_rejected"}) from error


@pytest.fixture
def vl(tmp_path, monkeypatch):
    tokenizer = TinyTokenizer()
    backend = TorchBackend(
        _tiny_model(),
        tokenizer,
        checkpoint_dir=tmp_path,
        max_tokens=256,
        processor=TinyProcessor(tokenizer),
    )
    session = TrainingSession(backend)
    monkeypatch.setattr(
        "flash.custom.client.ApiClient", lambda *args, **kwargs: InProcessHTTP(session)
    )
    return SimpleNamespace(
        client=TrainingClient("http://localhost", api_key="test"),
        backend=backend,
        session=session,
    )


def _row(client, images, text="ab{img}cd"):
    return client.encode(text.replace("{img}", PLACEHOLDER), images=images)


def test_encode_expands_each_placeholder_to_its_image_run(vl):
    tokens = _row(vl.client, [_uri("red")])
    # 100x80 resizes to 112x84 -> 8x6 patches -> 12 merged image tokens.
    assert tokens.count(PAD) == 12
    assert tokens[:2] == [6, 7]
    assert tokens[-2:] == [8, 9]
    assert vl.backend.info()["images"] is True


def test_encode_rejects_a_placeholder_count_mismatch_without_failing_the_session(vl):
    with pytest.raises((ValueError, ApiError), match="one <\\|image_pad\\|> placeholder per image"):
        vl.client.encode("ab" + PLACEHOLDER + "cd", images=[_uri("red"), _uri("blue")])
    assert not vl.session.failed
    vl.client.encode("ab")


def test_forward_reads_the_pixels(vl):
    red = _row(vl.client, [_uri("red")])
    blue = _row(vl.client, [_uri("blue")])
    assert red == blue  # same sizes, same placeholder runs
    out = vl.client.forward([Datum(red, images=[_uri("red")]), Datum(blue, images=[_uri("blue")])])
    first, second = (row["logprobs"] for row in out["outputs"])
    assert len(first) == len(red) - 1
    # tokens after the image depend on its content, so a different image changes them.
    assert first[-1] != pytest.approx(second[-1])


def test_forward_matches_the_model_called_directly(vl):
    tokens = _row(vl.client, [_uri("green")])
    scored = vl.client.forward([Datum(tokens, images=[_uri("green")])])["outputs"][0]["logprobs"]
    vision = vl.backend.vision_inputs(tokens, [_uri("green")])
    ids = torch.tensor([tokens[:-1]])
    with torch.no_grad():
        logits = vl.backend.model(
            input_ids=ids,
            pixel_values=vision["pixel_values"],
            image_grid_thw=vision["image_grid_thw"],
            mm_token_type_ids=(ids == PAD).int(),
            use_cache=False,
        ).logits[0]
    expected = torch.log_softmax(logits.float(), -1).gather(-1, torch.tensor(tokens[1:])[:, None])
    assert scored == pytest.approx(expected.squeeze(-1).tolist(), abs=1e-5)


def test_image_rows_train_and_the_session_continues(vl):
    tokens = _row(vl.client, [_uri("red")])
    weights = [0.0] * (len(tokens) - 3) + [1.0, 1.0]
    before = vl.client.forward([Datum(tokens, images=[_uri("red")])])["outputs"][0]["logprobs"]
    vl.client.forward_backward([Datum(tokens, weights, images=[_uri("red")])])
    stepped = vl.client.optim_step(learning_rate=1e-2)
    assert stepped["metrics"]["grad_norm"] > 0
    after = vl.client.forward([Datum(tokens, images=[_uri("red")])])["outputs"][0]["logprobs"]
    assert sum(after[-2:]) > sum(before[-2:])
    vision_tower = [p for n, p in vl.backend.model.named_parameters() if "visual" in n]
    assert vision_tower
    assert all(p.grad is None for p in vision_tower)


def test_custom_loss_carries_images_through_both_passes(vl):
    tokens = _row(vl.client, [_uri("red")])

    def loss_fn(data, logprobs):
        assert data[0].images == [_uri("red")]
        return -logprobs[0][-1], {"rows": 1.0}

    result = vl.client.forward_backward_custom([Datum(tokens, images=[_uri("red")])], loss_fn)
    assert result["metrics"]["rows"] == 1.0
    assert vl.backend.pending_backward == 1


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda t: t[: t.index(PAD)] + t[t.index(PAD) + 1 :], id="run-one-token-short"),
        pytest.param(lambda t: [*t[:3], PAD, *t[3:]], id="run-one-token-long"),
        pytest.param(lambda t: [x for x in t if x != PAD], id="no-placeholders"),
        pytest.param(lambda t: [*t[:-2], PAD], id="trailing-pad"),
    ],
)
def test_mismatched_placeholder_runs_are_rejected_before_gpu_work(vl, mutate):
    tokens = _row(vl.client, [_uri("red")])
    with pytest.raises((ValueError, ApiError), match="image placeholder tokens"):
        vl.client.forward([Datum(mutate(tokens), images=[_uri("red")])])
    assert not vl.session.failed
    assert vl.client.pending_request is None
    vl.client.forward([Datum(tokens, images=[_uri("red")])])


def test_sample_conditions_on_images(vl):
    prompt = _row(vl.client, [_uri("red")])
    out = vl.client.sample(prompt, max_tokens=3, temperature=0, images=[_uri("red")])
    sequence = out["sequences"][0]
    assert 1 <= len(sequence["tokens"]) <= 3
    assert len(sequence["logprobs"]) == len(sequence["tokens"])


def test_text_only_sessions_reject_images(tmp_path, monkeypatch):
    tokenizer = TinyTokenizer()
    backend = TorchBackend(_tiny_model(), tokenizer, checkpoint_dir=tmp_path, max_tokens=256)
    session = TrainingSession(backend)
    monkeypatch.setattr(
        "flash.custom.client.ApiClient", lambda *args, **kwargs: InProcessHTTP(session)
    )
    client = TrainingClient("http://localhost", api_key="test")
    assert backend.info()["images"] is False
    with pytest.raises((ValueError, ApiError), match="does not accept images"):
        client.forward([Datum([6, 7, 8], images=[_uri("red")])])
    with pytest.raises((ValueError, ApiError), match="does not accept images"):
        client.encode("a" + PLACEHOLDER, images=[_uri("red")])
    assert not session.failed


@pytest.mark.parametrize(
    "images",
    [
        [],
        ["https://example.com/cat.png"],
        [123],
        ["data:image/png;base64,AAAA"] * (MAX_IMAGES_PER_DATUM + 1),
    ],
)
def test_image_lists_are_shape_checked(images):
    with pytest.raises((ValueError, ApiError), match="images must be a list"):
        parse_batch(
            [{"tokens": [1, 2], "images": images}], vocab_size=64, max_tokens=8, max_batch=1
        )


def test_datum_serializes_images():
    row = Datum([1, 2], [0.5], images=["data:image/png;base64,AAAA"])
    assert row.to_dict() == {
        "tokens": [1, 2],
        "weights": [0.5],
        "images": ["data:image/png;base64,AAAA"],
    }
    assert "images" not in Datum([1, 2]).to_dict()
