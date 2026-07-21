from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.engine.worker.grpo_multimodal import (
    AtomicMultimodalProcessor,
    SingleTurnMultimodalGRPOMixin,
    validate_image_token_grid_invariant,
)
from flash.engine.worker.rl import select_grpo_trainer_class

torch = pytest.importorskip("torch")

_IMAGE_PAD_ID = 99


class _DynamicImageProcessor:
    merge_size = 2


class _DynamicProcessor:
    image_token_id = _IMAGE_PAD_ID

    def __init__(self):
        self.image_processor = _DynamicImageProcessor()
        self.tokenizer = SimpleNamespace()
        self.separate_forward_calls = 0

    @staticmethod
    def _images(conversations):
        return [
            next(
                part["image"]
                for message in conversation
                for part in message["content"]
                if part["type"] == "image"
            )
            for conversation in conversations
        ]

    @staticmethod
    def _grid(image):
        return [1, image.height // 28, image.width // 28]

    def apply_chat_template(self, conversation, **kwargs):
        if not kwargs.get("tokenize"):
            return "rendered"
        grids = [self._grid(image) for image in self._images(conversation)]
        feature_counts = [grid[0] * grid[1] * grid[2] // 4 for grid in grids]
        input_ids = [[11, *([_IMAGE_PAD_ID] * count), 13] for count in feature_counts]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
            "pixel_values": [[index] for index in range(sum(grid[1] * grid[2] for grid in grids))],
            "image_grid_thw": grids,
            "mm_token_type_ids": [
                [int(token_id == _IMAGE_PAD_ID) for token_id in ids] for ids in input_ids
            ],
        }

    def __call__(self, *, images, text, padding, return_tensors):
        self.separate_forward_calls += 1
        grids = [self._grid(image_list[0]) for image_list in images]
        grids[0] = [1, grids[0][1] - 4, grids[0][2]]
        return {
            "input_ids": [[1], [1]],
            "attention_mask": [[1], [1]],
            "pixel_values": [[0]],
            "image_grid_thw": grids,
            "mm_token_type_ids": [[0], [0]],
        }


def _large_heterogeneous_prompts():
    image_module = pytest.importorskip("PIL.Image")
    images = [
        image_module.new("RGB", (1120, 840), "red"),
        image_module.new("RGB", (1344, 896), "blue"),
    ]
    prompts = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "locate the target"},
                ],
            }
        ]
        for image in images
    ]
    return prompts, images


def _padded_prompt_completion(input_ids):
    max_prompt = max(len(ids) for ids in input_ids)
    prompt_completion_ids = []
    attention_mask = []
    for ids in input_ids:
        padding = max_prompt - len(ids)
        prompt_completion_ids.append([0] * padding + ids + [21, 22])
        attention_mask.append([0] * padding + [1] * len(ids) + [1, 1])
    return prompt_completion_ids, attention_mask


def test_atomic_processor_preserves_large_heterogeneous_rollout_fields():
    prompts, images = _large_heterogeneous_prompts()
    delegate = _DynamicProcessor()
    processor = AtomicMultimodalProcessor(delegate)

    rollout = processor.apply_chat_template(
        conversation=prompts,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    forward = processor(
        images=[[image] for image in images],
        text=["first render", "second render"],
        padding=True,
        return_tensors="pt",
    )

    assert delegate.separate_forward_calls == 0
    assert forward["pixel_values"].tolist() == rollout["pixel_values"]
    assert forward["image_grid_thw"].tolist() == rollout["image_grid_thw"]
    prompt_completion_ids, attention_mask = _padded_prompt_completion(rollout["input_ids"])

    desynced = _DynamicProcessor()(
        images=[[image] for image in images],
        text=["first render", "second render"],
        padding=True,
        return_tensors="pt",
    )
    with pytest.raises(
        ValueError, match=r"sample 0, image 0: active_image_tokens=300, grid_features=260"
    ):
        validate_image_token_grid_invariant(
            input_ids=prompt_completion_ids,
            attention_mask=attention_mask,
            image_grid_thw=desynced["image_grid_thw"],
            num_images=[1, 1],
            image_pad_token_id=_IMAGE_PAD_ID,
            merge_size=2,
        )

    validate_image_token_grid_invariant(
        input_ids=prompt_completion_ids,
        attention_mask=attention_mask,
        image_grid_thw=forward["image_grid_thw"],
        num_images=[1, 1],
        image_pad_token_id=_IMAGE_PAD_ID,
        merge_size=2,
    )


def test_image_token_grid_mismatch_is_rejected_before_forward():
    class _ForwardBase:
        def __init__(self, processing_class):
            self.processing_class = processing_class
            self._tokenizer = processing_class.tokenizer
            self._video_pad_token_id = None
            self.forward_calls = 0
            self.forward_mm_token_type_ids = None

        def _get_per_token_logps_and_entropies(self, *args, **kwargs):
            self.forward_calls += 1
            self.forward_mm_token_type_ids = kwargs["mm_token_type_ids"]
            return "forwarded"

    class _Trainer(SingleTurnMultimodalGRPOMixin, _ForwardBase):
        pass

    trainer = _Trainer(_DynamicProcessor())
    input_ids = torch.tensor(
        [
            [0, 11, _IMAGE_PAD_ID, _IMAGE_PAD_ID, 13, 21],
            [11, _IMAGE_PAD_ID, _IMAGE_PAD_ID, _IMAGE_PAD_ID, _IMAGE_PAD_ID, 21],
        ]
    )
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    valid_grids = torch.tensor([[1, 2, 4], [1, 4, 4]])

    assert (
        trainer._get_per_token_logps_and_entropies(
            object(),
            input_ids,
            attention_mask,
            logits_to_keep=1,
            image_grid_thw=valid_grids,
            num_images=[1, 1],
        )
        == "forwarded"
    )
    assert trainer.forward_calls == 1
    assert trainer.forward_mm_token_type_ids.tolist() == [
        [0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
    ]

    mismatched_grids = torch.tensor([[1, 2, 4], [1, 3, 4]])
    with pytest.raises(
        ValueError, match=r"sample 1, image 0: active_image_tokens=4, grid_features=3"
    ):
        trainer._get_per_token_logps_and_entropies(
            object(),
            input_ids,
            attention_mask,
            logits_to_keep=1,
            image_grid_thw=mismatched_grids,
            num_images=[1, 1],
        )
    assert trainer.forward_calls == 1


class _MultiImageProcessor:
    image_token_id = _IMAGE_PAD_ID

    def __init__(self, *, swapped_runs: bool = False, merge_size: object = 2):
        self.image_processor = SimpleNamespace(merge_size=merge_size)
        self.tokenizer = SimpleNamespace()
        self.separate_forward_calls = 0
        self.swapped_runs = swapped_runs

    def apply_chat_template(self, conversation, **kwargs):
        if not kwargs.get("tokenize"):
            return "rendered"
        assert len(conversation) == 2
        run_lengths = [4, 2] if self.swapped_runs else [2, 4]
        first_ids = [11, *([_IMAGE_PAD_ID] * run_lengths[0]), 12]
        first_ids.extend([_IMAGE_PAD_ID] * run_lengths[1])
        first_ids.append(13)
        return {
            "input_ids": [first_ids, [21, 22]],
            "attention_mask": [[1] * len(first_ids), [1, 1]],
            "pixel_values": [[float(index)] for index in range(24)],
            "image_grid_thw": [[1, 2, 4], [1, 4, 4]],
        }

    def __call__(self, *, images, text, padding, return_tensors):
        self.separate_forward_calls += 1
        raise AssertionError("trl's independent image-processor pass must not run")


class _TRL160LogprobCallSequence:
    def __init__(self, processing_class):
        self.processing_class = processing_class
        self._tokenizer = processing_class.tokenizer
        self._video_pad_token_id = None
        self.forward_calls = []

    def exercise_logprob_paths(self, prompts, images):
        rollout = self.processing_class.apply_chat_template(
            conversation=prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        forward = self.processing_class(
            images=images,
            text=["multi-image", "image-free"],
            padding=True,
            return_tensors="pt",
        )
        prompt_completion_ids, attention_mask = _padded_prompt_completion(rollout["input_ids"])
        input_ids = torch.tensor(prompt_completion_ids)
        attention_mask = torch.tensor(attention_mask)
        forward_kwargs = {
            "pixel_values": forward["pixel_values"],
            "image_grid_thw": forward["image_grid_thw"],
            "num_images": [2, 0],
        }

        for path in ("old", "reference", "current"):
            self._get_per_token_logps_and_entropies(
                path,
                input_ids,
                attention_mask,
                logits_to_keep=2,
                **forward_kwargs,
            )
        return forward

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        self.forward_calls.append((model, kwargs))
        return model


def _single_turn_multimodal_batch():
    images = [[object(), object()], []]
    prompts = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "no image"}]}],
    ]
    return prompts, images


def test_guarded_trainer_wiring_reuses_and_validates_atomic_multi_image_fields(monkeypatch):
    from flash.engine.worker import grpo_multimodal

    trainer_cls = select_grpo_trainer_class(
        _TRL160LogprobCallSequence,
        multimodal=True,
        is_multi_turn=False,
        tools=None,
    )
    assert issubclass(trainer_cls, SingleTurnMultimodalGRPOMixin)

    validated = []
    original_validate = grpo_multimodal.validate_image_token_grid_invariant

    def record_validation(**kwargs):
        validated.append(kwargs)
        original_validate(**kwargs)

    monkeypatch.setattr(
        grpo_multimodal,
        "validate_image_token_grid_invariant",
        record_validation,
    )
    prompts, images = _single_turn_multimodal_batch()
    processor = _MultiImageProcessor()
    trainer = trainer_cls(processor)
    forward = trainer.exercise_logprob_paths(prompts, images)

    assert processor.separate_forward_calls == 0
    assert [path for path, _ in trainer.forward_calls] == ["old", "reference", "current"]
    assert len(validated) == 3
    assert all(call["image_grid_thw"] is forward["image_grid_thw"] for call in validated)
    assert all(call["num_images"] == [2, 0] for call in validated)
    assert all(
        kwargs["image_grid_thw"] is forward["image_grid_thw"]
        and kwargs["pixel_values"] is forward["pixel_values"]
        and kwargs["num_images"] == [2, 0]
        for _, kwargs in trainer.forward_calls
    )

    swapped_processor = _MultiImageProcessor(swapped_runs=True)
    swapped_trainer = trainer_cls(swapped_processor)
    with pytest.raises(
        ValueError,
        match=r"sample 0, image 0: active_image_tokens=4, grid_features=2",
    ):
        swapped_trainer.exercise_logprob_paths(prompts, images)
    assert swapped_processor.separate_forward_calls == 0
    assert swapped_trainer.forward_calls == []


def test_guarded_trainer_fails_closed_without_real_merge_size():
    prompts, images = _single_turn_multimodal_batch()
    trainer_cls = select_grpo_trainer_class(
        _TRL160LogprobCallSequence,
        multimodal=True,
        is_multi_turn=False,
        tools=None,
    )
    processor = _MultiImageProcessor(merge_size=None)
    trainer = trainer_cls(processor)

    with pytest.raises(
        RuntimeError,
        match=r"requires image_processor\.merge_size to be a positive integer",
    ):
        trainer.exercise_logprob_paths(prompts, images)
    assert processor.separate_forward_calls == 0
    assert trainer.forward_calls == []
