from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.engine.worker.grpo_multimodal import (
    AtomicMultimodalProcessor,
    SingleTurnMultimodalGRPOMixin,
    validate_image_token_grid_invariant,
)

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
        ValueError, match=r"sample 0: active_image_tokens=300, grid_features=260"
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
    with pytest.raises(ValueError, match=r"sample 1: active_image_tokens=4, grid_features=3"):
        trainer._get_per_token_logps_and_entropies(
            object(),
            input_ids,
            attention_mask,
            logits_to_keep=1,
            image_grid_thw=mismatched_grids,
            num_images=[1, 1],
        )
    assert trainer.forward_calls == 1
