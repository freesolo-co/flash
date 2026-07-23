"""single-turn multimodal safeguards for trl grpo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from flash.multimodal import resolve_image_pad_token_id


def _batch_size(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 1:
        return int(shape[0])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if value and isinstance(value[0], Sequence):
            return len(value)
        return 1
    raise ValueError("multimodal processor output is missing batched input_ids")


class AtomicMultimodalProcessor:
    """reuse multimodal fields from the processor call that produced rollout ids."""

    def __init__(self, processor: Any):
        self._processor = processor
        self._pending_rollout: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        tokenized = self._processor.apply_chat_template(*args, **kwargs)
        if kwargs.get("tokenize") and kwargs.get("return_dict"):
            if not isinstance(tokenized, Mapping):
                raise TypeError("multimodal chat template must return a mapping")
            self._pending_rollout = dict(tokenized)
        return tokenized

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        is_forward_rebuild = (
            kwargs.get("images") is not None
            and kwargs.get("padding") is True
            and kwargs.get("return_tensors") == "pt"
        )
        if not is_forward_rebuild:
            return self._processor(*args, **kwargs)
        if self._pending_rollout is None:
            raise RuntimeError(
                "single-turn multimodal grpo forward preparation has no atomic rollout processor output"
            )

        tokenized = self._pending_rollout
        self._pending_rollout = None
        text = kwargs.get("text")
        expected_batch = len(text) if isinstance(text, list) else 1
        actual_batch = _batch_size(tokenized.get("input_ids"))
        if actual_batch != expected_batch:
            raise ValueError(
                "single-turn multimodal grpo processor batch changed between rollout and forward: "
                f"rollout={actual_batch}, forward={expected_batch}"
            )

        from transformers.feature_extraction_utils import BatchFeature

        forward_fields = {
            key: value
            for key, value in tokenized.items()
            if key not in {"input_ids", "attention_mask", "mm_token_type_ids"}
        }
        return BatchFeature(forward_fields, tensor_type="pt")


def _as_int_list(values: Any) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _active_image_token_runs(
    input_ids: Any, attention_mask: Any, image_pad_token_id: int
) -> list[list[int]]:
    if hasattr(input_ids, "dim"):
        active_image_tokens = (input_ids == image_pad_token_id) & attention_mask.bool()
        batch_runs = []
        for row in active_image_tokens:
            if row.numel() == 0:
                batch_runs.append([])
                continue
            previous = row.roll(1)
            previous[0] = False
            following = row.roll(-1)
            following[-1] = False
            starts = (row & ~previous).nonzero(as_tuple=False).flatten()
            ends = (row & ~following).nonzero(as_tuple=False).flatten()
            batch_runs.append(_as_int_list(ends - starts + 1))
        return batch_runs

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(attention_mask, "tolist"):
        attention_mask = attention_mask.tolist()

    batch_runs = []
    for ids, mask in zip(input_ids, attention_mask, strict=True):
        runs = []
        run_length = 0
        for token_id, active in zip(ids, mask, strict=True):
            if bool(active) and int(token_id) == image_pad_token_id:
                run_length += 1
            elif run_length:
                runs.append(run_length)
                run_length = 0
        if run_length:
            runs.append(run_length)
        batch_runs.append(runs)
    return batch_runs


def _grid_feature_counts(image_grid_thw: Any, merge_length: int) -> list[int]:
    if hasattr(image_grid_thw, "dim"):
        products = image_grid_thw.prod(dim=-1)
        remainders = products.remainder(merge_length)
        if bool(remainders.ne(0).any().item()):
            raise ValueError("image_grid_thw contains a grid not divisible by merge_size squared")
        return _as_int_list(products // merge_length)

    counts = []
    for grid in image_grid_thw:
        product = 1
        for dimension in grid:
            product *= int(dimension)
        if product % merge_length:
            raise ValueError("image_grid_thw contains a grid not divisible by merge_size squared")
        counts.append(product // merge_length)
    return counts


def validate_image_token_grid_invariant(
    *,
    input_ids: Any,
    attention_mask: Any,
    image_grid_thw: Any,
    num_images: Any,
    image_pad_token_id: int,
    merge_size: int,
) -> None:
    """reject samples whose active image tokens disagree with their processor grids."""
    if num_images is None:
        raise ValueError(
            "single-turn multimodal grpo requires num_images for image-grid validation"
        )
    images_per_sample = _as_int_list(num_images)
    active_runs = _active_image_token_runs(input_ids, attention_mask, image_pad_token_id)
    if len(images_per_sample) != len(active_runs):
        raise ValueError(
            "single-turn multimodal grpo image counts do not match the forward batch: "
            f"images={len(images_per_sample)}, samples={len(active_runs)}"
        )

    merge_length = int(merge_size) ** 2
    feature_counts = _grid_feature_counts(image_grid_thw, merge_length)
    if sum(images_per_sample) != len(feature_counts):
        raise ValueError(
            "single-turn multimodal grpo image_grid_thw rows do not match num_images: "
            f"grids={len(feature_counts)}, images={sum(images_per_sample)}"
        )

    grid_offset = 0
    for sample_index, (sample_runs, image_count) in enumerate(
        zip(active_runs, images_per_sample, strict=True)
    ):
        if len(sample_runs) != image_count:
            raise ValueError(
                "single-turn multimodal grpo image-token/grid invariant failed before forward for "
                f"sample {sample_index}: image_token_runs={len(sample_runs)}, images={image_count}"
            )
        sample_features = feature_counts[grid_offset : grid_offset + image_count]
        for image_index, (run_length, feature_count) in enumerate(
            zip(sample_runs, sample_features, strict=True)
        ):
            if run_length != feature_count:
                raise ValueError(
                    "single-turn multimodal grpo image-token/grid invariant failed before forward for "
                    f"sample {sample_index}, image {image_index}: "
                    f"active_image_tokens={run_length}, grid_features={feature_count}"
                )
        grid_offset += image_count


def _processor_merge_size(processing_class: Any) -> int:
    image_processor = getattr(processing_class, "image_processor", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if isinstance(merge_size, bool) or not isinstance(merge_size, Integral) or merge_size <= 0:
        raise RuntimeError(
            "single-turn multimodal grpo requires image_processor.merge_size "
            "to be a positive integer"
        )
    return int(merge_size)


class SingleTurnMultimodalGRPOMixin:
    """keep rollout multimodal tensors atomic and validate every model forward."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.processing_class = AtomicMultimodalProcessor(self.processing_class)
        self._flash_image_pad_token_id = getattr(self, "_image_pad_token_id", None)
        if self._flash_image_pad_token_id is None:
            self._flash_image_pad_token_id = resolve_image_pad_token_id(
                self.processing_class, self._tokenizer
            )

    def _get_per_token_logps_and_entropies(
        self,
        model: Any,
        input_ids: Any,
        attention_mask: Any,
        logits_to_keep: int,
        batch_size: int | None = None,
        compute_entropy: bool = False,
        pixel_values: Any = None,
        image_grid_thw: Any = None,
        num_images: Any = None,
        pixel_attention_mask: Any = None,
        image_sizes: Any = None,
        token_type_ids: Any = None,
        mm_token_type_ids: Any = None,
        image_position_ids: Any = None,
    ) -> Any:
        if image_grid_thw is not None:
            merge_size = _processor_merge_size(self.processing_class)
            validate_image_token_grid_invariant(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
                image_pad_token_id=self._flash_image_pad_token_id,
                merge_size=merge_size,
            )
            mm_token_type_ids = input_ids.new_zeros(input_ids.shape)
            mm_token_type_ids[input_ids == self._flash_image_pad_token_id] = 1
            video_pad_token_id = getattr(self, "_video_pad_token_id", None)
            if video_pad_token_id is not None:
                mm_token_type_ids[input_ids == video_pad_token_id] = 2

        return super()._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=batch_size,
            compute_entropy=compute_entropy,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            num_images=num_images,
            pixel_attention_mask=pixel_attention_mask,
            image_sizes=image_sizes,
            token_type_ids=token_type_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_position_ids=image_position_ids,
        )


def single_turn_multimodal_grpo_trainer(base_trainer: type) -> type:
    """build the flash trainer without importing the gpu-only trl dependency here."""
    return type(
        "SingleTurnMultimodalGRPOTrainer",
        (SingleTurnMultimodalGRPOMixin, base_trainer),
        {},
    )
