"""Image rows store no pixel tensors; the training child rebuilds them from the saved files."""

import numpy as np
import pytest

from flash.engine.worker.train.sft.child.dataset import FlashTokenizedSFTDataset

GRID = [[1, 2, 2]]


class _ImageProcessor:
    def __init__(self, grid=GRID):
        self.grid = grid
        self.calls = []

    def __call__(self, images, return_tensors):
        self.calls.append([(image.mode, image.size) for image in images])
        return {
            "pixel_values": np.full((4 * len(images), 6), 0.5, dtype=np.float32),
            "image_grid_thw": np.asarray(self.grid * len(images), dtype=np.int64),
        }


def _dataset(processor):
    dataset = object.__new__(FlashTokenizedSFTDataset)
    dataset.processor = processor
    return dataset


@pytest.fixture
def image_uris(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    uris = []
    for index, color in enumerate(((200, 10, 10), (10, 200, 10))):
        path = tmp_path / f"row-0-image-{index}.png"
        image_module.new("RGB", (6, 4), color).save(path, format="PNG")
        uris.append(path.as_uri())
    return uris


def test_missing_pixels_are_rebuilt_from_the_saved_images_and_grid_checked(image_uris):
    processor = _ImageProcessor()
    stored = {
        "image_grid_thw": np.asarray(GRID * 2, dtype=np.int64),
        "mm_token_type_ids": np.asarray([[0, 1, 1, 0]], dtype=np.int64),
    }

    inputs = _dataset(SimpleProcessor(processor))._with_image_pixels(stored, image_uris)

    assert processor.calls == [[("RGB", (6, 4)), ("RGB", (6, 4))]]
    assert inputs["pixel_values"].shape == (8, 6)
    assert inputs["image_grid_thw"] is stored["image_grid_thw"]
    assert inputs["mm_token_type_ids"] is stored["mm_token_type_ids"]


def test_a_rebuilt_grid_that_disagrees_with_the_tokens_is_rejected(image_uris):
    stored = {"image_grid_thw": np.asarray([[1, 4, 4]] * 2, dtype=np.int64)}

    with pytest.raises(ValueError, match="rebuilt image_grid_thw does not match"):
        _dataset(SimpleProcessor(_ImageProcessor()))._with_image_pixels(stored, image_uris)


def test_rows_that_already_carry_pixels_or_no_images_are_left_alone(image_uris):
    processor = _ImageProcessor()
    dataset = _dataset(SimpleProcessor(processor))
    legacy = {"pixel_values": np.zeros((8, 6)), "image_grid_thw": np.asarray(GRID * 2)}

    assert dataset._with_image_pixels(legacy, image_uris) is legacy
    assert dataset._with_image_pixels({}, []) == {}
    assert dataset._with_image_pixels({}, None) == {}
    assert processor.calls == []


def test_an_image_row_without_a_processor_fails_loudly(image_uris):
    with pytest.raises(ValueError, match="needs a processor"):
        _dataset(None)._with_image_pixels({}, image_uris)


class SimpleProcessor:
    def __init__(self, image_processor):
        self.image_processor = image_processor
