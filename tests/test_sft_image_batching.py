"""Image SFT rows pack into one micro-batch without crossing example boundaries.

The child dataset emits one example at a time; verl's no-padding layout then concatenates a
micro-batch into one ``(1, total)`` row, concatenates every row's ``pixel_values`` and
``image_grid_thw`` in row order, and flattens the 4-row mrope positions to ``(4, 1, total)``. These
tests build real dataset items for rows with different image counts and sizes and check the
invariants that layout relies on. The upstream collator itself is exercised in
``test_upstream_verl_contract.py``, which runs only against the pinned verl.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
image_module = pytest.importorskip("PIL.Image")
pd = pytest.importorskip("pandas")

VISION_START, IMAGE_PAD, VISION_END = 900, 901, 902
MERGE = 2


_TOKEN_IDS = {
    "<|vision_start|>": VISION_START,
    "<|image_pad|>": IMAGE_PAD,
    "<|video_pad|>": 903,
    "<|vision_end|>": VISION_END,
}


class _Tokenizer:
    def convert_tokens_to_ids(self, token):
        return _TOKEN_IDS[token]


class Qwen2VLImageProcessorStub:
    """Grid from the image size; each patch carries its image's red channel so order is visible."""

    merge_size = MERGE

    def __call__(self, images, return_tensors):
        pixels, grids = [], []
        for image in images:
            width, height = image.size
            grid = [1, height // 14, width // 14]
            patches = grid[0] * grid[1] * grid[2]
            pixels.append(np.full((patches, 6), float(image.getpixel((0, 0))[0]), np.float32))
            grids.append(grid)
        return {
            "pixel_values": torch.from_numpy(np.concatenate(pixels)),
            "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
        }


class _Processor:
    def __init__(self):
        self.image_processor = Qwen2VLImageProcessorStub()
        self.tokenizer = _Tokenizer()


# (image sizes, completion length): one image, two images of different sizes, and a larger image.
ROWS = (
    (((28, 28),), 3),
    (((56, 28), (28, 56)), 2),
    (((56, 56),), 4),
)


def _row(tmp_path, index, sizes, completion):
    from flash.engine.profiling.sft_image_rows import _serialize_multimodal_inputs

    ids, uris, grids = [10, 11], [], []
    for image_index, (width, height) in enumerate(sizes):
        path = tmp_path / f"row-{index}-image-{image_index}.png"
        image_module.new("RGB", (width, height), (10 * (index + 1) + image_index, 0, 0)).save(path)
        uris.append(path.as_uri())
        grid = [1, height // 14, width // 14]
        grids.append(grid)
        ids += [VISION_START, *[IMAGE_PAD] * (grid[1] * grid[2] // MERGE**2), VISION_END]
    ids += [12]
    mask = [0] * len(ids) + [1] * completion
    ids += [100 + index] * completion
    stored = {
        "image_grid_thw": np.asarray(grids, dtype=np.int64),
        "mm_token_type_ids": np.asarray([[int(t == IMAGE_PAD) for t in ids]], dtype=np.int64),
    }
    return {
        "input_ids": ids,
        "loss_mask": mask,
        "images": uris,
        "multimodal_inputs": _serialize_multimodal_inputs(stored),
    }


def _rope_index_stub(processor, input_ids, **_kwargs):
    """Stands in for verl's mrope builder where the pinned verl is absent; row 0 is unaffected."""
    return torch.arange(len(input_ids)).repeat(3, 1) + 7


@pytest.fixture
def items(tmp_path, monkeypatch):
    import sys
    import types

    from flash.engine.worker.train.sft.child.dataset import FlashTokenizedSFTDataset

    try:
        from verl.models.transformers.qwen2_vl import get_rope_index  # noqa: F401
    except ImportError:
        for name in ("verl", "verl.models", "verl.models.transformers"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        stub = types.ModuleType("verl.models.transformers.qwen2_vl")
        stub.get_rope_index = _rope_index_stub
        monkeypatch.setitem(sys.modules, stub.__name__, stub)

    dataset = object.__new__(FlashTokenizedSFTDataset)
    dataset.dataframe = pd.DataFrame(
        [_row(tmp_path, index, sizes, completion) for index, (sizes, completion) in enumerate(ROWS)]
    )
    dataset.processor = _Processor()
    dataset.max_length = 4096
    dataset.truncation = "error"
    return [dataset[index] for index in range(len(ROWS))]


def _packed(items):
    """The no-padding micro-batch layout verl's fsdp engine builds from these items."""
    lengths = [len(item["input_ids"]) for item in items]
    offsets = torch.tensor([0, *np.cumsum(lengths).tolist()])
    positions = torch.cat([item["position_ids"] for item in items], dim=1).unsqueeze(1)
    pixel_values = torch.cat([item["multi_modal_inputs"]["pixel_values"] for item in items])
    grids = torch.cat([item["multi_modal_inputs"]["image_grid_thw"] for item in items])
    return offsets, positions, pixel_values, grids


def test_each_row_rebuilds_its_own_pixels_and_mrope_positions(items):
    for index, item in enumerate(items):
        sizes = ROWS[index][0]
        inputs = item["multi_modal_inputs"]
        assert inputs["image_grid_thw"].shape == (len(sizes), 3)
        assert item["position_ids"].shape == (4, len(item["input_ids"]))
        # row 0 is the text position, which restarts at 0 for every example.
        assert item["position_ids"][0].tolist() == list(range(len(item["input_ids"])))


def test_packed_pixels_follow_the_image_tokens_in_row_order(items):
    _offsets, _positions, pixel_values, grids = _packed(items)
    # each patch carries its source image's red channel: row index * 10 + image index.
    expected = []
    for index, (sizes, _completion) in enumerate(ROWS):
        for image_index, (width, height) in enumerate(sizes):
            expected += [10 * (index + 1) + image_index] * ((height // 14) * (width // 14))
    assert pixel_values[:, 0].tolist() == expected
    # the merged-token count of every grid equals the image pads the concatenated ids carry, in
    # order, which is what masked_scatter needs to put each image's embeddings in its own row.
    ids = torch.cat([item["input_ids"] for item in items])
    pads = int((ids == IMAGE_PAD).sum())
    assert pads == int((grids.prod(dim=1) // MERGE**2).sum())


def test_text_positions_mark_every_example_boundary_for_the_gdn_reset(items):
    pytest.importorskip("transformers")
    from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

    from flash.engine.worker.train.core.child.model import _gdn_seq_idx

    offsets, positions, _pixels, _grids = _packed(items)
    # the hook reads row 0 of the (4, 1, total) mrope positions, exactly as configure_gdn does.
    text = positions[0]
    (cu_q, _cu_k), _max = prepare_fa_kwargs_from_position_ids(text)
    assert cu_q.tolist() == offsets.tolist()
    seq_idx = _gdn_seq_idx(text, cu_q)
    lengths = offsets.diff().tolist()
    assert seq_idx[0].tolist() == [i for i, n in enumerate(lengths) for _ in range(n)]


def test_packed_loss_equals_the_token_weighted_single_example_losses(items):
    """verl's no-padding sft loss: masked sum of the rolled mask over the packed tokens / tokens.

    Packing must supervise exactly the positions each example supervises alone, so the packed
    batch loss is the single-example masked sums over the same token denominator -- the per-token
    mean over the global batch the text path already trains with.
    """
    generator = torch.Generator().manual_seed(0)
    log_probs = [-torch.rand(len(item["input_ids"]), generator=generator) for item in items]
    masks = [item["loss_mask"] for item in items]
    # every row opens on prompt tokens, so the roll never supervises a neighbour's first token.
    assert all(int(mask[0]) == 0 for mask in masks)

    packed_mask = torch.roll(torch.cat(masks), shifts=-1, dims=0)
    tokens = int(torch.cat(masks).sum())
    packed = -(torch.cat(log_probs) * packed_mask).sum() / tokens

    singles = [
        -(log_prob * torch.roll(mask, shifts=-1, dims=0)).sum()
        for log_prob, mask in zip(log_probs, masks, strict=True)
    ]
    torch.testing.assert_close(packed, torch.stack(singles).sum() / tokens)
    assert torch.equal(
        packed_mask, torch.cat([torch.roll(mask, shifts=-1, dims=0) for mask in masks])
    )
