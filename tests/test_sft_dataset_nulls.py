"""Text-only rows keep their empty multimodal state through Arrow-backed parquet reads."""

import pyarrow as pa
import pytest

from flash.engine.worker.train.sft.child.dataset import FlashTokenizedSFTDataset
from flash.engine.worker.train.sft.setup.config import _write_sft_parquet


def test_text_only_null_survives_the_real_parquet_roundtrip(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "train.parquet"
    _write_sft_parquet(
        [{"input_ids": [1, 2], "loss_mask": [0, 1], "images": [], "multimodal_inputs": None}],
        str(path),
    )
    dataset = FlashTokenizedSFTDataset(str(path), None, {"max_length": 8})
    assert dataset[0]["multi_modal_inputs"] == {}
    assert dataset[0]["input_ids"].tolist() == [1, 2]


def test_arrow_null_scalar_has_no_multimodal_payload():
    assert FlashTokenizedSFTDataset._multimodal_inputs(pa.scalar(None, type=pa.binary())) == {}
