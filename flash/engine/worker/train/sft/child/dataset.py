from __future__ import annotations

import io

import numpy as np
import pandas as pd


class FlashTokenizedSFTDataset:
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        # verl hands data.train_files straight through from hydra, so this arrives as an omegaconf
        # ListConfig, not a list. an isinstance (list, tuple) check misses it and wraps the sequence
        # into [ListConfig([...])], which pandas rejects with "cannot construct a FileSource from
        # [...]" before the first step. branch on str instead: a bare path is the only scalar shape.
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(str(path), dtype_backend="pyarrow") for path in parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        if max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples]
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = int(config.get("max_length", 1024))
        self.truncation = config.get("truncation", "error")
        if config.get("ignore_input_ids_mismatch", False):
            raise ValueError(
                "flash tokenized sft requires input-id mismatch checks to stay enabled"
            )
        if self.truncation not in {"error", "right"}:
            raise ValueError("flash tokenized sft supports error or right truncation only")

    def __len__(self):
        return len(self.dataframe)

    @staticmethod
    def _list(value):
        if hasattr(value, "to_pylist"):
            value = value.to_pylist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        return [int(item) for item in value]

    @staticmethod
    def _multimodal_inputs(payload):
        if hasattr(payload, "as_py"):
            payload = payload.as_py()
        if payload is None or payload is pd.NA:
            return {}
        payload = bytes(payload)
        if not payload:
            return {}
        import torch

        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            return {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}

    def __getitem__(self, index):
        import torch

        row = self.dataframe.iloc[index]
        input_ids = self._list(row["input_ids"])
        loss_mask = self._list(row["loss_mask"])
        if len(input_ids) != len(loss_mask):
            raise ValueError("input_ids and loss_mask must have identical lengths")
        if len(input_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError("pretokenized row exceeds data.max_length")
            input_ids = input_ids[: self.max_length]
            loss_mask = loss_mask[: self.max_length]
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)
        multi_modal_inputs = self._multimodal_inputs(row["multimodal_inputs"])
        if (
            self.processor is not None
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids_tensor,
                image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
                second_per_grid_ts=multi_modal_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.arange(len(input_ids), dtype=torch.long)
        return {
            "input_ids": input_ids_tensor,
            "position_ids": position_ids,
            "loss_mask": loss_mask_tensor,
            "multi_modal_inputs": multi_modal_inputs,
        }
