"""Shared adapter artifact filenames used by training and control-plane code."""

from __future__ import annotations

# The PEFT adapter weights file a deployable adapter must carry. Modern saves use safetensors,
# but older PEFT/default settings may emit adapter_model.bin.
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
