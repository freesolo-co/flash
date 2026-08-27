"""Batch-layout helpers for the flash OPD distillation loss.

verl hands the loss a `TensorDict` whose fields are laid out per field rather than per batch:
`list_of_dict_to_tensordict` stacks a field when every sample's tensor shares one shape and
nests it only when the shapes differ. So `prompts`, `teacher_logprobs`, and `teacher_ids` can
each arrive nested or padded independently, and the loss has to read all of them either way.
"""

from __future__ import annotations


def signal_sequences(group_ids, response_mask):
    """Return the per-sequence mask for rows carrying at least one aligned student token."""
    return ((group_ids >= 0) & response_mask.bool()).any(dim=-1)


def full_sequence_signal_sequences(group_ids):
    """Detect aligned metadata in native full-sequence teacher tensors.

    Padded only: `flatten` raises on a nested tensor's ragged dim, so callers holding a nested
    tensor must reduce per row instead.
    """
    return group_ids.ge(0).flatten(start_dim=1).any(dim=-1)


def packed_full_sequence(tensor, data):
    """Put a full-sequence teacher tensor into the layout `no_padding_2_padding` documents.

    That helper wants a nested tensor or `(total_nnz, *)` and reads `.values()` only when
    nested, so a padded teacher measures as `bsz` and trips its `total_nnz` assertion. Lengths
    come from whichever field carries them, because `no_padding_2_padding` consults
    `attention_mask` only for strided prompts: a nested-prompt batch need not hold that key at
    all, and packing off the mask alone would turn its assertion into a `KeyError`.
    """
    import torch

    if tensor.is_nested:
        return tensor
    prompts = data["prompts"]
    if not prompts.is_nested:
        return tensor[data["attention_mask"].bool()]
    lengths = prompts.offsets().diff() + data["responses"].offsets().diff()
    return tensor[torch.arange(tensor.shape[1], device=tensor.device) < lengths.unsqueeze(1)]
