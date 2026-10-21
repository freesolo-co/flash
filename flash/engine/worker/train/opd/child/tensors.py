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
    nested, so a padded teacher measures as `bsz` and trips its `total_nnz` assertion.

    The mask decides which positions are real, never the row's shape. verl pads a teacher row on
    both sides (`_pad_teacher_outputs` left-pads by `prompt_width - prompt_length` and right-pads
    the response), so a length alone cannot say where the real tokens sit and selecting a prefix
    would silently train against the pad.
    """
    if tensor.is_nested:
        return tensor
    if "attention_mask" in data:
        return tensor[data["attention_mask"].bool()]
    # no mask: only a nested-prompt batch may omit that key, and only such a batch carries the
    # lengths needed to check the stack. verl stacks this field only when every sequence shares
    # one length, and pads a row only to the strided widths, so an unmasked stack is full.
    prompts = data["prompts"]
    if not prompts.is_nested:
        raise AssertionError(
            "flash OPD strided batch carries no attention mask, so real token positions "
            "cannot be located"
        )
    lengths = prompts.offsets().diff() + data["responses"].offsets().diff()
    if int(lengths.min()) != tensor.shape[1] or int(lengths.max()) != tensor.shape[1]:
        raise AssertionError(
            "flash OPD teacher tensor is padded but the batch carries no attention mask: "
            f"width {tensor.shape[1]} against lengths {int(lengths.min())}..{int(lengths.max())}"
        )
    return tensor.flatten(0, 1)
