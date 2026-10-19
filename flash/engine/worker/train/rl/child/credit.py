"""group-relative credit for the model-token spans of each turn."""

import torch


def rows(non_tensor_batch, batch_size):
    spans_column = non_tensor_batch.get("flash_turn_spans")
    rewards_column = non_tensor_batch.get("flash_turn_rewards")
    if spans_column is None or rewards_column is None:
        return None
    if len(spans_column) != batch_size or len(rewards_column) != batch_size:
        return None
    result = []
    for spans, turns in zip(spans_column, rewards_column, strict=True):
        if spans is None or turns is None or len(spans) != len(turns):
            result.append(None)
            continue
        result.append(
            (
                tuple((int(start), int(end)) for start, end in spans),
                tuple(float(value) for value in turns),
            )
        )
    return result


def per_turn_advantages(metadata_rows, index, episode_advantages):
    advantages = episode_advantages.clone()
    groups = {}
    for row_index, uid in enumerate(index):
        groups.setdefault(uid, []).append(row_index)
    for member_indexes in groups.values():
        if any(metadata_rows[row_index] is None for row_index in member_indexes):
            continue
        for row_index in member_indexes:
            advantages[row_index] = 0.0
        turn_total = max(len(metadata_rows[row_index][1]) for row_index in member_indexes)
        for turn_index in range(turn_total):
            scoring = [
                row_index
                for row_index in member_indexes
                if turn_index < len(metadata_rows[row_index][1])
                and metadata_rows[row_index][0][turn_index][1]
                > metadata_rows[row_index][0][turn_index][0]
            ]
            if not scoring:
                continue
            baseline = sum(metadata_rows[row_index][1][turn_index] for row_index in scoring) / len(
                scoring
            )
            for row_index in scoring:
                start, end = metadata_rows[row_index][0][turn_index]
                advantages[row_index, start:end] = (
                    metadata_rows[row_index][1][turn_index] - baseline
                )
    return advantages


def apply_per_turn_credit(data):
    episode = data.batch.get("advantages")
    if episode is None or episode.dim() != 2:
        return data
    batch_size, width = episode.shape
    metadata_rows = rows(data.non_tensor_batch, batch_size)
    if metadata_rows is None or all(row is None for row in metadata_rows):
        return data
    index = data.non_tensor_batch.get("uid")
    if index is None or len(index) != batch_size:
        return data
    for row in metadata_rows:
        if row is None:
            continue
        for start, end in row[0]:
            if not 0 <= start <= end <= width:
                raise ValueError(f"turn span [{start}, {end}) exceeds the response width {width}")
    advantages = per_turn_advantages(metadata_rows, index, episode)
    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("per-turn advantages must be finite")
    response_mask = data.batch.get("response_mask")
    if response_mask is not None:
        advantages = advantages * response_mask.to(dtype=advantages.dtype)
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages
    return data
