"""Decode queued multimodal positions into the engine's sequence-ragged layout."""


def normalize_position_ids(data):
    import torch

    positions = data.get("position_ids")
    if positions is None or not positions.is_nested or positions.dim() != 3:
        return
    tokens = data["input_ids"]
    if not tokens.is_nested or tokens.layout != torch.jagged:
        raise ValueError("multimodal positions require sequence-ragged input ids")
    lengths = tokens.offsets().diff().tolist()
    if positions.layout == torch.strided:
        rows = list(positions.unbind())
    else:
        values = positions.values()
        offsets = positions.offsets()
        sizes = offsets.diff().tolist()
        # TransferQueue packs equal-length (rope axes, tokens) rows along axis 0.
        # Upstream may have already relabeled that axis without rebuilding offsets.
        # Read the encoded storage geometry, then construct the actual sequence axis.
        if sizes == lengths and int(offsets[-1]) == values.shape[1]:
            rows = list(values.split(sizes, dim=1))
        elif int(offsets[-1]) == values.shape[0] and len(set(lengths)) == 1:
            rows = list(values.split(sizes, dim=0))
        else:
            raise ValueError("queued multimodal position offsets do not match token lengths")
    if len(rows) != len(lengths) or any(
        row.shape[-1] != n for row, n in zip(rows, lengths, strict=True)
    ):
        raise ValueError("queued multimodal positions do not match token lengths")
    data["position_ids"] = torch.nested.nested_tensor_from_jagged(
        torch.cat(rows, dim=1), offsets=tokens.offsets(), jagged_dim=2
    )
