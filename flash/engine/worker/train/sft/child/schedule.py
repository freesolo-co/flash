"""linear SFT warmup with a trainable first checkpoint."""

from functools import partial


def linear_warmup_factor(step: int, *, warmup: int, total: int) -> float:
    # LambdaLR initializes at index zero, before the first optimizer update. a zero factor
    # leaves standard zero-B LoRA unchanged, so a requested step-1 checkpoint fails export.
    # floor only that initial warmup factor; later indices and resumed state keep their meaning.
    if step < warmup:
        return max(1, step) / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def linear_warmup_scheduler(optimizer, *, num_warmup_steps: int, num_training_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    return LambdaLR(
        optimizer,
        partial(linear_warmup_factor, warmup=num_warmup_steps, total=num_training_steps),
    )
