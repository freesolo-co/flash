"""Automatic Spot recovery policy shared by all training backends."""

from dataclasses import replace

SPOT_CHECKPOINT_SECONDS = 600


def spot_checkpoint_train(train):
    """Keep requested model exports separate from the runtime recovery clock.

    Ten was the legacy automatic interval. Accept it on old persisted specs but
    normalize it away; newly authored Spot jobs should omit save_every entirely.
    """
    if train.save_every not in (None, 10):
        raise ValueError("Spot checkpointing is time-based and automatic; remove train.save_every")
    return replace(train, save_every=None)


def is_spot_spec(spec) -> bool:
    return bool(getattr(getattr(spec, "gpu", None), "spot", False))
