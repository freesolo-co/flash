"""Per-GPU idle windows for hosted serving engine containers."""

# Hosted models use their qualified GPU tier from the catalog. The slowest measured
# cold boot is 1821s (27b); retain
# its 2100s idle window so warm replicas survive normal gaps between requests.
SCALEDOWN_WINDOW_SECONDS_BY_GPU: dict[str, int] = {
    "L4": 2100,
    "L40S": 2100,
    "H100": 2100,
    "H200": 2100,
    "A10G": 2100,
    "A100-80GB": 2100,
}
# The tiers this app is allowed to run an engine on. `gpu` in the serving catalog is a plain string,
# so a typo ("b200", "B2OO") or an unvalidated new card used to fall through `dict.get` to a default
# window and deploy anyway, at that card's real hourly rate. Membership here is the one gate.
SUPPORTED_GPUS: frozenset[str] = frozenset(SCALEDOWN_WINDOW_SECONDS_BY_GPU)


def scaledown_window_for(gpu: str) -> int:
    """Idle seconds before ``gpu``'s engine containers scale down (see the table above).

    Raises ``ValueError`` for a tier this app has no window for, rather than silently applying a
    default to an unrecognized card.
    """
    try:
        return SCALEDOWN_WINDOW_SECONDS_BY_GPU[gpu]
    except KeyError:
        supported = ", ".join(sorted(SUPPORTED_GPUS))
        raise ValueError(
            f"Unsupported serving GPU tier {gpu!r}; supported tiers: {supported}. "
            "Add the tier to SCALEDOWN_WINDOW_SECONDS_BY_GPU with its measured cold-boot window."
        ) from None
