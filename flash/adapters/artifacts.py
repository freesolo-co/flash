"""Artifact filenames the worker writes and the control plane reads.

Every name here crosses a machine boundary: the worker uploads under it, the control plane fetches
by it, and the two never share a call stack to check. Spelling one on each side makes them agree by
coincidence, so they are defined once, here.
"""

# The PEFT adapter weights file a deployable adapter must carry. Modern saves use safetensors,
# but older PEFT/default settings may emit adapter_model.bin.
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")

# A PEFT save past its shard size splits the weights across `adapter_model-0000N-of-0000M.<ext>`
# beside an `adapter_model.<ext>.index.json` mapping every tensor key to the shard holding it.
ADAPTER_SHARD_PREFIX = "adapter_model-"
ADAPTER_WEIGHT_SUFFIXES = (".safetensors", ".bin")
ADAPTER_WEIGHT_INDEX_FILES = tuple(f"{name}.index.json" for name in ADAPTER_WEIGHT_FILES)


def is_adapter_weight_filename(filename: str) -> bool:
    """True for every adapter weight file shape the pipeline accepts, single-file or sharded.

    One predicate for the same reason the filenames above are defined once: serving validation and
    the exporter never share a call stack, and each spelling the shapes for itself makes them agree
    by coincidence. A disagreement fails silently, because validation accepts a shape the exporter
    skips and the export then ships weights peft loads as a no-op.
    """
    name = filename.rsplit("/", 1)[-1]
    if name in ADAPTER_WEIGHT_FILES:
        return True
    return name.startswith(ADAPTER_SHARD_PREFIX) and name.endswith(ADAPTER_WEIGHT_SUFFIXES)


# The largest attempt identity any artifact name may carry. An attempt number reaches a filename and
# an HF path, so it is bounded rather than merely nonnegative: an unbounded int would still format,
# and the resulting name is the one thing a later reader has to match exactly.
MAX_ATTEMPT_ID = (1 << 63) - 1


def attempt_scoped_artifact_name(kind: str, phase: str, attempt: int) -> str:
    """The name of one per-phase, per-attempt worker text artifact.

    The worker uploads these and the control plane reads them back, so the format is a contract
    between two processes that never share a call stack. It lives here, beside the adapter filenames,
    because a reader that spells the name for itself agrees with the writer only by coincidence --
    and the failure is silent: a missing artifact reads as "the worker uploaded nothing", which is
    exactly what it looks like when the worker crashed. That is the case these files exist to explain.

    Attempt-scoped because ``hf_prefix()`` is per-RUN, not per-attempt: without the suffix a retry
    would overwrite the attempt that actually reproduced the failure.
    """
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise ValueError("attempt must be a nonnegative integer")
    if attempt < 0 or attempt > MAX_ATTEMPT_ID:
        raise ValueError("attempt must be a bounded nonnegative integer")
    return f"{kind}_{phase}_attempt{attempt}.txt"
