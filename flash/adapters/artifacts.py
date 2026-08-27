"""Artifact filenames the worker writes and the control plane reads.

Every name here crosses a machine boundary: the worker uploads under it, the control plane fetches
by it, and the two never share a call stack to check. Spelling one on each side makes them agree by
coincidence, so they are defined once, here.
"""

from pathlib import Path

# the peft adapter weights file a deployable adapter must carry.
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors",)

# a peft save past its shard size splits the weights across `adapter_model-0000N-of-0000M.<ext>`
# beside an `adapter_model.<ext>.index.json` mapping every tensor key to the shard holding it.
ADAPTER_SHARD_PREFIX = "adapter_model-"
ADAPTER_WEIGHT_SUFFIXES = (".safetensors",)


def is_adapter_weight_filename(filename: str) -> bool:
    """True for every adapter weight file shape the pipeline accepts, single-file or sharded.

    One predicate for the same reason the filenames above are defined once: serving validation and
    the exporter never share a call stack, and each spelling the shapes for itself makes them agree
    by coincidence. A disagreement fails silently, because validation accepts a shape the exporter
    skips and the export then ships weights peft loads as a no-op.

    Filename shape only. Whether a SET of these files is loadable is a different question, because a
    lone shard is not: see :func:`has_loadable_adapter_weights`.
    """
    name = filename.rsplit("/", 1)[-1]
    if name in ADAPTER_WEIGHT_FILES:
        return True
    return name.startswith(ADAPTER_SHARD_PREFIX) and name.endswith(ADAPTER_WEIGHT_SUFFIXES)


def loadable_adapter_weight_files(filenames) -> list[str]:
    """The files peft would load out of ``filenames``, or ``[]`` when it could load none.

    The set, not the filename, is what decides this, and the difference is the whole point. peft
    binds one representation per suffix -- the single ``adapter_model.<ext>`` when present, otherwise
    the shards -- and it discovers the sharded form ONLY through ``adapter_model.<ext>.index.json``.
    A directory holding ``adapter_model-00001-of-00002.safetensors`` and no index therefore has no
    loadable representation at all: nothing tells peft the sharded form exists, or which shards
    complete it.

    Accepting that orphan is what let an incomplete upload pass validation. Export then falls back to
    shipping every matching shard when it finds no index, so deployment and export both "succeeded"
    while producing weights peft loads as a no-op -- the silent-disagreement failure this module
    exists to prevent, arriving one level up at the set instead of at the name.

    A partial shard set is rejected for the same reason: the index names the complete set, so a
    listing missing one of its members cannot be loaded either. Residue carrying a second shard total
    also fails closed. Filenames cannot reveal which total the index references, and guessing risks
    binding stale weights; cleaning the residue or re-uploading makes the representation decidable.

    Returned in peft's own precedence order, so the caller that reads tensor keys reads exactly the
    files the serving engine will.
    """
    names = {str(name).rsplit("/", 1)[-1] for name in filenames}
    for suffix in ADAPTER_WEIGHT_SUFFIXES:
        single = f"adapter_model{suffix}"
        if single in names:
            return [single]
        referenced = _index_shard_names(names, suffix)
        if referenced:
            return sorted(referenced)
    return []


def has_loadable_adapter_weights(filenames) -> bool:
    """True when ``filenames`` holds weights peft can actually load."""
    return bool(loadable_adapter_weight_files(filenames))


def directory_holds_loadable_adapter_weights(path: Path) -> bool:
    """True when ``path`` on disk holds a config and weights the engine about to read it can load.

    Both serving runtimes ask this before handing a directory to vLLM, and both used to answer it by
    looking for any one file whose NAME looked like adapter weights. A name is not the question: a
    directory holding a lone ``adapter_model-00001-of-00002.safetensors`` and no index has no
    loadable representation at all, which is exactly what a half-finished download looks like. Asking
    :func:`loadable_adapter_weight_files` instead keeps the runtime verdict identical to the one
    deployment admission and the exporter already reach from a remote listing.

    Size is checked AFTER that selection, never before it. Filtering empty files out of the listing
    first would describe a directory that does not exist: an empty ``adapter_model.safetensors``
    beside a complete shard set would disappear, the shards would be selected, and the caller would
    be told the directory loads -- while peft, reading the real directory, still binds the empty
    single file by precedence and fails. So the representation is chosen from every name present, and
    then every file that representation actually reads has to carry bytes.
    """
    if not (path / "adapter_config.json").is_file():
        return False
    names = [child.name for child in path.iterdir() if child.is_file()]
    selected = loadable_adapter_weight_files(names)
    if not selected:
        return False
    return all((path / name).stat().st_size > 0 for name in _files_peft_reads(selected))


def _files_peft_reads(selected: list[str]) -> list[str]:
    """``selected`` plus the shard index, when the selected representation is a sharded one.

    :func:`loadable_adapter_weight_files` returns the weights, but peft reaches a shard only through
    the index naming it, so an empty index makes the same representation unloadable. It returns
    shards only when that index is present, so it does not have to be looked for again.
    """
    shard_suffixes = {
        suffix
        for suffix in ADAPTER_WEIGHT_SUFFIXES
        for name in selected
        if name.startswith(ADAPTER_SHARD_PREFIX) and name.endswith(suffix)
    }
    return selected + [f"adapter_model{suffix}.index.json" for suffix in sorted(shard_suffixes)]


def _index_shard_names(names: set[str], suffix: str) -> set[str]:
    """The shard names an index for ``suffix`` would have to be accompanied by, from names alone.

    Derived from the ``adapter_model-{k}-of-{n}.<ext>`` convention rather than by reading the index
    file, because every caller here works from a LISTING: serving validates against a remote repo
    listing and the worker against a directory listing, and neither should have to download and
    parse json to answer "is this set complete". The exporter, which already has the files locally,
    still reads the real ``weight_map`` -- that is the authority on which shards are live.

    Empty when no index is present, the sole candidate shard set is incomplete, or more than one
    candidate total appears. The last case is genuinely undecidable from filenames alone: the index
    names exactly one set but the listing cannot say which, so rejecting loudly is safer than silently
    serving stale weights.
    """
    if f"adapter_model{suffix}.index.json" not in names:
        return set()
    totals = set()
    oversized_total = False
    for name in names:
        if not name.startswith(ADAPTER_SHARD_PREFIX) or not name.endswith(suffix):
            continue
        stem = name[len(ADAPTER_SHARD_PREFIX) : -len(suffix)]
        shard, sep, total = stem.partition("-of-")
        if not (sep and shard.isdigit() and total.isdigit()):
            continue
        parsed_total = int(total)
        if parsed_total <= 0:
            continue
        if parsed_total > len(names):
            # this total cannot be complete, but it still proves the listing contains residue from a
            # different save. keep that ambiguity evidence without materializing an unbounded set.
            oversized_total = True
            continue
        totals.add((total, len(shard)))
    if oversized_total or len(totals) != 1:
        return set()
    total, width = next(iter(totals))
    candidate = {
        f"{ADAPTER_SHARD_PREFIX}{index:0{width}d}-of-{total}{suffix}"
        for index in range(1, int(total) + 1)
    }
    return candidate if candidate <= names else set()


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
