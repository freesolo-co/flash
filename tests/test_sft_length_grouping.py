"""cpu regression for length-grouped sampling on the unpacked openrlhf sft path.

length-grouped sampling restores trl's ``group_by_length`` parity that openrlhf's plain
``DistributedSampler`` shuffle drops. its correctness is fully cpu-provable and is proved here:

* ``length_grouped_indices`` is a deterministic permutation of ``range(n)`` (lossless) that surfaces
  the longest example first and actually shrinks padding on high-variance length distributions;
* ``distributed_shard`` is byte-identical to openrlhf's ``DistributedSampler`` pad/subsample/skip
  tail, so a grouped permutation shards to the same per-rank counts and global coverage as the plain
  shuffle it replaces (loss-neutral);
* the two compose into the sampler's per-epoch stream: deterministic per (seed, epoch), lossless
  across ranks, and epoch-varying.

the throughput win and peak-memory behaviour on the chunked 32k path are gpu-deferred (opt-in via
``FLASH_SFT_SAMPLING_MODE=group_by_length``); this file covers correctness only.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from flash.engine.worker.sft_openrlhf import (
    distributed_shard,
    length_grouped_indices,
    render_openrlhf_sft_runtime,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed in offline CI",
)


# --- reference helpers -------------------------------------------------------------------------


def _sizes(n: int, num_replicas: int, drop_last: bool) -> tuple[int, int]:
    """openrlhf DistributedSampler.__init__ num_samples/total_size math (distributed_sampler.py)."""
    if drop_last and n % num_replicas != 0:
        num_samples = math.ceil((n - num_replicas) / num_replicas)
    else:
        num_samples = math.ceil(n / num_replicas)
    return num_samples, num_samples * num_replicas


def _openrlhf_reference_shard(
    full_indices, *, num_replicas, rank, total_size, num_samples, drop_last, consumed_indices
):
    """verbatim copy of openrlhf DistributedSampler.__iter__ tail (lines 115-133), the parity oracle."""
    indices = list(full_indices)
    if not drop_last:
        padding_size = total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
    else:
        indices = indices[:total_size]
    assert len(indices) == total_size
    indices = indices[rank:total_size:num_replicas]
    indices = indices[consumed_indices:]
    assert len(indices) == num_samples - consumed_indices
    return indices


def _padding(order, lengths, batch_size):
    """total pad tokens if ``order`` is consumed in contiguous micro-batches of ``batch_size``."""
    total = 0
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        widest = max(lengths[i] for i in batch)
        total += sum(widest - lengths[i] for i in batch)
    return total


# --- length_grouped_indices: permutation / determinism / longest-first -------------------------


@requires_torch
@pytest.mark.parametrize(("n", "batch_size"), [(0, 2), (1, 2), (3, 4), (14, 2), (100, 8), (257, 16)])
def test_length_grouped_indices_is_a_permutation(n, batch_size):
    import torch

    lengths = [(i * 7 + 3) % 97 + 1 for i in range(n)]
    generator = torch.Generator()
    generator.manual_seed(1234)
    order = length_grouped_indices(lengths, batch_size, generator=generator)
    # lossless: every index exactly once, nothing added or dropped.
    assert sorted(order) == list(range(n))


@requires_torch
def test_length_grouped_indices_is_deterministic_per_seed():
    import torch

    lengths = [(i * 13 + 5) % 200 + 1 for i in range(120)]

    def run(seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return length_grouped_indices(lengths, 4, generator=generator)

    assert run(7) == run(7)
    # a different seed reshuffles the mega-batches, so the order genuinely changes.
    assert run(7) != run(8)


@requires_torch
def test_length_grouped_indices_surfaces_longest_first():
    import torch

    lengths = [3, 9, 1, 42, 7, 5, 20, 2]
    generator = torch.Generator()
    generator.manual_seed(0)
    order = length_grouped_indices(lengths, 2, generator=generator)
    # oom early-detection: the very first example is a global-max-length one.
    assert lengths[order[0]] == max(lengths)


@requires_torch
def test_length_grouped_indices_shrinks_padding_on_high_variance_lengths():
    import torch

    # adversarial interleave: identity order pairs a short with a long example in every micro-batch.
    lengths = [1, 100] * 40
    identity = list(range(len(lengths)))
    generator = torch.Generator()
    generator.manual_seed(3)
    grouped = length_grouped_indices(lengths, batch_size=2, generator=generator)
    identity_pad = _padding(identity, lengths, 2)
    grouped_pad = _padding(grouped, lengths, 2)
    # grouping clusters similar lengths, so padding drops sharply versus the interleaved order.
    assert grouped_pad < identity_pad
    assert grouped_pad <= identity_pad // 2


@requires_torch
def test_length_grouped_indices_respects_mega_batch_mult():
    import torch

    lengths = [(i * 3) % 50 + 1 for i in range(64)]
    generator = torch.Generator()
    generator.manual_seed(0)
    # a mega_batch_mult of 1 means each mega-batch is a single micro-batch; still a permutation.
    order = length_grouped_indices(lengths, 4, generator=generator, mega_batch_mult=1)
    assert sorted(order) == list(range(len(lengths)))


# --- distributed_shard: byte-parity with openrlhf + losslessness -------------------------------


@pytest.mark.parametrize(("n", "num_replicas"), [(14, 3), (16, 4), (17, 2), (100, 8), (5, 5)])
@pytest.mark.parametrize("drop_last", [False, True])
def test_distributed_shard_matches_openrlhf_tail_byte_for_byte(n, num_replicas, drop_last):
    full = list(range(n))  # any permutation; identity is enough to exercise the tail
    num_samples, total_size = _sizes(n, num_replicas, drop_last)
    for rank in range(num_replicas):
        got = distributed_shard(
            full,
            num_replicas=num_replicas,
            rank=rank,
            total_size=total_size,
            num_samples=num_samples,
            drop_last=drop_last,
            consumed_indices=0,
        )
        want = _openrlhf_reference_shard(
            full,
            num_replicas=num_replicas,
            rank=rank,
            total_size=total_size,
            num_samples=num_samples,
            drop_last=drop_last,
            consumed_indices=0,
        )
        assert got == want


@pytest.mark.parametrize(("n", "num_replicas"), [(14, 3), (16, 4), (100, 8)])
def test_distributed_shard_is_lossless_across_ranks(n, num_replicas):
    full = list(reversed(range(n)))  # stand-in for a grouped permutation
    num_samples, total_size = _sizes(n, num_replicas, drop_last=False)
    union: list[int] = []
    for rank in range(num_replicas):
        shard = distributed_shard(
            full,
            num_replicas=num_replicas,
            rank=rank,
            total_size=total_size,
            num_samples=num_samples,
            drop_last=False,
            consumed_indices=0,
        )
        assert len(shard) == num_samples  # every rank draws an equal share
        union += shard
    assert len(union) == total_size
    # drop_last=False duplicates a few tail slots but never drops a real example.
    assert set(union) == set(range(n))


def test_distributed_shard_consumed_indices_resumes_the_tail():
    full = list(range(30))
    num_samples, total_size = _sizes(30, num_replicas=2, drop_last=False)
    full_rank0 = distributed_shard(
        full,
        num_replicas=2,
        rank=0,
        total_size=total_size,
        num_samples=num_samples,
        drop_last=False,
        consumed_indices=0,
    )
    resumed = distributed_shard(
        full,
        num_replicas=2,
        rank=0,
        total_size=total_size,
        num_samples=num_samples,
        drop_last=False,
        consumed_indices=5,
    )
    # resuming after k consumed samples yields exactly the remaining tail of the same rank stream.
    assert resumed == full_rank0[5:]


def test_distributed_shard_drop_last_truncates_remainder():
    full = list(range(17))
    num_samples, total_size = _sizes(17, num_replicas=4, drop_last=True)
    assert total_size == 16  # the ragged 17th example is dropped for even divisibility
    union: list[int] = []
    for rank in range(4):
        union += distributed_shard(
            full,
            num_replicas=4,
            rank=rank,
            total_size=total_size,
            num_samples=num_samples,
            drop_last=True,
            consumed_indices=0,
        )
    assert len(union) == 16
    assert len(set(union)) == 16


# --- composition: what the sampler's __iter__ produces (grouped perm -> shard) ------------------


@requires_torch
def test_grouped_sampler_stream_is_deterministic_lossless_and_epoch_varying():
    import torch

    lengths = [(i * 11 + 1) % 130 + 1 for i in range(64)]
    n = len(lengths)
    num_replicas, seed = 4, 42
    num_samples, total_size = _sizes(n, num_replicas, drop_last=False)

    def epoch_stream(epoch):
        # mirror install_length_grouped_distributed_sampler.__iter__: seed+epoch grouped perm, then
        # the exact openrlhf shard tail per rank.
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        full = length_grouped_indices(lengths, 4, generator=generator)
        return [
            distributed_shard(
                full,
                num_replicas=num_replicas,
                rank=rank,
                total_size=total_size,
                num_samples=num_samples,
                drop_last=False,
                consumed_indices=0,
            )
            for rank in range(num_replicas)
        ]

    epoch0 = epoch_stream(0)
    # deterministic: same (seed, epoch) reproduces the identical per-rank streams.
    assert epoch_stream(0) == epoch0
    # lossless: the epoch covers every real example across ranks.
    union = [i for shard in epoch0 for i in shard]
    assert set(union) == set(range(n))
    assert len(union) == total_size
    # epoch-varying: set_epoch reshuffles, so the next epoch differs.
    assert epoch_stream(1) != epoch0


@requires_torch
def test_grouped_and_plain_shuffle_are_equally_lossless():
    """grouping only reorders: it shards to the same per-rank counts and coverage as a plain shuffle."""
    import torch

    lengths = [(i * 5 + 2) % 90 + 1 for i in range(50)]
    n = len(lengths)
    num_replicas = 3
    num_samples, total_size = _sizes(n, num_replicas, drop_last=False)

    generator = torch.Generator()
    generator.manual_seed(9)
    grouped = length_grouped_indices(lengths, 2, generator=generator)
    plain = torch.randperm(n, generator=torch.Generator().manual_seed(9)).tolist()

    def coverage(full):
        counts = [0] * n
        for rank in range(num_replicas):
            for idx in distributed_shard(
                full,
                num_replicas=num_replicas,
                rank=rank,
                total_size=total_size,
                num_samples=num_samples,
                drop_last=False,
                consumed_indices=0,
            ):
                counts[idx] += 1
        return counts

    grouped_counts = coverage(grouped)
    plain_counts = coverage(plain)
    # both cover every example at least once and use exactly total_size slots.
    assert all(c >= 1 for c in grouped_counts)
    assert sum(grouped_counts) == total_size == sum(plain_counts)
    assert all(c >= 1 for c in plain_counts)


# --- child runtime rendering -------------------------------------------------------------------


def test_child_runtime_embeds_sampler_code_and_compiles():
    import ast

    src = render_openrlhf_sft_runtime()
    ast.parse(src)  # the rendered child module is valid python
    # the pure helpers are injected verbatim (one source of truth with this module) ...
    assert "def length_grouped_indices(" in src
    assert "def distributed_shard(" in src
    assert "def install_length_grouped_distributed_sampler(" in src
    # ... and the child-only glue + opt-in gate are present.
    assert "def _sft_sampling_mode(" in src
    assert "def example_lengths(" in src
    assert 'if _sft_sampling_mode() == "group_by_length" and _sft_packing_mode() == "unpacked":' in src
    assert "install_length_grouped_distributed_sampler(int(CONFIG[" in src
    # unknown modes fail loud rather than silently defaulting.
    assert "unknown sft_sampling_mode" in src
