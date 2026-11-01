"""partition collected tests by whole file without changing pytest discovery."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

# slow-file weights come from offline junit timings; unmeasured files still participate.
FILE_SECONDS: dict[str, float] = {}


def assign_files(paths: Iterable[str], count: int) -> dict[str, int]:
    loads = [0.0] * count
    assignments = {}
    for path in sorted(paths, key=lambda path: (-FILE_SECONDS.get(path, 1.0), path)):
        shard = min(range(count), key=lambda index: (loads[index], index))
        assignments[path] = shard
        loads[shard] += FILE_SECONDS.get(path, 1.0)
    return assignments


def pytest_addoption(parser):
    group = parser.getgroup("ci sharding")
    group.addoption("--ci-shard", type=int, default=None)
    group.addoption("--ci-shard-count", type=int, default=2)


def pytest_configure(config):
    shard = config.getoption("--ci-shard")
    count = config.getoption("--ci-shard-count")
    if shard is None or count < 1 or not 0 <= shard < count:
        raise pytest.UsageError("ci shard must satisfy 0 <= --ci-shard < --ci-shard-count")


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(config, items):
    # partition after marker selection so every worker assigns the same eligible file set.
    result = yield
    count = config.getoption("--ci-shard-count")
    shard = config.getoption("--ci-shard")
    paths = {item.path.relative_to(config.rootpath).as_posix() for item in items}
    assignments = assign_files(paths, count)
    kept, removed = [], []
    for item in items:
        path = item.path.relative_to(config.rootpath).as_posix()
        (kept if assignments[path] == shard else removed).append(item)
    config.hook.pytest_deselected(items=removed)
    items[:] = kept
    return result
