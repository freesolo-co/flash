"""Reloadable experiment commands; all ranks receive exactly one command per phase."""

import time
from pathlib import Path

import torch


def execute(native, command):
    op = command["op"]
    if op == "clear":
        native.clear()
        return {"cleared": True}
    if op == "warm":
        return native.warm(command["key"], command["tokens"], command["cut"])
    if op == "batch":
        return native.batch(command["requests"])
    if op == "snapshot":
        return native.snapshot(command["key"])
    if op == "drop":
        native.drop(command["key"])
        return {"dropped": True}
    if op == "restore":
        return native.restore(command["key"])
    if op == "save":
        path = Path(command["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        torch.save(native.snapshots[command["key"]], path)
        return {"save_seconds": time.perf_counter() - started, "bytes": path.stat().st_size}
    if op == "load":
        started = time.perf_counter()
        snapshot = torch.load(command["path"], weights_only=True, map_location="cpu")
        loaded = time.perf_counter() - started
        return {"load_seconds": loaded, **native.restore(command["key"], snapshot)}
    if op == "extension":
        import importlib

        module = importlib.import_module(command["module"])
        return importlib.reload(module).execute(native, command)
    raise ValueError(op)
