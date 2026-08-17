from __future__ import annotations


def valid_source_snapshot() -> dict:
    digest = "a" * 64
    return {
        "kind": "flash-source-snapshot",
        "format_version": 1,
        "archive_path": f"source/{digest}/flash-source.zip",
        "sha256": digest,
        "size": 123,
        "revision": "b" * 40,
    }
