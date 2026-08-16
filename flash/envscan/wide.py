"""Scanning the narrowed ASCII views of UTF-16 and UTF-32 content."""

from __future__ import annotations

from collections.abc import Callable

from flash.envscan.buffers import _wide_runs

Detector = Callable[..., str | None]


def credential_in_wide_runs(
    data: bytes,
    *,
    detector: Detector,
    deadline: float | None,
    depth: int,
    truncated: bool,
    shell: bool,
    literal_syntax: str | None,
    rejoin: bool,
) -> str | None:
    """The credential kind in any genuine wide-text run, or None."""
    for width, keep in ((2, (0, 1)), (4, (0, 3))):
        for offset in keep:
            # take every `width`-th byte, selecting either byte order's ascii column.
            for run in _wide_runs(data, width, offset):
                # narrowed runs need their own paired scan because nul padding makes the raw paired
                # tokenizer blind. truncation still follows the source window into base64 handling.
                if kind := detector(
                    run,
                    deadline=deadline,
                    depth=depth,
                    truncated=truncated,
                    paired=True,
                    shell=shell,
                    literal_syntax=literal_syntax,
                    rejoin=rejoin,
                ):
                    return kind
    return None
