"""Trace export CLI parser registration."""

from __future__ import annotations

import argparse

from flash.cli.commands.ops.traces import (
    DEFAULT_EXPORT_PATH,
    EXPORT_FORMATS,
    RAW_EXPORT_PATH,
    RECORDS_FORMAT,
    cmd_traces_export,
)


def _add_traces_commands(sub: argparse._SubParsersAction) -> None:
    """`traces export`."""
    traces = sub.add_parser("traces", help="work with traces recorded by the freesolo SDK")
    traces_sub = traces.add_subparsers(dest="traces_cmd", required=True)
    traces_export = traces_sub.add_parser(
        "export",
        help="export a project's traces as an environment dataset",
        description=(
            "export a project's recorded traces as freesolo environment records, ready to train on"
        ),
    )
    traces_export.add_argument(
        "--project",
        help="project id to export from; omit to pick one from your projects",
    )
    traces_export.add_argument(
        "-o",
        "--output",
        help=(
            f"output JSONL file; defaults to {DEFAULT_EXPORT_PATH}, "
            f"or {RAW_EXPORT_PATH} for --format raw"
        ),
    )
    traces_export.add_argument(
        "-f", "--force", action="store_true", help="overwrite existing output"
    )
    traces_export.add_argument(
        "--format",
        choices=EXPORT_FORMATS,
        default=RECORDS_FORMAT,
        help=(
            "records: {input, output} env records (default). "
            "prompts: {input} only, for GRPO. "
            "raw: the stored rows, converted by nothing. "
            "SFT and GRPO both read records, so there is no per-algorithm format"
        ),
    )
    traces_export.set_defaults(func=cmd_traces_export)
