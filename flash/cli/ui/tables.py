"""Table renderers for `flash` list commands.

The primitives these build on (`_table`, `_paint`, `header`, ...) stay in
`flash.cli.ui.render`; this module owns the per-command table layouts.
"""

from __future__ import annotations

from flash._internal.channel import CLI_NAME
from flash.cli.ui import cost
from flash.cli.ui.render import (
    _ACCENT2,
    _AMBER,
    _FAINT,
    _GRAY,
    _GREEN,
    _RED,
    _STATE_STYLE,
    _TEAL,
    _dim,
    _glyph,
    _humanize_ts,
    _paint,
    _safe,
    _table,
    arrow,
    gpu_label,
    header,
)


def models_table(rows: list[dict]) -> str:
    """Supported base models — a clean themed list of ids (the CLI lists ids only)."""
    dot = _glyph("•", "-")
    ids = "\n".join(f"  {_paint(dot, _FAINT)} {_paint(r['id'], _ACCENT2)}" for r in rows)
    foot = arrow(f"train one with: {CLI_NAME} train configs/sft.toml")
    return _safe(f"{header('models', 'supported base models')}\n{ids}\n\n{foot}")


def projects_table(rows: list[dict]) -> str:
    """Freesolo projects with copyable canonical ids."""
    body = [
        [
            (str(row.get("name") or ""), _GRAY),
            (str(row.get("id") or ""), _ACCENT2),
        ]
        for row in rows
    ]
    if not body:
        return _safe(
            f"{header('projects list', 'Freesolo projects')}\n"
            f"{_dim(f'  no projects yet; create one with `{CLI_NAME} projects create NAME`')}"
        )
    return _safe(
        f"{header('projects list', 'Freesolo projects')}\n{_table(['NAME', 'PROJECT ID'], body)}"
    )


def gpus_table(rows: list[tuple[str, int, float | None]], tip: str) -> str:
    """GPU classes: (name, vram_gb, $/hr or None)."""
    body = []
    for name, vram, rate in rows:
        rate_cell = (f"${rate:.2f}", _TEAL) if rate else ("-", _FAINT)
        body.append([(name, _ACCENT2), (f"{vram} GB", _GRAY), rate_cell])
    table = _table(["GPU", "VRAM", "$/HR"], body, aligns=["l", "r", "r"])
    return _safe(f"{header('gpus', 'managed GPU classes')}\n{table}\n\n{_dim(tip)}")


def runs_table(runs: list[dict]) -> str:
    """Runs list: state badges + cost, newest first."""
    body = []
    for r in sorted(runs, key=lambda r: r.get("updated_at", 0), reverse=True):
        spec = r.get("spec") or {}
        model = spec.get("model", "")
        algorithm = str(spec.get("algorithm") or "-").upper()
        where = gpu_label(spec, r.get("remote") or {})
        color, uni, ascii_dot = _STATE_STYLE.get(str(r.get("state", "")).lower(), (_GRAY, "•", "-"))
        amount, is_estimate = cost.run_cost(r)
        # `~` marks the submit-time quote for a run that has not settled, so a column of live runs
        # does not read as a column of free ones.
        body.append(
            [
                (r["run_id"], _ACCENT2),
                (f"{_glyph(uni, ascii_dot)} {r.get('state', '')}", color),
                (algorithm, _GRAY),
                (f"{'~' if is_estimate else ''}${amount:.4f}", _TEAL),
                (where, _GRAY),
                model,
            ]
        )
    table = _table(
        ["RUN ID", "STATE", "ALGO", "COST", "GPU", "MODEL"],
        body,
        aligns=["l", "l", "l", "r", "l", "l"],
    )
    return _safe(f"{header('runs', f'{len(runs)} run(s)')}\n{table}")


def deployments_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        deployment = row.get("deployment") or {}
        run_id = str(deployment.get("run_id") or row.get("run_id") or "")
        state = str(deployment.get("state") or "?")
        color = _GREEN if state in {"ready", "deployed"} else _RED if state == "failed" else _AMBER
        step = deployment.get("checkpoint_step")
        verified_at = deployment.get("verified_at")
        detail = str(deployment.get("error") or deployment.get("detail") or "")
        if len(detail) > 64:
            detail = detail[:61] + "..."
        body.append(
            [
                (run_id, _ACCENT2),
                ("final" if step is None else str(step), _TEAL),
                (str(deployment.get("checkpoint_id") or "-"), _ACCENT2),
                (state, color),
                (
                    "-" if verified_at is None else (_humanize_ts(verified_at) or str(verified_at)),
                    _GRAY,
                ),
                (str(deployment.get("openai_model") or run_id), _GREEN),
                (str(deployment.get("openai_base_url") or "-"), _TEAL),
                (detail, _GRAY),
            ]
        )
    table = _table(
        [
            "RUN ID",
            "STEP",
            "CHECKPOINT ID",
            "STATE",
            "VERIFIED AT",
            "OPENAI MODEL",
            "OPENAI BASE URL",
            "DETAIL",
        ],
        body,
    )
    return _safe(f"{header('deployments', f'{len(rows)} active')}\n{table}")


def checkpoints_table(run_id: str, rows: list[dict]) -> str:
    """Deployable per-step RL checkpoints: step number + the canonical `<run_id>/step-N` ref."""
    from flash.schema import format_checkpoint_ref

    body = [
        [
            (str(c.get("step", "")), _TEAL),
            (format_checkpoint_ref(run_id, c.get("step", 0)), _ACCENT2),
        ]
        for c in sorted(rows, key=lambda c: c.get("step", 0))
    ]
    table = _table(["STEP", "CHECKPOINT"], body, aligns=["r", "l"])
    foot = arrow(f"deploy one with: {CLI_NAME} models deploy {run_id}/step-<STEP>")
    return _safe(f"{header('checkpoints', f'{len(rows)} deployable')}\n{table}\n\n{foot}")
