"""Agent backends that drive Flash from a scaffolded workspace.

``AgentBackend`` is the one seam between the harness and "the agent": given a ready
workspace, produce the config to train and the env id to publish. The harness's run pipeline
(``drive/runner.py``) then performs the actual Flash lifecycle (dry-run validate, cost gate,
publish, submit, poll) the same way for every backend — so swapping a real LLM agent for the
deterministic control changes only the authoring, never the measurement.

- ``ScriptedBaseline``: no LLM. Uses the scaffolded env + config verbatim — the control that
  proves the loop and a floor every real agent should beat.
- ``ClaudeAgent`` (stub): a real agentic loop seeded with the workspace ``TRAINING.md``. Left
  unimplemented here; it requires the Anthropic SDK + filesystem/CLI tools and is wired up in
  a later milestone (look up the current model id via the ``claude-api`` skill at build time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from autoenv.drive.workspace import Workspace
from autoenv.manifest import PaperCase


@dataclass
class AgentOutcome:
    """What an agent produced for one case."""

    status: str  # "ok" | "timeout" | "error"
    config_path: str
    env_id: str
    notes: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@runtime_checkable
class AgentBackend(Protocol):
    name: str

    def run(self, ws: Workspace, case: PaperCase) -> AgentOutcome: ...


class ScriptedBaseline:
    """Deterministic, no-LLM control: train the scaffolded env + config as-is."""

    name = "scripted"

    def run(self, ws: Workspace, case: PaperCase) -> AgentOutcome:
        if not ws.environment_py.is_file():
            return AgentOutcome(
                status="error",
                config_path=str(ws.config_path),
                env_id=ws.env_id_placeholder,
                notes="environment.py missing from workspace",
            )
        return AgentOutcome(
            status="ok",
            config_path=str(ws.config_path),
            env_id=ws.env_id_placeholder,
            notes="scripted baseline: scaffolded environment + config used verbatim",
        )


class ClaudeAgent:
    """Real agentic backend (stub). Implemented in a later milestone."""

    name = "claude"

    def run(self, ws: Workspace, case: PaperCase) -> AgentOutcome:  # pragma: no cover - stub
        raise NotImplementedError(
            "ClaudeAgent is not implemented yet; use ScriptedBaseline. The real loop drives the "
            "`flash` CLI seeded with the workspace TRAINING.md (see plan milestone M3)."
        )


def get_backend(name: str) -> AgentBackend:
    """Resolve a backend by name."""
    backends: dict[str, AgentBackend] = {
        ScriptedBaseline.name: ScriptedBaseline(),
        ClaudeAgent.name: ClaudeAgent(),
    }
    try:
        return backends[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown agent backend {name!r}; known: {', '.join(sorted(backends))}"
        ) from exc
