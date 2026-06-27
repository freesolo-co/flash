"""Generate model responses for an eval split via the Flash serving endpoint.

This is the live piece Flash defers to "the serving side": deploy the trained adapter, then
call ``ApiClient.chat`` once per eval row at ``temperature=0`` (deterministic), and collect the
responses. The prompt is built the same way the scaffolded easy-mode ``environment.py`` builds
it (a single user turn with the row's input), so eval prompts match training prompts.

Needs network + a Flash client; imported lazily by the eval orchestrator, never at package
import (keeps ``autoenv gate``/``drive --dry-run`` offline).
"""

from __future__ import annotations

from collections.abc import Callable

# A prompt builder maps a canonical ``{"input","output"}`` row to chat messages. The default
# mirrors the scaffolded env's ``build_prompt_messages``: one user turn carrying the input.
PromptBuilder = Callable[[dict], list[dict]]


def default_prompt(row: dict) -> list[dict]:
    return [{"role": "user", "content": str(row.get("input", ""))}]


def _extract_text(response: dict) -> str:
    """Pull the assistant text out of an OpenAI-shaped chat-completions response."""
    try:
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return str(message.get("content") or "")
    except AttributeError:
        pass
    # Some serving paths return {"content": ...} directly.
    return str(response.get("content") or "")


def generate_for_rows(
    client,
    run_id: str,
    rows: list[dict],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    prompt_builder: PromptBuilder | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[str]:
    """Generate one response per row from the deployed ``run_id`` adapter.

    Deterministic at ``temperature=0``. The caller is responsible for deploying the run before
    and (optionally) undeploying after — this only drives inference.
    """
    build = prompt_builder or default_prompt
    responses: list[str] = []
    total = len(rows)
    for i, row in enumerate(rows):
        resp = client.chat(run_id, build(row), temperature=temperature, max_tokens=max_tokens)
        text = _extract_text(resp)
        responses.append(text)
        if on_progress is not None:
            on_progress(i + 1, total, text)
    return responses
