"""customer-owned serving documentation contract tests."""

from __future__ import annotations

import argparse
import pathlib
import re
import shlex
import tomllib

from flash.cli.parsing.serve_parser import _add_serve_commands
from flash.serve.deployment.profiles import get_profile, placement_for


def test_self_hosting_docs_document_a_command_that_exists() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    doc = (root / "SELF_HOSTING.md").read_text(encoding="utf-8")
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    for gone in ("flash serve setup", "serve-modal"):
        assert gone not in doc, gone
    assert "serve-modal" not in extras

    # every documented flag must exist and every required flag must appear.
    example = doc.split("flash serve deploy \\")[1].split("```")[0]
    documented = set(re.findall(r"--[a-z-]+", example))
    parser = argparse.ArgumentParser()
    _add_serve_commands(parser.add_subparsers(dest="cmd", required=True))
    deploy = (
        parser._subparsers._group_actions[0]
        .choices["serve"]
        ._subparsers._group_actions[0]
        .choices["deploy"]
    )
    accepted = {option for action in deploy._actions for option in action.option_strings}
    required = {
        action.option_strings[0]
        for action in deploy._actions
        if action.required and action.option_strings
    }
    assert documented <= accepted, documented - accepted
    assert required <= documented, required - documented

    # placement validation must accept the documented modal inputs.
    parsed = deploy.parse_args(shlex.split(example.replace("\\\n", " ")))
    placement_for(
        get_profile(parsed.model),
        parsed.provider,
        workspace_name=parsed.modal_workspace,
        environment=parsed.modal_environment,
        region=parsed.modal_region,
        web_suffix=(parsed.modal_web_suffix or None),
    )
    assert "pip install 'freesolo-flash[server]'" in doc


def test_self_hosting_docs_do_not_route_the_plane_key_to_a_customer_endpoint() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    doc = (root / "SELF_HOSTING.md").read_text(encoding="utf-8")
    app = (root / "flash" / "serve" / "app" / "http.py").read_text(encoding="utf-8")
    assert "internal-key" not in app.casefold(), "the packaged app must not read the plane key"
    assert '"bearer"' in app.casefold(), "the packaged app must authenticate with bearer"

    # the deployment section must show bearer auth and never export its url as the plane backend.
    section = doc.split("## Serving")[1]
    assert "Authorization: Bearer $FLASH_SERVING_KEY" in section
    exported = re.findall(r"export FREESOLO_SERVING_URL=(\S+)", section)
    assert not any("modal.run" in value for value in exported), exported
