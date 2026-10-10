"""`flash env list`: the org's published environments plus local sources.

Published ids come from the control plane (`GET /v1/envs`), which reads the GitHub-backed hub that
`flash env push` writes. Local sources are the scaffold directories in the working tree that have
not necessarily been pushed yet, so the two halves answer different questions and are reported
separately.
"""

from __future__ import annotations

from pathlib import Path

from flash._internal.channel import CLI_NAME
from flash.cli.ui import env_panels, render
from flash.client import ClientError
from flash.client.config import load_credentials


def _local_env_sources() -> list[str]:
    paths: list[str] = []
    if Path("environment.py").is_file():
        paths.append(".")
    local = Path("environments")
    if local.is_dir():
        for p in local.iterdir():
            if p.name.startswith("__"):
                continue
            if p.is_dir():
                stem = p.name.replace("-", "_")
                module = p / f"{stem}.py"
                canonical = p / "environment.py"
                if canonical.is_file() or module.is_file():
                    paths.append(f"environments/{p.name}")
            elif p.suffix == ".py":
                paths.append(f"environments/{p.name}")
    return sorted(paths)


def _published_envs() -> tuple[list[str], str | None]:
    """The org's published environment ids, or ``(<empty>, reason)`` when they can't be listed.

    A reason is NOT an error: local sources are still worth printing. But it must be reported
    instead of being folded into the empty state, because "nothing published" and "didn't check" are
    the same output otherwise, and reading the first as the second is what makes a successful
    publish look like it silently failed.
    """
    _, api_key = load_credentials()
    if not api_key:
        return [], f"not logged in - run `{CLI_NAME} login` to list published environments"
    # deliberately not gated on `has_freesolo_backend(api_url)`. unlike the other callers of that
    # helper, this request goes to the control plane, and the plane is what owns the hub: it derives
    # the namespace from the authenticated key and reads github with its own server-side
    # github_token. a self-hosted plane configured with both is a perfectly valid list target, but
    # its api-url is not under freesolo.co and its freesolo_base_url lives in the server's
    # environment, so a client-side check sees neither signal and answers "no hub" for a plane that
    # has one. the plane already reports the real answer -- 403 for an org-agnostic internal key,
    # 503 when github_token is unset -- and those reasons are more accurate than anything decidable
    # here, so ask and report what comes back.
    from flash.client import client_from_config

    try:
        return client_from_config().list_envs(), None
    except ClientError as exc:
        return [], str(exc)


def cmd_env_list(args) -> int:
    paths = _local_env_sources()
    published, unavailable = _published_envs()
    if render.styled():
        print(env_panels.env_list(paths, published=published, unavailable=unavailable))
        return 0
    if published:
        print('published environments (reference one with `[environment] id = "<id>"`):')
        for env_id in published:
            print(f"  {env_id}")
    elif unavailable:
        print(f"published environments unavailable: {unavailable}")
    if paths:
        print(
            f"local env sources (publish with `{CLI_NAME} env push --project <project-uuid> "
            "--name <name> <path>`):"
        )
        for path in paths:
            print(f"  {path}")
    if not paths and not published:
        # still worth printing next to an `unavailable` line: that line already says the published
        # list was not checked, so this reads as "and you have nothing local yet" rather than as a
        # claim about the hub.
        print(f"no environments yet - scaffold one with `{CLI_NAME} env setup`")
    return 0
