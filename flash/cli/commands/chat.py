"""`flash models chat`: stream one reply from a deployed run and exit.

The bulk of this is the empty-response contract. A serving path that stopped applying the run's
chat template still returns a well-formed stream carrying no assistant text, so exiting 0 there
would make a broken deployment indistinguishable from a model that answered nothing, and this
surface could no longer be used as a health check.

Split out of `flash.cli.commands` to keep that module under the file-size limit.
"""

from __future__ import annotations

import sys

from flash.cli.ui import render


def _commands():
    """The parent package, imported lazily because it re-exports this module.

    `client_from_config` and `CLI_NAME` are patched as attributes of `flash.cli.commands` by the
    cli tests -- the first to drive the stream from a fake client, the second to prove the dev
    channel's `flash-dev` name reaches the hint printed here. Binding either by value would
    capture the original before the patch lands.
    """
    from flash.cli import commands

    return commands


def cmd_chat(args) -> int:
    from flash.schema import parse_adapter_revision, parse_checkpoint_ref

    revision = parse_adapter_revision(args.run_id)
    parsed = parse_checkpoint_ref(args.run_id) if revision is None else None
    if revision is None and parsed is None:
        print(
            f"invalid chat target {args.run_id!r} "
            "(expected a bare <run_id>, <run_id>/step-N, or full immutable adapter revision)",
            file=sys.stderr,
        )
        return 1
    chat_target = args.run_id
    client = _commands().client_from_config()
    messages = [{"role": "user", "content": args.message}]
    system = getattr(args, "system", None)
    if system:
        messages.insert(0, {"role": "system", "content": system})
    wrote = False
    pending: list[str] = []
    for chunk in client.chat_stream(
        chat_target,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ):
        # delay the label and blank chunks until real text arrives. otherwise an empty response has
        # non-empty stdout and cannot serve as a health check. release buffered blanks verbatim;
        # flash/cli/commands/env/eval.py grades emptiness the same way.
        if not wrote:
            pending.append(chunk)
            if not chunk.strip():
                continue
            if render.styled():
                print(render.chat_label())
            chunk = "".join(pending)
            wrote = True
        print(chunk, end="", flush=True)
    if not wrote:
        # the request succeeded at the transport level but carried no assistant text, which is what
        # a serving path that stopped applying the run's chat template looks like from here. exiting
        # 0 with an empty stdout makes that indistinguishable from a model that answered nothing, so
        # this surface cannot be used as a health check -- say what happened and fail.
        print(
            f"no response text from {chat_target}: the request succeeded but the model returned "
            "nothing. the deployment may be unhealthy or still starting; check "
            f"`{_commands().CLI_NAME} models deployments` and retry.",
            file=sys.stderr,
        )
        return 1
    print()
    return 0
