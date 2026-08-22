"""cost quotes and config compatibility notes printed before `flash train` launches.

rl methods use the offline analytical calculation. sft asks the authenticated control-plane preview
for the quote and renders its packaged-dataset aggregates, including the raw-record estimate caveat.

split out of `flash.cli.commands` to keep that module under the file-size limit.
"""

from __future__ import annotations

import sys

from flash import __version__
from flash.cli.ui import render
from flash.client import ClientError
from flash.client.runtime_secrets import runtime_secrets_from_local_env
from flash.client.specs import spec_payload
from flash.cost.spec import runconfig_from_spec
from flash.engine.profiling.workload_profile import (
    rendered_reasoning_loss_warning,
    unpacked_batch_warning,
)
from flash.schema import spec_and_train_keys_from_file, train_schema_metadata


def _commands():
    """The parent package, imported lazily because it re-exports this module.

    `client_from_config`, `shadowed_login_warning` and `CLI_NAME` are patched as attributes of
    `flash.cli.commands` by the cli estimate tests -- the first two to prove the offline quote
    never builds a client and still surfaces the shadowed-key warning, the third to prove the dev
    channel's `flash-dev` name reaches the hints printed here. Binding any of them by value would
    capture the original before the patch lands.
    """
    from flash.cli import commands

    return commands


def _cmd_train_cost(args) -> int:
    """`flash train --cost`: print the pre-flight USD cost for the config and exit (no submit).

    grpo and opd quote offline from the catalog. sft asks the server to read the pinned packaged
    dataset and statically readable training contract without executing environment code. tokens,
    retention, truncation, and steps are estimates that can miss other environment transformations.
    """
    from flash.adapters.lora_rank import preflight_train_context_within_serving

    spec, authored_train_keys = spec_and_train_keys_from_file(
        args.config,
        run_id=None,
        overrides=args.overrides,
        extra_configs=args.extra_configs,
        project_required=True,
    )
    preflight_train_context_within_serving(spec)
    if spec.algorithm == "sft":
        return _cmd_train_cost_sft(args, spec, authored_train_keys)
    return _cmd_train_cost_offline(spec)


def _cmd_train_cost_offline(spec) -> int:
    """Catalog-only quote for the algorithms that do not need workload evidence yet (grpo, opd)."""
    from flash.cost import estimate_cost

    if spec.train.init_from_adapter:
        # --cost is offline/catalog-only and cannot read the source adapter, so the rank stays at the
        # local default. Warm starts train and are priced at the SOURCE adapter's authoritative rank
        # (resolved server-side at submit/dry-run), which can be higher — so this estimate may
        # under-quote. stderr keeps stdout clean for machine-readable callers.
        print(
            "warning: warm-start (train.init_from_adapter) cost uses the default LoRA rank; the "
            "source adapter's rank is authoritative and resolved at submit, so a higher-rank source "
            f"may cost more than this estimate. Run `{_commands().CLI_NAME} train --dry-run` "
            "for a source-rank quote.",
            file=sys.stderr,
        )
    estimate = estimate_cost(runconfig_from_spec(spec))
    if render.styled():
        print(render.cost_panel(estimate))
    else:
        print(estimate.breakdown())
    from flash.cli.commands.prompt_budget import warn_cost_prompt_budget

    warn_cost_prompt_budget(spec)
    return 0


def _cmd_train_cost_sft(args, spec, authored_train_keys: frozenset[str]) -> int:
    """SFT estimate served by the same authenticated path that freezes a real submit's quote."""
    # cli._warn_if_login_shadowed() suppresses this warning for `--cost` because the catalog path
    # never reaches an organization. the sft path does: it authenticates, resolves the project, and
    # requests the server-side packaged-dataset estimate, so the warning has to fire here after all.
    message = _commands().shadowed_login_warning()
    if message:
        print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)
    client = _commands().client_from_config()
    status = client.create_run(
        spec_payload(spec, authored_train_keys=authored_train_keys),
        runtime_secrets=runtime_secrets_from_local_env(args.config, keys=spec.environment.secrets)
        or None,
        dry_run=True,
        client_train_schema=_client_train_schema(authored_train_keys),
    )
    _print_sft_cost(status, spec)
    return 0


def _client_train_schema(authored_train_keys: frozenset[str]) -> dict:
    return {
        "version": __version__,
        "fields": train_schema_metadata(),
        "authored_keys": sorted(authored_train_keys),
    }


def _sft_cost_rows(spec, profile: dict) -> list[tuple[str, str | None]]:
    """Rows describing the published packaged-dataset estimate behind an SFT quote.

    Only aggregates the server actually returned are shown. A field that is absent is dropped
    rather than defaulted, so the panel never reports a count the profile did not compute.
    """

    def count(key: str) -> int | None:
        value = profile.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def text(key: str) -> str:
        value = profile.get(key)
        return value.strip() if isinstance(value, str) else ""

    steps = count("authoritative_steps")
    source = count("source_examples")
    retained, selected = count("retained_examples"), count("selected_examples")
    dropped = count("dropped_examples")
    compute, supervised = (
        count("authoritative_compute_tokens"),
        count("authoritative_supervised_tokens"),
    )
    packing = text("packing_mode")
    architecture = text("architecture_mode")
    environment_id = text("environment_id")
    environment_revision = text("environment_revision")
    digest = text("content_digest")
    examples = None
    if retained is not None and selected is not None:
        examples = f"{retained:,} trained of {selected:,} selected from "
        examples += (
            f"{source:,} source rows in published copy" if source is not None else "published copy"
        )
        if dropped:
            examples += f" ({dropped:,} dropped)"
    tokens = None
    if compute is not None:
        tokens = f"{compute:,} compute"
        if supervised is not None:
            tokens += f", {supervised:,} supervised"
    return [
        ("run", f"{spec.model}  [SFT{f', {steps} steps' if steps is not None else ''}]"),
        ("env", f"published environment {environment_id}" if environment_id else None),
        (
            "revision",
            f"{environment_revision[:12]} (published commit)" if environment_revision else None,
        ),
        ("workload", f"{packing} ({architecture})" if packing and architecture else None),
        ("examples", examples),
        ("tokens", tokens),
        ("digest", digest[:12] or None),
    ]


def _print_unpacked_batch_warning(status: object, spec) -> None:
    """Warn that an unpacked SFT run trains 1 example per update, ignoring `batch_size`.

    The quote/dry-run response already carries the frozen packing decision, so the override is
    knowable before any training GPU is allocated. The reason travels on the profile's
    `architecture_mode`, which is what the packing decision froze.
    """
    profile = status.get("workload_profile") if isinstance(status, dict) else None
    if not isinstance(profile, dict):
        return
    examples_per_update = profile.get("examples_per_update")
    if isinstance(examples_per_update, bool) or not isinstance(examples_per_update, int):
        return
    message = unpacked_batch_warning(
        packing_mode=str(profile.get("packing_mode") or ""),
        architecture_mode=str(profile.get("architecture_mode") or ""),
        examples_per_update=examples_per_update,
        configured_batch_size=getattr(spec.train, "batch_size", None),
    )
    if not message:
        return
    print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)


def _print_reasoning_loss_warning(status: object) -> None:
    """Warn that the chat template drops authored reasoning before a training GPU is allocated.

    Control-plane profiling runs inside the server, so the measurement's own stderr never reaches
    the submitting client. The counts travel on the quote's profile and the line is rendered here,
    which is the only place the user can act on it while restructuring is still free.
    """
    profile = status.get("workload_profile") if isinstance(status, dict) else None
    if not isinstance(profile, dict):
        return

    # tolerant, unlike every other reader of this profile, because of WHERE the dict comes from and
    # what this function is for. it is the quote response of a control plane the CLI does not ship
    # with, so a field this build expects can legitimately be absent from a peer's reply during a
    # rolling upgrade -- forward compatibility, not legacy debt. and the whole function is one
    # advisory warning line: these call sites are not wrapped, so raising here would abort `train
    # --cost` and the SFT dry run AFTER the server already returned a valid quote. losing a warning
    # is a far smaller harm than failing the command that carries it.
    def optional_count(key: str) -> int | None:
        value = profile.get(key)
        return None if isinstance(value, bool) or not isinstance(value, int) else value

    authored = optional_count("authored_reasoning_turns")
    rendered = optional_count("rendered_reasoning_spans")
    rows = optional_count("reasoning_rows")
    if authored is None or rendered is None or rows is None:
        return
    message = rendered_reasoning_loss_warning(
        authored_turns=authored,
        rendered_spans=rendered,
        # absent on a profile from a producer that had not yet split the two causes apart. zero
        # reads as "none were truncated", which is exactly the pre-split behaviour.
        truncated_spans=optional_count("truncated_reasoning_spans") or 0,
        rows=rows,
    )
    if not message:
        return
    print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)


def _print_published_sft_environment_note(status: object, spec=None) -> None:
    """Attribute SFT preview counts to whatever actually produced them.

    Which is not always the published package: inline `[environment.params] records` are read from
    the request body instead, and telling that user to publish local files would send them to fix
    something the counts never came from.
    """
    profile = status.get("workload_profile") if isinstance(status, dict) else None
    if not isinstance(profile, dict):
        return
    environment_id = profile.get("environment_id")
    revision = profile.get("environment_revision")
    if not isinstance(environment_id, str) or not environment_id.strip():
        return
    if not isinstance(revision, str) or not revision.strip():
        return

    from flash.envs.identity import is_github_environment_ref, is_managed_environment_slug

    source = profile.get("source_examples")
    source_suffix = (
        f" ({source:,} source rows)"
        if isinstance(source, int) and not isinstance(source, bool)
        else ""
    )
    environment_id = environment_id.strip()
    revision = revision.strip()
    params = getattr(getattr(spec, "environment", None), "params", None)
    if isinstance(params, dict) and params.get("records") is not None:
        # the rows were submitted inline, so the resolved package supplied the environment but not
        # the dataset. naming the published copy as their source would be simply wrong.
        message = (
            f"environment: {environment_id} @ {revision[:12]}{source_suffix}. SFT dataset counts "
            "come from the inline [environment.params] records in this config, not from the "
            "published environment's dataset files."
        )
        print(render.note(message) if render.styled() else message, file=sys.stderr)
        return
    message = (
        f"published environment: {environment_id} @ {revision[:12]}{source_suffix}. "
        "SFT dataset counts come from this resolved published copy, not local files."
    )
    if is_managed_environment_slug(environment_id):
        message += (
            f" For this managed id, that copy is the last successful `{_commands().CLI_NAME} env "
            "push`; push again after local dataset edits."
        )
    elif is_github_environment_ref(environment_id):
        # the plane resolves this ref from the REMOTE repository, so a local commit is invisible to
        # it until pushed. for a branch ref the id does not change at all, which is why "update the
        # id" was the wrong instruction: it is unnecessary here and, if taken as pinning a new sha,
        # names a commit the remote does not have yet.
        message += (
            " Push the commit to the remote branch this ref resolves, then update [environment] id "
            "only to pin a different ref."
        )
    else:
        message += " Commit local dataset edits and update [environment] id before relying on them."
    print(render.note(message) if render.styled() else message, file=sys.stderr)


def _print_sft_cost(status: dict, spec) -> None:
    total = status.get("estimated_cost_usd") if isinstance(status, dict) else None
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        raise ClientError(
            "the server accepted this SFT config but returned no cost estimate; "
            f"run `{_commands().CLI_NAME} train --dry-run` to see the full server response"
        )
    profile = status.get("workload_profile")
    rows = _sft_cost_rows(spec, profile if isinstance(profile, dict) else {})
    if render.styled():
        print(render.sft_cost_panel(rows, float(total)))
    else:
        for key, value in rows:
            if value is not None:
                print(f"{key.ljust(8)}: {value}")
        print(f"{'TOTAL'.ljust(8)}: ${float(total):.2f}")
    _print_published_sft_environment_note(status, spec)
    print(
        "tokens, retained rows, truncation, and optimizer steps are estimated from packaged "
        "input/output fields plus contract_text, contract_path, or TRAINING_CONTRACT.md. other "
        "environment-added prompts, few-shot examples, tool schemas, filtering, or transformations "
        "are not executed here, so actual training may retain fewer rows and cost more. no training "
        "gpu was allocated and nothing was charged for training.",
        file=sys.stderr,
    )
    _print_unpacked_batch_warning(status, spec)
    _print_reasoning_loss_warning(status)


def _print_train_schema_compatibility(result: object) -> None:
    if not isinstance(result, dict):
        message = "client/server [train] schema compatibility is unverifiable (legacy server)"
    elif result.get("status") == "agreement":
        message = "client/server [train] schemas agree exactly"
    else:
        differences = []
        for label, key in (
            ("client-only keys", "client_only"),
            ("server-only keys", "server_only"),
        ):
            values = result.get(key)
            if isinstance(values, list) and values:
                differences.append(f"{label}: {', '.join(str(value) for value in values)}")
        metadata = result.get("introduced_in_differences")
        if isinstance(metadata, list) and metadata:
            rendered = ", ".join(
                f"{item['key']} (client {item['client']}, server {item['server']})"
                for item in metadata
                if isinstance(item, dict)
                and all(isinstance(item.get(key), str) for key in ("key", "client", "server"))
            )
            if rendered:
                differences.append(f"introduced_in differences: {rendered}")
        suffix = f"; {'; '.join(differences)}" if differences else ""
        message = f"client/server [train] schemas disagree{suffix}"
    text = f"train schema: {message}"
    print(render.note(text) if render.styled() else text, file=sys.stderr)


def _warn_if_wandb_requested_without_key(
    spec, runtime_secrets: dict | None, *, dry_run: bool
) -> None:
    """Warn when a config asks for W&B but no ``WANDB_API_KEY`` was found locally.

    ``WANDB_API_KEY`` is an optional runtime secret, and discovery only looks at the process env and
    the ``.env``/``.env.local`` files beside the cwd and the config. A key one directory up is not
    found, and an absent optional secret is not an error, so the run trains to completion with
    logging silently off, which is only discoverable after the GPU spend, when the curve is gone.
    """
    if not (spec.wandb.project or spec.wandb.run_name):
        return
    # a dry-run allocates no gpu and trains nothing, so "this run will train" would contradict the
    # dry-run notice printed moments later and can read as though a paid run started.
    subject = "a run submitted with this config will train" if dry_run else "this run will train"
    # strip before deciding: the server's _runtime_secrets() strips and drops the value, so a
    # whitespace-only key reaches the worker as no key at all -- the exact silent-no-logging
    # failure this warning exists to prevent, but with the warning suppressed.
    if str((runtime_secrets or {}).get("WANDB_API_KEY") or "").strip():
        return
    message = (
        "[wandb] is configured but no WANDB_API_KEY was found in the environment, "
        f"./.env(.local), or the .env(.local) beside the config; {subject} with W&B logging "
        "DISABLED. Export WANDB_API_KEY or put it in a .env next to the config."
    )
    print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)
