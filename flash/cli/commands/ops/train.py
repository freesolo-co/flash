"""cost quotes and config compatibility notes printed before `flash train` launches.

every method asks the authenticated control-plane preview for the quote submission would freeze.
sft also renders packaged-dataset aggregates, including the raw-record estimate caveat.

split out of `flash.cli.commands` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import sys

from flash import __version__
from flash._internal.channel import CLI_NAME
from flash._internal.logging import get_logger
from flash.cli.commands.ops.prompt_budget import (
    print_status_prompt_budget_warning,
    print_warmstart_context_supplement,
    prompt_budget_validation_suffix,
    warn_before_paid_submit,
)
from flash.cli.ui import render
from flash.client import ClientError, client_from_config
from flash.client.config import shadowed_login_warning
from flash.client.runtime_secrets import runtime_secrets_from_local_env
from flash.client.specs import spec_payload
from flash.cost.currency import format_usd
from flash.engine.profiling.workload_profile import (
    rendered_reasoning_loss_warning,
    unpacked_batch_warning,
)
from flash.schema import spec_and_train_keys_from_file, train_schema_metadata

logger = get_logger("flash.cli")


def _cmd_train_cost(args) -> int:
    """`flash train --cost`: print the pre-flight USD cost for the config and exit (no submit).

    every method asks the server to run the same read-only preparation that freezes a real submit's
    quote. sft also reads the pinned packaged dataset and statically readable training contract
    without executing environment code. tokens, retention, truncation, and steps are estimates that
    can miss other environment transformations.
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
    return _cmd_train_cost_rl(args, spec, authored_train_keys)


def _request_server_quote(args, spec, authored_train_keys: frozenset[str]) -> dict:
    """Run authenticated submit preparation without allocating a training gpu."""
    # the generic cli hook suppresses this warning for `--cost` because the algorithm is not known
    # until this file parses the config. every quote now reaches an organization, so restore it here.
    message = shadowed_login_warning()
    if message:
        print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)
    client = client_from_config()
    return client.create_run(
        spec_payload(spec, authored_train_keys=authored_train_keys),
        runtime_secrets=runtime_secrets_from_local_env(args.config, keys=spec.environment.secrets)
        or None,
        dry_run=True,
        client_train_schema=_client_train_schema(authored_train_keys),
    )


def _cmd_train_cost_rl(args, spec, authored_train_keys: frozenset[str]) -> int:
    """Authoritative grpo/opd quote from the same preparation used by submission."""
    from flash.cli.commands.ops.prompt_budget import print_status_prompt_budget_warning

    status = _request_server_quote(args, spec, authored_train_keys)
    _print_server_cost(status, spec)
    print_status_prompt_budget_warning(status)
    return 0


def _cmd_train_cost_sft(args, spec, authored_train_keys: frozenset[str]) -> int:
    """SFT estimate served by the same authenticated path that freezes a real submit's quote."""
    status = _request_server_quote(args, spec, authored_train_keys)
    _print_sft_cost(status, spec)
    return 0


def _client_train_schema(authored_train_keys: frozenset[str]) -> dict:
    return {
        "version": __version__,
        "fields": train_schema_metadata(),
        "authored_keys": sorted(authored_train_keys),
    }


def _has_inline_records(spec) -> bool:
    """Whether this config supplied its SFT rows inline rather than from the resolved package.

    One predicate for every surface that attributes the counts -- the cost rows and the provenance
    note sit within a few lines of each other, so disagreeing about where the data came from would
    print two contradictory answers in one quote.
    """
    params = getattr(getattr(spec, "environment", None), "params", None)
    return isinstance(params, dict) and params.get("records") is not None


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
    # inline `[environment.params] records` supply the rows from the request body, so the resolved
    # package contributed the environment but not the dataset. labelling those counts "published
    # copy" names a source they did not come from. one adjective decides all three rows, so they
    # cannot describe the same quote two different ways.
    inline_records = _has_inline_records(spec)
    origin = "inline records" if inline_records else "published copy"
    resolved = "resolved" if inline_records else "published"
    examples = None
    if retained is not None and selected is not None:
        examples = f"{retained:,} trained of {selected:,} selected from "
        examples += f"{source:,} source rows in {origin}" if source is not None else origin
        if dropped:
            examples += f" ({dropped:,} dropped)"
    tokens = None
    if compute is not None:
        tokens = f"{compute:,} compute"
        if supervised is not None:
            tokens += f", {supervised:,} supervised"
    return [
        ("run", f"{spec.model}  [SFT{f', {steps} steps' if steps is not None else ''}]"),
        ("env", f"{resolved} environment {environment_id}" if environment_id else None),
        (
            "revision",
            f"{environment_revision[:12]} ({resolved} commit)" if environment_revision else None,
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


def _print_published_sft_environment_note(status: object, spec) -> None:
    """Warn when SFT counts came from a published managed or GitHub environment."""
    if _has_inline_records(spec):
        return

    profile = status.get("workload_profile") if isinstance(status, dict) else None
    if not isinstance(profile, dict):
        return
    environment_id = profile.get("environment_id")
    revision = profile.get("environment_revision")
    if not isinstance(environment_id, str) or not environment_id.strip():
        return
    if not isinstance(revision, str) or not revision.strip():
        return

    environment_id = environment_id.strip()
    advice = _republish_advice(environment_id)
    if advice is None:
        return

    source = profile.get("source_examples")
    source_suffix = (
        f" ({source:,} source rows)"
        if isinstance(source, int) and not isinstance(source, bool)
        else ""
    )
    message = (
        f"published environment: {environment_id} @ {revision.strip()[:12]}{source_suffix}. "
        "SFT dataset counts come from this resolved published copy, not local files. "
        f"{advice}"
    )
    print(render.note(message) if render.styled() else message, file=sys.stderr)


def _republish_advice(environment_id: str) -> str | None:
    """Return conditional remediation for a managed or GitHub environment id."""
    from flash.envs.meta.identity import (
        canonical_managed_environment_slug,
        github_environment_ref_is_pinned,
        is_github_environment_ref,
    )

    prefix = "If you expected local dataset edits to be included, "
    try:
        managed_slug = canonical_managed_environment_slug(environment_id)
    except ValueError:
        return None
    if managed_slug is not None:
        return (
            f"{prefix}run `{CLI_NAME} env push --name NAME "
            "--project PROJECT_UUID [path]` again for this managed environment."
        )
    if not is_github_environment_ref(environment_id):
        return None
    if github_environment_ref_is_pinned(environment_id):
        return (
            f"{prefix}publish the new commit, then update [environment] id to the full new commit "
            "SHA; the existing immutable SHA cannot move."
        )
    return (
        f"{prefix}make the named remote branch or tag point at the new commit. A tag may instead "
        "need a new tag and an updated [environment] id rather than moving the existing tag."
    )


def _server_quote_total(status: object, algorithm: str) -> float:
    total = status.get("estimated_cost_usd") if isinstance(status, dict) else None
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        message = (
            f"the server accepted this {algorithm.upper()} config but returned no cost estimate"
        )
        if algorithm == "sft":
            message += f"; run `{CLI_NAME} train --dry-run` to see the full server response"
        raise ClientError(message)
    return float(total)


def _print_server_cost(status: dict, spec) -> None:
    """Render a server-prepared quote without inventing an offline hardware breakdown."""
    total = _server_quote_total(status, spec.algorithm)
    rows = [("run", f"{spec.model}  [{spec.algorithm.upper()}]")]
    if render.styled():
        print(render.server_cost_panel(rows, total, "authoritative server quote"))
    else:
        print(f"{'run'.ljust(8)}: {rows[0][1]}")
        print(f"{'TOTAL'.ljust(8)}: {format_usd(total)}")


def _print_sft_cost(status: dict, spec) -> None:
    total = _server_quote_total(status, "sft")
    profile = status.get("workload_profile")
    rows = _sft_cost_rows(spec, profile if isinstance(profile, dict) else {})
    if render.styled():
        print(render.sft_cost_panel(rows, float(total)))
    else:
        for key, value in rows:
            if value is not None:
                print(f"{key.ljust(8)}: {value}")
        print(f"{'TOTAL'.ljust(8)}: {format_usd(total)}")
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


def cmd_train(args) -> int:
    if getattr(args, "cost", False):
        return _cmd_train_cost(args)
    spec, authored_train_keys = spec_and_train_keys_from_file(
        args.config,
        run_id=None,
        overrides=args.overrides,
        extra_configs=args.extra_configs,
        project_required=True,
    )
    payload = spec_payload(spec, authored_train_keys=authored_train_keys)
    client = client_from_config()
    client_train_schema = _client_train_schema(authored_train_keys)
    runtime_secrets = (
        runtime_secrets_from_local_env(args.config, keys=spec.environment.secrets) or None
    )
    _warn_if_wandb_requested_without_key(spec, runtime_secrets, dry_run=bool(args.dry_run))
    if args.dry_run:
        # dry-run runs submit-time server preflights without allocating a training gpu or charging
        # for training. a rejection surfaces as the server's error with exit status 1. for sft the
        # server reads the packaged dataset file and builds the quote without executing environment.py.
        status = client.create_run(
            payload,
            runtime_secrets=runtime_secrets,
            dry_run=True,
            client_train_schema=client_train_schema,
        )
        compatibility = status.pop("train_schema_compatibility", None)
        _print_train_schema_compatibility(compatibility)
        # the server fails open on a billing-infra problem, so "cost" is only in the validated list
        # when it was actually checked. absent key = a server that predates the signal, which is
        # equally not a verification -- so treat anything but an explicit True as unverified.
        affordability_verified = status.pop("affordability_verified", None) is True
        cost = "and cost" if affordability_verified else "but NOT cost"
        budget_validated = prompt_budget_validation_suffix(status)
        environment = (
            "it did NOT import or run your environment.py. packaged input/output fields and the "
            "statically readable training contract were tokenized together; tokens, retention, "
            "truncation, and steps can still miss other environment transformations. environment "
            "execution, model load, and gpu/training are first exercised on the worker after cold-start."
            if spec.algorithm == "sft"
            else "it did NOT import or run your environment.py; dataset loading, "
            "start_episode/episode shapes, reward/scorer, worker imports, model load, and "
            "gpu/training are first exercised on the worker after cold-start."
        )
        print(
            "dry-run validated: config/schema, model+algorithm compatibility, lora rank, "
            f"runtime-secret presence, warm-start source, serving context cap{budget_validated}, "
            f"{cost}. {environment}",
            file=sys.stderr,
        )
        if not affordability_verified:
            print(
                "affordability was NOT verified (billing check unavailable); this run can still be "
                "rejected for insufficient balance when you submit it for real.",
                file=sys.stderr,
            )
        if render.styled():
            print(
                render.object_panel(
                    "train", status, "dry run — validated by the server, not submitted"
                )
            )
        else:
            print(json.dumps(status, indent=2))
        if spec.algorithm == "sft":
            _print_published_sft_environment_note(status, spec)
        _print_unpacked_batch_warning(status, spec)  # after the payload, so stdout stays parseable
        print_status_prompt_budget_warning(status)
        _print_reasoning_loss_warning(status)
        return 0
    local_budget = warn_before_paid_submit(client, spec)
    status = client.create_run(
        payload,
        runtime_secrets=runtime_secrets,
        client_train_schema=client_train_schema,
    )
    run_id = status["run_id"]
    _print_unpacked_batch_warning(status, spec)  # a real submit overrides batch_size the same way
    _print_reasoning_loss_warning(status)  # and trains on whatever reasoning the template kept
    print_warmstart_context_supplement(local_budget, status)
    logger.info(
        "submitted run %s: model=%s algorithm=%s gpu=%s",
        run_id,
        spec.model,
        spec.algorithm,
        spec.gpu.type or "auto",
    )
    if args.background:
        if render.styled():
            print(render.object_panel("train", status, "submitted (running in background)"))
        else:
            print(json.dumps(status, indent=2))
        return 0
    if render.styled():
        print(render.submitted(run_id), file=sys.stderr)
    else:
        print(
            f"run {run_id} submitted; following logs (Ctrl-C detaches and the run keeps "
            f"billing, `{CLI_NAME} runs log {run_id} --follow` resumes, "
            f"`{CLI_NAME} runs cancel {run_id}` stops it)",
            file=sys.stderr,
        )
    from flash.cli.commands.ops.runs import _follow_run

    return _follow_run(client, run_id)
