"""What `flash train` prints before it launches: the cost quote and the config compatibility notes.

These run against the plane's estimate endpoint, fall back to an offline calculation when there is
no backend, and print the exact-SFT rows when a workload profile already exists. The workload
profile handling is the bulky part: a pending profile is not an error the caller can act on, so it
is turned into an explanation of which run to follow and what it will be billed.

Split out of `flash.cli.commands` to keep that module under the file-size limit.
"""

from __future__ import annotations

import re
import sys

from flash import __version__
from flash.cli.ui import render
from flash.client import ApiClient, ApiError, ClientError
from flash.client.runtime_secrets import runtime_secrets_from_local_env
from flash.client.specs import spec_payload
from flash.cost.spec import runconfig_from_spec
from flash.engine.profiling.workload_profile import unpacked_batch_warning
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


# moved here with its only reader: the plane rejects a pre-schema `[train]` table with this exact
# message, and the rejection detail below turns it into a per-key explanation.
_LEGACY_TRAIN_UNKNOWN_KEYS_RE = re.compile(
    r"\A\[train\] unknown key\(s\): "
    r"(?P<keys>[A-Za-z_][A-Za-z0-9_]*(?:, [A-Za-z_][A-Za-z0-9_]*)*) "
    r"\(allowed: [A-Za-z_][A-Za-z0-9_]*(?:, [A-Za-z_][A-Za-z0-9_]*)*\)\Z"
)


def _cmd_train_cost(args) -> int:
    """`flash train --cost`: print the pre-flight USD cost for the config and exit (no submit).

    grpo and opd quote offline from the catalog. sft has no offline quote at all: its cost is
    derived from the exact tokenized dataset, and only a workload profile knows that, so sft asks
    the server for the same profile-backed quote a real submit would freeze. There is deliberately
    no analytical sft fallback -- a guessed row count is what this whole path exists to remove.
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
    return 0


def _cmd_train_cost_sft(args, spec, authored_train_keys: frozenset[str]) -> int:
    """Exact sft quote, served by the same authenticated path that freezes a real submit's quote."""
    # cli._warn_if_login_shadowed() suppresses this warning for `--cost` because the catalog path
    # never reaches an organization. the sft path does: it authenticates, resolves the project, and
    # can start a billed profile run, so the warning has to fire here after all.
    message = _commands().shadowed_login_warning()
    if message:
        print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)
    client = _commands().client_from_config()
    try:
        status = client.create_run(
            spec_payload(spec, authored_train_keys=authored_train_keys),
            runtime_secrets=runtime_secrets_from_local_env(
                args.config, keys=spec.environment.secrets
            )
            or None,
            dry_run=True,
            client_train_schema=_client_train_schema(authored_train_keys),
        )
    except ApiError as exc:
        _raise_if_workload_profile_pending(client, exc)
        detail = _legacy_train_key_rejection_detail(exc, authored_train_keys)
        if detail is None:
            raise
        raise ApiError(exc.status, detail, detail=detail) from exc
    _print_exact_sft_cost(status, spec)
    return 0


def _client_train_schema(authored_train_keys: frozenset[str]) -> dict:
    return {
        "version": __version__,
        "fields": train_schema_metadata(),
        "authored_keys": sorted(authored_train_keys),
    }


def _raise_if_workload_profile_pending(client: ApiClient, exc: ApiError) -> None:
    """Explain a profile-pending rejection and fail, or return so the caller keeps handling `exc`.

    A miss is not a validation error: the server started a real, separately billed profile run that
    tokenizes the exact dataset. Saying only "409" would leave the user with a charge they cannot
    see the reason for, and re-running blindly would look like the same request failing twice.
    """
    detail = exc.detail
    if exc.code != "workload_profile_pending" or not isinstance(detail, dict):
        return
    profile_run_id = str(detail.get("profile_run_id") or "")
    state = str(detail.get("state") or "unknown")
    # the profile id is deterministic in the workload, so this config's profile may already be
    # running under another key. that run is not readable here and is not billed here either, so
    # both the follow-up command and the charge sentence have to change.
    owned = detail.get("owned") is not False
    # only a request that WON the claim started and billed a profile. an owner re-running `train`,
    # `--cost` or `--dry-run` while its own profile is still queued/running joins that run: nothing
    # is launched and nothing is charged again, so the start-and-bill wording would name a second
    # charge that does not exist. absent reads as launched, matching `owned` above: an older server
    # omits the field, and telling a user who WAS charged that nothing happened is the worse error.
    launched = detail.get("launched") is not False
    charge = _profile_charge(client, profile_run_id) if owned and launched else None
    if owned and launched:
        lines = [
            (
                "no exact workload profile exists for this config yet, so there is no training quote "
                "to print. the server started a separate profile run that loads your environment and "
                "tokenizes the exact dataset this training would consume."
            ),
            "that profile run is real work and is billed on its own"
            + (f" (estimated ${charge:.2f})" if charge is not None else "")
            + "; no training run was created, no training gpu was allocated, and nothing was "
            "charged for training.",
            f"follow it with `{_commands().CLI_NAME} runs status {profile_run_id}`, then re-run this command "
            "once it reports done."
            if profile_run_id
            else "re-run this command once the profile reports done.",
        ]
    elif owned:
        lines = [
            (
                "no exact workload profile exists for this config yet, so there is no training quote "
                f"to print. the profile run you already started is still {state}; this command "
                "launched nothing and charged nothing."
            ),
            f"follow it with `{_commands().CLI_NAME} runs status {profile_run_id}`, then re-run this command "
            "once it reports done."
            if profile_run_id
            else "re-run this command once the profile reports done.",
        ]
    else:
        lines = [
            (
                "no exact workload profile exists for this config yet, so there is no training quote "
                "to print. one is already being measured for this exact config and will be reused, so "
                "nothing was started or charged here."
            ),
            "re-run this command in a few minutes.",
        ]
    for line in lines:
        print(render.note(line) if render.styled() else line, file=sys.stderr)
    raise ClientError(
        f"workload profile {profile_run_id or '(unknown)'} is {state}; "
        "the exact quote is available once it succeeds"
    )


def _profile_charge(client: ApiClient, profile_run_id: str) -> float | None:
    """The profile run's own quote, or None when it cannot be read (not owned by this key, etc)."""
    if not profile_run_id:
        return None
    try:
        status = client.get_run(profile_run_id)
    except (ApiError, ClientError):
        return None
    quote = status.get("estimated_cost_usd") if isinstance(status, dict) else None
    # bool is an int subclass, so an unchecked isinstance would render a JSON `true` as "$1.00" --
    # a charge the user is told to expect that no profile ever quoted.
    if not isinstance(quote, (int, float)) or isinstance(quote, bool):
        return None
    return float(quote)


def _exact_sft_cost_rows(spec, profile: dict) -> list[tuple[str, str | None]]:
    """Rows describing the exact measured workload behind an sft quote.

    Only aggregates the server actually returned are shown. A field that is absent is dropped
    rather than defaulted, so the panel never reports a count the profile did not measure.
    """

    def count(key: str) -> int | None:
        value = profile.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    steps = count("authoritative_steps")
    retained, selected = count("retained_examples"), count("selected_examples")
    dropped = count("dropped_examples")
    compute, supervised = (
        count("authoritative_compute_tokens"),
        count("authoritative_supervised_tokens"),
    )
    packing = str(profile.get("packing_mode") or "")
    architecture = str(profile.get("architecture_mode") or "")
    digest = str(profile.get("content_digest") or "")
    examples = None
    if retained is not None and selected is not None:
        examples = f"{retained:,} trained of {selected:,} selected"
        if dropped:
            examples += f" ({dropped:,} dropped)"
    tokens = None
    if compute is not None:
        tokens = f"{compute:,} compute"
        if supervised is not None:
            tokens += f", {supervised:,} supervised"
    return [
        ("run", f"{spec.model}  [SFT{f', {steps} steps' if steps is not None else ''}]"),
        ("workload", f"{packing} ({architecture})" if packing and architecture else None),
        ("examples", examples),
        ("tokens", tokens),
        ("profile", digest[:12] or None),
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


def _print_exact_sft_cost(status: dict, spec) -> None:
    total = status.get("estimated_cost_usd") if isinstance(status, dict) else None
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        raise ClientError(
            "the server accepted this SFT config but returned no cost estimate; "
            f"run `{_commands().CLI_NAME} train --dry-run` to see the full server response"
        )
    profile = status.get("workload_profile")
    rows = _exact_sft_cost_rows(spec, profile if isinstance(profile, dict) else {})
    if render.styled():
        print(render.exact_cost_panel(rows, float(total)))
    else:
        for key, value in rows:
            if value is not None:
                print(f"{key.ljust(8)}: {value}")
        print(f"{'TOTAL'.ljust(8)}: ${float(total):.2f}")
    print(
        "quoted from the exact tokenized workload, not an assumed row count. no training gpu was "
        "allocated and nothing was charged for training.",
        file=sys.stderr,
    )
    _print_unpacked_batch_warning(status, spec)


def _legacy_train_key_rejection_detail(
    exc: ApiError, authored_train_keys: frozenset[str]
) -> str | None:
    if exc.status != 400:
        return None
    match = _LEGACY_TRAIN_UNKNOWN_KEYS_RE.fullmatch(str(exc))
    if match is None:
        return None
    metadata = train_schema_metadata()
    unsupported = sorted(set(match.group("keys").split(", ")) & authored_train_keys & set(metadata))
    if not unsupported:
        return None
    declared = ", ".join(
        f"{key} (minimum released Flash version {metadata[key]})" for key in unsupported
    )
    return (
        f"{exc}. Unsupported authored [train] key(s): {declared}; "
        "client/server [train] schemas disagree"
    )


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
