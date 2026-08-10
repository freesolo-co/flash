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
from flash._internal.logging import get_logger
from flash.cli.ui import render
from flash.client import ApiClient, ApiError, ClientError
from flash.client.runtime_secrets import runtime_secrets_from_local_env
from flash.client.specs import spec_payload
from flash.cost.spec import runconfig_from_spec
from flash.schema import spec_and_train_keys_from_file, train_schema_metadata

logger = get_logger("flash.cli")


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

    if _measured_quote_would_differ(spec):
        # this command prices rollouts at the declared completion cap, but a dry-run or submit on
        # the same config measures them and quotes the measured length instead -- so the two
        # deliberately disagree, and this is the one meant for pre-spend decisions. it stays
        # cap-based rather than measuring: measuring imports and runs the user's environment.py and
        # can call a paid external scorer, which is not what `--cost` promises. say which number
        # this is instead of letting the difference surface as an unexplained change at submit.
        print(
            "note: this quote prices generation at the declared completion cap. A sampler key is "
            "configured, so `flash train --dry-run` measures real rollout length and can quote a "
            "different (usually lower) amount for this same config.",
            file=sys.stderr,
        )
    if spec.train.init_from_adapter:
        # --cost is offline/catalog-only and cannot read the source adapter, so the rank stays at the
        # local default. Warm starts train and are priced at the SOURCE adapter's authoritative rank
        # (resolved server-side at submit/dry-run), which can be higher — so this estimate may
        # under-quote. stderr keeps stdout clean for machine-readable callers.
        print(
            "warning: warm-start (train.init_from_adapter) cost uses the default LoRA rank; the "
            "source adapter's rank is authoritative and resolved at submit, so a higher-rank source "
            "may cost more than this estimate. Run `flash train --dry-run` for a source-rank quote.",
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


def _measured_quote_would_differ(spec) -> bool:
    """whether a dry-run on this same config would quote from measured rollouts instead.

    only true when a measurement could actually happen: the algorithm samples rollouts, a sampler
    key is configured, and nothing about the config makes a hosted draw unrepresentative. a config
    that would be declined is quoted from the cap on BOTH paths, so warning about it would describe
    a disagreement that does not exist.
    """
    from flash.core.catalog import samples_on_policy

    if not samples_on_policy(spec.algorithm):
        return False
    from flash.cli.commands.rollout_profile import unsamplable_reason
    from flash.engine.profiling.rollout_sampler import sampler_credentials

    _base_url, api_key = sampler_credentials()
    return bool(api_key) and not unsamplable_reason(spec)


def _client_train_schema(authored_train_keys: frozenset[str]) -> dict:
    return {
        "version": __version__,
        "fields": train_schema_metadata(),
        "authored_keys": sorted(authored_train_keys),
    }


def _dry_run_preview_line(
    *,
    algorithm: str,
    affordability_verified: bool,
    rollout_evidence: dict | None,
    environment_stages_run: tuple[str, ...] = (),
    rollout_evidence_accepted: bool = False,
) -> str:
    """What a dry run actually checked, and what it actually executed on this machine.

    Three-way, because what ran locally differs per path. sft required a matching workload profile
    to get this far, and that profile run already imported environment.py and tokenized the dataset,
    so claiming otherwise would understate what has been checked and already billed. A grpo/opd
    quote that reached profiling imported environment.py too. Only the path that never ran it
    locally ran nothing.

    ``environment_stages_run`` names the env hooks that actually executed, and is tracked separately
    from ``rollout_evidence`` because profiling can run the user's module and still return nothing --
    a multi-turn env, no samplable prompts, or an unreachable sampler. Both facts are things a user
    could act on: what of their code ran here, and whether this price came from a measurement.

    Per-stage rather than one flag, because the declines differ: a multi-turn env is refused after
    the import alone, and an empty dataset before prompt_messages(). Naming hooks that never ran
    would misstate exactly the disclosure this line exists to make.
    """
    # the server fails open on a billing-infra problem, so "cost" is only in the validated list
    # when it was actually checked. absent key = a server that predates the signal, which is
    # equally not a verification -- so treat anything but an explicit True as unverified.
    cost = "and cost" if affordability_verified else "but NOT cost"
    if algorithm == "sft":
        environment = (
            "your environment.py and the exact dataset were already loaded and tokenized by "
            "the workload profile this quote is built on; model load and gpu/training are "
            "first exercised on the worker after cold-start."
        )
    elif environment_stages_run:
        # what ran is reported from the stages that actually executed, not from whether a payload
        # came back. a multi-turn env, a dataset with no samplable prompts, or a dead sampler all
        # import environment.py and then return no evidence -- and they stop at different points,
        # so naming a fixed pair of hooks would overclaim on exactly those paths.
        #
        # a stage that STARTED but did not finish arrives marked and reaches this branch -- the
        # user's code ran, which is what this disclosure is about -- but is excluded from `called`.
        # a dataset() still running at the deadline did not "run on a small sample", and saying it
        # did is the same overclaim as the reverse.
        #
        # the marker is IMPORTED from the producer rather than spelled again here: two copies of a
        # sentinel is one rename away from this branch silently reading every stage as completed.
        from flash.cli.commands.rollout_profile import _ATTEMPTED_SUFFIX as _ATTEMPTED

        completed = [stage for stage in environment_stages_run if not stage.endswith(_ATTEMPTED)]
        unfinished = [
            stage.removesuffix(_ATTEMPTED)
            for stage in environment_stages_run
            if stage.endswith(_ATTEMPTED)
        ]
        unfinished = [stage for stage in unfinished if stage not in completed]
        called = [f"{stage}()" for stage in completed if stage != "import"]
        ran = f" {' and '.join(called)} ran on a small sample." if called else ""
        if unfinished:
            # named, because this is the one case where the user's code may still be RUNNING:
            # the local deadline abandons the call but the daemon thread carries on. it is also the
            # likeliest thing they can act on -- a raising import is why the quote fell to the cap.
            hooks = " and ".join(f"{stage}()" for stage in unfinished)
            ran += f" {hooks} did not finish: it raised or hit the local deadline."
        # the SERVER's verdict, not whether the client produced a payload. the client's only gate
        # is "at least one draw came back", so a deadline-truncated pass of 31 draws, or one whose
        # truncation rate or prompt coverage fails the trust gate, is evidence here and rejected
        # there -- and the run is then priced at the cap. reporting the client's optimism as the
        # outcome tells the user the opposite of what they will be billed against.
        priced = (
            "this quote is priced from the rollouts it measured"
            if rollout_evidence_accepted
            else "no usable measurement came back, so this quote still uses the declared "
            "completion cap"
        )
        # the lead clause turns on whether the import COMPLETED, not on whether it was attempted. an
        # environment.py that raises at module scope has executed -- the user needs to know that --
        # but "WAS imported" would be the same false claim in the other direction.
        opened = (
            "your environment.py WAS imported locally to measure this quote"
            if "import" in completed
            else "flash STARTED importing your environment.py locally to measure this quote"
        )
        environment = (
            f"{opened}:{ran} {priced}. "
            "reward() was NOT called, so its grading cost is not in this quote. worker imports, "
            "model load, and gpu/training are first exercised on the worker after cold-start."
        )
    else:
        environment = (
            "it did NOT import or run your environment.py; dataset loading, "
            "start_episode/episode shapes, reward/scorer, worker imports, model load, and "
            "gpu/training are first exercised on the worker after cold-start."
        )
    return (
        "dry-run validated: config/schema, model+algorithm compatibility, lora rank, "
        f"runtime-secret presence, warm-start source, serving context cap, {cost}. "
        f"{environment}"
    )


def _rollout_evidence_for(
    client: ApiClient, spec, runtime_secrets: dict | None = None
) -> tuple[dict | None, tuple[str, ...]]:
    """Measured rollout aggregates for a grpo/opd submit, or None when nothing was measured.

    Advisory, and silent on every failure. The server re-derives the digest and re-applies the same
    trust verdict a first-party measurement passes, so this can only ever supply evidence that is
    rejected or evidence that survives those checks. Returning None leaves the quote on the declared
    completion cap, which is the pricing this path had before any of it existed -- so a submit must
    never fail because a sampler was unreachable, slow, or unconfigured.

    ``runtime_secrets`` are forwarded so the env code executed locally sees the same declared
    secrets the worker will; without them an external-judge reward() raises and the measurement is
    silently lost.

    Returns the evidence and the env stages that actually ran here ("import", "dataset",
    "prompt_messages"). They are separate facts: profiling imports and runs the module before it can
    discover that a config is not measurable, so a None payload does not mean nothing ran -- and
    which hooks ran differs per decline, so one flag would overclaim.
    """
    from flash.cli.commands.rollout_profile import collect_for_submit

    executed: list[str] = []

    evidence = collect_for_submit(
        client, spec, runtime_secrets=runtime_secrets, on_environment_loaded=executed.append
    )
    if evidence:
        logger.info(
            "measured %s rollouts for the quote: mean %.0f completion tokens (cap %s)",
            evidence.get("completed_rollouts"),
            evidence.get("completion_tokens_mean") or 0.0,
            spec.train.max_completion_tokens or "recipe default",
        )
    return evidence, tuple(executed)


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
            "no exact workload profile exists for this config yet, so there is no training quote "
            "to print. the server started a separate profile run that loads your environment and "
            "tokenizes the exact dataset this training would consume.",
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
            "no exact workload profile exists for this config yet, so there is no training quote "
            f"to print. the profile run you already started is still {state}; this command "
            "launched nothing and charged nothing.",
            f"follow it with `{_commands().CLI_NAME} runs status {profile_run_id}`, then re-run this command "
            "once it reports done."
            if profile_run_id
            else "re-run this command once the profile reports done.",
        ]
    else:
        lines = [
            "no exact workload profile exists for this config yet, so there is no training quote "
            "to print. one is already being measured for this exact config and will be reused, so "
            "nothing was started or charged here.",
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
