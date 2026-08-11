"""Warn about scaffold files a rerun retains from a differently-configured plane.

Split out of setup.py to keep that module under the 1000-line cap. The concern is self-contained:
setup never overwrites a file the user may have edited, so when a rerun targets the other plane
kind the only available remedy is to name the retained file that now describes the wrong workflow.

`warn` is threaded in rather than imported, because it lives in setup.py and importing it back
would make the two modules circular.
"""

import tomllib
from collections.abc import Callable
from pathlib import Path


def _warn_if_retained_starter_files_describe_another_plane(
    starter_files: tuple[Path, ...],
    *,
    can_publish: bool,
    project_id: str,
    warn: Callable[[str], None],
) -> None:
    """Warn when a retained `environment.py` / `evaluations.py` documents the other plane kind.

    The same idempotence that preserves the configs preserves these: `_for_self_hosted_plane`
    rewrites the generated docstrings, but the result is only written under `if not
    starter_env_exists`, and `evaluations.py` is nested one level deeper inside that same guard. So
    a hosted-then-self-hosted rerun keeps files telling the operator to run `flash env push` -- a
    command their plane cannot use -- while the configs and the printed next step describe the new
    plane.

    Detect by the marker each rewrite targets rather than by a plane flag, so the warning tracks
    what is actually ON DISK: an operator who already hand-edited the guidance is not warned, and a
    file the rewrite missed still is. `_for_self_hosted_plane` matches text carrying the real uuid,
    so this probe has to as well.

    Warn rather than rewrite, matching every other retained-file path in this command: these are
    files the user is expected to edit, and setup has never overwritten one.
    """
    push_marker = f"`flash env push --project {project_id} --name my-env .`"
    # the self-hosted rewrite's own replacement text, so a same-plane rerun stays silent
    self_hosted_marker = "this plane is self-hosted, so publishing"
    stale = "hosted" if can_publish else "self-hosted"
    looking_for = self_hosted_marker if can_publish else push_marker
    mismatched = [
        path
        for path in starter_files
        if path.exists() and looking_for in path.read_text(encoding="utf-8")
    ]
    if not mismatched:
        return
    names = ", ".join(str(path) for path in mismatched)
    if can_publish:
        warn(
            f"existing {names} document a self-hosted plane, but this one publishes to the managed "
            "hub; keeping the files unchanged. Their guidance to commit the environment to your own "
            "git repo does not apply here -- run `flash env push` and use the returned id instead"
        )
        return
    warn(
        f"existing {names} still tell you to run `flash env push`, which this plane cannot do; "
        f"keeping the files unchanged. Ignore that {stale} guidance: commit the environment to a "
        "git repo your plane can read and name it in [environment] id"
    )


def _warn_if_environment_form_disagrees(
    configs: tuple[Path, ...], *, can_publish: bool, warn: Callable[[str], None]
) -> None:
    """Warn when configs left over from a run against the OTHER plane kind are kept as-is.

    Setup is idempotent: the `_write_*` helpers skip a config that already exists, and
    `_validate_existing_config_projects` only compares the project uuid. So a rerun in a directory
    scaffolded against the other plane kind keeps its old `[environment]` block on disk while
    printing this plane's next step -- guidance the retained file contradicts. Warn rather than
    refuse or rewrite: refusing would break the idempotent rerun, and every helper here
    deliberately preserves what the user has edited.

    Classify with the loader's own predicates, not a `github:` prefix test. The loader accepts a
    plain `https://github.com/OWNER/REPO/...` URL as the same self-hosted form, so a prefix test
    would warn about an id this plane resolves fine.

    Three states, not two. A blank id is neither form: it is the hosted branch's own placeholder
    (`id = ""`), which `validate_spec` rejects outright. Collapsing it into "not github" would let
    the self-hosted branch tell an identity-backend operator their blank ids are "already right",
    which is the one thing they certainly are not.
    """
    from flash.client import ClientError
    from flash.envs.loader import is_github_environment_ref, is_managed_environment_slug

    github: list[Path] = []
    managed: list[Path] = []
    unfilled: list[Path] = []
    for cfg in configs:
        if not cfg.exists():
            continue
        try:
            raw = tomllib.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ClientError(f"cannot read existing {cfg}: {exc}") from exc
        environment = raw.get("environment")
        environment_id = environment.get("id") if isinstance(environment, dict) else None
        env_id = environment_id.strip() if isinstance(environment_id, str) else ""
        if is_github_environment_ref(env_id):
            github.append(cfg)
        elif is_managed_environment_slug(env_id):
            managed.append(cfg)
        else:
            unfilled.append(cfg)

    def _names(paths: list[Path]) -> str:
        return ", ".join(str(cfg) for cfg in paths)

    if can_publish:
        # An unfilled id needs no warning here: it is what this branch scaffolds, and the printed
        # next step is already `flash env push`.
        if github:
            warn(
                f"existing {_names(github)} use `github:` [environment] ids, but this hosted plane "
                "requires managed hub ids; keeping the files unchanged. Run `flash env push`, then "
                "replace each [environment] id with the returned id"
            )
        return

    if managed:
        # hedged, unlike the hosted branch above: which form a self-hosted plane can USE depends on
        # server-side FLASH_STANDALONE, which the CLI cannot read. An identity-backed plane takes
        # managed slugs, so a flat "replace it" would be advice to break a working config.
        #
        # "cannot resolve", not "rejects": standalone does not reject a slug at validation --
        # `_require_hosted_environment_form` returns early under `auth.standalone()`, and so does
        # `require_environment_project`. It fails one layer later, at fetch:
        # `managed_slug_to_github_ref` maps every slug onto `freesolo-co/environment-hub`, which is
        # an internal repo an external operator's GITHUB_TOKEN cannot read. Saying "rejects" would
        # send someone hunting for a validation error that never appears in their logs.
        warn(
            f"existing {_names(managed)} use managed hub [environment] ids, which a standalone "
            "plane accepts but cannot fetch -- they resolve to Freesolo's internal environment-hub "
            "repo; keeping the files unchanged. If this plane runs with `FLASH_STANDALONE=1`, "
            "replace each [environment] id with a `github:OWNER/REPO@REF:PATH` form pointing at a "
            "repo your plane's GITHUB_TOKEN can read; if it runs against an identity backend, "
            "managed hub ids are the accepted form and these are already right"
        )
    if unfilled:
        # Unhedged, because no plane accepts a blank id: `validate_spec` fails the submit before the
        # standalone-vs-identity question is ever reached. Both forms are still named, since which
        # one to fill in does depend on that server setting.
        warn(
            f"existing {_names(unfilled)} have no usable [environment] id, which fails validation on "
            "any plane; keeping the files unchanged. Fill each one in -- with a "
            "`github:OWNER/REPO@REF:PATH` form if this plane runs with `FLASH_STANDALONE=1`, or with "
            "a managed hub id if it runs against an identity backend"
        )
