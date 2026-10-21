"""Warn about scaffold files a rerun retains from a differently-configured plane.

setup never overwrites a file the user may have edited, so when a rerun targets the other plane
kind the only available remedy is to name the retained file that now describes the wrong workflow.

`warn` is threaded in rather than imported, because it lives in setup.py and importing it back
would make the two modules circular.
"""

import tomllib
from collections.abc import Callable
from pathlib import Path

from flash._internal.channel import CLI_NAME


def _text(path: Path) -> str:
    """The file's text, or "" if it cannot be decoded as UTF-8.

    This whole path only decides whether to PRINT an advisory, and setup leaves the retained files
    untouched either way, so a file it cannot read is a file it has nothing to say about. Reading
    strictly would let a valid PEP 263 latin-1 `environment.py` abort `env setup` outright.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _warn_if_retained_starter_files_describe_another_plane(
    starter_files: tuple[Path, ...],
    *,
    can_publish: bool,
    warn: Callable[[str], None],
) -> None:
    """Warn when a retained `environment.py` / `evaluations.py` documents the other plane kind.

    Detect by the marker each rewrite targets rather than by a plane flag, so the warning tracks
    what is actually ON DISK: an operator who already hand-edited the guidance is not warned, and a
    file the rewrite missed still is.

    Deliberately project-INDEPENDENT, unlike `_render_starter`, which interpolates the real uuid: a
    directory scaffolded under one project and rerun under another still holds the OLD uuid, so an
    interpolated marker would miss it. Match the project-invariant prefix instead.

    CLI-name-independent for the same reason: the guidance names the executable the operator invoked
    (`flash`, `flash-cli`, `python -m flash.cli`), so a scaffold written under the alias carries a
    different command than one written under `flash`. Anchoring on `flash` would silently stop
    recognising those files as hosted scaffolds -- the retained-file warning is the only thing that
    tells the operator their guidance now describes the wrong plane, so it must not depend on which
    entry point happened to write it.
    """
    push_marker = "env push --project "
    # `_render_starter` writes DIFFERENT guidance into each file, so the reverse direction needs
    # both markers: matching only environment.py's would warn about that file while silently
    # keeping a stale evaluations.py beside it.
    self_hosted_markers = (
        "this plane is self-hosted, so publishing",  # environment.py
        "in the git repo named by [environment] id",  # evaluations.py
    )
    wanted = self_hosted_markers if can_publish else (push_marker,)
    mismatched = [
        path
        for path in starter_files
        if path.exists()
        if any(marker in _text(path) for marker in wanted)
    ]
    if not mismatched:
        return
    names = ", ".join(str(path) for path in mismatched)
    if can_publish:
        warn(
            f"existing {names} document a self-hosted plane, but this one publishes to the managed "
            "hub; keeping the files unchanged. Their guidance to commit the environment to your own "
            f"git repo does not apply here -- run `{CLI_NAME} env push` and use the returned id "
            "instead"
        )
        return
    warn(
        f"existing {names} still tell you to run `{CLI_NAME} env push`, which this plane cannot do; "
        "keeping the files unchanged. Commit the environment to a git repo your plane can read and "
        "name it in [environment] id"
    )


def _warn_if_environment_form_disagrees(
    configs: tuple[Path, ...], *, can_publish: bool, warn: Callable[[str], None]
) -> None:
    """Warn when configs left over from a run against the OTHER plane kind are kept as-is.

    Setup is idempotent: the `_write_*` helpers skip a config that already exists, so a rerun in a
    directory scaffolded against the other plane kind keeps its old `[environment]` block on disk
    while printing this plane's next step. Warn rather than refuse or rewrite: refusing would break
    the idempotent rerun.

    Classify with the loader's own predicates, not a `github:` prefix test. The loader accepts a
    plain `https://github.com/OWNER/REPO/...` URL as the same self-hosted form, so a prefix test
    would warn about an id this plane resolves fine.

    Three states, not two: a blank id is neither form, and `validate_spec` rejects it outright.
    """
    from flash.client import ClientError
    from flash.envs.loading.loader import is_github_environment_ref, is_managed_environment_slug

    github: list[Path] = []
    managed: list[Path] = []
    unfilled: list[Path] = []
    for cfg in configs:
        if not cfg.exists():
            continue
        try:
            raw = tomllib.loads(cfg.read_text(encoding="utf-8"))
        # UnicodeDecodeError too: it is not an OSError, so without it a non-UTF-8 config escapes
        # as a traceback instead of the actionable message this branch exists to give.
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
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
        # "cannot fetch", not "rejects": a slug passes validation and fails one layer later, at
        # fetch -- `managed_slug_to_github_ref` maps every slug onto `freesolo-co/environment-hub`,
        # which an external operator's GITHUB_TOKEN cannot read.
        warn(
            f"existing {_names(managed)} use managed hub [environment] ids, which this plane "
            "accepts but cannot fetch -- they resolve to Freesolo's internal environment-hub "
            "repo; keeping the files unchanged. Replace each [environment] id with a "
            "`github:OWNER/REPO@REF:PATH` form pointing at a repo your plane's GITHUB_TOKEN can read"
        )
    if unfilled:
        warn(
            f"existing {_names(unfilled)} have no usable [environment] id, which fails validation on "
            "any plane; keeping the files unchanged. Fill each one in with a "
            "`github:OWNER/REPO@REF:PATH` form pointing at a repo your plane can read"
        )
