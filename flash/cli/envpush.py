"""Environment publish/pull machinery for the `flash env` subcommands.

`flash env push` packages a local Freesolo environment and uploads it through the
managed Flash control plane.
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from flash._channel import CLI_NAME

from . import render
from ._tty import TtyStatusLine

if TYPE_CHECKING:
    from flash.client.http import ProgressCallback


# windows shell metacharacters. `list2cmdline` implements MS C-runtime *argv* quoting, which is what
# a program's argument parser undoes -- it is not command-line escaping, so it leaves these
# untouched and the shell would act on them before the program ever runs.
#
# the set covers cmd.exe AND powershell, because we cannot tell which one the user will paste into:
# `os.name == "nt"` says nothing about the shell, and powershell is the default terminal on current
# windows. `foo;calc` passed the cmd-only set and rendered as `--output=foo;calc`, which powershell
# splits at the semicolon and runs `calc` as its own statement (codex). the extra members --
# `;{}[]$``,'` and whitespace-adjacent `@#` -- are powershell-only operators, quoting characters,
# and expansion sigils.
_CMD_METACHARACTERS = frozenset('&|<>^()!"%' + ";{}[]$`,'@#")


def _on_windows() -> bool:
    """Whether to quote for a Windows shell.

    A module-local seam so a test can simulate Windows without assigning to ``os.name``. That
    assignment is process-wide, and on python 3.11 -- which CI runs -- ``pathlib`` reads it at
    instantiation, so every later ``Path(...)`` raised ``NotImplementedError: cannot instantiate
    'WindowsPath' on your system`` and the offline job failed before reaching the assertions. It
    does NOT raise on 3.12, so the pattern looks fine locally and only breaks in CI (codex).
    """
    return os.name == "nt"


def _quote_shell_token(token: str) -> str:
    """Quote one argv token for the shell the user is most likely to paste it back into.

    ``shlex.quote`` is POSIX-only: it wraps in single quotes, which cmd.exe passes through as
    literal characters rather than grouping, so a suggested command with a space would split
    there. This package is declared OS-independent, so pick the quoting per platform.

    Returns an empty string when the token cannot be made safely pasteable, so the caller can drop
    the copy-pasteable form rather than emit a command that would execute something else.
    """
    if not _on_windows():
        return shlex.quote(token)
    import subprocess

    # a destination named `foo&bar` comes back from list2cmdline unquoted, and pasting that hint
    # runs `bar` as a second command. there is no quoting that both survives cmd.exe and reaches
    # the C runtime intact for every one of these, so refuse to suggest a command at all rather
    # than print one that executes unintended text.
    if _CMD_METACHARACTERS & set(token):
        return ""
    return subprocess.list2cmdline([token])


def _atomic_write_bytes(out: Path, data: bytes | bytearray) -> None:
    """Write data via a sibling temp file so existing files are not truncated on failure."""
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".flash-env-pull-")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, out)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _err(msg: str) -> int:
    """Print a command error (themed red ✗ on a TTY, plain on the machine path) and return 1.

    These `env` failures `return 1` directly rather than raising, so they never reach main()'s
    themed handler; this keeps them on the same red ✗ idiom while leaving the machine path's exact
    text untouched for scripts and the env tests."""
    print(render.error(msg) if render.styled() else msg, file=sys.stderr)
    return 1


_MANAGED_HUB_ID_PART_RE = re.compile(r"[a-z0-9._-]+")
_ManagedHubIdError: TypeAlias = Literal["not-managed", "not-canonical"]


def _normalize_managed_hub_id(raw: object) -> tuple[str | None, _ManagedHubIdError | None]:
    from flash.envs.adapter import is_managed_environment_slug

    env_id = str(raw or "").strip()
    if not is_managed_environment_slug(env_id):
        return None, "not-managed"
    if env_id != env_id.lower() or not all(
        _MANAGED_HUB_ID_PART_RE.fullmatch(part) for part in env_id.split("/")
    ):
        return None, "not-canonical"
    return env_id, None


def cmd_env_pull(args) -> int:
    """Download a published Freesolo environment (or a single file from it) to local disk.

    Environments are addressed by their managed hub slug ``namespace/name`` and pulled as package
    tarballs through the authenticated Flash control plane.
    """
    from flash.client import ClientError, client_from_config
    from flash.envs.pull import (
        download_environment_file_from_archive,
        ensure_environment_pull_destination_available,
        environment_local_dirname,
        pull_environment_package_from_archive,
    )

    env_id = str(args.env_id or "").strip()
    managed_env_id, managed_error = _normalize_managed_hub_id(env_id)
    if managed_error == "not-canonical":
        return _err(
            f'env id must be lowercase "namespace/name" with no spaces (got {args.env_id!r})'
        )
    if managed_env_id is None:
        print(
            'env id must be a managed Freesolo hub slug "your-name/your-env" '
            f"(got {args.env_id!r})",
            file=sys.stderr,
        )
        return 1
    env_id = managed_env_id
    try:
        if args.path:
            positional = Path(args.path.replace("\\", "/"))
            default_name = positional.name
            out = Path(args.output) if args.output else Path(default_name)
            # the common way to land here is `flash env pull ns/env ./somedir`, meaning "put the
            # env in ./somedir". but the second positional is a path INSIDE the env, not a
            # destination -- so ./somedir became the file to fetch and its basename the output.
            #
            # test the ORIGINAL positional, not `out`: an absolute /tmp/into-here is already
            # reduced to the basename `into-here` by the time `out` exists, so comparing against
            # `out` never matches and the pull runs on to fail with an invalid-env-path error
            # instead of the hint that would explain it.
            #
            # only when -o is ABSENT, though. with an explicit `-o somedir` the user named the
            # in-env path deliberately, so telling them to drop it would abandon the file they
            # asked for and silently turn this into a whole-environment download.
            #
            # ...and only the POSITIONAL itself. a bare basename collision is not evidence the user
            # meant a destination: `env pull ns/env assets/config` with a local ./config/ is a real
            # single-file pull, and telling that user to drop `assets/config` would abandon the file
            # they asked for and aim --force at an unrelated directory.
            #
            # ...and only a positional that could BE a destination. a multi-component relative path
            # names a location inside the environment, and `assets/config` happening to exist
            # locally as a directory does not make it one: on `is_dir()` alone that real single-file
            # pull was refused, and its nonempty branch aimed a whole-env --force replace at the
            # local ./assets/config the user never mentioned. a destination is written as its own
            # name (`into here`, `./-dest`, `.`) or absolute, which is exactly `positional == out`
            # plus the absolute form that `out` has already reduced to a basename (cursor).
            #
            # a path that traverses upward is the third form. `..` cannot name anything inside the
            # environment -- `_safe_repo_relative_path` rejects every component of it -- so there is
            # no in-env reading to protect and no ambiguity to resolve. without this
            # `env pull ns/env ../into-here` downloaded the package only to fail with an invalid
            # environment path, instead of explaining `--output` (codex[bot], cursor). tested on the
            # parts rather than a leading `..` because `assets/../config` is rejected just the same.
            traverses_up = ".." in positional.parts
            could_be_destination = positional == out or positional.is_absolute() or traverses_up
            mistaken_dest = (
                positional
                if (
                    not args.output
                    and could_be_destination
                    and positional.is_dir()
                    and not positional.is_symlink()
                )
                else None
            )
            if mistaken_dest is not None:
                # attached `--output=X`, not `-o X`: a directory whose basename starts with a
                # dash (./-dest -> -dest) is read by argparse as an option and rejected with
                # "expected one argument". quoting alone does not help -- the attached form
                # does, because the value cannot start the token. quote the whole token so it
                # also survives spaces and metacharacters on the way back through a shell.
                quoted = _quote_shell_token(f"--output={mistaken_dest}")
                # ...and when even that cannot be made safe (cmd.exe metacharacters), name the
                # option without the value rather than print a command that would run something
                # else when pasted.
                suggestion = quoted or "--output=DEST"
                # the whole-environment path refuses any nonempty destination, so recommending
                # it verbatim into a populated directory just trades this error for the next.
                if any(mistaken_dest.iterdir()):
                    # ...and --force is not the answer either when the destination contains the
                    # cwd: ensure_environment_pull_destination_available() rejects that outright
                    # rather than deleting the directory the user is standing in. ask the guard
                    # itself so this cannot drift from the rule it is describing.
                    from flash.envs.pull import _cwd_is_inside

                    if _cwd_is_inside(mistaken_dest):
                        hint = (
                            f"\nhint: {mistaken_dest}{os.sep} is a directory, and the second "
                            f"positional is a path INSIDE the environment, not a destination. to "
                            f"download the whole environment, pass a destination with -o -- but not "
                            f"{mistaken_dest}{os.sep}, which contains the current working directory "
                            f"and cannot be replaced even with --force. choose a separate path: "
                            f"{CLI_NAME} env pull {env_id} --output=DEST"
                        )
                    else:
                        hint = (
                            f"\nhint: {mistaken_dest}{os.sep} is a directory, and the second "
                            f"positional is a path INSIDE the environment, not a destination. to "
                            f"download the whole environment, pass a destination with -o; "
                            f"{mistaken_dest}{os.sep} is not empty, so choose a new path or replace "
                            f"its contents: {CLI_NAME} env pull {env_id} {suggestion} --force"
                        )
                else:
                    hint = (
                        f"\nhint: to download the whole environment into {mistaken_dest}{os.sep}, "
                        f"drop the second positional and pass it as the destination: "
                        f"{CLI_NAME} env pull {env_id} {suggestion}"
                    )
                print(
                    f"refusing to overwrite directory {mistaken_dest} with a file "
                    "(a single-file pull needs -o to be a FILE path, not a directory)" + hint,
                    file=sys.stderr,
                )
                return 1
            if out.is_dir() and not out.is_symlink():
                print(
                    f"refusing to overwrite directory {out} with a file "
                    "(a single-file pull needs -o to be a FILE path, not a directory)",
                    file=sys.stderr,
                )
                return 1
            if (out.exists() or out.is_symlink()) and not args.force:
                print(f"refusing to overwrite {out} (pass --force)", file=sys.stderr)
                return 1
            package = client_from_config().download_env_package(env_id)
            data = download_environment_file_from_archive(package, args.path)
            out.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(out, data)
            if render.styled():
                print(render.env_pulled(str(out), f"{args.path} · {len(data):,} bytes"))
            else:
                print(f"pulled {args.path} from {env_id} -> {out} ({len(data):,} bytes)")
        else:
            out = Path(args.output) if args.output else Path(environment_local_dirname(env_id))
            ensure_environment_pull_destination_available(out, overwrite=args.force)
            package = client_from_config().download_env_package(env_id)
            pull_environment_package_from_archive(package, out, overwrite=args.force)
            if render.styled():
                print(render.env_pulled(f"{out}/", env_id))
            else:
                print(f"pulled {env_id} -> {out}/")
        return 0
    except FileExistsError:
        print(
            f"refusing to overwrite existing {args.output or environment_local_dirname(env_id)!r} "
            "(pass --force)",
            file=sys.stderr,
        )
        return 1
    except ClientError as exc:
        print(f"env pull failed: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"env pull failed: {exc}", file=sys.stderr)
        return 1


def cmd_env_delete(args) -> int:
    from flash.client import ClientError, client_from_config
    from flash.spec import require_project_id

    try:
        project_id = require_project_id(getattr(args, "project", None))
    except (TypeError, ValueError) as exc:
        detail = str(exc).replace("project", "project id", 1)
        return _err(f"{detail}: pass `--project <project-uuid>`")

    # Delete only targets MANAGED hub ids ("namespace/name") — not github: refs or local paths, which
    # don't live on the hub. And enforce the hub's canonical form (lowercase, no surrounding
    # whitespace) BEFORE the network call: the server's slug validator is lowercase-only, so a
    # mixed-case / padded id would otherwise make a pointless request and return a confusing 400.
    env_id, validation_error = _normalize_managed_hub_id(args.env_id)
    if validation_error == "not-managed":
        return _err(
            f'env id must be a managed Freesolo hub id "namespace/name" (got {args.env_id!r}); '
            "github refs and local paths can't be deleted from the hub"
        )
    if validation_error == "not-canonical" or env_id is None:
        return _err(
            f'env id must be lowercase "namespace/name" with no spaces (got {args.env_id!r})'
        )
    if not getattr(args, "yes", False):
        prompt = f"delete environment {env_id}? this removes it from the hub [y/N] "
        try:
            answer = input(render.warn(prompt) if render.styled() else prompt)
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("aborted; environment not deleted", file=sys.stderr)
            return 1
    try:
        result = client_from_config().delete_env(env_id, project_id=project_id)
    except ClientError as exc:
        return _err(str(exc))
    deleted = bool(result["deleted"])
    msg = f"deleted {env_id}" if deleted else f"{env_id} was not found on the hub (already deleted)"
    print(render.ok(msg) if render.styled() else msg)
    return 0


_ENV_ENTRYPOINT = "environment.py"
# the held-out evaluation sidecar (flash/envs/evaluations.py). a known filename beside the
# entrypoint rather than an environment module: never an entrypoint candidate, and carried
# alongside the entrypoint on a single-file push the way its docs and dataset are.
_ENV_EVALUATIONS_SIDECAR = "evaluations.py"
_ENV_SYSPATH_BOOTSTRAP = (
    "import os as _flash_os, sys as _flash_sys\n"
    "_flash_sys.path.insert(0, _flash_os.path.dirname(__file__))\n"
)
_ENV_PUSH_IGNORED_NAMES = frozenset(
    {
        ".prime",
        ".git",
        ".github",
        "__pycache__",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        ".ds_store",
        ".git-worktrees",
        ".env",
        ".netrc",
        ".aws",
        ".ssh",
        "venv",
        "node_modules",
        "credentials",
        "secrets",
    }
)
_ENV_PUSH_ROOT_IGNORED_NAMES = frozenset({"pyproject.toml", "source"})
_ENV_PUSH_SECRET_PATTERNS = (
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "credentials*",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
)
# secret patterns match secret data FILES only (see _ignore_env_push_path): they never prune a
# directory, so a package tree like credentials_store/ still ships, and python source is exempt,
# so helper modules like credentials.py or id_ed25519_loader.py travel instead of being silently
# dropped and breaking the worker with ModuleNotFoundError. that keeps the prefixes broad enough
# to catch backup variants (credentials_backup, id_rsa_backup) without eating code or packages.
_ENV_PUSH_CODE_SUFFIXES = frozenset({".py", ".pyi"})
_ENV_PUSH_MAX_TOTAL_BYTES = 256 * 1024 * 1024

try:
    from flash.envs.archive_policy import ARCHIVE_MEMBER_LIMIT as _ENV_PUSH_MAX_FILES
except ImportError:
    _ENV_PUSH_MAX_FILES = 5000


def _normalize_env_name(raw: str) -> str | None:
    """Normalize only the NAME segment; a namespace prefix is passed through verbatim.

    The server is the sole authority on namespace grammar and ownership — rewriting the
    namespace client-side would silently target a different (possibly forbidden) slug.
    """
    from flash.schema import normalize_env_name_segment

    text = str(raw or "").strip()
    if "/" not in text:
        return normalize_env_name_segment(text)
    parts = [part.strip() for part in text.split("/")]
    if len(parts) != 2 or not all(parts):
        return None
    name = normalize_env_name_segment(parts[1])
    if name is None:
        return None
    return f"{parts[0]}/{name}"


def _with_syspath_bootstrap(env_source: str) -> str:
    """Prepend a sys.path bootstrap so a published env can resolve shipped sibling helpers."""
    import ast

    try:
        tree = ast.parse(env_source)
    except SyntaxError:
        return _ENV_SYSPATH_BOOTSTRAP + env_source
    insert_after = 0
    body = tree.body
    i = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        insert_after = body[0].end_lineno or 0
        i = 1
    while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__":
        insert_after = body[i].end_lineno or insert_after
        i += 1
    lines = env_source.splitlines(keepends=True)
    return "".join(lines[:insert_after]) + _ENV_SYSPATH_BOOTSTRAP + "".join(lines[insert_after:])


def _raise_walk_error(error: OSError) -> None:
    """fail env packaging loudly instead of silently skipping an unreadable directory."""
    raise error


def _tar_b64(directory: Path) -> str:
    """Pack a directory into a base64 tarball, excluding caches and metadata directories."""
    import base64
    import io
    import os
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            dirs[:] = sorted(d for d in dirs if d.lower() not in _ENV_PUSH_IGNORED_NAMES)
            for name in sorted(files):
                child = root_path / name
                tar.add(child, arcname=str(child.relative_to(directory)), recursive=False)
    return base64.b64encode(buf.getvalue()).decode()


def _ignore_env_push_path(path: Path, *, env_root: Path, entrypoint: Path) -> bool:
    """Return whether a source path must not be included in an env package."""
    import fnmatch

    name = path.name
    lowered_name = name.lower()
    if path.is_symlink():
        return True
    if path.is_dir() and (path / "pyvenv.cfg").is_file():
        return True
    if path == entrypoint or (path.parent == env_root and lowered_name == _ENV_ENTRYPOINT.lower()):
        return True
    if lowered_name.startswith(".") or lowered_name in _ENV_PUSH_IGNORED_NAMES:
        return True
    if path.parent == env_root and lowered_name in _ENV_PUSH_ROOT_IGNORED_NAMES:
        return True
    if (
        path.is_file()
        and path.suffix.lower() not in _ENV_PUSH_CODE_SUFFIXES
        and any(fnmatch.fnmatchcase(lowered_name, pattern) for pattern in _ENV_PUSH_SECRET_PATTERNS)
    ):
        return True
    return not (path.is_dir() or path.is_file())


# the functions that import a module named at runtime. one definition, because the binding walk
# and the call-site matcher below must agree on what counts as one.
_DYNAMIC_IMPORT_FUNCTIONS = frozenset({"import_module", "__import__"})


def _dynamic_import_callees(tree) -> frozenset[str]:
    """Bare names that call `importlib.import_module` in this module, aliases included.

    `from importlib import import_module as load` binds the same function to a different
    identifier, so the call site reads `load("judge")` and matching on the canonical name alone
    skipped it -- leaving `judge.py` out of the archive for a suite that passes locally, its
    directory being importable there, and raises ModuleNotFoundError on its first published case
    (Cursor). The binding is in the same file as the call, so the alias is statically knowable.

    An assignment binds it just as well: `load = importlib.import_module` is not an `import ... as`
    and so appears in no import statement at all, which left the same helper unpackaged for the
    same reason (codex[bot]). Chains are followed too (`a = importlib.import_module` then `b = a`),
    in either declaration order, since nothing requires the alias to appear before its use.

    Chains resolve through a work queue rather than repeated sweeps. Re-scanning every assignment
    per pass discovers one link at a time when they are declared in reverse order -- the very order
    this supports -- so a generated 5,000-link sidecar spent ~3.4s here, twice, because `env push`
    walks again while copying (codex[bot]). Each binding is instead indexed by the name it reads
    and visited when that name becomes known, so the walk costs one visit per binding.

    Deliberately flow-insensitive: a name rebound to something else later still counts here. The
    two errors are not symmetric -- an extra name ships a file the archive did not need, while a
    missed one is a published environment that raises on its first case.

    Covers the two spellings that occur: a plain or annotated assignment. Destructuring
    (`load, x = importlib.import_module, 1`), walrus, star targets and augmented assignment are
    not followed, so a helper named only through one of those is still missed. Said plainly rather
    than left to look like full coverage -- the symptom is the ModuleNotFoundError above, and
    knowing which spellings are walked is what makes it diagnosable."""
    import ast
    from collections import deque

    names = set(_DYNAMIC_IMPORT_FUNCTIONS)
    # names bound from another name, keyed by the name they read: `b = a` waits under "a" until
    # "a" is known to be an importer, whether that happens before or after this line was parsed.
    pending: dict[str, list[str]] = {}
    newly_known: deque[str] = deque()

    def learn(name: str) -> None:
        """Record a name as an importer, and queue it so anything waiting on it can resolve."""
        if name not in names:
            names.add(name)
            newly_known.append(name)

    def bind(targets, value) -> None:
        if isinstance(value, ast.Attribute):
            # `importlib.import_module`, under any alias of the module itself. Matched on the
            # canonical attribute name only, exactly as `_dynamic_import_name` matches the call
            # site, so the two cannot disagree about what counts as a dynamic import.
            if value.attr not in _DYNAMIC_IMPORT_FUNCTIONS:
                return
            source = None  # bound straight from the importer, so it needs nothing else first
        elif isinstance(value, ast.Name):
            source = value.id  # bound from another name, which may not be known yet
        else:
            return
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if source is None:
                learn(target.id)
            else:
                pending.setdefault(source, []).append(target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    learn(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            bind(node.targets, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind([node.target], node.value)

    # seeded with the canonical names too, so `load = import_module` resolves like any other link.
    newly_known.extend(_DYNAMIC_IMPORT_FUNCTIONS)
    while newly_known:
        # each binding is released exactly once, when the name it reads becomes known, so the
        # whole walk costs one visit per binding however the chain is ordered in the source.
        for target_id in pending.pop(newly_known.popleft(), ()):
            learn(target_id)
    return frozenset(names)


def _dynamic_import_name(node, callees: frozenset[str]) -> str | None:
    """The module a literal `import_module("x")` or `__import__("x")` call names, if any.

    A sidecar importing a sibling dynamically passes local test and eval -- the scope makes the
    directory importable -- and then fails on its first published case, because the helper was
    never packaged (codex[bot]). A literal argument is statically knowable, so it is followed like
    any other import. A computed name is not, and is left to the runtime rather than guessed at."""
    import ast

    func = node.func
    if isinstance(func, ast.Attribute):
        # `importlib.import_module(...)`, under any alias of the module itself. Attribute access is
        # matched on the canonical name only: a module that happens to bind `load` at top level
        # does not make an unrelated `obj.load("x")` a dynamic import.
        matched = func.attr in _DYNAMIC_IMPORT_FUNCTIONS
    else:
        matched = getattr(func, "id", None) in callees
    if not matched:
        return None
    # both accept the module as the keyword `name`, and that form imports identically. reading
    # only the positional argument left `import_module(name="judge")` out of the archive, so the
    # suite that passed locally -- its directory is importable there -- raised ModuleNotFoundError
    # on its first published case (codex[bot]).
    first = (
        node.args[0]
        if node.args
        else next((kw.value for kw in node.keywords if kw.arg == "name"), None)
    )
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    # `import_module(".helper", package=...)` is relative; the sidecar directory is not a package,
    # so only an absolute name can name a sibling this push would carry.
    return first.value.split(".", 1)[0] or None


def _imported_module_names(tree, *, relative_names_are_siblings: bool = True) -> set[str]:
    """Top-level sibling names a parsed module imports, by statement or literal call.

    Every name returned is resolved against the env root by the caller, so a relative import
    only belongs here when the parsed file sits AT that root. Pass
    `relative_names_are_siblings=False` for a file nested deeper: `from . import config` inside
    `pkg/__init__.py` names `pkg/config.py`, which the package walk already ships, and reading it
    as a root-level name published an unrelated top-level `config.py` instead (codex[bot]).
    """
    import ast

    names: set[str] = set()
    callees = _dynamic_import_callees(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    names.add(node.module.split(".", 1)[0])
            elif node.level == 1 and relative_names_are_siblings:
                # `from . import config` and `from .utils import load` name siblings of this file
                # just as the absolute spellings do. an entrypoint written to work both as a
                # package member and as a loose module puts the relative form first and falls back
                # to the absolute one, so reading only absolute imports left the helper unpackaged
                # whenever the relative spelling came first -- and the fallback then had no file to
                # import. a bare `from . import x` carries the name in `names`, not in `module`.
                if node.module:
                    names.add(node.module.split(".", 1)[0])
                else:
                    names.update(alias.name.split(".", 1)[0] for alias in node.names)
            # level >= 2 climbs above the package this push carries, so it can name no sibling.
        elif isinstance(node, ast.Call):
            dynamic = _dynamic_import_name(node, callees)
            if dynamic:
                names.add(dynamic)
    return names


def _helper_imports(helper: Path, *, env_root: Path) -> set[str]:
    """Top-level names a packaged helper imports, for following its own dependencies.

    A helper that fails to parse is still shipped verbatim: refusing the push over a file the
    sidecar may never execute would be a harsher failure than the ModuleNotFoundError this
    lookahead exists to avoid, and the sidecar's own syntax is validated separately.

    Only a helper sitting directly at `env_root` has root-level siblings, so a relative import
    from deeper in a packaged subdirectory is dropped rather than resolved against the wrong
    directory -- the package walk that reached it already ships its true siblings."""
    import ast

    if helper.suffix != ".py":
        return set()
    try:
        return _imported_module_names(
            ast.parse(helper.read_text(encoding="utf-8"), filename=str(helper)),
            relative_names_are_siblings=helper.parent == env_root,
        )
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()


def _iter_import_closure(
    seed_modules: set[str], *, env_root: Path, entrypoint: Path, yielded: set[Path]
) -> Iterator[tuple[Path, Path]]:
    """Yield the sibling helpers reachable from `seed_modules`, following their imports too.

    A helper that imports another sibling needs that sibling too: shipping `scorer.py` without
    the `thresholds` it imports published an environment that raised ModuleNotFoundError on its
    first case, having passed every local check because the source directory was importable
    (cursor[bot]). The queue grows only through siblings actually reached, so it stays a closure
    over files this push already carries rather than a general dependency resolver -- an
    unimported neighbour stays local.

    `yielded` is shared with the caller's other walks so a helper that IS `dataset/` (or lives
    under it) is not emitted twice; `_check_env_push_limits` would otherwise charge those bytes
    and members twice and reject a tree that is actually under the limit (codex[bot]).
    """
    import os

    pending = sorted(seed_modules)
    visited: set[str] = set()
    while pending:
        module_name = pending.pop()
        if module_name in visited:
            continue
        visited.add(module_name)
        # `from graders.rules import score` resolves to a sibling PACKAGE, not graders.py.
        # matching only the module spelling published an environment whose sidecar raises
        # ModuleNotFoundError on its first case, having passed every local check.
        #
        # resolution follows python's own order, which is NOT simply "directories first".
        # a directory holding __init__.py is a regular package and outranks a same-named
        # module, so shipping the .py and skipping the package published the file the
        # sidecar never imports while dropping the one it does (cursor[bot]). but a PEP 420
        # namespace directory (no __init__.py) is only a fallback portion: `graders.py`
        # wins over a bare `graders/`. requiring the marker unconditionally sent a
        # namespace helper to the .py fallback, which did not exist either, and published
        # an archive missing the helper entirely (codex[bot]). verified by probe: with both
        # present, `import graders` binds graders.py.
        package = env_root / module_name
        helper = env_root / f"{module_name}.py"
        ships_package = (
            package.is_dir()
            and not package.is_symlink()
            and ((package / "__init__.py").is_file() or not helper.is_file())
        )
        if ships_package and not _ignore_env_push_path(
            package, env_root=env_root, entrypoint=entrypoint
        ):
            for root, dirs, files in os.walk(
                package, topdown=True, followlinks=False, onerror=_raise_walk_error
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    name
                    for name in dirs
                    if not _ignore_env_push_path(
                        root_path / name, env_root=env_root, entrypoint=entrypoint
                    )
                )
                for name in sorted(files):
                    child = root_path / name
                    if child in yielded or _ignore_env_push_path(
                        child, env_root=env_root, entrypoint=entrypoint
                    ):
                        continue
                    yielded.add(child)
                    pending.extend(_helper_imports(child, env_root=env_root))
                    yield child, child.relative_to(env_root)
            continue
        if (
            helper.is_file()
            and helper != entrypoint
            and helper not in yielded
            and not _ignore_env_push_path(helper, env_root=env_root, entrypoint=entrypoint)
        ):
            yielded.add(helper)
            pending.extend(_helper_imports(helper, env_root=env_root))
            yield helper, helper.relative_to(env_root)


def _iter_env_sidecar_files(
    env_root: Path, *, entrypoint: Path, include_full_tree: bool
) -> Iterator[tuple[Path, Path]]:
    """Yield publishable sidecar files and their package-relative paths."""
    import os

    roots = [env_root]
    # shared with the dataset walk below: the import closure can reach a helper package that IS
    # `dataset/` (or lives under it), and yielding it twice makes `_check_env_push_limits` charge
    # those bytes and members twice -- rejecting a tree that is actually under the limit
    # (codex[bot]). the copy pass is idempotent, so only the limit check ever saw the double.
    yielded: set[Path] = set()
    if not include_full_tree:
        import ast

        # a single-file push uses bounded sidecar selections. root *.toml and recursive
        # configs/**/*.toml are handled explicitly below rather than widening to neighbouring trees.
        # ship user-authored docs so the synthesized stub readme does not replace real guidance.
        for doc_name in ("README.md", "TRAINING.md"):
            doc = env_root / doc_name
            if doc.is_file() and not _ignore_env_push_path(
                doc, env_root=env_root, entrypoint=entrypoint
            ):
                yielded.add(doc)
                yield doc, doc.relative_to(env_root)
        # root config files and the conventional configs tree are runtime inputs, not a request to
        # widen an exact-file push to unrelated neighbouring trees. apply the shared ignore policy
        # and yielded set so the limit and copy passes select the identical safe members.
        for config in sorted(env_root.iterdir()):
            if (
                config.is_file()
                and config.suffix.lower() == ".toml"
                and config not in yielded
                and not _ignore_env_push_path(config, env_root=env_root, entrypoint=entrypoint)
            ):
                yielded.add(config)
                yield config, config.relative_to(env_root)
        configs = env_root / "configs"
        if configs.is_dir() and not _ignore_env_push_path(
            configs, env_root=env_root, entrypoint=entrypoint
        ):
            for root, dirs, files in os.walk(
                configs, topdown=True, followlinks=False, onerror=_raise_walk_error
            ):
                root_path = Path(root)
                dirs[:] = sorted(
                    name
                    for name in dirs
                    if not _ignore_env_push_path(
                        root_path / name, env_root=env_root, entrypoint=entrypoint
                    )
                )
                for name in sorted(files):
                    config = root_path / name
                    if (
                        config.suffix.lower() != ".toml"
                        or config in yielded
                        or _ignore_env_push_path(config, env_root=env_root, entrypoint=entrypoint)
                    ):
                        continue
                    yielded.add(config)
                    yield config, config.relative_to(env_root)
        # the entrypoint's OWN imports ship, exactly as its evaluation sidecar's do. seeding the
        # closure only from evaluations.py published an environment whose entrypoint imported
        # siblings without them: `env push` exits 0 and prints an id, so nothing surfaces until a
        # GPU has been rented and the worker imports the module. parsed with the same reader as
        # every other module here, so a lazily imported helper (inside load_environment) counts.
        try:
            entry_tree = ast.parse(entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint))
        except (OSError, SyntaxError, UnicodeDecodeError):
            # the entrypoint is validated for syntax on its own path, and a push that cannot read
            # it fails there with a better message. an unreadable file here just means no closure.
            entry_tree = None
        if entry_tree is not None:
            yield from _iter_import_closure(
                _imported_module_names(entry_tree),
                env_root=env_root,
                entrypoint=entrypoint,
                yielded=yielded,
            )
        # the eval sidecar ships with its entrypoint. `env eval` loads evaluations.py next to an
        # exact .py target, so omitting it here would publish an environment whose suite passed
        # locally and is simply absent once pushed.
        sidecar = env_root / _ENV_EVALUATIONS_SIDECAR
        if (
            sidecar.is_file()
            and sidecar != entrypoint
            and not _ignore_env_push_path(sidecar, env_root=env_root, entrypoint=entrypoint)
        ):
            # the entrypoint closure above shares `yielded` and runs first, so an entrypoint that
            # imports `evaluations` already shipped it. yielding it a second time made
            # `_check_env_push_limits` charge those bytes and that member twice, rejecting a tree
            # that is actually under the limit (codex[bot]). only the yield is skipped: the syntax
            # check and import walk below must still run either way.
            if sidecar not in yielded:
                yielded.add(sidecar)
                yield sidecar, sidecar.relative_to(env_root)
            try:
                tree = ast.parse(sidecar.read_text(encoding="utf-8"), filename=str(sidecar))
            except SyntaxError as exc:
                raise ValueError(
                    f"{sidecar}: invalid evaluation sidecar syntax: {exc.msg}"
                ) from exc
            # walk the whole tree, not just tree.body: the loader deliberately keeps the package
            # dir on sys.path so a sidecar can import its helpers lazily inside cases() or score()
            # (flash/envs/evaluations.py). scanning only top-level nodes skips exactly that
            # pattern, so `env test` passes locally and the pushed environment is missing the
            # helper the sidecar imports the moment it grades a case.
            yield from _iter_import_closure(
                _imported_module_names(tree),
                env_root=env_root,
                entrypoint=entrypoint,
                yielded=yielded,
            )
        roots = [
            env_root / name
            for name in ("dataset", "datasets")
            if (env_root / name).is_dir() and not (env_root / name).is_symlink()
        ]

    for walk_root in roots:
        for root, dirs, files in os.walk(
            walk_root, topdown=True, followlinks=False, onerror=_raise_walk_error
        ):
            root_path = Path(root)
            dirs[:] = sorted(
                name
                for name in dirs
                if not _ignore_env_push_path(
                    root_path / name, env_root=env_root, entrypoint=entrypoint
                )
            )
            for name in sorted(files):
                child = root_path / name
                if child in yielded or _ignore_env_push_path(
                    child, env_root=env_root, entrypoint=entrypoint
                ):
                    continue
                yielded.add(child)
                yield child, child.relative_to(env_root)


def _entrypoint_alias_source(module_name: str) -> str:
    """The alias module that rebinds a noncanonical entrypoint name to environment.py."""
    return (
        "# generated by `flash env push`: this environment's entrypoint was published as\n"
        f"# {_ENV_ENTRYPOINT}, so `import {module_name}` keeps resolving to the same module.\n"
        "import sys\n"
        "\n"
        "import environment as _environment\n"
        "\n"
        "sys.modules[__name__] = _environment\n"
    )


def _write_entrypoint_alias(pkg: Path, *, entrypoint: Path) -> None:
    """Keep a noncanonical entrypoint importable under the name it had locally.

    Packaging writes the entrypoint's contents as environment.py whatever it was called, so a
    sidecar doing `from custom import SCORER` -- which resolves locally, and is the only way to
    reach the entrypoint when it is not named environment.py -- raised ModuleNotFoundError once
    published (cursor[bot]). The alias rebinds sys.modules rather than re-importing, so the
    sidecar and the runner share ONE module object and one set of module-level state; importing
    the source twice would give the sidecar a second copy of every constant it scores against.

    No sibling can collide with the alias: the entrypoint lives in the package root, and
    `_ignore_env_push_path` excludes it from the sidecar walk, so its name is unclaimed."""
    if entrypoint.name == _ENV_ENTRYPOINT:
        return
    (pkg / entrypoint.name).write_text(_entrypoint_alias_source(entrypoint.stem))


def _check_env_push_limits(
    env_root: Path, *, entrypoint: Path, include_full_tree: bool, env_name: str
) -> None:
    """Reject oversized source trees before copying or building an in-memory archive."""
    files = 1
    total_bytes = entrypoint.stat().st_size + len(_ENV_SYSPATH_BOOTSTRAP.encode())
    if entrypoint.name != _ENV_ENTRYPOINT:
        # the import alias `cmd_env_push` writes beside environment.py counts against the same
        # member cap the server enforces, so it cannot be free here.
        files += 1
        total_bytes += len(_entrypoint_alias_source(entrypoint.stem).encode())
    directories: set[Path] = set()
    has_readme = False
    for child, relative in _iter_env_sidecar_files(
        env_root, entrypoint=entrypoint, include_full_tree=include_full_tree
    ):
        files += 1
        total_bytes += child.stat().st_size
        has_readme = has_readme or relative == Path("README.md")
        # the server counts every file AND directory against the same member cap when it
        # repackages the checkout for download, so count unique parent dirs here too.
        directories.update(parent for parent in relative.parents if parent != Path("."))
    if not has_readme:
        files += 1
        total_bytes += len(f"# {env_name}\n\nFlash Freesolo environment.\n".encode())

    members = files + len(directories)
    if total_bytes > _ENV_PUSH_MAX_TOTAL_BYTES:
        raise ValueError(
            f"environment package totals {_human_bytes(total_bytes)} "
            f"(limit {_human_bytes(_ENV_PUSH_MAX_TOTAL_BYTES)}); "
            "remove large artifacts or use a smaller dataset"
        )
    if members > _ENV_PUSH_MAX_FILES:
        raise ValueError(
            f"selected environment contains {members:,} files and directories "
            f"(limit {_ENV_PUSH_MAX_FILES:,}); "
            "remove large artifacts or use a smaller dataset"
        )


def _copy_env_sidecars(
    env_root: Path, dest: Path, *, entrypoint: Path, include_full_tree: bool
) -> None:
    """Copy publishable sidecar files beside environment.py, creating directories lazily."""
    import shutil

    for child, relative in _iter_env_sidecar_files(
        env_root, entrypoint=entrypoint, include_full_tree=include_full_tree
    ):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, target)


def _human_bytes(n: int) -> str:
    """Human-readable byte count."""
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class _UploadProgress(TtyStatusLine):
    """Carriage-return upload progress bar on stderr; no-op off a TTY."""

    _BAR_WIDTH = 24

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._last_pct = -1

    @property
    def callback(self) -> ProgressCallback | None:
        return self.update if self._enabled else None

    def status(self, message: str) -> None:
        if self._enabled:
            self._write(message)

    def update(self, sent: int, total: int) -> None:
        if not self._enabled:
            return
        pct = 100 if total <= 0 else min(100, sent * 100 // total)
        # avoid thousands of redraws: skip if percent unchanged mid-upload
        if pct == self._last_pct and sent < total:
            return
        self._last_pct = pct
        filled = (
            self._BAR_WIDTH if total <= 0 else min(self._BAR_WIDTH, self._BAR_WIDTH * sent // total)
        )
        bar = "#" * filled + "-" * (self._BAR_WIDTH - filled)
        self._write(
            f"uploading {self._name} [{bar}] {pct:3d}% {_human_bytes(sent)}/{_human_bytes(total)}"
        )


def _upload_and_report(
    name: str, *, package_b64: str, bar: _UploadProgress, project_id: str
) -> int:
    """Upload a packaged env to the managed control plane and print the returned id."""
    from flash.client import ClientError, client_from_config

    try:
        result = client_from_config().publish_env(
            name=name, package_b64=package_b64, project_id=project_id, progress=bar.callback
        )
    except ClientError as exc:
        bar.clear()
        return _err(str(exc))
    bar.clear()
    slug = result.get("id")
    if not slug:
        # a publish that can't report an id has not done its job (no `id = "..."` snippet to show),
        # so this is a hard failure — route it through _err's red ✗ idiom, not the amber ⚠ warning
        # idiom (which means "succeeded / still proceeding") it used to borrow.
        return _err("the env was uploaded but the server returned no id")
    if render.styled():
        print(render.env_published(slug))
    else:
        print(f"published {slug}")
        print(f'reference it in your config:\n\n  [environment]\n  id = "{slug}"')
    return 0


def _resolve_local_env_entrypoint(path: str | Path) -> tuple[Path, Path, Path, bool]:
    """Resolve a local environment path to its Python entrypoint and package root."""
    src = Path(path)
    if not src.exists():
        raise ValueError(f"no such path: {src}")
    if src.is_symlink():
        raise ValueError(f"cannot publish {src}: symlinks are not allowed")

    include_full_tree = src.is_dir()
    if include_full_tree:
        canonical_entrypoint = src / _ENV_ENTRYPOINT
        if canonical_entrypoint.is_file():
            entrypoint = canonical_entrypoint
            env_root = src
        elif (src / "pyproject.toml").is_file():
            raise ValueError(f"{src} has a pyproject.toml but no environment.py entrypoint")
        else:
            # evaluations.py is a known sidecar, never an entrypoint. counting it here would
            # make adding one break a legacy single-module package that resolved fine before:
            # the directory would suddenly hold "multiple top-level .py modules" and be rejected
            # before either file is read.
            modules = [
                p
                for p in sorted(src.glob("*.py"))
                if not p.name.startswith("__") and p.name != _ENV_EVALUATIONS_SIDECAR
            ]
            if len(modules) != 1:
                raise ValueError(
                    f"{src} has no environment.py and "
                    f"{'no' if not modules else 'multiple'} top-level .py module(s); "
                    "add an environment.py entrypoint or pass the exact .py file "
                    "for a single-file smoke test."
                )
            env_root = src
            entrypoint = modules[0]
    elif src.is_file() and src.suffix == ".py":
        env_root = src.parent
        entrypoint = src
    else:
        raise ValueError(
            f"cannot publish {src}: expected a Freesolo .py module or an env directory."
        )

    if entrypoint.is_symlink():
        raise ValueError(f"cannot publish {entrypoint}: symlinks are not allowed")
    return src, env_root, entrypoint, include_full_tree


def cmd_env_push(args) -> int:
    env_name = _normalize_env_name(str(getattr(args, "name", "") or ""))
    if not env_name:
        return _err("env name required: pass `--name <name>`")

    from flash.spec import require_project_id

    try:
        project_id = require_project_id(getattr(args, "project", None))
    except (TypeError, ValueError) as exc:
        detail = str(exc).replace("project", "project id", 1)
        return _err(f"{detail}: pass `--project <project-uuid>`")

    try:
        src, env_root, entrypoint, include_full_tree = _resolve_local_env_entrypoint(args.path)
    except ValueError as exc:
        return _err(str(exc))

    try:
        _check_env_push_limits(
            env_root,
            entrypoint=entrypoint,
            include_full_tree=include_full_tree,
            env_name=env_name,
        )
    except (OSError, ValueError) as exc:
        return _err(f"cannot publish {src}: {exc}")

    with tempfile.TemporaryDirectory(prefix="flash-env-push-") as tmp:
        pkg = Path(tmp)
        module_source = entrypoint.read_text()
        (pkg / _ENV_ENTRYPOINT).write_text(_with_syspath_bootstrap(module_source))
        _write_entrypoint_alias(pkg, entrypoint=entrypoint)
        _copy_env_sidecars(
            env_root,
            pkg,
            entrypoint=entrypoint,
            include_full_tree=include_full_tree,
        )
        # Only synthesize a stub README when the env didn't ship its own (now carried as a
        # ``.md`` sidecar) — don't clobber a user-authored README with boilerplate.
        readme = pkg / "README.md"
        if not readme.exists():
            readme.write_text(f"# {env_name}\n\nFlash Freesolo environment.\n")
        # One progress widget spans both phases the user otherwise waits through silently:
        # packaging (walk + gzip, slow for large datasets) and the upload itself.
        bar = _UploadProgress(env_name)
        bar.status("packaging environment")
        try:
            package_b64 = _tar_b64(pkg)
            return _upload_and_report(
                env_name, package_b64=package_b64, bar=bar, project_id=project_id
            )
        finally:
            bar.clear()
