"""No prose comment in `flash/` may be parsed as a PEP 484 type comment.

A comment whose first token is `type:` is not a comment to mypy, it is a type annotation. When
prose wraps so a continuation line starts that way -- as it did in `flash/_internal/channel.py`
after "tell the operator what to" / "type: the styled wordmark" -- mypy reports `Invalid syntax`
and STOPS: `errors prevented further checking`. Nothing after it in the tree is checked.

That failure is invisible by construction. The `mypy (advisory)` job ends in `|| true` and prints
mypy's own summary line, so the run reports `Found 1 error in 1 file` and stays green while the
real count (412 across 95 files) goes unmeasured and the 90% of functions carrying return
annotations stop being verified. The gate reads exactly the same as a healthy one.

The rule mypy applies, established by probing it directly rather than assumed:

  * the check is on the first token after `#`, so leading whitespace does not help --
    `#  type: ...` is misparsed exactly like `# type: ...`;
  * it is case-sensitive: `# Type: ...` is ordinary prose;
  * `# type: ignore` is a real directive and must keep working;
  * at module level the error ABORTS the whole run; inside a function body it is reported
    against the enclosing `def` and checking continues. Both are worth preventing, so this
    scans every comment rather than only module-level ones.

Tokenizing rather than grepping is what makes the scan trustworthy: `tokenize` yields only real
COMMENT tokens, so a string literal or docstring that happens to contain the same characters
cannot trip it, and a genuinely misparsed comment cannot hide inside one.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "flash"

SKIP_DIRS = {"__pycache__", ".venv", ".ruff_cache", "build"}

# Comment bodies that ARE legitimate tooling directives rather than prose. `type: ignore` is the
# only `type:` form the codebase uses on purpose; anything else starting with `type:` is prose
# that mypy will misread.
_ALLOWED = re.compile(r"^type:\s*ignore\b")

# mypy strips the `#` and any following whitespace before testing for the `type:` prefix, so the
# scan must too -- otherwise `#  type: ...` (two spaces) reads as safe here while mypy still
# misparses it.
_TYPE_PREFIX = re.compile(r"^type:")


def _source_files() -> list[Path]:
    paths = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        paths.append(path)
    assert paths, f"no python sources found under {PACKAGE_DIR}"
    return paths


def _misparsed_comments(path: Path) -> list[tuple[int, str]]:
    """Every comment in `path` that mypy would read as a type comment rather than prose."""
    source = path.read_text(encoding="utf-8")
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        body = token.string.lstrip("#").lstrip()
        if _TYPE_PREFIX.match(body) and not _ALLOWED.match(body):
            found.append((token.start[0], token.string.strip()))
    return found


def test_no_prose_comment_is_read_as_a_type_comment():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}  {text}"
        for path in _source_files()
        for line, text in _misparsed_comments(path)
    ]
    assert not offenders, (
        "comment(s) mypy will parse as a type comment, not prose:\n  "
        + "\n  ".join(offenders)
        + "\n\nRewrap the line so it does not begin with `type:` (moving the preceding word down "
        "is usually enough). At module level this aborts the entire mypy run."
    )


@pytest.mark.parametrize(
    ("comment", "misparsed"),
    [
        ("# type: the styled wordmark, `flash version`", True),
        # mypy skips whitespace after `#`, so extra spaces do not make prose safe.
        ("#  type: the styled wordmark", True),
        # Case-sensitive: capitalized prose is never a type comment.
        ("# Type: the styled wordmark", False),
        # The one `type:` form that is a real directive.
        ("# type: ignore[attr-defined]", False),
        ("# ordinary prose about a type: something", False),
    ],
)
def test_detector_matches_mypys_own_rule(comment: str, misparsed: bool, tmp_path: Path):
    """The detector agrees with the behavior probed out of mypy itself.

    Without this, the scan above could be quietly wrong in either direction -- missing the
    two-space form that actually breaks, or rejecting `# type: ignore`, which would make the
    guard unlivable and get it deleted.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(f"X = 1\n{comment}\nY = 2\n", encoding="utf-8")
    assert bool(_misparsed_comments(probe)) is misparsed


def test_detector_flags_the_regression_that_prompted_this_guard(tmp_path: Path):
    """The exact wrap from `flash/_internal/channel.py` is caught if it comes back."""
    probe = tmp_path / "channel_like.py"
    probe.write_text(
        "# The product's name, for places that IDENTIFY the tool rather than tell the operator "
        "what to\n"
        "# type: the styled wordmark, `flash version`, `--version`.\n"
        'BRAND_NAME = "flash"\n',
        encoding="utf-8",
    )
    found = _misparsed_comments(probe)
    assert [line for line, _ in found] == [2]
