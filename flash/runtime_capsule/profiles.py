"""Which modules each capsule profile carries."""

from __future__ import annotations

from dataclasses import dataclass

_LIFECYCLE = "flash/providers/_lifecycle"


@dataclass(frozen=True)
class Profile:
    """A named, versioned set of modules plus the entrypoint that runs them."""

    name: str
    version: int
    entrypoint: str
    # (repo-relative source path, member path inside the capsule)
    sources: tuple[tuple[str, str], ...]


# the bootstrap trio the rented-box providers execute. bootstrap.py imports its two siblings as
# bare top-level modules when it runs outside a package (its `if __package__:` branch), which is
# exactly how it is imported from inside the capsule.
_INSTANCE_BOOTSTRAP_SOURCES = (
    (f"{_LIFECYCLE}/bootstrapping/bootstrap.py", "bootstrap.py"),
    (f"{_LIFECYCLE}/bootstrapping/attempt.py", "bootstrap_attempt.py"),
    (f"{_LIFECYCLE}/bootstrapping/secrets.py", "bootstrap_secrets.py"),
    (f"{_LIFECYCLE}/bootstrapping/console.py", "bootstrap_console.py"),
    (f"{_LIFECYCLE}/bootstrapping/pip.py", "bootstrap_pip.py"),
    (f"{_LIFECYCLE}/bootstrapping/processes.py", "bootstrap_processes.py"),
    ("flash/snapshot/archive.py", "archive.py"),
)

PROFILES: dict[str, Profile] = {
    "instance-bootstrap": Profile(
        name="instance-bootstrap",
        version=7,
        entrypoint="bootstrap.py",
        sources=_INSTANCE_BOOTSTRAP_SOURCES,
    ),
}


def get_profile(name: str) -> Profile:
    """Return the named profile, or raise with the set of known names."""
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown capsule profile {name!r} (known: {known})") from None
