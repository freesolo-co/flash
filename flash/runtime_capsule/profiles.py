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
    (f"{_LIFECYCLE}/bootstrap.py", "bootstrap.py"),
    (f"{_LIFECYCLE}/bootstrap_secrets.py", "bootstrap_secrets.py"),
    (f"{_LIFECYCLE}/bootstrap_console.py", "bootstrap_console.py"),
    (f"{_LIFECYCLE}/bootstrap_pip.py", "bootstrap_pip.py"),
    ("flash/source_snapshot.py", "source_snapshot.py"),
)

# the host-side helpers the launch script invokes as standalone programs, alongside the bootstrap.
_HOST_HELPER_SOURCES = (
    (f"{_LIFECYCLE}/host/deadline_sleep.py", "deadline_sleep.py"),
    (f"{_LIFECYCLE}/host/hostlog.py", "hostlog.py"),
    (f"{_LIFECYCLE}/host/failmark.py", "failmark.py"),
)

PROFILES: dict[str, Profile] = {
    "instance-bootstrap": Profile(
        name="instance-bootstrap",
        version=2,
        entrypoint="bootstrap.py",
        sources=_INSTANCE_BOOTSTRAP_SOURCES + _HOST_HELPER_SOURCES,
    ),
}


def get_profile(name: str) -> Profile:
    """Return the named profile, or raise with the set of known names."""
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown capsule profile {name!r} (known: {known})") from None
