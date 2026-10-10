"""Which modules each capsule profile carries.

Profiles are deliberately separate and minimal rather than one capsule with everything. The
Lambda host runs its payload through a hard-capped cloud-init budget, so it must not carry
members only the RunPod handler needs; and a smaller profile is a smaller thing to audit.

Each profile names its members by repository path. The build resolves them against the repo root,
so a rename that misses this table fails the build rather than shipping a capsule with a module
missing.
"""

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
    (f"{_LIFECYCLE}/bootstrapping/secrets.py", "bootstrap_secrets.py"),
    (f"{_LIFECYCLE}/bootstrapping/console.py", "bootstrap_console.py"),
    (f"{_LIFECYCLE}/bootstrapping/pip.py", "bootstrap_pip.py"),
    (f"{_LIFECYCLE}/bootstrapping/processes.py", "bootstrap_processes.py"),
    ("flash/snapshot/archive.py", "archive.py"),
)

# the host-side helpers the launch script invokes as standalone programs, alongside the bootstrap.
_HOST_HELPER_SOURCES = (
    (f"{_LIFECYCLE}/host/deadline_sleep.py", "deadline_sleep.py"),
    (f"{_LIFECYCLE}/host/hostlog.py", "hostlog.py"),
    (f"{_LIFECYCLE}/host/failmark.py", "failmark.py"),
)

PROFILES: dict[str, Profile] = {
    # rented-box providers (Lambda, Vast): the shared bootstrap plus the host helpers the launch
    # script runs directly. one profile, because both providers execute the same programs.
    "instance-bootstrap": Profile(
        name="instance-bootstrap",
        version=4,
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
