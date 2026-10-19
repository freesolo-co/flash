"""Runtime capsule: determinism, execution, and every rejection path.

The rejection tests carry the weight here. A verifier that accepts everything is indistinguishable
from no verifier at all, so each way a capsule can be wrong gets its own case that proves the
build actually refuses it.
"""

from __future__ import annotations

import inspect
import io
import json
import re
import subprocess
import sys
import zipfile

import pytest

from flash.providers._lifecycle.instances import instance
from flash.providers.vast.jobs import builders as vast_builders
from flash.runtime_capsule import (
    MANIFEST_NAME,
    CapsuleError,
    build_capsule,
    read_capsule,
    sha256_bytes,
    verify_capsule,
    write_capsule,
)
from flash.runtime_capsule.manifest import parse_manifest, validate_member_path
from flash.runtime_capsule.profiles import get_profile

PROFILE = "instance-bootstrap"


def _repack(archive: bytes, mutate) -> bytes:
    """Rebuild an archive after `mutate` edits its {name: bytes} payload."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        payload = {name: zf.read(name) for name in zf.namelist()}
    mutate(payload)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(payload):
            zf.writestr(name, payload[name])
    return buffer.getvalue()


def test_capsule_build_is_deterministic():
    """Same sources, same bytes -- a digest stamped at image build must still match at run time."""
    first, _ = build_capsule(PROFILE)
    second, _ = build_capsule(PROFILE)
    assert first == second


def test_capsule_declares_every_member_with_a_digest():
    archive, manifest = build_capsule(PROFILE)
    _parsed, contents = read_capsule(archive)
    assert {m.path for m in manifest.members} == set(contents)
    for member in manifest.members:
        assert member.size == len(contents[member.path])
        assert member.sha256 == sha256_bytes(contents[member.path])


def test_capsule_verifies_against_an_external_digest():
    archive, _ = build_capsule(PROFILE)
    assert (
        verify_capsule(
            archive, expected_sha256=sha256_bytes(archive), expected_profile=PROFILE
        ).profile
        == PROFILE
    )


def test_capsule_rejects_a_wrong_external_digest():
    """The digest comes from the trusted payload, so a mismatch means the bytes are not ours."""
    archive, _ = build_capsule(PROFILE)
    with pytest.raises(CapsuleError, match="digest mismatch"):
        verify_capsule(archive, expected_sha256="0" * 64)


def test_capsule_rejects_a_mutated_member():
    """A SAME-LENGTH edit, so the size check cannot be what catches it.

    Length-preserving is the realistic tampering shape (flip a comparison, swap a constant); a
    longer payload would trip the cheaper size check and leave the digest check unproven.
    """
    archive, _ = build_capsule(PROFILE)

    def mutate(payload):
        original = payload["bootstrap_pip.py"]
        payload["bootstrap_pip.py"] = b"#" + original[1:]
        assert len(payload["bootstrap_pip.py"]) == len(original)

    with pytest.raises(CapsuleError, match="sha256 mismatch"):
        read_capsule(_repack(archive, mutate))


def test_capsule_rejects_a_truncated_member():
    archive, _ = build_capsule(PROFILE)

    def mutate(payload):
        payload["bootstrap_pip.py"] = payload["bootstrap_pip.py"][:-20]

    with pytest.raises(CapsuleError, match="size mismatch"):
        read_capsule(_repack(archive, mutate))


def test_capsule_rejects_a_missing_member():
    archive, _ = build_capsule(PROFILE)

    def mutate(payload):
        payload.pop("bootstrap_pip.py")

    with pytest.raises(CapsuleError, match="missing declared member"):
        read_capsule(_repack(archive, mutate))


def test_capsule_rejects_an_undeclared_member():
    """The direction that matters most: an ADDED module the manifest never named.

    Checking only that declared members match would let this through, and the loader would then
    have an importable module nobody reviewed sitting next to the real ones.
    """
    archive, _ = build_capsule(PROFILE)

    def mutate(payload):
        payload["sitecustomize.py"] = b"import os\nos.environ['OWNED'] = '1'\n"

    with pytest.raises(CapsuleError, match="undeclared member"):
        read_capsule(_repack(archive, mutate))


def test_capsule_rejects_a_consistently_replaced_manifest_only_by_digest():
    """A manifest replaced ALONGSIDE its members is internally consistent.

    This is exactly why the expected digest lives outside the archive: read_capsule cannot catch
    it, and verify_capsule must.
    """
    archive, _ = build_capsule(PROFILE)

    def mutate(payload):
        payload["bootstrap_pip.py"] = b"# replaced wholesale\n"
        manifest = json.loads(payload[MANIFEST_NAME].decode())
        for member in manifest["members"]:
            if member["path"] == "bootstrap_pip.py":
                member["size"] = len(payload["bootstrap_pip.py"])
                member["sha256"] = sha256_bytes(payload["bootstrap_pip.py"])
        payload[MANIFEST_NAME] = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    forged = _repack(archive, mutate)
    # internally consistent: the cheap check passes
    read_capsule(forged)
    # but it is not the archive the control plane published
    with pytest.raises(CapsuleError, match="digest mismatch"):
        verify_capsule(forged, expected_sha256=sha256_bytes(archive))


def test_capsule_rejects_a_wrong_profile():
    archive, _ = build_capsule(PROFILE)
    with pytest.raises(CapsuleError, match="profile mismatch"):
        verify_capsule(archive, expected_sha256=sha256_bytes(archive), expected_profile="verl-sft")


def test_capsule_rejects_a_symlink_member():
    archive, _ = build_capsule(PROFILE)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        payload = {name: zf.read(name) for name in zf.namelist()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(payload):
            zf.writestr(name, payload[name])
        info = zipfile.ZipInfo("evil.py")
        info.external_attr = (0o120777 << 16) | 0o200000
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(CapsuleError, match="symlink"):
        read_capsule(buffer.getvalue())


def test_capsule_rejects_a_non_zip_payload():
    with pytest.raises(CapsuleError, match="not a readable zip"):
        read_capsule(b"this is not an archive at all")


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../escape.py",
        "a/../../escape.py",
        "dir\\evil.py",
        "c:evil.py",
        "a//b.py",
        "./b.py",
        "",
    ],
)
def test_member_paths_that_escape_or_alias_are_rejected(path):
    with pytest.raises(CapsuleError):
        validate_member_path(path)


def test_manifest_rejects_an_unknown_format_version():
    """An unknown version is refused, never best-effort parsed."""
    archive, _ = build_capsule(PROFILE)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME).decode())
    manifest["format_version"] = 99
    with pytest.raises(CapsuleError, match="unsupported capsule format_version"):
        parse_manifest((json.dumps(manifest) + "\n").encode())


def test_manifest_rejects_an_unknown_field():
    archive, _ = build_capsule(PROFILE)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME).decode())
    manifest["run_this_too"] = "payload.py"
    with pytest.raises(CapsuleError, match="unknown field"):
        parse_manifest((json.dumps(manifest) + "\n").encode())


def test_manifest_rejects_a_duplicated_member():
    archive, _ = build_capsule(PROFILE)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME).decode())
    manifest["members"].append(dict(manifest["members"][0]))
    with pytest.raises(CapsuleError, match="more than once"):
        parse_manifest((json.dumps(manifest) + "\n").encode())


def test_manifest_rejects_an_entrypoint_that_is_not_a_member():
    archive, _ = build_capsule(PROFILE)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME).decode())
    manifest["entrypoint"] = "not_shipped.py"
    with pytest.raises(CapsuleError, match="not a declared member"):
        parse_manifest((json.dumps(manifest) + "\n").encode())


def test_capsule_members_import_with_flash_absent(tmp_path):
    """The point of the capsule: the far end runs it with stock Python and no flash package.

    Imports run under `-S` with the repo off sys.path, so a member that secretly needed the
    installed flash package would fail here rather than on a paid GPU.
    """
    archive, _ = build_capsule(PROFILE)
    capsule = tmp_path / "instance-bootstrap.pyz"
    write_capsule(capsule, archive)

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import bootstrap\n"
        "assert not bootstrap.__package__, bootstrap.__package__\n"
        "assert callable(bootstrap.main)\n"
        "import pickle\n"
        "assert pickle.loads(pickle.dumps(bootstrap._console_upload_loop)) "
        "is bootstrap._console_upload_loop\n"
        # EVERY non-entry member, deadline_sleep included. Importing a member must be INERT: these
        # are programs, and an unguarded one reads argv, reads the box payload, and can exit the
        # importing process or fire a real HF upload. Anything that walks flash's modules (a
        # packaging probe, a test collector) imports them, so a missing guard is a live defect --
        # and an omission HERE is how it reaches main.
        "for name in ('bootstrap_secrets', 'bootstrap_pip', 'deadline_sleep', 'hostlog', "
        "'failmark'):\n"
        "    importlib.import_module(name)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(probe), str(capsule)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_importing_any_member_is_inert(tmp_path):
    """No member may DO anything when imported -- derived from the profile, not a hand list.

    The helpers are programs; the capsule runs them with `runpy.run_module(run_name="__main__")`.
    Without a `__main__` guard, an ordinary import runs the program: deadline_sleep exits the
    importing process with 125, and hostlog attempts a real HF upload while its blanket `except`
    hides that it tried. `tests/test_client_import_purity.py` imports every flash module, so this
    is a live defect, not a hypothetical.

    Enumerated from the profile so a member added later is covered automatically -- a hand-written
    list is exactly how deadline_sleep slipped through in the first place.
    """
    from flash.runtime_capsule import get_profile

    members = [
        member.removesuffix(".py")
        for _source, member in get_profile(PROFILE).sources
        if member.endswith(".py")
    ]
    assert len(members) > 3, members

    archive, _ = build_capsule(PROFILE)
    capsule = tmp_path / "capsule.pyz"
    write_capsule(capsule, archive)
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "for name in sys.argv[2:]:\n"
        "    importlib.import_module(name)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(probe), str(capsule), *members],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    # a program that runs on import exits non-zero (or hangs); the entrypoint is included, because
    # importing the bootstrap must not start a training run either.
    assert result.returncode == 0, f"a member ran on import:\n{result.stderr}"
    assert "OK" in result.stdout


def test_every_program_the_launch_scripts_invoke_is_a_capsule_member():
    """Each `capsule.pyz <name>` in the provider launch text must name something the capsule runs.

    The dispatcher test below proves an unknown name is REFUSED. That is only half the contract: it
    says nothing about whether the names the launch scripts actually pass are known ones. Before the
    capsule the two were the same text -- a helper was a heredoc written right where it was invoked,
    so they could not desync. Now the name is a bare string in a shell line and the code is a member
    resolved at run time, and nothing structural holds them together.

    A rename that misses a call site is invisible locally: two of the invocations end in
    `|| true` (hostlog) or `|| true` inside `fail()` (failmark), so on a rented box the program
    simply never runs and the launch continues, losing the console log or the failure marker with no
    error anywhere.
    """
    invoked = set()
    for module in (instance, vast_builders):
        for name in re.findall(r"capsule\.pyz ([a-z_]+)", inspect.getsource(module)):
            invoked.add(name)
    # the sources are (repo path, member path); a program name is the member stem.
    profile = get_profile(PROFILE)
    runnable = {member.removesuffix(".py") for _, member in profile.sources}

    assert invoked, "found no capsule invocations to check -- the regex stopped matching"
    assert invoked <= runnable, f"launch text invokes non-members: {sorted(invoked - runnable)}"


@pytest.mark.parametrize(
    "argv",
    [
        ["nonesuch"],  # a plain typo
        ["hostlogg"],  # a near-miss on a real member
        ["--run=nonesuch"],  # the explicit form
        ["bootstrap.py"],  # the member PATH rather than its program name
    ],
)
def test_capsule_refuses_an_unknown_program_instead_of_running_the_default(tmp_path, argv):
    """An unrecognized program name must exit non-zero, NOT fall through to the entrypoint.

    The entrypoint is the training bootstrap: it installs packages, fetches code from HF, and starts
    a run. Treating an unknown first argument as an argument TO that default means a typo'd or
    stale launch line silently starts training on a box rented for something else, and the exit
    status still looks like the host helper succeeded.
    """
    archive, _ = build_capsule(PROFILE)
    capsule = tmp_path / "capsule.pyz"
    write_capsule(capsule, archive)

    result = subprocess.run(
        [sys.executable, "-S", str(capsule), *argv],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode != 0, result.stdout
    assert "unknown program" in result.stderr, result.stderr
    # the refusal names what IS runnable, so the fix is obvious from the box's log.
    assert "bootstrap" in result.stderr


def test_capsule_runs_a_named_host_helper_and_passes_its_arguments(tmp_path):
    """The dispatch is only fail-closed if the real names still work -- and still get their args.

    deadline_sleep takes the number of seconds to sleep, so an argv that arrives shifted (or not at
    all) turns every bounded wait on the box into an immediate return or a hang.
    """
    archive, _ = build_capsule(PROFILE)
    capsule = tmp_path / "capsule.pyz"
    write_capsule(capsule, archive)
    (tmp_path / "payload.json").write_text("{}")

    # no payload at the box path -> the helper exits 125 (its "cannot read the deadline" code)
    # rather than sleeping. What matters here is that it RAN: an unknown-name refusal exits 1 with
    # "unknown program", and the default (bootstrap) would fail differently again.
    result = subprocess.run(
        [sys.executable, "-S", str(capsule), "deadline_sleep", "0"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert "unknown program" not in result.stderr, result.stderr
    assert result.returncode == 125, (result.returncode, result.stderr)


def test_capsule_entry_does_not_own_spawn_targets():
    """`__main__` must only dispatch.

    multiprocessing's spawn start method re-imports the entry as `__mp_main__`, so a target
    defined in a zipapp's `__main__` cannot be unpickled by the child -- the bootstrap's console
    uploader would die on a rented box while every local test passed.
    """
    archive, _ = build_capsule(PROFILE)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        main_src = zf.read("__main__.py").decode()
    assert "def " not in main_src
    assert "runpy.run_module" in main_src


def test_write_capsule_is_atomic_and_executable(tmp_path):
    archive, _ = build_capsule(PROFILE)
    destination = tmp_path / "nested" / "capsule.pyz"
    write_capsule(destination, archive)
    assert destination.read_bytes() == archive
    assert destination.stat().st_mode & 0o111
    # no temp files left behind
    assert sorted(p.name for p in destination.parent.iterdir()) == ["capsule.pyz"]
