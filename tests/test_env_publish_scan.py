"""The credential scan the control plane runs before committing an environment to the hub.

The CLI runs the same check before uploading, but the CLI is not the trust boundary: an older
client, a direct `Client.publish_env`, or a raw `POST /v1/envs` arrives here having skipped it
entirely. The server is what writes to the shared hub, whose history is permanent.
"""

from __future__ import annotations

import pytest

envs = pytest.importorskip("flash.server.domain.envs", reason="server extra not installed")

# assembled rather than written whole so the literal never exists in the repository.
FREESOLO_KEY = "fslo_" + "A1bCdEfGhIjKlMnOpQrS"


def test_a_package_carrying_a_credential_is_refused(tmp_path):
    """The server rescans the extracted tree, so a client that skipped its own check still fails."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "environment.py").write_text("def build():\n    return None\n")
    (package / "env.sh").write_text(f"export FREESOLO_API_KEY={FREESOLO_KEY}\n")
    with pytest.raises(envs.EnvPublishError) as caught:
        envs._reject_credentials(package)
    assert FREESOLO_KEY not in str(caught.value)


def test_a_clean_package_passes(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "environment.py").write_text("def build():\n    return None\n")
    envs._reject_credentials(package)


def test_a_credential_in_the_env_name_is_refused():
    """The name becomes the hub path and the commit message, so it is published just as
    permanently as a file, and it never reaches the package scan."""
    with pytest.raises(envs.EnvPublishError):
        envs._reject_credential_name(f"env-{FREESOLO_KEY}")


def test_an_ordinary_env_name_passes():
    envs._reject_credential_name("my-training-env")


def test_a_name_normalising_into_a_credential_is_refused():
    """Normalisation is what reaches the hub: it folds separators, so a name the patterns reject
    can become one they match."""
    body = "A1bCdEfGhIjKlMnOpQrS"
    with pytest.raises(envs.EnvPublishError):
        envs._reject_credential_name("fslo_" + body[:5] + "!" + body[5:])


def test_a_qualified_id_is_normalised_per_segment():
    """`publish_slug_for_name` writes the separators as directory boundaries, so no token spans
    them. Folding the whole id into one string welded unrelated segments into a credential nobody
    published."""
    envs._reject_credential_name("acme/fslo_/AbCdEf0123456789")


def test_an_over_long_name_is_refused():
    with pytest.raises(envs.EnvPublishError):
        envs.validate_publish_inputs(package_b64="", name="x" * (envs._MAX_ENV_NAME_CHARS + 1))


def test_an_unreadable_member_names_the_file_and_survives_a_bare_oserror(tmp_path, monkeypatch):
    """A member the scan cannot open is a refusal, and the refusal has to say which member.

    The message previously carried only `exc.strerror`, which names the errno and not the file, so
    a publisher was told a file could not be read without being told which one. `strerror` is also
    unset on an `OSError` raised without an errno, which rendered the refusal as "...: None".
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "environment.py").write_text("def build():\n    return None\n")
    locked = package / "secrets" / "config.json"
    locked.parent.mkdir()
    locked.write_text("{}")

    def refuse_to_open(*_args, **_kwargs):
        raise OSError("no errno on this one")

    monkeypatch.setattr(envs, "reject_credential_bearing_package", refuse_to_open)
    with pytest.raises(envs.EnvPublishError) as caught:
        envs._reject_credentials(package)
    assert "None" not in str(caught.value)
    assert "could not be read to check it for credentials" in str(caught.value)


def test_an_unreadable_member_is_named_relative_to_the_package(tmp_path, monkeypatch):
    """The absolute path is the control plane's staging directory, which is the server's business
    rather than the publisher's, so only the package-relative part is shown."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "environment.py").write_text("def build():\n    return None\n")

    def refuse_to_open(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", str(package / "secrets" / "config.json"))

    monkeypatch.setattr(envs, "reject_credential_bearing_package", refuse_to_open)
    with pytest.raises(envs.EnvPublishError) as caught:
        envs._reject_credentials(package)
    assert "secrets/config.json" in str(caught.value)
    assert str(tmp_path) not in str(caught.value), "the staging path must not be echoed"


def test_an_unreadable_member_name_holding_a_credential_is_redacted(tmp_path, monkeypatch):
    """A member name can itself be the credential, so naming it in the refusal has to redact it
    the same way every other message in the scan does."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "environment.py").write_text("def build():\n    return None\n")

    def refuse_to_open(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", str(package / f"{FREESOLO_KEY}.json"))

    monkeypatch.setattr(envs, "reject_credential_bearing_package", refuse_to_open)
    with pytest.raises(envs.EnvPublishError) as caught:
        envs._reject_credentials(package)
    message = str(caught.value)
    assert FREESOLO_KEY not in message
    # the member has to be named, masked -- asserting only the absence of the key would also pass
    # on a message that never mentioned the file, which is the bug this pairs with.
    assert ".json" in message
