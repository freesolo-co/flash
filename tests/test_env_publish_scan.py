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
