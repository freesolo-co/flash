"""The credential content scan that `flash env push` runs over the staged package.

Filename filters answer the wrong question: `env.sh` and a helper `.py` are named like ordinary
tooling, so a key in either published untouched. These cover the scan that reads what is about to
be uploaded, plus the false-positive controls that keep an ordinary environment publishable.
"""

from __future__ import annotations

import gzip
import io
import tarfile
import time
import zipfile

import pytest

from flash.cli.commands.env.push import _check_env_push_credentials
from flash.envscan.secrets import _Unscannable, credential_in_file, credential_in_name

# assembled rather than written whole so the literal never exists in the repository: push
# protection blocks some complete token shapes, and a fixture is not worth a blocked push.
FREESOLO_KEY = "fslo_" + "A1bCdEfGhIjKlMnOpQrS"
HF_TOKEN = "hf_" + "aBcDeFgHiJkLmNoPqRsTuVwXyZ012345"


def _staged(tmp_path, name: str, body: str):
    """A staged package holding one ordinary entrypoint plus `name`."""
    pkg = tmp_path / "pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "environment.py").write_text("def build():\n    return None\n")
    (pkg / name).write_text(body)
    return pkg


def _refusal(pkg) -> str:
    from pathlib import Path

    with pytest.raises(ValueError, match=r"credential|key|token") as caught:
        _check_env_push_credentials(pkg, entrypoint=Path("environment.py"))
    return str(caught.value)


# --- the original leak -------------------------------------------------------------------------


def test_sourceable_shell_file_contents_are_refused(tmp_path):
    """`env.sh` is dropped by no filename pattern, so only a content scan catches it."""
    message = _refusal(_staged(tmp_path, "env.sh", f"export FREESOLO_API_KEY={FREESOLO_KEY}\n"))
    assert "env.sh" in message
    assert "Freesolo API key" in message


def test_python_helper_contents_are_refused(tmp_path):
    """Python source is exempt from the name filter by design, so a key in one had nothing between
    it and the hub."""
    message = _refusal(_staged(tmp_path, "helper.py", f'TOKEN = "{FREESOLO_KEY}"\n'))
    assert "helper.py" in message


def test_generated_readme_contents_are_refused(tmp_path):
    """The README is synthesized during staging, so scanning the source tree never saw it."""
    message = _refusal(_staged(tmp_path, "README.md", f"run with {FREESOLO_KEY}\n"))
    assert "README.md" in message


def test_third_party_tokens_are_refused(tmp_path):
    """Coverage is not Freesolo-only: a published hub repo leaks any issuer's key just as far."""
    assert "Hugging Face" in _refusal(
        _staged(tmp_path, "setup.sh", f"export HF_TOKEN={HF_TOKEN}\n")
    )


def test_the_refusal_never_echoes_the_secret(tmp_path):
    """The message is printed to terminals and collected logs, so it names the category only."""
    message = _refusal(_staged(tmp_path, "env.sh", f"export FREESOLO_API_KEY={FREESOLO_KEY}\n"))
    assert FREESOLO_KEY not in message
    assert "rotate" in message.lower()


# --- false-positive controls -------------------------------------------------------------------


def test_a_clean_package_still_publishes(tmp_path):
    """The check has to be invisible to an ordinary environment or it is not shippable."""
    from pathlib import Path

    pkg = _staged(tmp_path, "train.py", "import torch\n\nprint('training')\n")
    _check_env_push_credentials(pkg, entrypoint=Path("environment.py"))


@pytest.mark.parametrize(
    "body",
    [
        "export FREESOLO_API_KEY=fslo_xxxxxxxxxxxxxxxxxxxx\n",
        "export FREESOLO_API_KEY=<your-key-here>\n",
        "from huggingface_hub import hf_hub_download\n",
        "set your FREESOLO_API_KEY before running this environment\n",
    ],
)
def test_placeholders_and_prose_publish(tmp_path, body):
    """A placeholder is what documentation looks like; refusing it would train people to bypass."""
    from pathlib import Path

    _check_env_push_credentials(
        _staged(tmp_path, "README.md", body), entrypoint=Path("environment.py")
    )


# --- what a scan cannot read is refused, never reported clean ----------------------------------


def test_a_credential_inside_a_gzip_member_is_found(tmp_path):
    """A compressed member holds its key nowhere in its own bytes."""
    path = tmp_path / "data.gz"
    path.write_bytes(gzip.compress(f"export FREESOLO_API_KEY={FREESOLO_KEY}".encode()))
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_credential_inside_a_zip_member_is_found(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("env.sh", f"export FREESOLO_API_KEY={FREESOLO_KEY}")
    path = tmp_path / "bundle.zip"
    path.write_bytes(buf.getvalue())
    assert credential_in_file(path) == "a Freesolo API key"


def test_a_credential_inside_a_tar_member_is_found(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        payload = f"export FREESOLO_API_KEY={FREESOLO_KEY}".encode()
        info = tarfile.TarInfo("env.sh")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    path = tmp_path / "bundle.tar"
    path.write_bytes(buf.getvalue())
    assert credential_in_file(path) == "a Freesolo API key"


@pytest.mark.parametrize(
    ("name", "data", "expected"),
    [
        ("shard.parquet", b"PAR1" + b"\x00" * 64 + b"PAR1", "Parquet"),
        ("blob.zst", b"\x28\xb5\x2f\xfd" + b"\x00" * 64, "zstd"),
    ],
)
def test_a_recognised_but_unexpandable_format_is_refused(tmp_path, name, data, expected):
    """Unverifiable is not clean. Reporting a container clean because nothing opened it is the one
    outcome that publishes a key."""
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(_Unscannable) as caught:
        credential_in_file(path)
    assert expected in str(caught.value)


def test_a_filename_alone_does_not_refuse(tmp_path):
    """Naming a file `.parquet` is not evidence: the refusal is structural, so an ordinary text
    file with a misleading extension still publishes."""
    path = tmp_path / "notes.parquet"
    path.write_bytes(b"this is ordinary text, not a parquet file\n")
    assert credential_in_file(path) is None


# --- bounds ------------------------------------------------------------------------------------


def test_an_expired_deadline_refuses_rather_than_passing(tmp_path):
    """Every bound in the scan refuses when it bites. Returning "nothing found" would make the
    cheapest bypass of the whole check "make it expensive"."""
    path = tmp_path / "data.gz"
    path.write_bytes(gzip.compress(b"x" * 4096))
    with pytest.raises(_Unscannable):
        credential_in_file(path, deadline=time.monotonic() - 1.0)


def test_a_credential_in_a_member_name_is_found():
    """The name is committed to the hub path as surely as the contents."""
    assert credential_in_name(f"data/{FREESOLO_KEY}.txt") == "a Freesolo API key"


def test_an_ordinary_name_is_clean():
    assert credential_in_name("data/train.jsonl") is None
