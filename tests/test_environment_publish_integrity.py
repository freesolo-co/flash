"""Malformed package transport and identifiers remain validation failures."""

import io
import tarfile

import pytest

from flash.server.domain.registry import envs


@pytest.mark.parametrize("slug", ["org/project/name\n", "org/project\n/name"])
def test_environment_id_rejects_trailing_newline_in_segment(slug):
    with pytest.raises(envs.EnvPublishError, match="invalid env id segment"):
        envs.canonical_env_id(slug)


def test_truncated_gzip_package_reports_validation_error(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        entry = tarfile.TarInfo("environment.py")
        entry.size = 20000
        tar.addfile(entry, io.BytesIO(b"x" * entry.size))
    with pytest.raises(envs.EnvPublishError):
        envs._safe_extract(data.getvalue()[:15], tmp_path)
