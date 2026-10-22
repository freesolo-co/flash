"""Runtime secret validation must not modify nonblank credential contents."""

import pytest
from fastapi import HTTPException

from flash.core.spec import EnvironmentSpec, JobSpec
from flash.server.platform.deps import _runtime_secrets


@pytest.mark.parametrize("value", [" padded ", "\tpassword\n", "token'", '"token"'])
def test_runtime_secret_validation_preserves_exact_value(value):
    spec = JobSpec(environment=EnvironmentSpec(secrets=("AUDIT_SECRET",)))
    assert _runtime_secrets({"runtime_secrets": {"AUDIT_SECRET": value}}, spec) == {
        "AUDIT_SECRET": value
    }


@pytest.mark.parametrize("value", ["", " \t\n", None])
def test_blank_required_runtime_secret_still_fails(value):
    spec = JobSpec(environment=EnvironmentSpec(secrets=("AUDIT_SECRET",)))
    with pytest.raises(HTTPException, match="missing runtime secret"):
        _runtime_secrets({"runtime_secrets": {"AUDIT_SECRET": value}}, spec)
