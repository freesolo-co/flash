"""Namespace isolation and storage-error propagation for managed environment listing."""

from contextlib import contextmanager

import pytest

from flash.envs.loading import loader
from flash.envs.registry import azure
from flash.server.domain.registry import envs


def test_domain_uses_authenticated_namespace_without_github_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr(envs.blob_envs, "list_slugs", lambda ns: calls.append(ns) or [f"{ns}/p/e"])
    assert envs.list_namespace_slugs(key={"org_slug": "acme"}) == ["acme/p/e"]
    assert calls == ["acme"]


def test_listing_refuses_org_agnostic_internal_key(monkeypatch):
    monkeypatch.setattr(
        envs.blob_envs, "list_slugs", lambda ns: pytest.fail("storage must not be touched")
    )
    with pytest.raises(envs.EnvPublishError, match="org slug"):
        envs.list_namespace_slugs(key={"auth_kind": "internal"})


@pytest.mark.parametrize("status", [429, 503])
def test_listing_preserves_storage_failure_instead_of_returning_empty(monkeypatch, status):
    @contextmanager
    def unavailable():
        raise azure.RegistryError("unavailable", status=status)
        yield

    monkeypatch.setattr(azure, "open_registry", unavailable)
    with pytest.raises(envs.EnvPublishError) as error:
        envs.list_namespace_slugs(key={"org_slug": "acme"})
    assert error.value.status == status


def test_the_worker_still_reschedules_on_either_transient_cause():
    """Splitting the type must not change what the worker does: both still retry.

    The worker classifies on the shared base, so an outage that used to arrive as the rate-limit
    type keeps rescheduling the run rather than failing it permanently.
    """
    from flash.engine.worker.entry.worker import _worker_failure_flags

    for exc in (
        loader.GitHubRateLimitError("quota"),
        loader.GitHubUnavailableError("outage"),
    ):
        assert _worker_failure_flags(exc)["retriable"]
