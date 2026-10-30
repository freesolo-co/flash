"""Server-side managed Freesolo env download from Azure."""

from __future__ import annotations

import pytest

from flash.server.domain.registry import envs


def test_download_package_user_can_download_own_namespace(monkeypatch):
    seen: dict[str, object] = {}

    def fake_download(slug):
        seen.update(slug=slug)
        return b"package"

    monkeypatch.setattr(envs.blob_envs, "download", fake_download)

    assert (
        envs.download_package(slug="acme/checkout-bot/my-env", key={"org_slug": "acme"})
        == b"package"
    )
    assert seen == {"slug": "acme/checkout-bot/my-env"}


def test_download_package_rejects_other_users_namespace(monkeypatch):
    monkeypatch.setattr(
        envs.blob_envs, "download", lambda *a, **k: pytest.fail("storage must not be touched")
    )

    with pytest.raises(envs.EnvPublishError) as excinfo:
        envs.download_package(slug="someone-else/project/env", key={"org_slug": "acme"})

    assert excinfo.value.status == 403


def test_download_package_internal_key_can_download_any_namespace(monkeypatch):
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        envs.blob_envs,
        "download",
        lambda slug: seen.update(slug=slug) or b"package",
    )

    assert (
        envs.download_package(slug="acme/project/paper-foo", key={"auth_kind": "internal"})
        == b"package"
    )
    assert seen == {"slug": "acme/project/paper-foo"}
