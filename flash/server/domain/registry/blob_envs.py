"""Translate managed Blob Storage operations into the environment API's errors."""

from __future__ import annotations

from contextlib import contextmanager

from flash.envs.registry import azure, packages


@contextmanager
def api_errors():
    from flash.server.domain.registry.envs import EnvPublishError

    try:
        yield
    except azure.RegistryError as exc:
        raise EnvPublishError(str(exc), status=exc.status) from exc


def publish(slug, directory, *, create_only=False):
    with api_errors():
        packages.publish_directory(slug, directory, create_only=create_only)
    return slug


def download(slug):
    with api_errors(), azure.open_registry() as registry:
        return registry.download(registry.resolve(slug))


def delete(slug):
    with api_errors(), azure.open_registry() as registry:
        return registry.delete(slug)


def list_slugs(namespace):
    with api_errors(), azure.open_registry() as registry:
        return registry.list(namespace)
