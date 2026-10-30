"""List managed environments from the private Azure registry."""

from flash.envs.registry.azure import open_registry


def list_managed_namespace_slugs(namespace: str) -> list[str]:
    with open_registry() as registry:
        return registry.list(namespace)
