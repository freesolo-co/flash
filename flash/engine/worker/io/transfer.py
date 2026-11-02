"""Construct the Azure artifact publisher."""

from flash.artifacts.api import ArtifactStore


def upload_api(*, token=None):
    return ArtifactStore(token=token)
