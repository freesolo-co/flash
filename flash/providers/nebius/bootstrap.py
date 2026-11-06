"""Publish a content-addressed capsule outside Nebius's 32 KiB user-data limit."""

import hashlib
import logging

from flash.artifacts.config import open_container
from flash.artifacts.manifest import repo_prefix
from flash.artifacts.storage import BlobArtifacts
from flash.providers.artifacts.channel import artifact_call


def publish_capsule(payload, capsule, *, deadline_at):
    token = payload.get("env", {}).get("FLASH_ARTIFACT_BLOB_SAS_URL")
    if not token:
        raise ValueError("Nebius worker requires private Azure artifact credentials")
    digest = hashlib.sha256(capsule).hexdigest()
    blob = f"{repo_prefix(payload['hf_repo'])}/capsules/{digest}.pyz"

    def publish():
        with open_container(token) as container:
            if container.get_container_properties().public_access:
                raise ValueError("Nebius capsule container must be private")
            BlobArtifacts(container).immutable(blob, capsule)

    artifact_call(
        publish,
        "publish Nebius capsule",
        deadline_at=deadline_at,
        logger=logging.getLogger(__name__),
    )
    return {"blob": blob, "sha256": digest}


# Stdlib only: runs on the host before the training container is available.
DOWNLOAD = """import hashlib,json,pathlib,urllib.request,urllib.parse
root=pathlib.Path('/opt/flash')
payload=json.loads((root/'payload.env').read_text().split('=',1)[1])
ref=json.loads((root/'capsule.json').read_text())
sas=urllib.parse.urlsplit(payload['env']['FLASH_ARTIFACT_BLOB_SAS_URL'])
url=urllib.parse.urlunsplit((sas.scheme,sas.netloc,sas.path+'/'+urllib.parse.quote(ref['blob'],safe='/'),sas.query,''))
request=urllib.request.Request(url)
with urllib.request.urlopen(request,timeout=60) as response:
    data=response.read(2*1024*1024)
if hashlib.sha256(data).hexdigest()!=ref['sha256']:
    raise RuntimeError('Flash capsule checksum mismatch')
(root/'capsule.pyz').write_bytes(data)
(root/'capsule.pyz').chmod(0o600)
"""
