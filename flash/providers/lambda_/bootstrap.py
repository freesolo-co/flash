"""Private, checksum-verified cloud-init bootstrap for a Lambda training VM.

The same scheme as Mithril Spot: the capsule and a launch script (carrying the worker payload and
its runtime secrets) are published to private Azure storage, and the cloud-init ``user_data`` only
carries a small downloader plus the blob reference. The downloader runs from a 0600 file, so no
credential is ever in argv, a process listing or the cloud-init log.

Unlike Mithril there is no ``/mnt/local`` scratch disk to move Docker onto: Lambda's root volume is
the large local NVMe, so Docker keeps its default data-root.
"""

import base64
import hashlib
import json
import logging

from flash.artifacts.config import open_container
from flash.artifacts.manifest import repo_prefix
from flash.artifacts.storage import BlobArtifacts
from flash.providers.artifacts.channel import artifact_call
from flash.providers.mithril.bootstrap import DOWNLOAD

BOOTSTRAP_PATH = "/opt/flash/bootstrap.py"
UNIT = "flash-training"


def _publish(payload, data, suffix, *, deadline_at):
    token = payload.get("env", {}).get("FLASH_ARTIFACT_BLOB_SAS_URL")
    if not token:
        raise ValueError("Lambda worker requires private Azure artifact credentials")
    digest = hashlib.sha256(data).hexdigest()
    blob = f"{repo_prefix(payload['hf_repo'])}/capsules/{digest}.{suffix}"

    def publish():
        with open_container() as container:
            if container.get_container_properties().public_access:
                raise ValueError("Lambda capsule container must be private")
        with open_container(token) as container:
            BlobArtifacts(container).immutable(blob, data)

    artifact_call(
        publish,
        "publish Lambda bootstrap",
        deadline_at=deadline_at,
        logger=logging.getLogger(__name__),
    )
    return {"blob": blob, "sha256": digest}


def user_data(payload, capsule, image, *, deadline_at):
    """The cloud-init ``user_data`` for one attempt (``#cloud-config``)."""
    capsule_ref = _publish(payload, capsule, "pyz", deadline_at=deadline_at)
    bundle = {
        "payload": payload,
        "capsule": capsule_ref,
        "image": image,
        "deadline_at": deadline_at,
        "claim_blob": (
            f"{repo_prefix(payload['hf_repo'])}/{payload['phase']}/{payload['run_id']}"
            f"/lambda_claim_{payload['attempt']}"
        ),
    }
    # Source descriptors and runtime secrets stay in private storage, not in Lambda's user_data.
    source = "BUNDLE = " + repr(bundle) + "\n" + WORKER
    ref = _publish(payload, source.encode(), "py", deadline_at=deadline_at)
    ref["sas"] = payload["env"]["FLASH_ARTIFACT_BLOB_SAS_URL"]
    encoded = base64.b64encode(json.dumps(ref).encode()).decode()
    downloader = (
        "import base64,json\nREF=json.loads(base64.b64decode(" + repr(encoded) + "))\n" + DOWNLOAD
    )
    config = {
        "write_files": [
            {
                "path": BOOTSTRAP_PATH,
                "permissions": "0600",
                "owner": "root:root",
                "encoding": "b64",
                "content": base64.b64encode(downloader.encode()).decode(),
            }
        ],
        # A transient unit, so a training run that lasts hours never holds cloud-init's final stage.
        "runcmd": [
            ["systemd-run", f"--unit={UNIT}", "--collect", "python3", BOOTSTRAP_PATH],
        ],
    }
    return "#cloud-config\n" + json.dumps(config)


# Runs as root on the host. No shell tracing, credentials in argv, or public blob access.
WORKER = """import hashlib,json,os,pathlib,subprocess,time,urllib.request,urllib.parse
os.umask(0o077)
root=pathlib.Path('/opt/flash')
payload=BUNDLE['payload']
sas=urllib.parse.urlsplit(payload['env']['FLASH_ARTIFACT_BLOB_SAS_URL'])
def blob_url(blob):
    return urllib.parse.urlunsplit((sas.scheme,sas.netloc,sas.path+'/'+urllib.parse.quote(blob,safe='/'),sas.query,''))
def ready(command):
    return subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False).returncode==0
def run():
    if time.time() >= BUNDLE['deadline_at']:
        raise RuntimeError('deadline reached')
    # Atomically claim this attempt, even if a duplicate launch starts it twice.
    claim=urllib.request.Request(blob_url(BUNDLE['claim_blob']),data=b'claimed',method='PUT',
        headers={'x-ms-blob-type':'BlockBlob','If-None-Match':'*','x-ms-version':'2023-11-03'})
    with urllib.request.urlopen(claim,timeout=60):
        pass
    with urllib.request.urlopen(blob_url(BUNDLE['capsule']['blob']),timeout=60) as response:
        data=response.read(8*1024*1024)
    if hashlib.sha256(data).hexdigest()!=BUNDLE['capsule']['sha256']:
        raise RuntimeError('Flash capsule checksum mismatch')
    (root/'capsule.pyz').write_bytes(data)
    (root/'payload.env').write_text('FLASH_BOOTSTRAP_PAYLOAD='+json.dumps(payload,separators=(',',':'))+'\\n')
    # cloud-init can run before Docker and the NVIDIA driver finish starting.
    for _ in range(120):
        if ready(['docker','info']) and ready(['nvidia-smi']):
            break
        if time.time() >= BUNDLE['deadline_at']:
            raise RuntimeError('deadline reached')
        time.sleep(5)
    else:
        raise RuntimeError('docker or the GPU never became ready')
    seconds=int(BUNDLE['deadline_at']-time.time())
    if seconds<=0:
        raise RuntimeError('deadline reached')
    command=['timeout','--signal=TERM','--kill-after=30',str(seconds),
        'docker','run','--rm','--gpus','all','--ipc=host','--env-file',str(root/'payload.env'),
        '-v','/opt/flash/capsule.pyz:/opt/flash/capsule.pyz:ro',
        '--workdir','/root/flash',
        BUNDLE['image'],'sh','-c',
        "python -m pip install --disable-pip-version-check --no-input "
        "'azure-storage-blob>=12.24,<13' 'azure-identity>=1.19,<2' "
        "&& exec python -u /opt/flash/capsule.pyz bootstrap"]
    subprocess.run(command,check=True)
try:
    run()
except Exception:
    # URLs can contain SAS credentials: never print a host traceback.
    print('Flash worker bootstrap failed',flush=True)
finally:
    # Halting does not stop Lambda billing; the control plane terminates the instance. This only
    # stops a finished host from doing anything else.
    subprocess.run(['shutdown','-h','now'],check=False)
"""
