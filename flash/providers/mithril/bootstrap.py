"""Private, checksum-verified bootstrap artifacts within Mithril's 10,000-character limit."""

import base64
import hashlib
import json
import logging

from flash.artifacts.config import open_container
from flash.artifacts.manifest import repo_prefix
from flash.artifacts.storage import BlobArtifacts
from flash.providers.artifacts.channel import artifact_call


def _publish(payload, data, suffix, *, deadline_at):
    token = payload.get("env", {}).get("FLASH_ARTIFACT_BLOB_SAS_URL")
    if not token:
        raise ValueError("Mithril worker requires private Azure artifact credentials")
    digest = hashlib.sha256(data).hexdigest()
    blob = f"{repo_prefix(payload['hf_repo'])}/capsules/{digest}.{suffix}"

    def publish():
        with open_container() as container:
            if container.get_container_properties().public_access:
                raise ValueError("Mithril capsule container must be private")
        with open_container(token) as container:
            BlobArtifacts(container).immutable(blob, data)

    artifact_call(
        publish,
        "publish Mithril bootstrap",
        deadline_at=deadline_at,
        logger=logging.getLogger(__name__),
    )
    return {"blob": blob, "sha256": digest}


def publish_capsule(payload, capsule, *, deadline_at):
    return _publish(payload, capsule, "pyz", deadline_at=deadline_at)


def startup_script(payload, capsule, image, *, owner_line, deadline_at):
    capsule_ref = publish_capsule(payload, capsule, deadline_at=deadline_at)
    bundle = {
        "payload": payload,
        "capsule": capsule_ref,
        "image": image,
        "deadline_at": deadline_at,
        "claim_blob": f"{repo_prefix(payload['hf_repo'])}/{payload['phase']}/{payload['run_id']}/mithril_claim_{payload['attempt']}",
    }
    # Large source descriptors and runtime secrets stay in private storage, not the API's script.
    source = "BUNDLE = " + repr(bundle) + "\n" + WORKER
    ref = _publish(payload, source.encode(), "py", deadline_at=deadline_at)
    ref["sas"] = payload["env"]["FLASH_ARTIFACT_BLOB_SAS_URL"]
    encoded = base64.b64encode(json.dumps(ref).encode()).decode()
    script = (
        f"#!/bin/bash\n{owner_line}\nset -eu\numask 077\nsudo -n python3 - <<'FLASH_BOOTSTRAP'\n"
        "import base64,json\nREF=json.loads(base64.b64decode("
        + repr(encoded)
        + "))\n"
        + DOWNLOAD
        + "\nFLASH_BOOTSTRAP\n"
    )
    if len(script) > 10_000:
        raise ValueError("Mithril startup script exceeds 10,000 characters")
    return script


# Runs as root on the host. No shell tracing, credentials in argv, or public blob access.
DOWNLOAD = """import hashlib,os,pathlib,subprocess,urllib.request,urllib.parse
os.umask(0o077)
root=pathlib.Path('/opt/flash')
root.mkdir(parents=True,exist_ok=True)
sas=urllib.parse.urlsplit(REF['sas'])
url=urllib.parse.urlunsplit((sas.scheme,sas.netloc,sas.path+'/'+urllib.parse.quote(REF['blob'],safe='/'),sas.query,''))
try:
    with urllib.request.urlopen(url,timeout=60) as response:
        data=response.read(8*1024*1024)
    if hashlib.sha256(data).hexdigest()!=REF['sha256']:
        raise RuntimeError('checksum mismatch')
    path=root/'launch.py'
    path.write_bytes(data)
    path.chmod(0o600)
except Exception:
    raise SystemExit('Flash bootstrap download failed') from None
subprocess.run(['python3',str(path)],check=True)
"""

WORKER = """import hashlib,json,os,pathlib,subprocess,time,urllib.request,urllib.parse
os.umask(0o077)
root=pathlib.Path('/opt/flash')
payload=BUNDLE['payload']
sas=urllib.parse.urlsplit(payload['env']['FLASH_ARTIFACT_BLOB_SAS_URL'])
def blob_url(blob):
    return urllib.parse.urlunsplit((sas.scheme,sas.netloc,sas.path+'/'+urllib.parse.quote(blob,safe='/'),sas.query,''))
def run():
    if time.time() >= BUNDLE['deadline_at']:
        raise RuntimeError('deadline reached')
    # Atomically claim this attempt, even if a duplicate bid or replacement VM starts it.
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
    # Dedicated GPU host: keep image layers and training scratch off the small boot disk.
    scratch=pathlib.Path('/mnt/local')
    if not scratch.is_mount():
        raise RuntimeError('Mithril local scratch is not mounted')
    daemon=pathlib.Path('/etc/docker/daemon.json')
    config=json.loads(daemon.read_text()) if daemon.exists() else {}
    config['data-root']='/mnt/local/docker'
    daemon.parent.mkdir(parents=True,exist_ok=True)
    daemon.write_text(json.dumps(config))
    subprocess.run(['systemctl','restart','docker'],check=True,timeout=60)
    custom=json.loads(payload['job_spec_json']).get('algorithm')=='custom'
    if custom:
        subprocess.Popen(['foundrypf','8000'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    seconds=int(BUNDLE['deadline_at']-time.time())
    if seconds<=0:
        raise RuntimeError('deadline reached')
    command=['timeout','--signal=TERM','--kill-after=30',str(seconds),
        'docker','run','--rm','--gpus','all','--ipc=host','--env-file',str(root/'payload.env'),
        '-v','/opt/flash/capsule.pyz:/opt/flash/capsule.pyz:ro',
        '--workdir','/root/flash']
    if custom:
        command+=['-p','8000:8000']
    command += [BUNDLE['image'],'sh','-c',
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
    subprocess.run(['shutdown','-h','now'],check=False)
"""
