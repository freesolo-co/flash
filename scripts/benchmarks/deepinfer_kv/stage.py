import json
import os
import sys
from pathlib import Path

import modal
from control import execute

root = Path(os.environ.get("KV_AB_WORKDIR", "/tmp/flash-deepinfer-kv-ab"))
source = Path(__file__).resolve().parent
entry = next(
    e
    for e in json.loads((root / "ledger.json").read_text())["allocations"]
    if e["label"] == sys.argv[1]
)
s = modal.Sandbox.from_id(entry["id"])
for p in source.glob("*.py"):
    s.filesystem.write_text(p.read_text(), "/opt/DeepInfer/" + p.name)
s.filesystem.write_text(
    (source.parents[2] / "flash/serving/deepinfer/cache_routing.py").read_text(),
    "/opt/DeepInfer/cache_routing.py",
)
code = """import subprocess,os,json
from pathlib import Path
os.chdir('/opt/DeepInfer')
subprocess.run(['python','workloads.py'],check=True)
with open('/results/service.log','w') as log:
 p=subprocess.Popen(['python','-u','service.py'],env=dict(os.environ,BENCH_EXPERT_FP8='1',PYTORCH_ALLOC_CONF='expandable_segments:True'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
Path('/results/service-pid.json').write_text(json.dumps({'pid':p.pid}))
print('SERVICE_STARTED',p.pid,flush=True)
"""

for name in ["relay.json", "relay-token.secret"]:
    if (root / name).exists():
        s.filesystem.write_text((root / name).read_text(), "/opt/DeepInfer/" + name)
s.filesystem.write_text((source / "sampler.py").read_text(), "/opt/DeepInfer/sampler.py")
assert execute(s, ["python", "-u", "-c", code]) == 0
assert (
    execute(
        s,
        [
            "python",
            "-c",
            "import subprocess; subprocess.Popen(['python','/opt/DeepInfer/sampler.py'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)",
        ],
    )
    == 0
)
