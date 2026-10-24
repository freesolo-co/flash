"""Functional persistence probe: restore relay KV after receiver process restart.

One controlled restart, not a three-repeat performance result. The relay itself
is not restarted, so this tests process-loss survival rather than durable disk.
"""

import argparse
import json
import time
from pathlib import Path

import modal
from driver import correctness
from multi_driver import Replica, clear_relay, transfer


def run(root):
    entries = json.loads((root / "ledger.json").read_text())["allocations"]
    source = Replica(next(x for x in entries if x["label"] == "primary2"))
    destination = Replica(next(x for x in entries if x["label"] == "peer2"))
    cases = json.loads(source.sandbox.filesystem.read_text("/results/workload-8192.json"))
    source.all("restart-source-clear", {"op": "clear"})
    destination.all("restart-peer-clear", {"op": "clear"})
    clear_relay(root)
    source.phase(
        "restart-source-warm",
        [
            {
                "op": "warm",
                "key": f"p{r}",
                "tokens": cases[r * 4]["tokens"],
                "cut": cases[r * 4]["cut"],
            }
            for r in range(4)
        ],
    )
    transfer(source, destination, "restart-proof", [[f"p{r}"] for r in range(4)])
    commands = [
        {"op": "batch", "requests": [dict(cases[r * 4 + q], key=f"p{r}") for q in range(4)]}
        for r in range(4)
    ]
    before = destination.phase("restart-reference", commands)
    print("SOURCE_NO_LONGER_REQUIRED", flush=True)
    started = time.monotonic()
    code = """import os,signal,time,json,subprocess
from pathlib import Path
root=Path('/results');pid=json.loads((root/'service-pid.json').read_text())['pid']
assert 'service.py' in Path(f'/proc/{pid}/cmdline').read_text()
os.killpg(pid,signal.SIGKILL)
(root/'ready.json').unlink(missing_ok=True)
for name in ['commands','responses']:
 (root/name).rename(root/(name+'-before-restart'))
# Wait for this campaign's four model processes to release their CUDA contexts.
end=time.monotonic()+120
while True:
 used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 if max(float(x) for x in used.splitlines())<100:break
 assert time.monotonic()<end,used
 time.sleep(1)
os.chdir('/opt/DeepInfer')
with open('/results/restart-service.log','w') as log:
 p=subprocess.Popen(['python','-u','service.py'],env=dict(os.environ,BENCH_EXPERT_FP8='1',PYTORCH_ALLOC_CONF='expandable_segments:True'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'service-pid.json').write_text(json.dumps({'pid':p.pid}))
print('RESTART_PID',p.pid,flush=True)
"""
    process = destination.sandbox.exec("python", "-u", "-c", code, timeout=180)
    process.wait()
    assert process.returncode == 0, process.stderr.read()
    deadline = time.monotonic() + 1200
    while True:
        try:
            destination.sandbox.filesystem.read_text("/results/ready.json")
            break
        except modal.exception.SandboxFilesystemError:
            assert destination.sandbox.poll() is None, "Receiver sandbox terminated"
            assert time.monotonic() < deadline, "Restart did not become ready"
            time.sleep(5)
    restart_seconds = time.monotonic() - started
    cold = destination.phase(
        "restart-cold-reference",
        [
            {"op": "batch", "requests": [dict(cases[r * 4 + q], key=None) for q in range(4)]}
            for r in range(4)
        ],
    )
    restore = destination.all(
        "restart-restore-existing-object",
        {
            "op": "extension",
            "module": "sharing_actions",
            "operation": "load",
            "object": "restart-proof",
        },
    )
    after = destination.phase("restart-restored-candidate", commands)
    result = {
        "restart_seconds": restart_seconds,
        "before": before,
        "cold": cold,
        "restore": restore,
        "after": after,
        "correctness": correctness(before, after),
    }
    (root / "restart-result.json").write_text(json.dumps(result))
    print(
        "RESTART_COMPLETE",
        sum(x["exact_tokens"] for x in result["correctness"]),
        len(result["correctness"]),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
