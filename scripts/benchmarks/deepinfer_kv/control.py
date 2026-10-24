"""Campaign-owned resources only; reserve worst-case spend before allocation."""

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("KV_AB_WORKDIR", "/tmp/flash-deepinfer-kv-ab"))
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE = Path(__file__).resolve().parent / "runtime"
LEDGER = ROOT / "ledger.json"
RATE = (4 * 0.001736 + 16 * 0.00003942 + 512 * 0.00000667) * 3600


def save(value):
    temporary = LEDGER.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(LEDGER)


def execute(sb, command, timeout=2400):
    proc = sb.exec(*command, timeout=timeout)

    def drain(reader):
        for line in reader:
            print(line, end="", flush=True)

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        tasks = [pool.submit(drain, proc.stdout), pool.submit(drain, proc.stderr)]
        proc.wait()
        for task in tasks:
            task.result()
    return proc.returncode


def main():
    action, *args = sys.argv[1:]
    ledger = (
        json.loads(LEDGER.read_text())
        if LEDGER.exists()
        else {
            "budget_usd": 500,
            "allocation_limit_usd": 420,
            "hourly_reserved_usd": RATE,
            "pricing": "https://modal.com/pricing",
            "base_sha": "b6a4019ae7a8c1cd4d6967ba5ae8e22cbd0e58ad",
            "allocations": [],
            "experiments": [
                "routing",
                "tiering",
                "sharing",
                "retention",
                "disaggregation",
                "blend",
                "pruning",
            ],
        }
    )
    ledger["hourly_reserved_usd"] = RATE
    if action == "launch":
        label = args[0]
        duration = int(args[1]) if len(args) > 1 else 7200
        assert 0 < duration <= 14400
        reservation = RATE * duration / 3600
        assert (
            sum(x.get("elapsed_cost_upper", x["reserved_usd"]) for x in ledger["allocations"])
            + reservation
            <= 420
        )
        app = modal.App.lookup(
            "deepinfer-kv-ab-20261023", create_if_missing=True, environment_name="main"
        )
        active = list(modal.Sandbox.list(app_id=app.app_id))
        assert (
            sum(x.object_id in {v.get("id") for v in ledger["allocations"]} for x in active) < 2
        ), "At most eight campaign GPUs"
        assert all(x["label"] != label for x in ledger["allocations"])
        now = time.time()
        entry = {
            "label": label,
            "started": now,
            "deadline": now + duration,
            "reserved_usd": reservation,
            "hourly_rate_upper_usd": RATE,
            "host_memory_cap_gib": 512,
            "app_id": app.app_id,
            "state": "allocating",
        }
        ledger["allocations"].append(entry)
        save(ledger)
        env = {
            "BENCH_DEADLINE": str(entry["deadline"]),
            "DEEPINFER_ACCELERATOR": "nvidia",
            "PATH": "/opt/deepinfer-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "VIRTUAL_ENV": "/opt/deepinfer-venv",
            "PYTHONPATH": "/opt/DeepInfer",
            "PYTHONUNBUFFERED": "1",
        }
        guard = "import os,time; end=float(os.environ['BENCH_DEADLINE']); print('DEADLINE_ACTIVE',end,flush=True)\nwhile time.time()<end-30: time.sleep(5)"
        with modal.enable_output():
            sb = modal.Sandbox.create(
                "python",
                "-u",
                "-c",
                guard,
                app=app,
                image=modal.Image.from_id("im-01M5GXCE5NCMK4TBY608BEPZ1Y"),
                gpu="B200:4",
                cpu=(16, 16),
                memory=(524288, 524288),
                timeout=duration,
                volumes={
                    "/weights": modal.Volume.from_name(
                        "deepinfer-b200-1017-benchmark-weights", environment_name="main"
                    ).read_only()
                },
                env=env,
            )
        entry.update(id=sb.object_id, state="running")
        save(ledger)
        print(json.dumps(entry), flush=True)
        for name in [
            "runtime-patches.diff",
            "patch_detach.py",
            "restore_fast.py",
            "bench_fp8.py",
            "engram_local.py",
            "serving_init.py",
        ]:
            sb.filesystem.write_text(
                (SOURCE / (name + ".txt" if name.endswith(".py") else name)).read_text(),
                "/opt/DeepInfer/" + name,
            )
        sys.exit(
            execute(
                sb,
                [
                    "python",
                    "-u",
                    "-c",
                    """
import subprocess,os
from pathlib import Path
r=Path('/opt/DeepInfer');os.chdir(r)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='9bd9cf243fc53ef9afbf8f04d391d989117d6ab9'
subprocess.run(['git','apply','runtime-patches.diff'],check=True)
subprocess.run(['python','patch_detach.py'],check=True)
p=r/'csrc/engine/pybind_sequence.cpp';s=p.read_text()
anchor='    bind_sequence_property(sequence, "state_id",'
assert s.count(anchor)==1
s=s.replace(anchor,'    bind_sequence_property(sequence, "block_table", [](Sequence &seq) -> auto & { return seq.cache.block_table; });\\n'+anchor)
p.write_text(s)
subprocess.run(['python','setup.py','build_ext','--inplace'],env=dict(os.environ,MAX_JOBS='8',CXX='/usr/bin/g++-13'),check=True)
subprocess.run(['python','restore_fast.py'],check=True)
Path('/results').mkdir(exist_ok=True)
print('KV_AB_RUNTIME_READY',flush=True)
""",
                ],
            )
        )
    entry = next(x for x in ledger["allocations"] if x["label"] == args[0])
    sb = modal.Sandbox.from_id(entry["id"])
    if action == "exec":
        sys.exit(
            execute(sb, args[1:], max(1, min(3600, int(entry["deadline"] - time.time() - 40))))
        )
    elif action == "upload":
        sb.filesystem.write_text(Path(args[1]).read_text(), args[2])
    elif action == "read":
        print(sb.filesystem.read_text(args[1]))
    elif action == "pull":
        Path(args[2]).write_bytes(sb.filesystem.read_bytes(args[1]))
    elif action == "status":
        print(
            json.dumps(
                {
                    "id": sb.object_id,
                    "exit_code": sb.poll(),
                    "elapsed_cost_upper": entry.get("hourly_rate_upper_usd", RATE)
                    * (time.time() - entry["started"])
                    / 3600,
                }
            )
        )
    elif action == "stop":
        sb.terminate(wait=True)
        entry.update(state="stopped", stopped=time.time(), exit_code=sb.poll())
        entry["elapsed_cost_upper"] = (
            entry.get("hourly_rate_upper_usd", RATE) * (entry["stopped"] - entry["started"]) / 3600
        )
        save(ledger)
        print(json.dumps(entry))


if __name__ == "__main__":
    main()
