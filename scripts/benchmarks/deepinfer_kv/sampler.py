import json
import os
import subprocess
import time

with open("/results/memory.jsonl", "a") as stream:
    while time.time() < float(os.environ["BENCH_DEADLINE"]) - 60:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        stream.write(
            json.dumps({"time": time.time(), "gpus": p.stdout.strip().splitlines()}) + "\n"
        )
        stream.flush()
        time.sleep(2)
