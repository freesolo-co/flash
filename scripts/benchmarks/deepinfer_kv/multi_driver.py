"""Equal-hardware two-replica A/B experiments with measured HTTP KV transfer."""

import concurrent.futures
import json
import time
import urllib.request

import modal


class Replica:
    def __init__(self, entry):
        self.sandbox = modal.Sandbox.from_id(entry["id"])

    def phase(self, name, ranks):
        filename = f"{time.time_ns()}.json"
        self.sandbox.filesystem.write_text(
            json.dumps({"name": name, "ranks": ranks}), "/results/commands/" + filename + ".tmp"
        )
        proc = self.sandbox.exec(
            "python",
            "-c",
            "import os,sys; os.replace(sys.argv[1],sys.argv[2])",
            "/results/commands/" + filename + ".tmp",
            "/results/commands/" + filename,
        )
        proc.wait()
        assert proc.returncode == 0
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                return json.loads(
                    self.sandbox.filesystem.read_text("/results/responses/" + filename)
                )
            except modal.exception.SandboxFilesystemNotFoundError:
                time.sleep(0.3)
        raise TimeoutError(name)

    def all(self, name, command):
        return self.phase(name, [command] * 4)


def parallel(functions):
    with concurrent.futures.ThreadPoolExecutor(len(functions)) as pool:
        futures = [pool.submit(function) for function in functions]
        return [future.result() for future in futures]


def batch(cases, suffixes=range(4), cached=False):
    return [
        {
            "op": "batch",
            "requests": [
                dict(cases[r * 4 + q], key=cases[r * 4 + q]["case"] if cached else None)
                for q in suffixes
            ],
        }
        for r in range(4)
    ]


def transfer(source, destination, object_name, keys):
    push = source.phase(
        "push-" + object_name,
        [
            {
                "op": "extension",
                "module": "sharing_actions",
                "operation": "store",
                "object": object_name,
                "keys": keys[r],
            }
            for r in range(4)
        ],
    )
    pull = destination.all(
        "pull-" + object_name,
        {
            "op": "extension",
            "module": "sharing_actions",
            "operation": "load",
            "object": object_name,
        },
    )
    return push, pull


def clear_relay(root):
    relay = json.loads((root / "relay.json").read_text())
    token = (root / "relay-token.secret").read_text()
    request = urllib.request.Request(
        relay["url"], method="DELETE", headers={"Authorization": "Bearer " + token}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 204
