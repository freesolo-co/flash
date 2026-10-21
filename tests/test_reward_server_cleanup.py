"""Stopping a reward bridge releases its listener and request workers."""

import json
import threading
import urllib.request

from flash.engine.worker.train.rl.rollout.multi_turn import start_reward_server


def test_reward_server_shutdown_releases_socket_and_worker_threads():
    existing_threads = set(threading.enumerate())
    server, url = start_reward_server(lambda index, solution: 1.0, example_count=1)
    try:
        request = urllib.request.Request(
            f"{url}/score",
            data=json.dumps({"index": 0, "solution_str": "answer"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.load(response) == {"score": 1.0}
        request_workers = [
            thread
            for thread in set(threading.enumerate()) - existing_threads
            if thread.name.startswith("flash-bridge")
        ]
        assert request_workers
        server.shutdown()
        assert server.socket.fileno() == -1
        for thread in request_workers:
            thread.join(timeout=1)
            assert not thread.is_alive()
    finally:
        server.shutdown()
        server.server_close()
