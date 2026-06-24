"""Bounded poller: wait for the H100 worker to start writing heartbeats (or finish/fail), then exit.
Re-invokes the agent on exit so monitoring stays on a ~check-on-state-change cadence, not a tight loop."""
import os
import time

REPO = "Freesolo-Co/flashrun-fa3probeh100a"
MATCH = "a3probeh100a"
DEADLINE = time.time() + 11 * 60


def main():
    from huggingface_hub import HfApi

    from flash.providers.runpod import api as rp

    tok = os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    eid = None
    for e in rp.list_endpoints():
        if MATCH in (e.get("name", "") or ""):
            eid = e.get("id")
    i = 0
    while time.time() < DEADLINE:
        i += 1
        health = {}
        try:
            health = rp.endpoint_health(eid) if eid else {}
        except Exception as ex:
            health = {"err": str(ex)[:80]}
        try:
            files = set(api.list_repo_files(REPO, repo_type="dataset"))
        except Exception:
            files = set()
        non_code = [f for f in files if not f.startswith("code/") and f != ".gitattributes"]
        w = (health.get("workers") or {}) if isinstance(health, dict) else {}
        j = (health.get("jobs") or {}) if isinstance(health, dict) else {}
        print(f"[{i}] workers={w} jobs={j} hf_new={non_code}", flush=True)
        # Exit on any meaningful state change: heartbeat/metrics/DONE present, or job done/failed.
        if non_code:
            print("STATE: worker writing artifacts ->", non_code, flush=True)
            return 0
        if isinstance(j, dict) and (j.get("completed") or j.get("failed")):
            print("STATE: job terminal", j, flush=True)
            return 0
        time.sleep(45)
    print("STATE: still cold-starting after 11 min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
