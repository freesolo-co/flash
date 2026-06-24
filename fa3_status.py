"""Status of the live FA3 H100 probe: RunPod endpoint/job state + HF heartbeat/console/metrics."""
import json
import os
import sys

REPO = "Freesolo-Co/flashrun-fa3probeh100a"
MATCH = "a3probeh100a"


def main():
    from flash.providers.runpod import api as rp

    eid = jid = None
    try:
        for e in rp.list_endpoints():
            if MATCH in (e.get("name", "") or ""):
                eid = e.get("id")
                print(f"ENDPOINT {e.get('name')} id={eid}")
                try:
                    print("  health:", json.dumps(rp.endpoint_health(eid)))
                except Exception as ex:
                    print("  health err:", str(ex)[:160])
    except Exception as ex:
        print("list_endpoints err:", str(ex)[:200])

    # HF artifacts: heartbeat (live phase), console (the [attn] line), metrics (loss curve), DONE.
    from huggingface_hub import HfApi, hf_hub_download

    tok = os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    try:
        files = api.list_repo_files(REPO, repo_type="dataset")
        print("HF files:", [f for f in files if not f.startswith("code/")][:20])
    except Exception as ex:
        print("HF list err:", str(ex)[:160])
        return 0
    for f in ("heartbeat.json", "metrics.json", "DONE"):
        try:
            p = hf_hub_download(REPO, f, repo_type="dataset", token=tok)
            body = open(p).read()
            print(f"\n=== {f} ===")
            print(body[:1200])
        except Exception as ex:
            print(f"\n{f}: not present ({str(ex)[:60]})")
    # Console tail: the FA3 signal "[attn] sm90 -> attn_implementation=flash_attention_3".
    for cf in ("console.txt", "console.log", "logs/console.txt"):
        try:
            p = hf_hub_download(REPO, cf, repo_type="dataset", token=tok)
            lines = open(p, errors="replace").read().splitlines()
            attn = [ln for ln in lines if "[attn]" in ln or "flash_attention" in ln or "flash_attn" in ln.lower()]
            print(f"\n=== {cf} attn lines ===")
            print("\n".join(attn[-10:]) or "(none yet)")
            print(f"--- {cf} tail ---")
            print("\n".join(lines[-15:]))
            break
        except Exception:
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
