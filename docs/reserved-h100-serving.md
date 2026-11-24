# Reserved H100 serving

Serving code never launches or terminates reserved instances. This page is the **required operator procedure** for bringing the 8-slot fleet up, retiring the previous 15-slot 9B Base fleet, and owning the fleet afterwards.

## Fleet

| Slots | Model | Engine |
|---|---|---|
| 1–5 | `Qwen/Qwen3.8-27B` | official FP8 checkpoint `Qwen/Qwen3.8-27B-FP8`, TP=1, one H100 80 GB each |
| 6–8 | `Qwen/Qwen3.5-0.8B-Base` @ `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | BF16, spec-decoding drafter |

- **`Qwen/Qwen3.5-9B-Base` has no reserved slots.** It runs on the on-demand Lambda pool with its Modal GPU fallback, keeping an idle floor of 2 in main.
- **In main, the two reserved models are served only by their slots.** They get no lease-pool workers and no Modal fallback. Dev serves every model from the normal pools.
- **Source of truth:** `flash/serving/lambda_cloud/reserved.py` (`SLOT_MODELS`) and `reserved_watchdog.py`.

| Kind | Name | Holds |
|---|---|---|
| Modal Secret (main) | `flash-reserved-h100-workers-20261123` | `FLASH_RESERVED_H100_WORKERS` (the worker list) and `LAMBDA_API_KEY` (reboots only) |
| Modal Dict (main) | `flash-reserved-h100-health-20261123` | per-slot watchdog state and `configured_slots`, written by the watchdog |
| Lambda instances | `flash-h100-reservation-20261112-<n>` (kept names) | the 8 workers; the training provider never sweeps this name prefix |

**Workers secret entry**, one per slot:

```json
{"slot": 1, "model": "Qwen/Qwen3.8-27B", "instance_id": "…", "ip": "…", "revision": "<40-hex release>", "tls": {"cert": "…", "token": "…"}}
```

Every entry's `model` must be its slot's model. Otherwise the proxies and the watchdog reject the whole list, and the deploy serves nothing for these two models.

### Health and recovery
The watchdog, `reserved_h100_watchdog`, is scheduled every 120 s on the serving app in main.
- **Probe:** each worker gets `/health`, which must report its slot model, plus a one-token canary admitted through the worker's dedicated canary lane. A saturated but healthy worker still answers the canary.
- **Quarantine and reboot:** two failed probes quarantine a worker. After 10 minutes unhealthy it is rebooted through the Lambda API, at most one per cycle, only if another worker passed, and with a 30 minute cooldown.
- **Routing:** requests round-robin over the healthy, configured slots of the requested model, and fail open within that model's slots.

## Ownership and renewal (ongoing)
No code holds or renews these instances. The hold renewal and the `flash-h100-reservation-20261112` scheduler went with the 15-slot fleet.
- **Billing:** the instances are plain on-demand Lambda VMs. They run, and bill, until an operator terminates them, so treat them as a fixed monthly cost.
- **Review:** recheck the fleet whenever the spec-decoding work that needs it changes.
- **Lost or replaced instance:**
  1. Launch a replacement single-H100 VM with the cloud-init from `flash.serving.lambda_cloud.bootstrap.user_data(<slot model>, <release>, credentials, identity())`, mounting the region's hosting-cache filesystem at `/data/hosting-cache`.
  2. Name it `flash-h100-reservation-20261112-<n>`.
  3. Update its entry in the workers secret.
  4. Run `modal container stop` on the watchdog's warm container and the slot's proxy container, so they reread the secret.
- **New release:** re-bootstrap each worker (step 3 below) before or with the serving deploy. A proxy refuses a worker whose `deployment_sha` differs from its entry's `revision`.

## Rollout runbook (one-time, 2026-11)
These steps go from the 15 9B Base boxes to 8 boxes serving 27B and 0.8B Base.

1. **Deploy the on-demand 9B Base change first** (#1867's removal of the 9B reserved routing) and wait out the router drain (`ROUTER_TIMEOUT_SECONDS`, 2 h). After that, none of the 15 old boxes receives traffic. Do not terminate anything yet.
2. **Choose the 8 boxes to keep.** Criteria, in order:
   1. `gpu_1x_h100_sxm5` over PCIe.
   2. `healthy: true`, the fewest failures and no recent `restarted_at` in the old Dict `flash-reserved-h100-health-20261112`.
   3. Not in old slots 1–4 (the scheduler's), when ties remain.
   4. One region where possible, so the 27B downloads once into the shared `/data/hosting-cache`.

   Give 5 boxes to new slots 1–5 (27B) and 3 to slots 6–8 (0.8B Base). Each box's TLS cert and token are in `/opt/flash/cert.pem` and `FLASH_LAMBDA_TOKEN` in `/opt/flash/worker.env`.
3. **Stop the old scheduler.** If you keep any old slot 1–4, first set `state.mode = "adopted"` in the Dict `flash-h100-reservation-20261112`, under its `tick_lock`. Adopted reservations never terminate anything. Then `modal app stop flash-h100-reservation-20261112 --env main`.
4. **Re-bootstrap each kept box** onto release `R`. SSH as `ubuntu@<ip>` with key `freesolo-serving`.
   1. **Check disk:** `df -h / /data/hosting-cache`. Keep ≥ 60 GB free on the cache filesystem (27B FP8 weights are ~30 GB, plus vLLM caches) and ≥ 160 GB on the root disk (catalog `min_disk_gb` for 27B). Delete old core dumps if needed.
   2. **Edit `/opt/flash/worker.env`:** `FLASH_LAMBDA_MODEL=<slot model>` and `FREESOLO_DEPLOYMENT_SHA=R`.
   3. **Edit `/opt/flash/setup.sh`:** use revision `R` and `R`'s `flash.serving.lambda_cloud.config.RUNTIME_IMAGE`.
   4. **Restart:** `sudo docker rm -f flash-worker && sudo rm -rf /opt/flash-source/* && sudo bash /opt/flash/setup.sh`.
   5. **Verify:** `/health` (port 8443, bearer token) reports `ok`, the slot model, `deployment_sha == R`, `configured_gpu == "H100"` and `canary_lane: true`. A 27B cold boot takes up to ~30 minutes.
5. **Create the Secret `flash-reserved-h100-workers-20261123`** with the 8 entries and `LAMBDA_API_KEY`. Nothing needs pre-writing to the new health Dict.
6. **Deploy serving to main at `R`.**
   - Check the logs for `reserved H100 slot <n>: ok` for all 8.
   - Smoke-test one 27B and one 0.8B Base chat.
   - Until this deploy, the 27B is still served by its on-demand pool, so it has no downtime.
7. **Terminate the other 7 boxes.**
   1. For each, `POST https://cloud.lambda.ai/api/v1/instance-operations/terminate` with `{"instance_ids": ["<id>"]}`.
   2. Poll `GET /api/v1/instances/<id>` until it returns 404 or `terminated`.
   3. Confirm with `GET /api/v1/instances` that exactly 8 `flash-h100-reservation-20261112-*` instances remain.

   Do not use Flash's `terminate_confirmed`: it refuses serving VMs by design.
8. **Optional cleanup:** the old Dicts `flash-reserved-h100-health-20261112` and `flash-reserved-h100-expansion-20261121`, and the old Secrets `flash-reserved-h100-workers-20261112` and `flash-h100-reservation-20261112`. Keep the Dict `flash-h100-reservation-20261112` while you still want its record of old slots 1–4.

## Retiring the fleet entirely
1. Remove the slots from `SLOT_MODELS` (or the reserved code) and deploy. The two models then return to the on-demand pool and fallback.
2. Wait out the 2 h router drain.
3. Terminate the 8 instances with the Lambda API steps in step 7 above.
4. Delete `flash-reserved-h100-workers-20261123` and `flash-reserved-h100-health-20261123`.
