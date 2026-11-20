# Reserved H100 retention scheduler

Source of the Modal app `flash-h100-reservation-20261112`, which owns Lambda instances
`flash-h100-reservation-20261112-1` through `-4` (see `docs/reserved-h100-serving.md`).
It was deployed from outside this repository. This copy was recovered byte for byte from the
running container on 2026-11-20 (`reservation.py` sha256
`5c278e5e9cbd177842ef717379089441671c548b08417fc53ec3959aa56c465e`). It is kept unformatted
so it stays identical to what ran.

`tick()` runs every minute under the `tick_lock` in the Modal Dict of the same name. In mode
`serving` with four active instances it only records state. Once `hold_until` passes it
**terminates all four instances**. The serving watchdog (`reserved_watchdog.py`) renews
`hold_until`.

## Deploying (workspace moves included)

**The Dict must hold the live `state` before the app is deployed.** With no `state`, `tick()`
starts from `fresh_state()`, whose `hold_until` is `absolute_hold_until` from the config.
For this campaign that date has already passed, so the first tick terminates the fleet.

1. Stop the old scheduler app and wait for its containers to exit.
2. Copy every key of the Dict `flash-h100-reservation-20261112` into the target workspace,
   then confirm `state.mode == "serving"` and that `state.hold_until` is more than seven days out.
3. Create the Secret `flash-h100-reservation-20261112` (`LAMBDA_API_KEY`) in the target workspace.
4. Copy `config.example.json` to `config.json`, set `notification`, and deploy:
   `modal deploy --env main deploy/reserved-h100/reservation.py`.
5. Check `status.remote()`, or the first tick log line, for phase `serving_flash_9b_base` with
   four active slots.
