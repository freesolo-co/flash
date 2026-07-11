---
name: verify
summary: verify the local Flash CLI and control plane harness
---

# verify local Flash harness

Keep this verification local and no-spend. Do not submit a training run or call provider APIs.

1. identify the selected checkout. Treat its `.flashdev/` directory as the only authoritative runtime state root. Derive `.flashdev/home`, `.flashdev/server.pid`, `.flashdev/server.log`, `.flashdev/server.port`, and `.flashdev/lifecycle.lock` from that checkout.
2. confirm the selected checkout's `flash/server/app.py` contains the exact `FLASH_LOCAL_CONTROL_PLANE=1` safety mode. Probe a checkout without the server change and confirm `serve` fails closed before launch with the compatibility error.
3. snapshot the checksum of the real user's `~/.flash/config.json` if it exists.
4. run `./dev/flashdev serve`, optionally with `--checkout`. Confirm `.flashdev/server.port` contains one valid numeric port, derive the health URL as `http://127.0.0.1:<port>`, and confirm the lifecycle lock is released after the launcher exits.
5. read `.flashdev/server.pid`. Use `ps` only to confirm the recorded launcher command line contains `python -m flash.server`, the checkout-derived `HOME=<selected-checkout>/.flashdev/home` argument, and the exact `--port <port>` argument. This is an ownership marker, not proof of the live process environment or child behavior.
6. run `./dev/flashdev verify` and confirm health plus the internal account response. Pass a conflicting explicit port to `status`, `verify`, and `stop`, and confirm each refuses before associating the recorded PID with that port.
7. confirm `.flashdev/home/.flash/server.db` exists. If a config file was created, confirm it is under `.flashdev/home/.flash/`, not the real user's home.
8. while the server owns its recorded port, run another `serve` targeting that port and confirm it fails fast without replacing the recorded PID. Create a lifecycle lock owned by a live test PID and confirm both `serve` and `stop` refuse it; create one owned by a stopped numeric PID and confirm the next mutating command reclaims it. With a disposable slow-start checkout, send TERM to the launcher during its health wait and confirm it stops the tracked child, removes the matching PID file, and releases its lock.
9. confirm `cli login`, `cli --debug login`, `cli --verbose login`, `cli -v login`, `cli -vv login`, and `cli -- login` are rejected.
10. run `./dev/flashdev stop`. Confirm the managed process exited before `.flashdev/server.pid` disappeared and the lifecycle lock was released.
11. confirm the shared `~/.flash/config.json` checksum is unchanged.
