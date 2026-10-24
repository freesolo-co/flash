import json
import os
import secrets
import time
from pathlib import Path

import modal

root = Path(os.environ.get("KV_AB_WORKDIR", "/tmp/flash-deepinfer-kv-ab"))
root.mkdir(parents=True, exist_ok=True)
app = modal.App.lookup("deepinfer-kv-ab-20261023", environment_name="main")
token = secrets.token_urlsafe(32)
p = root / "relay-token.secret"
p.write_text(token)
p.chmod(0o600)
with modal.enable_output():
    sb = modal.Sandbox.create(
        "python",
        "-u",
        "-c",
        Path(__file__).with_name("relay_server.py").read_text(),
        app=app,
        image=modal.Image.debian_slim(python_version="3.12"),
        cpu=(1, 1),
        memory=(4096, 4096),
        timeout=10800,
        encrypted_ports=[8443],
        env={"KV_RELAY_TOKEN": token},
    )
url = sb.tunnels()[8443].url
record = {
    "id": sb.object_id,
    "url": url,
    "app_id": app.app_id,
    "started": time.time(),
    "reserved_usd": 2,
}
(root / "relay.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
