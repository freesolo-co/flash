from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Literal

SchemaStatus = Literal["ready", "retry", "authorization", "terminal"]
_RETRYABLE_STATUSES = {0, 400, 404, 406}


def classify_schema_status(status: int) -> SchemaStatus:
    if status == 200:
        return "ready"
    if status in {401, 403}:
        return "authorization"
    if status in _RETRYABLE_STATUSES or 500 <= status <= 599:
        return "retry"
    return "terminal"


def _probe(endpoint: str, service_role_key: str, timeout_seconds: float) -> int:
    request = urllib.request.Request(
        endpoint,
        headers={
            "apikey": service_role_key,
            "Accept-Profile": "flash",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except OSError:
        return 0


def wait_for_schema(
    *,
    supabase_url: str,
    service_role_key: str,
    table: str,
    columns: str,
    label: str,
    timeout_seconds: float = 300.0,
    interval_seconds: float = 10.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    probe: Callable[[str, str, float], int] = _probe,
) -> None:
    query = urllib.parse.urlencode({"select": columns, "limit": "1"})
    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{table}?{query}"
    deadline = clock() + timeout_seconds
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise RuntimeError(
                f"{label} did not become queryable within {timeout_seconds:g} seconds"
            )
        status = probe(endpoint, service_role_key, min(10.0, remaining))
        classification = classify_schema_status(status)
        if classification == "ready":
            print(f"{label} is queryable.", flush=True)
            return
        if classification == "authorization":
            raise RuntimeError(f"{label} readiness check was rejected with HTTP {status}")
        if classification == "terminal":
            raise RuntimeError(f"{label} readiness check failed with HTTP {status}")
        print(f"{label} is not ready yet (HTTP {status:03d}); retrying.", flush=True)
        remaining = deadline - clock()
        if remaining > 0:
            sleep(min(interval_seconds, remaining))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--columns", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    try:
        wait_for_schema(
            supabase_url=os.environ["SUPABASE_URL"],
            service_role_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"],
            table=args.table,
            columns=args.columns,
            label=args.label,
            timeout_seconds=args.timeout_seconds,
        )
    except RuntimeError as error:
        print(f"::error::{error}", flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
