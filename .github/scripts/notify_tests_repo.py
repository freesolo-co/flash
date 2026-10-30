"""Dispatch integration tests, respecting GitHub's primary and secondary rate limits."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

DISPATCH_URL = "https://api.github.com/repos/freesolo-co/tests/dispatches"
MAX_ATTEMPTS = 5
MAX_WAIT_SECONDS = 65 * 60


def retry_delay(error: urllib.error.HTTPError, message: str, attempt: int) -> float | None:
    """A permission-denied 403 is permanent; a rate-limit 403 is retryable."""
    if error.code in (403, 429):
        if value := error.headers.get("Retry-After"):
            return max(1, float(value))
        if error.headers.get("X-RateLimit-Remaining") == "0" and (
            reset := error.headers.get("X-RateLimit-Reset")
        ):
            return max(1, float(reset) - time.time() + 1)
        if error.code == 429 or "rate limit" in message.lower():
            return 60 * 2**attempt
    if 500 <= error.code < 600:
        return 2 ** (attempt + 1)
    return None


def dispatch(token: str, sha: str, ref: str) -> None:
    payload = json.dumps({"event_type": "flash-merged", "client_payload": {"sha": sha, "ref": ref}})
    request = urllib.request.Request(
        DISPATCH_URL,
        data=payload.encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 204:
                    raise RuntimeError(f"Unexpected dispatch response: HTTP {response.status}")
            print(f"Dispatched flash-merged for {ref} at {sha}.")
            return
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                message = str(json.loads(body)["message"])
            except (ValueError, KeyError, TypeError):
                message = body[:1000]
            message = message.replace(token, "***")
            delay = retry_delay(error, message, attempt)
            if delay is None:
                raise RuntimeError(
                    f"GitHub dispatch failed (HTTP {error.code}): {message}. "
                    "Check FREESOLO_GITHUB_PAT access to freesolo-co/tests "
                    "(Contents: write for a fine-grained PAT, or repo for a classic PAT)."
                ) from None
            if attempt + 1 == MAX_ATTEMPTS or time.monotonic() + delay > deadline:
                raise RuntimeError(
                    f"GitHub dispatch retry budget exhausted (HTTP {error.code}): {message}. "
                    "Re-run this workflow after the rate limit or service outage clears."
                ) from None
            print(f"GitHub returned HTTP {error.code}; retrying in {delay:.0f}s.", flush=True)
            time.sleep(delay)


def main() -> int:
    token = os.environ.get("PAT", "")
    try:
        if not token:
            raise RuntimeError(
                "FREESOLO_GITHUB_PAT is not set. Add a credential with access to "
                "freesolo-co/tests; the integration suites have NOT been notified."
            )
        dispatch(token, os.environ["SHA"], os.environ["REF"])
    except (RuntimeError, urllib.error.URLError, TimeoutError, ValueError) as error:
        message = str(error)
        if token:
            message = message.replace(token, "***")
        message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
