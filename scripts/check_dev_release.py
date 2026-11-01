"""Decide whether the dev version is absent from PyPI without trusting cached 404s."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def is_published(version: str) -> bool:
    """Retry briefly for index propagation; lookup errors never authorize an upload."""
    url = f"https://pypi.org/pypi/freesolo-flash-dev/{urllib.parse.quote(version, safe='')}/json"
    for delay in (0, 2, 5):
        if delay:
            time.sleep(delay)
        # The preceding publish decision can cache a 404 before the version is uploaded.
        # A later dev push must not reuse that negative response (PyPI caches for 900s).
        # Use a fresh cache key on every attempt as well as requesting revalidation.
        request = urllib.request.Request(
            f"{url}?flash_release_probe={uuid.uuid4().hex}",
            headers={"Cache-Control": "no-cache"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"Unexpected PyPI release status: {response.status}")
                return True
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
    return False


def main() -> None:
    # stdout is appended to GITHUB_OUTPUT by the workflow. Do not emit a decision on error.
    published = is_published(os.environ["VERSION"])
    print(f"publish={'false' if published else 'true'}")


if __name__ == "__main__":
    main()
