"""Shared test helpers for the flash suite (importable via ``tests._helpers``)."""


def wait_until_leading(client, seconds: float = 10.0) -> None:
    """Wait until a TestClient's plane leads: it serves before its background recovery finishes."""
    import time

    deadline = time.monotonic() + seconds
    while getattr(client.app.state, "plane_standby", False):
        assert time.monotonic() < deadline, "control plane never finished recovery"
        time.sleep(0.01)
