"""Bounded framed IPC shared by deployment-smoke worker processes."""

from __future__ import annotations

import base64
import json
import os
import select
import struct
import time

from flash.serve.contract.errors import ServingError

_IPC_FRAME_HEADER_BYTES = 4
_IPC_MAX_PAYLOAD_BYTES = 1_048_576


class _IpcDeadlineExceeded(Exception):
    """the framed IPC reader received no frame before its deadline."""


def _write_all(fd: int, value: bytes, *, deadline: float, description: str) -> None:
    view = memoryview(value)
    was_blocking = os.get_blocking(fd)
    if was_blocking:
        os.set_blocking(fd, False)
    try:
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _IpcDeadlineExceeded
            _, writable, _ = select.select([], [fd], [], remaining)
            if not writable:
                raise _IpcDeadlineExceeded
            try:
                written = os.write(fd, view)
            except BlockingIOError:
                continue
            if written <= 0:
                raise OSError(f"{description} IPC write made no progress")
            view = view[written:]
            if view and time.monotonic() >= deadline:
                raise _IpcDeadlineExceeded
    finally:
        if was_blocking:
            os.set_blocking(fd, True)


def _encode_json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_encode_json_value(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "value": [_encode_json_value(item) for item in value]}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "value": [
                [_encode_json_value(key), _encode_json_value(item)] for key, item in value.items()
            ],
        }
    raise TypeError("unsupported isolated worker IPC value")


def _decode_json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if not isinstance(value, dict) or set(value) != {"type", "value"}:
        raise ValueError("invalid IPC value envelope")
    kind = value["type"]
    encoded = value["value"]
    if kind == "bytes" and isinstance(encoded, str):
        return base64.b64decode(encoded, validate=True)
    if kind in {"tuple", "list"} and isinstance(encoded, list):
        items = [_decode_json_value(item) for item in encoded]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict" and isinstance(encoded, list):
        decoded = {}
        for pair in encoded:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid IPC dictionary item")
            key, item = (_decode_json_value(part) for part in pair)
            decoded[key] = item
        return decoded
    raise ValueError("invalid IPC value type")


def _send_framed_ipc(
    connection,
    value,
    *,
    deadline: float,
    description: str,
) -> None:
    """send one bounded JSON frame under one absolute deadline."""
    payload = json.dumps(
        _encode_json_value(value),
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(payload) > _IPC_MAX_PAYLOAD_BYTES:
        raise ServingError(
            f"{description} produced an oversized IPC frame ({len(payload)} bytes; "
            f"maximum {_IPC_MAX_PAYLOAD_BYTES})"
        )
    if time.monotonic() >= deadline:
        raise _IpcDeadlineExceeded
    fd = connection.fileno()
    _write_all(
        fd,
        struct.pack("!I", len(payload)),
        deadline=deadline,
        description=description,
    )
    _write_all(fd, payload, deadline=deadline, description=description)


def _read_exact(
    fd: int,
    size: int,
    *,
    deadline: float,
    description: str,
    frame_started: bool = False,
) -> bytes:
    value = bytearray()
    while len(value) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if frame_started or value:
                raise ServingError(f"{description} returned a truncated IPC frame")
            raise _IpcDeadlineExceeded
        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            if frame_started or value:
                raise ServingError(f"{description} returned a truncated IPC frame")
            raise _IpcDeadlineExceeded
        chunk = os.read(fd, size - len(value))
        if not chunk:
            raise ServingError(f"{description} returned a truncated IPC frame")
        value.extend(chunk)
    return bytes(value)


def _receive_framed_ipc(connection, *, deadline: float, description: str):
    """receive one size-limited frame while enforcing the deadline on every read."""
    fd = connection.fileno()
    header = _read_exact(
        fd,
        _IPC_FRAME_HEADER_BYTES,
        deadline=deadline,
        description=description,
    )
    (size,) = struct.unpack("!I", header)
    if size > _IPC_MAX_PAYLOAD_BYTES:
        raise ServingError(
            f"{description} returned an oversized IPC frame ({size} bytes; "
            f"maximum {_IPC_MAX_PAYLOAD_BYTES})"
        )
    payload = _read_exact(
        fd,
        size,
        deadline=deadline,
        description=description,
        frame_started=True,
    )
    try:
        encoded = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
        return _decode_json_value(encoded)
    except (TypeError, ValueError) as exc:
        raise ServingError(f"{description} returned a malformed IPC frame") from exc
