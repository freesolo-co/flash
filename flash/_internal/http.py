"""private stdlib http transport policies shared across flash clients."""

from __future__ import annotations

import copy
import inspect
import os
import struct
import threading
import types
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_UrlOpen = Callable[..., Any]
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_STDLIB_URLOPEN_ARGUMENTS = (
    "url",
    "data",
    "timeout",
    "cafile",
    "capath",
    "cadefault",
    "context",
)
_STDLIB_URLOPEN_NAMES = (
    "warnings",
    "warn",
    "DeprecationWarning",
    "ValueError",
    "_have_ssl",
    "ssl",
    "create_default_context",
    "Purpose",
    "SERVER_AUTH",
    "set_alpn_protocols",
    "HTTPSHandler",
    "build_opener",
    "_opener",
    "open",
)
_HANDLER_COPY_ERROR = "installed urllib handler cannot be copied safely"
_OPENER_COPY_ERROR = "installed urllib opener cannot be copied safely"
_SNAPSHOT_DEPTH_MAX = 8
_SNAPSHOT_ITEMS_MAX = 256
_ABSENT_SLOT = object()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    handler_order = 0

    def http_error_redirect(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)

    def http_response(self, request, response):
        if 300 <= response.code < 400:
            raise urllib.error.HTTPError(
                request.full_url,
                response.code,
                response.msg,
                response.info(),
                response,
            )
        return response

    http_error_301 = http_error_redirect
    http_error_302 = http_error_redirect
    http_error_303 = http_error_redirect
    http_error_307 = http_error_redirect
    http_error_308 = http_error_redirect
    https_response = http_response


@dataclass(frozen=True)
class _InstalledOpenerCache:
    installed: urllib.request.OpenerDirector
    handler_signature: tuple[int, tuple[tuple[int, object], ...]]
    addheader_signature: tuple[int, tuple[tuple[str, str], ...]]
    private: urllib.request.OpenerDirector


def _build_no_redirect_opener(*handlers: object) -> urllib.request.OpenerDirector:
    """build a private opener that returns redirects as httperrors."""

    blocker = _NoRedirectHandler()
    opener = urllib.request.build_opener(blocker, *handlers)
    for protocol in ("http", "https"):
        processors = opener.process_response.get(protocol)
        if processors is not None:
            processors.remove(blocker)
            processors.insert(0, blocker)
    return opener


_OPENER_LOCK = threading.Lock()
_DEFAULT_NO_REDIRECT_OPENER: urllib.request.OpenerDirector | None = None
_INSTALLED_OPENER_CACHE: _InstalledOpenerCache | None = None


def _handles_redirect_error(handler: urllib.request.BaseHandler) -> bool:
    try:
        return any(
            callable(inspect.getattr_static(handler, f"http_error_{status}", None))
            for status in _REDIRECT_STATUSES
        )
    except Exception:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None


def _snapshot_value(value: object, depth: int = 0) -> object:
    value_type = type(value)
    if depth > _SNAPSHOT_DEPTH_MAX:
        raise ValueError
    if value is None or value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type is float:
        return (float, struct.pack("!d", value))
    if value_type in (list, tuple):
        if len(value) > _SNAPSHOT_ITEMS_MAX:
            raise ValueError
        return (value_type, tuple(_snapshot_value(item, depth + 1) for item in value))
    if value_type is dict:
        if len(value) > _SNAPSHOT_ITEMS_MAX:
            raise ValueError
        return (
            dict,
            frozenset(
                (_snapshot_value(key, depth + 1), _snapshot_value(item, depth + 1))
                for key, item in value.items()
            ),
        )
    if value_type in (set, frozenset):
        if len(value) > _SNAPSHOT_ITEMS_MAX:
            raise ValueError
        return (value_type, frozenset(_snapshot_value(item, depth + 1) for item in value))
    return ("opaque", id(value_type), id(value))


def _slot_names(declaration: object) -> tuple[str, ...]:
    if type(declaration) is str:
        names = (declaration,)
    elif type(declaration) in (list, tuple, set, frozenset):
        names = tuple(declaration)
    elif type(declaration) is dict:
        names = tuple(declaration.keys())
    else:
        raise TypeError
    if any(type(name) is not str for name in names):
        raise TypeError
    return names


def _mangled_slot_name(owner: type, name: str) -> str:
    if name.startswith("__") and not name.endswith("__"):
        class_name = type.__getattribute__(owner, "__name__").lstrip("_")
        if class_name:
            return f"_{class_name}{name}"
    return name


def _slot_config_signature(handler: urllib.request.BaseHandler) -> tuple[object, ...]:
    try:
        found = []
        seen: set[int] = set()
        handler_type = type(handler)
        for owner in type.__getattribute__(handler_type, "__mro__"):
            namespace = type.__getattribute__(owner, "__dict__")
            declaration = namespace.get("__slots__", ())
            for declared_name in _slot_names(declaration):
                if declared_name in {"__dict__", "__weakref__", "parent"}:
                    continue
                slot_name = _mangled_slot_name(owner, declared_name)
                descriptor = namespace.get(slot_name)
                if type(descriptor) is not types.MemberDescriptorType:
                    raise TypeError
                if id(descriptor) in seen:
                    continue
                seen.add(id(descriptor))
                try:
                    value = types.MemberDescriptorType.__get__(descriptor, handler, handler_type)
                except AttributeError:
                    snapshot = _ABSENT_SLOT
                else:
                    snapshot = _snapshot_value(value)
                found.append((id(owner), slot_name, snapshot))
        return tuple(found)
    except Exception:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None


def _handler_config_signature(handler: urllib.request.BaseHandler) -> object:
    try:
        state = object.__getattribute__(handler, "__dict__")
        if type(state) is not dict or any(type(name) is not str for name in state):
            raise TypeError
        dictionary = frozenset(
            (name, _snapshot_value(value)) for name, value in state.items() if name != "parent"
        )
    except Exception:
        raise urllib.error.URLError(_OPENER_COPY_ERROR) from None
    return (dictionary, _slot_config_signature(handler))


def _copy_installed_handler(
    handler: urllib.request.BaseHandler,
    opener: urllib.request.OpenerDirector,
) -> urllib.request.BaseHandler:
    try:
        copied = copy.copy(handler)
        if isinstance(handler, urllib.request.ProxyHandler):
            urllib.request.ProxyHandler.__init__(copied, handler.proxies)
        invalid = (
            copied is handler
            or type(copied) is not type(handler)
            or copied.__dict__ is handler.__dict__
            or getattr(copied, "parent", None) is not opener
        )
    except Exception:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None
    if invalid:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR)
    return copied


def _validate_installed_opener(opener: urllib.request.OpenerDirector) -> None:
    try:
        state = object.__getattribute__(opener, "__dict__")
        class_open = inspect.getattr_static(type(opener), "open", None)
        instance_open = inspect.getattr_static(opener, "open", None)
    except Exception:
        raise urllib.error.URLError(_OPENER_COPY_ERROR) from None
    standard_open = urllib.request.OpenerDirector.open
    if (
        type(state) is not dict
        or "open" in state
        or class_open is not standard_open
        or instance_open is not standard_open
    ):
        raise urllib.error.URLError(_OPENER_COPY_ERROR)


def _installed_config(
    opener: urllib.request.OpenerDirector,
) -> tuple[
    tuple[urllib.request.BaseHandler, ...],
    list[tuple[str, str]],
    tuple[int, tuple[tuple[int, object], ...]],
    tuple[int, tuple[tuple[str, str], ...]],
]:
    _validate_installed_opener(opener)
    try:
        handlers = tuple(opener.handlers)
        addheaders = []
        for item in opener.addheaders:
            name, value = item
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError
            addheaders.append((name, value))
    except Exception:
        raise urllib.error.URLError(_OPENER_COPY_ERROR) from None
    handler_signature = (
        id(opener.handlers),
        tuple((id(handler), _handler_config_signature(handler)) for handler in handlers),
    )
    addheader_signature = (id(opener.addheaders), tuple(addheaders))
    return handlers, addheaders, handler_signature, addheader_signature


def _clone_installed_opener(
    installed: urllib.request.OpenerDirector,
    handlers: tuple[urllib.request.BaseHandler, ...],
    addheaders: list[tuple[str, str]],
) -> urllib.request.OpenerDirector:
    copied = [
        _copy_installed_handler(handler, installed)
        for handler in handlers
        if not _handles_redirect_error(handler)
    ]
    private = _build_no_redirect_opener(*copied)
    private.addheaders = list(addheaders)
    return private


def _default_no_redirect_opener() -> urllib.request.OpenerDirector:
    global _DEFAULT_NO_REDIRECT_OPENER

    if _DEFAULT_NO_REDIRECT_OPENER is None:
        _DEFAULT_NO_REDIRECT_OPENER = _build_no_redirect_opener()
    return _DEFAULT_NO_REDIRECT_OPENER


def _active_no_redirect_opener() -> urllib.request.OpenerDirector:
    global _INSTALLED_OPENER_CACHE

    with _OPENER_LOCK:
        for _attempt in range(3):
            installed = urllib.request._opener
            if installed is None:
                return _default_no_redirect_opener()
            handlers, addheaders, handler_signature, addheader_signature = _installed_config(
                installed
            )
            cached = _INSTALLED_OPENER_CACHE
            if (
                cached is not None
                and cached.installed is installed
                and cached.handler_signature == handler_signature
            ):
                if cached.addheader_signature != addheader_signature:
                    cached.private.addheaders = list(addheaders)
                    cached = _InstalledOpenerCache(
                        installed,
                        handler_signature,
                        addheader_signature,
                        cached.private,
                    )
                    _INSTALLED_OPENER_CACHE = cached
                return cached.private
            private = _clone_installed_opener(installed, handlers, addheaders)
            current = _installed_config(installed)
            if urllib.request._opener is installed and current[2:] == (
                handler_signature,
                addheader_signature,
            ):
                _INSTALLED_OPENER_CACHE = _InstalledOpenerCache(
                    installed,
                    handler_signature,
                    addheader_signature,
                    private,
                )
                return private
        raise urllib.error.URLError(_OPENER_COPY_ERROR)


def _is_stdlib_urlopen(transport: object) -> bool:
    if type(transport) is not types.FunctionType:
        return False
    try:
        code = object.__getattribute__(transport, "__code__")
        module_file = os.path.realpath(urllib.request.__file__)
        code_file = os.path.realpath(code.co_filename)
        arguments = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        return (
            object.__getattribute__(transport, "__module__") == urllib.request.__name__
            and object.__getattribute__(transport, "__name__") == "urlopen"
            and object.__getattribute__(transport, "__qualname__") == "urlopen"
            and code_file == module_file
            and code.co_name == "urlopen"
            and code.co_argcount == 3
            and code.co_kwonlyargcount == 4
            and arguments == _STDLIB_URLOPEN_ARGUMENTS
            and code.co_names == _STDLIB_URLOPEN_NAMES
        )
    except Exception:
        return False


def _urlopen_no_redirect(
    request: urllib.request.Request,
    *,
    timeout: float,
    urlopen: _UrlOpen | None = None,
):
    """open one authenticated request without allowing a credential-bearing redirect."""

    transport = urllib.request.urlopen if urlopen is None else urlopen
    if not _is_stdlib_urlopen(transport):
        return transport(request, timeout=timeout)
    opener = _active_no_redirect_opener()
    return opener.open(request, timeout=timeout)
