"""private stdlib http transport policies shared across flash clients."""

from __future__ import annotations

import enum
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

from .http_refs import (
    _SNAPSHOT_ITEMS_MAX,
    _TRAVERSAL_NODES_MAX,
    _getattr_type_static,
)
from .http_refs import (
    _references_target as _find_references_target,
)

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
_URLOPEN_CLASSIFICATION_ERROR = "stdlib urllib transport cannot be classified safely"
_HANDLER_COPY_METHODS = (
    "__copy__",
    "__getattribute__",
    "__getattr__",
    "__getnewargs__",
    "__getnewargs_ex__",
    "__getstate__",
    "__reduce__",
    "__reduce_ex__",
    "__setstate__",
    "__setattr__",
    "add_parent",
)
_OPENER_REQUEST_METHODS = (
    "__getattribute__",
    "__getattr__",
    "open",
    "_open",
    "_call_chain",
    "error",
)
_ABSENT_SLOT = object()
_BASE_HANDLER_DICT_DESCRIPTOR = inspect.getattr_static(urllib.request.BaseHandler, "__dict__")
_OPENER_DICT_DESCRIPTOR = inspect.getattr_static(urllib.request.OpenerDirector, "__dict__")
_STANDARD_HANDLER_METHODS = tuple(
    (
        name,
        inspect.getattr_static(urllib.request.BaseHandler, name, _ABSENT_SLOT),
    )
    for name in _HANDLER_COPY_METHODS
)


def _getattr_handler_static(handler: object, name: str, default: object) -> object:
    state = types.GetSetDescriptorType.__get__(
        _BASE_HANDLER_DICT_DESCRIPTOR,
        handler,
        type(handler),
    )
    if type(state) is not dict:
        raise TypeError
    if name in state:
        return state[name]
    return _getattr_type_static(type(handler), name, default)


def _require_standard_add_parent(handler: object) -> None:
    standard = urllib.request.BaseHandler.add_parent
    if _getattr_handler_static(handler, "add_parent", _ABSENT_SLOT) is not standard:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    handler_order = 0

    def http_error_redirect(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)

    def http_response(self, request, response):
        if response is None:
            return None
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


class _UrlopenKind(enum.Enum):
    STDLIB = enum.auto()
    INJECTED = enum.auto()
    UNKNOWN_STDLIB = enum.auto()


@dataclass(frozen=True)
class _InstalledOpenerCache:
    installed: urllib.request.OpenerDirector
    handler_signature: tuple[int, tuple[tuple[int, object], ...]]
    addheader_signature: tuple[int, tuple[tuple[str, str], ...]]
    private: urllib.request.OpenerDirector


def _build_no_redirect_opener(
    *handlers: object,
    add_default_handlers: bool = True,
) -> urllib.request.OpenerDirector:
    """build a private opener that returns redirects as httperrors."""

    blocker = _NoRedirectHandler()
    for handler in handlers:
        _require_standard_add_parent(handler)
    if add_default_handlers:
        opener = urllib.request.build_opener(blocker, *handlers)
    else:
        opener = urllib.request.OpenerDirector()
        opener.add_handler(blocker)
        for handler in handlers:
            _require_standard_add_parent(handler)
            opener.add_handler(handler)
    for protocol in ("http", "https"):
        processors = opener.process_response.get(protocol)
        if processors is not None:
            processors.remove(blocker)
            processors.insert(0, blocker)
    return opener


_OPENER_LOCK = threading.Lock()
_DEFAULT_NO_REDIRECT_OPENER: urllib.request.OpenerDirector | None = None
_INSTALLED_OPENER_CACHE: _InstalledOpenerCache | None = None
_STDLIB_INSTALL_OPENER = urllib.request.install_opener


def _install_opener_with_cache_reset(opener) -> None:
    global _DEFAULT_NO_REDIRECT_OPENER, _INSTALLED_OPENER_CACHE

    with _OPENER_LOCK:
        _STDLIB_INSTALL_OPENER(opener)
        if opener is None:
            _DEFAULT_NO_REDIRECT_OPENER = None
            _INSTALLED_OPENER_CACHE = None


urllib.request.install_opener = _install_opener_with_cache_reset


def _handles_redirect_error(handler: urllib.request.BaseHandler) -> bool:
    try:
        return any(
            callable(_getattr_handler_static(handler, f"http_error_{status}", None))
            for status in _REDIRECT_STATUSES
        )
    except Exception:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None


def _snapshot_value(
    value: object,
    seen: set[int] | None = None,
    active: set[int] | None = None,
    budget: list[int] | None = None,
) -> object:
    if seen is None:
        seen = set()
    if active is None:
        active = set()
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > _TRAVERSAL_NODES_MAX:
        raise ValueError
    value_type = type(value)
    if value is None or value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type is float:
        return (float, struct.pack("!d", value))
    if value_type in (list, tuple, dict, set, frozenset):
        value_id = id(value)
        if value_id in active:
            raise ValueError
        if value_id in seen:
            return ("alias", value_id)
        seen.add(value_id)
        active.add(value_id)
        try:
            if len(value) > _SNAPSHOT_ITEMS_MAX:
                raise ValueError
            if value_type in (list, tuple):
                snapshot = tuple(_snapshot_value(item, seen, active, budget) for item in value)
            elif value_type is dict:
                snapshot = frozenset(
                    (
                        _snapshot_value(key, seen, active, budget),
                        _snapshot_value(item, seen, active, budget),
                    )
                    for key, item in value.items()
                )
            else:
                snapshot = frozenset(_snapshot_value(item, seen, active, budget) for item in value)
        finally:
            active.remove(value_id)
        return (value_type, snapshot)
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


def _slot_entries(
    handler: urllib.request.BaseHandler,
) -> tuple[tuple[int, str, types.MemberDescriptorType, object], ...]:
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
                value = _ABSENT_SLOT
            found.append((id(owner), slot_name, descriptor, value))
    return tuple(found)


def _slot_values(
    handler: urllib.request.BaseHandler,
) -> tuple[tuple[int, str, object], ...]:
    return tuple(
        (owner_id, slot_name, value)
        for owner_id, slot_name, _descriptor, value in _slot_entries(handler)
    )


def _copy_slot_state(
    handler: urllib.request.BaseHandler,
    copied: urllib.request.BaseHandler,
) -> None:
    for _owner_id, _slot_name, descriptor, value in _slot_entries(handler):
        if value is not _ABSENT_SLOT:
            types.MemberDescriptorType.__set__(descriptor, copied, value)


def _slot_config_signature(
    handler: urllib.request.BaseHandler,
    seen: set[int],
    active: set[int],
    budget: list[int],
) -> tuple[object, ...]:
    try:
        return tuple(
            (
                owner_id,
                slot_name,
                value
                if value is _ABSENT_SLOT
                else _snapshot_value(value, seen=seen, active=active, budget=budget),
            )
            for owner_id, slot_name, value in _slot_values(handler)
        )
    except Exception:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None


def _handler_state(handler: urllib.request.BaseHandler) -> dict[str, object]:
    state = types.GetSetDescriptorType.__get__(
        _BASE_HANDLER_DICT_DESCRIPTOR,
        handler,
        type(handler),
    )
    if type(state) is not dict or any(type(name) is not str for name in state):
        raise TypeError
    return state


def _validate_handler_copy_protocol(
    handler: urllib.request.BaseHandler,
    state: dict[str, object],
) -> None:
    handler_type = type(handler)
    metaclass = type(handler_type)
    if (
        _getattr_type_static(metaclass, "__getattribute__", _ABSENT_SLOT)
        is not type.__getattribute__
    ):
        raise TypeError
    for name, standard in _STANDARD_HANDLER_METHODS:
        if (
            name in state
            or _getattr_type_static(handler_type, name, _ABSENT_SLOT) is not standard
            or _getattr_handler_static(handler, name, _ABSENT_SLOT) is not standard
        ):
            raise TypeError


def _handler_config_signature(handler: urllib.request.BaseHandler) -> object:
    try:
        state = _handler_state(handler)
        seen: set[int] = set()
        active: set[int] = set()
        budget = [0]
        dictionary = frozenset(
            (
                name,
                _snapshot_value(value, seen=seen, active=active, budget=budget),
            )
            for name, value in state.items()
            if name != "parent"
        )
    except Exception:
        raise urllib.error.URLError(_OPENER_COPY_ERROR) from None
    return (dictionary, _slot_config_signature(handler, seen, active, budget))


def _references_target(
    value: object,
    targets: tuple[object, ...],
    seen: set[int] | None = None,
    active: set[int] | None = None,
    budget: list[int] | None = None,
    inspect_global_object: bool = False,
    trusted_objects: tuple[object, ...] = (),
) -> bool:
    return _find_references_target(
        value,
        targets,
        seen,
        active,
        budget,
        inspect_global_object,
        _TRAVERSAL_NODES_MAX,
        trusted_objects,
    )


def _trusted_shallow_clone(
    handler: urllib.request.BaseHandler,
    state: dict[str, object],
) -> tuple[urllib.request.BaseHandler, dict[str, object]]:
    copied = object.__new__(type(handler))
    copied_state = dict.copy(state)
    types.GetSetDescriptorType.__set__(
        _BASE_HANDLER_DICT_DESCRIPTOR,
        copied,
        copied_state,
    )
    _copy_slot_state(handler, copied)
    return copied, copied_state


def _proxy_callback(
    method: types.MethodType,
    proxy: str,
    proxy_type: str,
) -> types.FunctionType:
    def callback(request, proxy=proxy, proxy_type=proxy_type, method=method):
        return method(request, proxy, proxy_type)

    return callback


def _rebuild_proxy_callbacks(
    copied: urllib.request.BaseHandler,
    copied_state: dict[str, object],
) -> None:
    proxies = copied_state.get("proxies", _ABSENT_SLOT)
    if type(proxies) is not dict or any(
        type(name) is not str or type(url) is not str for name, url in proxies.items()
    ):
        raise TypeError
    if "proxy_open" in copied_state:
        raise TypeError
    proxy_open = _getattr_type_static(type(copied), "proxy_open", _ABSENT_SLOT)
    if type(proxy_open) is not types.FunctionType:
        raise TypeError
    method = types.MethodType(proxy_open, copied)
    for proxy_type, proxy in proxies.items():
        normalized_type = str.lower(proxy_type)
        copied_state[f"{normalized_type}_open"] = _proxy_callback(
            method,
            proxy,
            normalized_type,
        )


def _registered_class_callbacks(
    handler: urllib.request.BaseHandler,
    state: dict[str, object],
) -> tuple[types.FunctionType, ...]:
    callbacks = []
    seen = set(state)
    for owner in type.__getattribute__(type(handler), "__mro__"):
        namespace = type.__getattribute__(owner, "__dict__")
        if namespace.get("__module__") == urllib.request.__name__:
            continue
        for name, value in namespace.items():
            if name in seen:
                continue
            seen.add(name)
            if name in {"redirect_request", "do_open", "proxy_open"} or "_" not in name:
                continue
            condition = name.split("_", 1)[1]
            if not (
                condition == "open"
                or condition == "request"
                or condition == "response"
                or condition.startswith("error")
            ):
                continue
            if type(value) is not types.FunctionType:
                raise TypeError
            if object.__getattribute__(value, "__module__") == urllib.request.__name__:
                continue
            callbacks.append(value)
    return tuple(callbacks)


def _copy_installed_handler(
    handler: urllib.request.BaseHandler,
    opener: urllib.request.OpenerDirector,
) -> urllib.request.BaseHandler:
    try:
        handler_state = _handler_state(handler)
        _validate_handler_copy_protocol(handler, handler_state)
        if handler_state.get("parent", _ABSENT_SLOT) is not opener:
            raise TypeError
        copied, copied_state = _trusted_shallow_clone(handler, handler_state)
        _validate_handler_copy_protocol(copied, copied_state)
        if isinstance(handler, urllib.request.ProxyHandler):
            _rebuild_proxy_callbacks(copied, copied_state)
        if copied_state is handler_state or copied_state.get("parent", _ABSENT_SLOT) is not opener:
            raise TypeError
        copied_values = (
            tuple(value for name, value in copied_state.items() if name != "parent")
            + tuple(
                value
                for _owner_id, _slot_name, value in _slot_values(copied)
                if value is not _ABSENT_SLOT
            )
            + _registered_class_callbacks(copied, copied_state)
        )
        reference_seen: set[int] = set()
        reference_active: set[int] = set()
        reference_budget = [0]
        retains_target = False
        for value in copied_values:
            retains_target = (
                _references_target(
                    value,
                    (handler, opener),
                    seen=reference_seen,
                    active=reference_active,
                    budget=reference_budget,
                    trusted_objects=(copied,),
                )
                or retains_target
            )
        if retains_target:
            raise TypeError
    except Exception:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None
    return copied


def _opener_state(opener: urllib.request.OpenerDirector) -> dict[str, object]:
    state = types.GetSetDescriptorType.__get__(
        _OPENER_DICT_DESCRIPTOR,
        opener,
        type(opener),
    )
    if type(state) is not dict:
        raise TypeError
    return state


def _validate_installed_opener(opener: urllib.request.OpenerDirector) -> None:
    try:
        state = _opener_state(opener)
        if any(name in state for name in _OPENER_REQUEST_METHODS):
            raise TypeError
        for name in _OPENER_REQUEST_METHODS:
            standard = inspect.getattr_static(urllib.request.OpenerDirector, name, _ABSENT_SLOT)
            if (
                inspect.getattr_static(type(opener), name, _ABSENT_SLOT) is not standard
                or inspect.getattr_static(opener, name, _ABSENT_SLOT) is not standard
            ):
                raise TypeError
    except Exception:
        raise urllib.error.URLError(_OPENER_COPY_ERROR) from None


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
        state = _opener_state(opener)
        raw_handlers = state["handlers"]
        raw_addheaders = state["addheaders"]
        if type(raw_handlers) is not list or type(raw_addheaders) is not list:
            raise TypeError
        handlers = tuple(raw_handlers)
        addheaders = []
        for item in raw_addheaders:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError
            name = item[0]
            value = item[1]
            if type(name) is not str or type(value) is not str:
                raise TypeError
            addheaders.append((name, value))
    except Exception:
        raise urllib.error.URLError(_OPENER_COPY_ERROR) from None
    handler_signature = (
        id(raw_handlers),
        tuple((id(handler), _handler_config_signature(handler)) for handler in handlers),
    )
    addheader_signature = (id(raw_addheaders), tuple(addheaders))
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
    private = _build_no_redirect_opener(*copied, add_default_handlers=False)
    try:
        if any(
            _handler_state(handler).get("parent", _ABSENT_SLOT) is not private for handler in copied
        ):
            raise TypeError
    except Exception:
        raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None
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
                _INSTALLED_OPENER_CACHE = None
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


def _classify_urlopen(transport: object) -> _UrlopenKind:
    if type(transport) is not types.FunctionType:
        return _UrlopenKind.INJECTED
    try:
        metadata_matches = (
            object.__getattribute__(transport, "__module__") == urllib.request.__name__
            and object.__getattribute__(transport, "__name__") == "urlopen"
            and object.__getattribute__(transport, "__qualname__") == "urlopen"
        )
    except Exception:
        return _UrlopenKind.INJECTED
    if not metadata_matches:
        return _UrlopenKind.INJECTED
    try:
        code = object.__getattribute__(transport, "__code__")
        module_file = os.path.realpath(urllib.request.__file__)
        code_file = os.path.realpath(code.co_filename)
        arguments = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        shape_matches = (
            code.co_name == "urlopen"
            and code.co_argcount == 3
            and code.co_kwonlyargcount == 4
            and arguments == _STDLIB_URLOPEN_ARGUMENTS
        )
    except Exception:
        return _UrlopenKind.UNKNOWN_STDLIB
    if not shape_matches:
        return _UrlopenKind.INJECTED
    if code_file != module_file or code.co_names != _STDLIB_URLOPEN_NAMES:
        return _UrlopenKind.UNKNOWN_STDLIB
    return _UrlopenKind.STDLIB


def _urlopen_no_redirect(
    request: urllib.request.Request,
    *,
    timeout: float,
    urlopen: _UrlOpen | None = None,
):
    """open one authenticated request without allowing a credential-bearing redirect."""

    transport = urllib.request.urlopen
    if urlopen is not None:
        if urlopen is not transport:
            return urlopen(request, timeout=timeout)
        transport = urlopen
    kind = _classify_urlopen(transport)
    if kind is _UrlopenKind.INJECTED:
        return transport(request, timeout=timeout)
    if kind is _UrlopenKind.UNKNOWN_STDLIB:
        raise urllib.error.URLError(_URLOPEN_CLASSIFICATION_ERROR)
    opener = _active_no_redirect_opener()
    return opener.open(request, timeout=timeout)
