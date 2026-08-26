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
    _function_append_only_capture_values,
    _function_bound_reference_values,
    _function_capture_values,
    _function_global_reference_values,
    _function_reference_values,
    _getattr_type_static,
    _slot_entries,
    _source_function_code,
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


def _is_registered_callback_name(name: str) -> bool:
    if name in {"redirect_request", "do_open", "proxy_open"} or "_" not in name:
        return False
    condition = name.split("_", 1)[1]
    return (
        condition == "open"
        or condition == "request"
        or condition == "response"
        or condition.startswith("error")
    )


_STDLIB_HANDLER_TYPES = (
    urllib.request.BaseHandler,
    urllib.request.HTTPErrorProcessor,
    urllib.request.HTTPDefaultErrorHandler,
    urllib.request.HTTPRedirectHandler,
    urllib.request.ProxyHandler,
    urllib.request.AbstractBasicAuthHandler,
    urllib.request.AbstractDigestAuthHandler,
    urllib.request.HTTPBasicAuthHandler,
    urllib.request.ProxyBasicAuthHandler,
    urllib.request.HTTPDigestAuthHandler,
    urllib.request.ProxyDigestAuthHandler,
    urllib.request.AbstractHTTPHandler,
    urllib.request.HTTPHandler,
    urllib.request.HTTPSHandler,
    urllib.request.HTTPCookieProcessor,
    urllib.request.UnknownHandler,
    urllib.request.FileHandler,
    urllib.request.FTPHandler,
    urllib.request.CacheFTPHandler,
    urllib.request.DataHandler,
)
_STDLIB_HANDLER_CALLBACKS = tuple(
    value
    for handler_type in _STDLIB_HANDLER_TYPES
    for name, value in type.__getattribute__(handler_type, "__dict__").items()
    if _is_registered_callback_name(name) and type(value) is types.FunctionType
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


def _function_implementation_signature(function: types.FunctionType) -> tuple[object, ...]:
    code = object.__getattribute__(function, "__code__")
    defaults = object.__getattribute__(function, "__defaults__")
    kwdefaults = object.__getattribute__(function, "__kwdefaults__")
    closure = object.__getattribute__(function, "__closure__")
    if (
        type(code) is not types.CodeType
        or (defaults is not None and type(defaults) is not tuple)
        or (kwdefaults is not None and type(kwdefaults) is not dict)
        or (closure is not None and type(closure) is not tuple)
    ):
        raise TypeError
    seen: set[int] = set()
    active: set[int] = set()
    budget = [0]
    closure_values = []
    for cell in closure or ():
        try:
            value = object.__getattribute__(cell, "cell_contents")
        except ValueError:
            value = _ABSENT_SLOT
        closure_values.append(_snapshot_value(value, seen, active, budget))
    return (
        code,
        _snapshot_value(defaults, seen, active, budget),
        _snapshot_value(kwdefaults, seen, active, budget),
        tuple(closure_values),
    )


def _stdlib_function_signature(function: types.FunctionType) -> tuple[object, ...]:
    qualname = object.__getattribute__(function, "__qualname__")
    if type(qualname) is not str or type(urllib.request.__file__) is not str:
        raise TypeError
    return (
        _source_function_code(urllib.request.__file__, qualname),
        _snapshot_value(None),
        _snapshot_value(None),
        (),
    )


_STDLIB_HANDLER_CALLBACK_SIGNATURES = tuple(
    (callback, _stdlib_function_signature(callback)) for callback in _STDLIB_HANDLER_CALLBACKS
)
_STANDARD_DO_OPEN = urllib.request.AbstractHTTPHandler.do_open
_STANDARD_DO_OPEN_SIGNATURE = _stdlib_function_signature(_STANDARD_DO_OPEN)
_STANDARD_PROXY_CALLBACK_CODE = next(
    value
    for value in urllib.request.ProxyHandler.__init__.__code__.co_consts
    if type(value) is types.CodeType and value.co_name == "<lambda>"
)


def _slot_values(
    handler: urllib.request.BaseHandler,
) -> tuple[tuple[int, str, object], ...]:
    return tuple(
        (owner_id, slot_name, value)
        for owner_id, slot_name, _descriptor, value in _slot_entries(handler, _ABSENT_SLOT)
    )


def _copy_slot_state(
    handler: urllib.request.BaseHandler,
    copied: urllib.request.BaseHandler,
) -> None:
    for _owner_id, _slot_name, descriptor, value in _slot_entries(handler, _ABSENT_SLOT):
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


def _redirect_config_signature(handler: urllib.request.BaseHandler) -> tuple[tuple[int, bool], ...]:
    return tuple(
        (id(value), callable(value))
        for status in _REDIRECT_STATUSES
        for value in (_getattr_handler_static(handler, f"http_error_{status}", None),)
    )


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
    return (
        dictionary,
        _slot_config_signature(handler, seen, active, budget),
        _redirect_config_signature(handler),
    )


def _references_target(
    value: object,
    targets: tuple[object, ...],
    seen: set[int] | None = None,
    active: set[int] | None = None,
    budget: list[int] | None = None,
    inspect_global_object: bool = False,
    inspect_method_receiver: bool = False,
    trusted_objects: tuple[object, ...] = (),
) -> bool:
    return _find_references_target(
        value,
        targets,
        seen,
        active,
        budget,
        inspect_global_object,
        inspect_method_receiver,
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


_REBUILT_PROXY_CALLBACK_CODE = next(
    value
    for value in _proxy_callback.__code__.co_consts
    if type(value) is types.CodeType and value.co_name == "callback"
)


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


def _is_trusted_stdlib_callback(value: types.FunctionType) -> bool:
    for callback, signature in _STDLIB_HANDLER_CALLBACK_SIGNATURES:
        if value is callback:
            if _function_implementation_signature(value) != signature:
                raise TypeError
            return True
    return False


def _validate_callback_helpers(handler: urllib.request.BaseHandler) -> None:
    handler_type = type(handler)
    for callback_name, standard in (
        ("http_open", urllib.request.HTTPHandler.http_open),
        ("https_open", urllib.request.HTTPSHandler.https_open),
    ):
        callback = _getattr_type_static(handler_type, callback_name, _ABSENT_SLOT)
        if callback is not standard:
            continue
        helper = _getattr_handler_static(handler, "do_open", _ABSENT_SLOT)
        if (
            helper is not _STANDARD_DO_OPEN
            or _function_implementation_signature(helper) != _STANDARD_DO_OPEN_SIGNATURE
        ):
            raise TypeError


def _proxy_callback_names(
    handler: urllib.request.BaseHandler,
    state: dict[str, object],
) -> frozenset[str]:
    if not isinstance(handler, urllib.request.ProxyHandler):
        return frozenset()
    proxies = state.get("proxies", _ABSENT_SLOT)
    if type(proxies) is not dict or any(
        type(name) is not str or type(url) is not str for name, url in proxies.items()
    ):
        raise TypeError
    return frozenset(f"{str.lower(name)}_open" for name in proxies)


def _validate_proxy_instance_callback(
    handler: urllib.request.BaseHandler,
    name: str,
    callback: object,
    state: dict[str, object],
) -> bool:
    if name not in _proxy_callback_names(handler, state):
        return False
    if type(callback) is not types.FunctionType:
        raise TypeError
    code = object.__getattribute__(callback, "__code__")
    defaults = object.__getattribute__(callback, "__defaults__")
    if (
        code not in {_STANDARD_PROXY_CALLBACK_CODE, _REBUILT_PROXY_CALLBACK_CODE}
        or type(defaults) is not tuple
        or len(defaults) != 3
    ):
        raise TypeError
    proxy, proxy_type, method = defaults
    expected_type = name.removesuffix("_open")
    expected_proxy = _ABSENT_SLOT
    for configured_type, configured_proxy in state["proxies"].items():
        if str.lower(configured_type) == expected_type:
            expected_proxy = configured_proxy
    if (
        type(proxy) is not str
        or proxy != expected_proxy
        or proxy_type != expected_type
        or type(method) is not types.MethodType
        or object.__getattribute__(method, "__self__") is not handler
        or object.__getattribute__(method, "__func__")
        is not _getattr_type_static(type(handler), "proxy_open", _ABSENT_SLOT)
    ):
        raise TypeError
    return True


def _registered_instance_callbacks(
    handler: urllib.request.BaseHandler,
    state: dict[str, object],
) -> tuple[object, ...]:
    callbacks = []
    for name, value in state.items():
        if not _is_registered_callback_name(name):
            continue
        if _validate_proxy_instance_callback(handler, name, value, state):
            continue
        callbacks.append(value)
    return tuple(callbacks)


def _registered_class_callbacks(
    handler: urllib.request.BaseHandler,
    state: dict[str, object],
) -> tuple[types.FunctionType, ...]:
    callbacks = []
    seen = set(state)
    for owner in type.__getattribute__(type(handler), "__mro__"):
        namespace = type.__getattribute__(owner, "__dict__")
        for name, value in namespace.items():
            if name in seen:
                continue
            seen.add(name)
            if not _is_registered_callback_name(name):
                continue
            if type(value) is not types.FunctionType:
                raise TypeError
            if _is_trusted_stdlib_callback(value):
                continue
            callbacks.append(value)
    return tuple(callbacks)


def _validate_instance_callbacks(
    handler: urllib.request.BaseHandler,
    state: dict[str, object],
    targets: tuple[object, ...],
    private_targets: tuple[object, ...] = (),
) -> None:
    seen: set[int] = set()
    active: set[int] = set()
    budget = [0]
    for callback in _registered_instance_callbacks(handler, state):
        if _references_target(
            callback,
            (*targets, *private_targets),
            seen=seen,
            active=active,
            budget=budget,
            inspect_method_receiver=True,
        ):
            raise TypeError
        if private_targets and type(callback) is types.MethodType:
            function = object.__getattribute__(callback, "__func__")
            callback_self = object.__getattribute__(callback, "__self__")
            for reference in _function_bound_reference_values(function, callback_self):
                if _references_target(reference, private_targets):
                    raise TypeError


def _validate_class_callbacks(
    handler: urllib.request.BaseHandler,
    state: dict[str, object],
    targets: tuple[object, ...],
    trusted_objects: tuple[object, ...] = (),
    private_targets: tuple[object, ...] = (),
) -> None:
    seen: set[int] = set()
    active: set[int] = set()
    budget = [0]
    _validate_callback_helpers(handler)
    for callback in _registered_class_callbacks(handler, state):
        for reference in _function_reference_values(callback, handler):
            if _references_target(
                reference,
                targets,
                seen=seen,
                active=active,
                budget=budget,
                trusted_objects=trusted_objects,
            ):
                raise TypeError
        if not private_targets:
            continue
        append_only = _function_append_only_capture_values(callback)
        for capture in _function_capture_values(callback):
            if _references_target(capture, private_targets) and not (
                type(capture) is list and any(capture is observer for observer in append_only)
            ):
                raise TypeError
        for reference in (
            *_function_global_reference_values(callback),
            *_function_bound_reference_values(callback, handler),
        ):
            if _references_target(reference, private_targets):
                raise TypeError


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
        _validate_callback_helpers(copied)
        _validate_instance_callbacks(copied, copied_state, (handler, opener))
        copied_values = (
            tuple(
                value
                for name, value in copied_state.items()
                if name != "parent" and not _is_registered_callback_name(name)
            )
            + tuple(
                value
                for _owner_id, _slot_name, value in _slot_values(copied)
                if value is not _ABSENT_SLOT
            )
            + tuple(
                reference
                for callback in _registered_class_callbacks(copied, copied_state)
                for reference in _function_reference_values(callback, copied)
            )
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
                try:
                    for handler in handlers:
                        if not _handles_redirect_error(handler):
                            state = _handler_state(handler)
                            private_targets = (cached.private, *cached.private.handlers)
                            _validate_instance_callbacks(
                                handler,
                                state,
                                (handler, installed),
                                private_targets,
                            )
                            _validate_class_callbacks(
                                handler,
                                state,
                                (handler, installed),
                                private_targets,
                                private_targets,
                            )
                except Exception:
                    raise urllib.error.URLError(_HANDLER_COPY_ERROR) from None
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
