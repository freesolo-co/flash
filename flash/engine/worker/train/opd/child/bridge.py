"""Teacher-bridge transport and failure reporting for the OPD verl child.

Copied into the isolated verl child workdir as `flash_opd_bridge.py`. Everything here is about
talking to the parent's localhost bridge and about what to do when that conversation fails: post a
json request, classify the error, and drop a fallback file on disk so the parent can attribute the
child's exit even when the bridge itself is what broke.

Split out of the OPD child plugin to keep that module under the file-size limit. It deliberately
does not import the plugin back: both files land flat in the child workdir, and a cycle there would
fail at import time inside verl rather than here.
"""

from __future__ import annotations

import contextlib
import functools
import http.client
import json
import os
import tempfile
import traceback
import types
import urllib.error
import urllib.request

if __name__ == "flash_opd_bridge":
    from flash_child_diagnostics import neutralize_control_chars, sanitize_diagnostic
else:
    from flash._internal.diagnostics import neutralize_control_chars, sanitize_diagnostic

# duplicated rather than imported from the plugin: this module is copied flat into the child
# workdir alongside it, and importing back would make the pair circular there. the parent reads
# these same two numbers off the child's exit status, so they must stay in step.
_PERMANENT_TEACHER_EXIT = 86
_TRANSIENT_TEACHER_EXIT = 87
_FAILURE_FALLBACK_MAX_CHARS = 8192
_ROLLOUT_FAILURE_KIND = "multiturn_rollout_exception"
_ROLLOUT_FAILURE_VERSION = 1
_ROLLOUT_FAILURE_TYPE_MAX_CHARS = 200
_ROLLOUT_FAILURE_MESSAGE_MAX_CHARS = 2000
_ROLLOUT_FAILURE_TRACEBACK_MAX_CHARS = 5000


def _plugin():
    """The child plugin module, imported lazily because it re-exports this one.

    Four names here are patched as attributes of `plugin` by the opd tests -- `_post_json`,
    `_mutation_distributed`, `_publish_mutation_notice`, `_coordinate_first_mutation_notice` --
    and every caller of theirs lives in this file. Resolving them through the plugin is what
    keeps those patches effective; a direct call would bind this module's own function. The flat
    name is the one that exists in the verl child workdir, where `flash` is not importable.
    """
    if __name__ == "flash_opd_bridge":
        import flash_opd_plugin as plugin
    else:
        from flash.engine.worker.train.opd.child import plugin

    return plugin


class FlashTeacherBridgeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        classification: str,
        delivery_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.delivery_unknown = delivery_unknown


class _DeferredScoreFailure(RuntimeError):
    def __init__(self, message: str, *, classification: str, exit_code: int) -> None:
        super().__init__(message)
        self.classification = classification
        self.exit_code = exit_code


def _post_json(url: str, token: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
            details = payload["error"]
            classification = str(details["classification"])
            message = str(details["message"])
        except (KeyError, TypeError, ValueError) as decode_error:
            raise FlashTeacherBridgeError(
                f"flash OPD bridge returned unclassified HTTP {error.code}",
                classification="permanent",
            ) from decode_error
        if classification not in {"permanent", "transient"}:
            raise FlashTeacherBridgeError(
                f"flash OPD bridge returned unknown teacher failure classification {classification!r}",
                classification="permanent",
            ) from error
        raise FlashTeacherBridgeError(message, classification=classification) from error
    except (OSError, http.client.HTTPException) as error:
        raise FlashTeacherBridgeError(
            f"flash OPD bridge transport failed: {type(error).__name__}",
            classification="transient",
            delivery_unknown=True,
        ) from error
    try:
        return json.loads(body.decode("utf-8"))
    except (TypeError, ValueError) as error:
        raise FlashTeacherBridgeError(
            "flash OPD bridge returned malformed success JSON",
            classification="permanent",
            delivery_unknown=True,
        ) from error


def _serialize_failure_fallback(classification: str, message: str) -> bytes:
    message = str(message).encode("utf-8", errors="replace").decode("utf-8")
    lower = 0
    upper = min(len(message), _FAILURE_FALLBACK_MAX_CHARS)
    serialized = json.dumps(
        {"classification": classification, "message": ""},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    while lower <= upper:
        length = (lower + upper) // 2
        candidate = json.dumps(
            {"classification": classification, "message": message[:length]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(candidate) <= _FAILURE_FALLBACK_MAX_CHARS:
            serialized = candidate
            lower = length + 1
        else:
            upper = length - 1
    return serialized


def _write_failure_fallback(
    base_path: str,
    classification: str,
    message: str,
) -> None:
    if not base_path:
        return
    process_id = os.getpid()
    record_path = f"{base_path}.{process_id}.{classification}.json"
    directory = os.path.dirname(base_path) or "."
    prefix = f".{os.path.basename(base_path)}.{process_id}.{classification}."
    payload = _serialize_failure_fallback(classification, message)
    temporary_path = ""
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            dir=directory,
            prefix=prefix,
            suffix=".tmp",
        )
        try:
            remaining = payload
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("mutation failure fallback write did not progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary_path, record_path)
        temporary_path = ""
    except OSError:
        if temporary_path:
            with contextlib.suppress(OSError):
                os.unlink(temporary_path)


def _rollout_failure_payload(
    classification: str,
    exception_type: str,
    message: str,
    traceback_text: str,
) -> bytes:
    record = {
        "version": _ROLLOUT_FAILURE_VERSION,
        "kind": _ROLLOUT_FAILURE_KIND,
        "classification": classification,
        "exception_type": exception_type,
        "message": message,
        "traceback": traceback_text,
    }

    def encode() -> bytes:
        return json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    payload = encode()
    for field in ("traceback", "message"):
        if len(payload) <= _FAILURE_FALLBACK_MAX_CHARS:
            break
        value = str(record[field])
        lower = 0
        upper = len(value)
        best = ""
        while lower <= upper:
            length = (lower + upper) // 2
            record[field] = value[:length]
            candidate = encode()
            if len(candidate) <= _FAILURE_FALLBACK_MAX_CHARS:
                best = str(record[field])
                lower = length + 1
            else:
                upper = length - 1
        record[field] = best
        payload = encode()
    return payload


def _serialize_rollout_failure(error: BaseException, classification: str) -> bytes:
    try:
        exception_type = neutralize_control_chars(
            sanitize_diagnostic(type(error).__name__, limit=_ROLLOUT_FAILURE_TYPE_MAX_CHARS)
        )
    except BaseException:
        exception_type = "BaseException"
    try:
        message = neutralize_control_chars(
            sanitize_diagnostic(str(error), limit=_ROLLOUT_FAILURE_MESSAGE_MAX_CHARS)
        )
        traceback_text = neutralize_control_chars(
            sanitize_diagnostic(
                "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                limit=_ROLLOUT_FAILURE_TRACEBACK_MAX_CHARS,
            )
        )
    except BaseException:
        message = "diagnostic rendering failed"
        traceback_text = ""
    return _rollout_failure_payload(
        classification,
        exception_type,
        message,
        traceback_text,
    )


def _write_rollout_failure_fallback(error: BaseException, classification: str) -> None:
    base_path = os.environ.get("FLASH_OPD_ROLLOUT_FAILURE_PATH", "")
    if not base_path or classification not in {"permanent", "transient"}:
        return
    record_path = f"{base_path}.json"
    directory = os.path.dirname(base_path) or "."
    try:
        payload = _serialize_rollout_failure(error, classification)
    except BaseException:
        return
    temporary_path = ""
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            dir=directory,
            prefix=f".{os.path.basename(base_path)}.",
            suffix=".tmp",
        )
        try:
            remaining = payload
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("rollout failure fallback write did not progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary_path, record_path)
        except FileExistsError:
            return
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError:
        return
    finally:
        if temporary_path:
            with contextlib.suppress(OSError):
                os.unlink(temporary_path)


def _read_rollout_failure_fallback(base_path: str) -> dict[str, str] | None:
    if not base_path:
        return None
    try:
        with open(f"{base_path}.json", encoding="utf-8") as file:
            encoded = file.read(_FAILURE_FALLBACK_MAX_CHARS + 1)
        if len(encoded.encode("utf-8")) > _FAILURE_FALLBACK_MAX_CHARS:
            return None
        record = json.loads(encoded)
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("version") != _ROLLOUT_FAILURE_VERSION:
        return None
    if record.get("kind") != _ROLLOUT_FAILURE_KIND:
        return None
    classification = record.get("classification")
    exception_type = record.get("exception_type")
    message = record.get("message")
    traceback_text = record.get("traceback")
    if classification not in {"permanent", "transient"}:
        return None
    if not isinstance(exception_type, str) or not exception_type.strip():
        return None
    if not isinstance(message, str) or not isinstance(traceback_text, str):
        return None
    return {
        "classification": classification,
        "exception_type": exception_type.strip(),
        "message": message.strip(),
        "traceback": traceback_text.strip(),
    }


def _render_rollout_failure(record: dict[str, str], *, limit: int = 8192) -> str:
    detail = f"{record['exception_type']}: {record['message']}"
    if record["traceback"]:
        detail = f"{detail}\n{record['traceback']}"
    return sanitize_diagnostic(detail, limit=limit)


def _write_mutation_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_MUTATION_FAILURE_PATH", ""),
        classification,
        message,
    )


def _write_score_delivery_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH", ""),
        classification,
        message,
    )


def _write_abandonment_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_ABANDONMENT_FAILURE_PATH", ""),
        classification,
        message,
    )


def _write_resample_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_RESAMPLE_FAILURE_PATH", ""),
        classification,
        message,
    )


def _write_cycle_commit_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH", ""),
        classification,
        message,
    )


def _fallback_classification(error: FlashTeacherBridgeError) -> str:
    return "transient" if error.delivery_unknown else error.classification


def _prepare_score_failure(error: FlashTeacherBridgeError) -> tuple[str, int, str]:
    """classify one score failure and persist only the evidence the parent cannot already have.

    an explicitly rejected score is already recorded parent-side as a teacher failure, so writing a
    delivery record for it would relabel a teacher rejection as a delivery failure. only a
    delivery-unknown outcome leaves the parent with nothing to read.
    """
    classification = _fallback_classification(error)
    message = sanitize_diagnostic(str(error), limit=_FAILURE_FALLBACK_MAX_CHARS)
    if error.delivery_unknown:
        _write_score_delivery_failure_fallback(classification, message)
    exit_code = (
        _PERMANENT_TEACHER_EXIT if classification == "permanent" else _TRANSIENT_TEACHER_EXIT
    )
    return classification, exit_code, message


def _defer_score_failure(error: FlashTeacherBridgeError) -> None:
    """record the score failure here, but let the async caller exit after marking the prompt.

    exiting on this executor thread would kill the process before the prompt is marked failed,
    which is the wedge itself: verl keeps polling a prompt that stays `running` forever.
    """
    classification, exit_code, message = _prepare_score_failure(error)
    raise _DeferredScoreFailure(
        message,
        classification=classification,
        exit_code=exit_code,
    ) from error


def _exit_for_score_failure(error: FlashTeacherBridgeError) -> None:
    _classification, exit_code, _message = _prepare_score_failure(error)
    os._exit(exit_code)


def _fatal_rollout_exit_code(error: BaseException) -> int:
    """map a fatal rollout error to the exit code the parent already classifies."""
    if isinstance(error, _DeferredScoreFailure):
        return error.exit_code
    if getattr(error, "classification", None) == "transient":
        return _TRANSIENT_TEACHER_EXIT
    return _PERMANENT_TEACHER_EXIT


def _unexpected_mutation_bridge_error(error: Exception) -> FlashTeacherBridgeError:
    return FlashTeacherBridgeError(
        f"unexpected mutation bridge failure: {type(error).__name__}",
        classification="permanent",
    )


def _publish_mutation_notice(url: str, token: str) -> None:
    payload = {"process_id": os.getpid()}
    try:
        _plugin()._post_json(url, token, "/mutation", payload)
        return
    except FlashTeacherBridgeError as error:
        if not error.delivery_unknown:
            _write_mutation_failure_fallback(error.classification, str(error))
            raise
    except Exception as error:
        bridge_error = _unexpected_mutation_bridge_error(error)
        _write_mutation_failure_fallback(
            bridge_error.classification,
            str(bridge_error),
        )
        raise bridge_error from error

    try:
        _plugin()._post_json(url, token, "/mutation", payload)
    except FlashTeacherBridgeError as error:
        _write_mutation_failure_fallback(error.classification, str(error))
        raise
    except Exception as error:
        bridge_error = _unexpected_mutation_bridge_error(error)
        _write_mutation_failure_fallback(
            bridge_error.classification,
            str(bridge_error),
        )
        raise bridge_error from error


def _post_no_signal_resample(url: str, token: str) -> None:
    try:
        _plugin()._post_json(url, token, "/no-signal/resample", {})
    except FlashTeacherBridgeError as error:
        _write_resample_failure_fallback(_fallback_classification(error), str(error))
        raise
    except Exception as error:
        _write_resample_failure_fallback(
            "permanent",
            f"unexpected resample bridge failure: {type(error).__name__}",
        )
        raise


def _post_no_signal_abandoned(url: str, token: str) -> dict:
    """Tell the bridge this step is abandoned; return its per-reason skip tally for the error text."""
    try:
        response = _plugin()._post_json(url, token, "/no-signal/abandoned", {})
        return dict((response or {}).get("skip_counts") or {})
    except FlashTeacherBridgeError as error:
        _write_abandonment_failure_fallback(
            _fallback_classification(error),
            str(error),
        )
        raise
    except Exception as error:
        _write_abandonment_failure_fallback(
            "permanent",
            f"unexpected abandonment bridge failure: {type(error).__name__}",
        )
        raise


def _post_teacher_cycle_committed(url: str, token: str) -> None:
    try:
        _plugin()._post_json(url, token, "/teacher-cycle/committed", {})
        return
    except FlashTeacherBridgeError as error:
        if not error.delivery_unknown:
            _write_cycle_commit_failure_fallback(error.classification, str(error))
            raise
    except Exception as error:
        _write_cycle_commit_failure_fallback(
            "permanent",
            f"unexpected cycle commitment bridge failure: {type(error).__name__}",
        )
        raise

    try:
        _plugin()._post_json(url, token, "/teacher-cycle/committed", {})
    except FlashTeacherBridgeError as error:
        _write_cycle_commit_failure_fallback(
            _fallback_classification(error),
            str(error),
        )
        raise
    except Exception as error:
        _write_cycle_commit_failure_fallback(
            "permanent",
            f"unexpected cycle commitment bridge failure: {type(error).__name__}",
        )
        raise


def _update_actor_after_teacher_cycle_commit(
    actor_rollout_wg,
    batch,
    url: str,
    token: str,
):
    _post_teacher_cycle_committed(url, token)
    return actor_rollout_wg.update_actor(batch)


def _mutation_distributed():
    import torch

    return torch.distributed


def _coordinate_first_mutation_notice(url: str, token: str) -> None:
    distributed = _plugin()._mutation_distributed()
    distributed.barrier()
    outcome: list[tuple[str, str] | None] = [None]
    if distributed.get_rank() == 0:
        try:
            _plugin()._publish_mutation_notice(url, token)
        except FlashTeacherBridgeError as error:
            outcome[0] = (error.classification, str(error))
    distributed.broadcast_object_list(outcome, src=0)
    distributed.barrier()
    if outcome[0] is not None:
        classification, message = outcome[0]
        raise FlashTeacherBridgeError(message, classification=classification)


def _wrap_optimizer_with_mutation_notice(optimizer, url: str, token: str):
    original_step = optimizer.step
    mutation_acknowledged = False

    # torch's lr scheduler patches optimizer.step to assert step ordering, and that patch reads
    # step_fn.__func__ and rebinds it with func.__get__(opt, opt.__class__). both only exist on a
    # bound method, so assigning a plain function here fails with "'function' object has no
    # attribute '__func__'" as soon as verl builds its warmup scheduler, before any gpu work
    # happens. bind the wrapper to the instance so the descriptor protocol keeps working.
    @functools.wraps(original_step)
    def step_with_notice(_optimizer, *args, **kwargs):
        nonlocal mutation_acknowledged
        if not mutation_acknowledged:
            _plugin()._coordinate_first_mutation_notice(url, token)
            mutation_acknowledged = True
        return original_step(*args, **kwargs)

    optimizer.step = types.MethodType(step_with_notice, optimizer)
    return optimizer
