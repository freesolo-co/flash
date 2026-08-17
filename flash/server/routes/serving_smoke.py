"""The deployment smoke test: generate once against the new adapter and validate what comes back.

A deployment is only marked ready after this passes, so everything here runs against untrusted
output under a hard wall-clock budget. Two of those bounds are the reason this is its own module:
a user-supplied structured-output regex is evaluated with a timeout, and a user-supplied JSON
schema is compiled in a spawned process that is reaped whether or not it answers -- a catastrophic
pattern or a schema that hangs the validator must not wedge the serving route.

Split out of `flash.server.routes.serving` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import multiprocessing
import time

import regex as safe_regex
from jsonschema import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable

from flash.adapters.lora_rank import serving_completion_token_capacity
from flash.content.structured_outputs import parse_structured_outputs
from flash.core.spec import JobSpec
from flash.serve.deploy import AliasThinkingSilent, RetryableServingUnavailable, ServingError
from flash.serve.preflight import (
    SERVING_PROMPT_TOKEN_ALLOWANCE,
    ExternalSchemaReference,
    reject_external_schema_reference,
    resolve_smoke_completion_tokens,
    validate_local_json_schema,
    validate_structured_output_patterns,
)
from flash.server import app as _app


def _serving():
    """The route module, imported lazily because it re-exports this one.

    `validator_for` is patched as an attribute of `flash.server.routes.serving` by the schema
    coverage test, which makes the validator factory raise to prove the spawned worker reports the
    failure instead of hanging. Binding it by value here would capture the real factory at import
    time, so the patch would rebind a name this module never reads.
    """
    from flash.server.routes import serving

    return serving


_SMOKE_PROMPT = "Deployment smoke test: answer in one short sentence. What is 2+2?"
_SMOKE_IMAGE_PROMPT = (
    "Look at the square in the attached image. What color is it? Reply with one color word."
)
_SMOKE_IMAGE_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR42u3NQQ0AAAgEoNP+"
    "nTWFDzcoQE1udQQCgUAgEAgEAsGTYAGjxAE/G/Q2tQAAAABJRU5ErkJggg=="
)
_SMOKE_IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": _SMOKE_IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": _SMOKE_IMAGE_DATA_URI}},
        ],
    }
]
# hard wall-clock budget for the trusted fixed-prompt generation and validation.
# it remains below the deployment stale threshold.
_SMOKE_BUDGET_SECONDS = 600.0


def _smoke_timeout_error(budget_s: float) -> ServingError:
    return ServingError(f"deployment_smoke_timeout: bounded smoke exceeded {budget_s:g}s")


def _bounded_call(fn, *, deadline: float, budget_s: float):
    """run the trusted smoke call within the remaining global deadline."""
    import threading

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _smoke_timeout_error(budget_s)
    result: dict = {}

    def _target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # re-raised on the caller thread below
            result["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=remaining)
    if thread.is_alive():
        raise _smoke_timeout_error(budget_s)
    if "error" in result:
        raise result["error"]
    return result.get("value")


_DIRECT_REGEX_TIMEOUT_SECONDS = 0.05
_JSON_SCHEMA_TIMEOUT_SECONDS = 3.0
_JSON_SCHEMA_PROCESS_NAME = "flash-json-schema-validation"


def _bounded_regex_fullmatch(pattern: str, value: str, *, deadline: float, budget_s: float):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _smoke_timeout_error(budget_s)
    global_deadline_limits = remaining <= _DIRECT_REGEX_TIMEOUT_SECONDS
    timeout = min(_DIRECT_REGEX_TIMEOUT_SECONDS, remaining)
    try:
        return safe_regex.fullmatch(pattern, value, timeout=timeout)
    except TimeoutError as exc:
        if global_deadline_limits:
            raise _smoke_timeout_error(budget_s) from exc
        raise ServingError(
            f"structured-output regex evaluation exceeded the {_DIRECT_REGEX_TIMEOUT_SECONDS:.2f}s deadline"
        ) from exc


def _sanitized_schema_error(exc: Exception) -> str:
    message = getattr(exc, "message", str(exc))
    return " ".join(str(message).split())[:500]


def _json_schema_validation_worker(connection, instance, schema) -> None:
    try:
        validator_class = validate_local_json_schema(
            schema, validator_factory=_serving().validator_for
        )
        registry = Registry(retrieve=reject_external_schema_reference)
        validator_class(schema, registry=registry).validate(instance)
    except (ExternalSchemaReference, Unresolvable) as exc:
        outcome = ("reference", _sanitized_schema_error(exc))
    except SchemaError as exc:
        outcome = ("schema", _sanitized_schema_error(exc))
    except ValidationError as exc:
        outcome = ("validation", _sanitized_schema_error(exc))
    except Exception as exc:
        outcome = ("error", f"{type(exc).__name__}: {_sanitized_schema_error(exc)}")
    else:
        outcome = ("ok", "")
    try:
        connection.send(outcome)
    finally:
        connection.close()


def _reap_schema_validation_process(process, *, deadline: float) -> None:
    remaining = max(0.0, deadline - time.monotonic())
    process.join(timeout=min(0.1, remaining))
    if process.is_alive():
        process.terminate()
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=min(0.2, remaining))
    if process.is_alive():
        process.kill()
        process.join()


def _validate_json_schema(instance, schema: dict, *, deadline: float, budget_s: float) -> None:
    now = time.monotonic()
    if deadline <= now:
        raise _smoke_timeout_error(budget_s)
    local_deadline = now + _JSON_SCHEMA_TIMEOUT_SECONDS
    validation_deadline = min(deadline, local_deadline)
    global_deadline_limits = deadline <= local_deadline
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_json_schema_validation_worker,
        args=(send_connection, instance, schema),
        name=_JSON_SCHEMA_PROCESS_NAME,
        daemon=True,
    )
    outcome: tuple[str, str] | None = None
    timed_out = False
    try:
        try:
            process.start()
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise _smoke_timeout_error(budget_s) from exc
            raise ServingError(f"could not start isolated JSON schema validation: {exc}") from exc
        finally:
            send_connection.close()
        remaining = max(0.0, validation_deadline - time.monotonic())
        if receive_connection.poll(remaining):
            try:
                outcome = receive_connection.recv()
            except EOFError:
                outcome = None
        else:
            timed_out = True
    finally:
        receive_connection.close()
        if process.pid is not None:
            _reap_schema_validation_process(process, deadline=deadline)
            process.close()

    if outcome is None:
        if timed_out and global_deadline_limits:
            raise _smoke_timeout_error(budget_s)
        raise ServingError(
            f"JSON schema validation exceeded the {_JSON_SCHEMA_TIMEOUT_SECONDS:.1f}s wall-clock deadline"
        )
    status, detail = outcome
    if status == "ok":
        return
    if status == "reference":
        raise ServingError(f"configured JSON schema reference could not be resolved: {detail}")
    if status == "schema":
        raise ServingError(f"configured JSON schema is invalid: {detail}")
    if status == "validation":
        raise ServingError(f"structured smoke output violates the configured JSON schema: {detail}")
    raise ServingError(f"JSON schema validation failed safely: {detail}")


def _strict_json_loads(value: str):
    def reject_constant(constant: str):
        raise ValueError(f"non-finite JSON constant {constant!r} is not allowed")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except ValueError as exc:
        raise ServingError(f"structured smoke output is not valid JSON: {exc}") from exc


def _smoke_provenance(result: dict, adapter_revision: str, checkpoint: str) -> tuple[str, object]:
    choice = (result.get("choices") or [{}])[0]
    content = str((choice.get("message") or {}).get("content") or "")
    finish = choice.get("finish_reason")
    if not content.strip():
        raise ServingError(f"smoke generation returned no content (finish_reason={finish!r})")
    provenance = result.get("freesolo")
    if not isinstance(provenance, dict):
        raise ServingError("smoke response omitted immutable revision provenance")
    if provenance.get("adapter_revision") != adapter_revision:
        raise ServingError("smoke response returned the wrong adapter revision")
    if provenance.get("checkpoint") != checkpoint:
        raise ServingError("smoke response returned the wrong checkpoint")
    hf_revision = adapter_revision.rsplit(".", 1)[-1]
    if provenance.get("hf_revision") != hf_revision:
        raise ServingError("smoke response returned the wrong Hub revision")
    headers = result.get("_freesolo_headers")
    expected_headers = {
        "adapter_revision": adapter_revision,
        "checkpoint": checkpoint,
        "hf_revision": hf_revision,
    }
    if headers != expected_headers:
        raise ServingError("smoke response returned mismatched provenance headers")
    return content, finish


def _smoke_answer(
    result: dict,
    spec: JobSpec,
    *,
    serving_model: str,
    expected_checkpoint: str,
) -> tuple[str, object, str]:
    content, finish = _smoke_provenance(result, serving_model, expected_checkpoint)
    if spec.thinking and finish == "length":
        raise ServingError("smoke generation was truncated at the maximum token length")
    answer = (
        _thinking_answer(content, require_tag=_thinking_tag_is_guaranteed(spec))
        if spec.thinking
        else content.strip()
    )
    return content, finish, answer


def _thinking_tag_is_guaranteed(spec) -> bool:
    """Whether the catalog vouches that this model's chat template opens a thinking block.

    A curated entry states its ``thinking`` capability, so the tag is required. Only a stale caller
    can present an uncataloged model -- submit rejects those -- and nothing vouches for its
    template, so the tag is not demanded of it.

    Asks the catalog DIRECTLY rather than through ``resolve_model``, which validates against an
    algorithm and raises for an uncataloged id. Whether a chat template opens a ``<think>`` block
    has nothing to do with either, and treating that raise as "guaranteed" would demand the tag
    from precisely the models that cannot promise it.
    """
    from flash.core.catalog import MODELS

    model = getattr(spec, "model", None)
    return isinstance(model, str) and model.strip() in MODELS


def _thinking_answer(content: str, *, require_tag: bool = True) -> str:
    """Return the answer a thinking adapter emitted after its reasoning, or reject the smoke.

    Stop sequences can end during reasoning while still reporting ``finish_reason=stop``; require an
    answer whenever the catalog guarantees a thinking tag. Unknown open models may omit the tag.
    """
    closed = content.find("</think>")
    if closed < 0:
        if not require_tag:
            return content.strip()
        raise ServingError(
            "smoke generation for a thinking adapter never closed its reasoning with </think>"
        )
    answer = content[closed + len("</think>") :].strip()
    if not answer:
        raise ServingError("smoke generation returned no answer after </think>")
    if answer == "</think>":
        # a compatibility backend that retains only the sampled close leaves the fold no answer to
        # place behind the block, so it emits the delimiter twice. that shape is indistinguishable
        # at the source from an adapter whose answer IS the tag, and folding deliberately defers the
        # call to here. neither is an answer to the smoke prompt, so reject both.
        raise ServingError("smoke generation returned only a close tag after </think>")
    return answer


def _validate_structured_smoke(
    answer: str, constraint: dict, *, deadline: float, budget_s: float
) -> None:
    if time.monotonic() >= deadline:
        raise _smoke_timeout_error(budget_s)
    try:
        validate_structured_output_patterns(constraint)
    except ValueError as exc:
        raise ServingError(f"configured structured-output {exc}") from exc
    try:
        if "json" in constraint:
            _validate_json_schema(
                _strict_json_loads(answer),
                constraint["json"],
                deadline=deadline,
                budget_s=budget_s,
            )
        elif constraint.get("json_object") is True:
            if not isinstance(_strict_json_loads(answer), dict):
                raise ServingError("structured smoke output is valid JSON but not a JSON object")
        elif "choice" in constraint:
            if answer not in constraint["choice"]:
                raise ServingError(
                    f"structured smoke output {answer!r} is not one of {constraint['choice']!r}"
                )
        elif (
            "regex" in constraint
            and _bounded_regex_fullmatch(
                str(constraint["regex"]), answer, deadline=deadline, budget_s=budget_s
            )
            is None
        ):
            raise ServingError("structured smoke output does not match the configured regex")
    except safe_regex.error as exc:
        raise ServingError(f"configured structured-output regex is invalid: {exc}") from exc


def _smoke_request_settings(spec: JobSpec) -> tuple[dict | None, int, list[str] | None]:
    train = getattr(spec, "train", None)
    constraint = parse_structured_outputs(getattr(train, "structured_outputs", ""))
    max_tokens = 256
    # thinking spends tokens before content regardless of grammar, so 256 can truncate the think
    # block and reject a healthy deployment. the resolver widens it to what the run reasons within,
    # bounded by a smoke-specific ceiling. a grammar lifts that ceiling because the smoke generates
    # under the adapter's serving default and its shortest legal answer may be longer.
    if spec.thinking:
        max_tokens = max(
            256, resolve_smoke_completion_tokens(spec, constrained=constraint is not None)
        )
        serving_capacity = serving_completion_token_capacity(
            spec, prompt_allowance=SERVING_PROMPT_TOKEN_ALLOWANCE
        )
        if serving_capacity is not None:
            max_tokens = min(max_tokens, serving_capacity)
    stop_sequences = [str(value) for value in (getattr(train, "stop_sequences", ()) or ())]
    return constraint, max_tokens, stop_sequences or None


def _bounded_smoke_chat(
    *,
    serving_model: str,
    thinking: bool,
    expected_checkpoint: str,
    messages: list[dict] | None = None,
    structured_outputs: dict | None = None,
    expected_adapter_revision: str,
    max_tokens: int,
    stop_sequences: list[str] | None,
    deadline: float,
    budget_s: float,
    error_context: str | None = None,
) -> dict:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _smoke_timeout_error(budget_s)
        try:

            def _chat_call(timeout_s: float = remaining):
                chat_kwargs = {
                    "run_id": serving_model,
                    "messages": (
                        messages
                        if messages is not None
                        else [{"role": "user", "content": _SMOKE_PROMPT}]
                    ),
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "thinking": thinking,
                    "expected_checkpoint": expected_checkpoint,
                    "expected_adapter_revision": expected_adapter_revision,
                    "timeout_s": timeout_s,
                    "retry_unavailable": True,
                    "stop": stop_sequences,
                }
                if structured_outputs is not None:
                    chat_kwargs["structured_outputs"] = structured_outputs
                return _app.serve_chat(**chat_kwargs)

            return _bounded_call(_chat_call, deadline=deadline, budget_s=budget_s)
        except RetryableServingUnavailable as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _smoke_timeout_error(budget_s) from exc
            time.sleep(min(exc.retry_after_seconds, remaining))
        except ServingError:
            raise
        except Exception as exc:
            if error_context is None:
                raise
            raise ServingError(f"{error_context}: {exc}") from exc


def _run_deployment_smoke(
    run_id: str,
    spec: JobSpec,
    *,
    serving_model: str,
    expected_checkpoint: str,
    budget_s: float = _SMOKE_BUDGET_SECONDS,
) -> dict:
    started = time.monotonic()
    deadline = started + budget_s
    constraint, max_tokens, stop_sequences = _smoke_request_settings(spec)
    from flash.core.catalog import supports_image_training

    image_capable = supports_image_training(spec.model)
    result = _bounded_smoke_chat(
        serving_model=serving_model,
        thinking=spec.thinking,
        expected_checkpoint=expected_checkpoint,
        expected_adapter_revision=serving_model,
        messages=_SMOKE_IMAGE_MESSAGES if image_capable else None,
        structured_outputs={} if image_capable and constraint is not None else None,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        deadline=deadline,
        budget_s=budget_s,
    )
    content, finish, answer = _smoke_answer(
        result,
        spec,
        serving_model=serving_model,
        expected_checkpoint=expected_checkpoint,
    )
    verify_turns = 1
    if image_capable:
        if answer.strip().upper() != "RED":
            raise ServingError("image deployment smoke did not identify the trusted red square")
        if constraint:
            structured_result = _bounded_smoke_chat(
                serving_model=serving_model,
                thinking=spec.thinking,
                expected_checkpoint=expected_checkpoint,
                expected_adapter_revision=serving_model,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                deadline=deadline,
                budget_s=budget_s,
            )
            structured_content, _structured_finish, structured_answer = _smoke_answer(
                structured_result,
                spec,
                serving_model=serving_model,
                expected_checkpoint=expected_checkpoint,
            )
            _validate_structured_smoke(
                structured_answer,
                constraint,
                deadline=deadline,
                budget_s=budget_s,
            )
            content = f"{content}\n{structured_content}"
            verify_turns = 2
    elif constraint:
        _validate_structured_smoke(answer, constraint, deadline=deadline, budget_s=budget_s)
    if time.monotonic() > deadline:
        raise _smoke_timeout_error(budget_s)
    return {
        "verified_at": time.time(),
        "verify_kind": "fixed_image" if image_capable else "fixed_prompt",
        "verify_turns": verify_turns,
        "verify_latency_s": time.monotonic() - started,
        "verify_finish_reason": finish,
        "thinking_tag": "<think>" in content or "</think>" in content,
        "verify_sample": answer[:160],
    }


def _alias_reasoning_content(result: dict, run_id: str, adapter_revision: str) -> str:
    choices = result.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        raise ServingError("alias thinking verification returned a malformed chat message")
    if "reasoning_content" not in message:
        raise AliasThinkingSilent(
            run_id,
            adapter_revision,
            detail=(
                "the alias returned no direct reasoning_content field while the immutable revision "
                "smoked with reasoning enabled"
            ),
        )
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        raise ServingError("alias thinking verification returned non-string reasoning_content")
    return reasoning


def _verify_alias_thinking(
    run_id: str,
    spec: JobSpec,
    adapter_revision: str,
    expected_checkpoint: str,
    *,
    budget_s: float = _SMOKE_BUDGET_SECONDS,
) -> dict:
    """Confirm the freshly activated alias serves the activated revision with reasoning metadata."""
    started = time.monotonic()
    deadline = started + budget_s
    _, max_tokens, stop_sequences = _smoke_request_settings(spec)
    result = _bounded_smoke_chat(
        serving_model=run_id,
        thinking=True,
        expected_checkpoint=expected_checkpoint,
        expected_adapter_revision=adapter_revision,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        deadline=deadline,
        budget_s=budget_s,
        error_context=f"alias thinking verification could not reach {run_id}",
    )

    _smoke_provenance(result, adapter_revision, expected_checkpoint)
    _alias_reasoning_content(result, run_id, adapter_revision)
    if time.monotonic() > deadline:
        raise _smoke_timeout_error(budget_s)
    return {
        "alias_thinking_tag": True,
        "alias_thinking_verified_at": time.time(),
        "alias_thinking_latency_s": time.monotonic() - started,
    }
