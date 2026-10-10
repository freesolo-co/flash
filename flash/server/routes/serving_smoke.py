"""The deployment smoke test: generate once against the new adapter and validate what comes back.

A deployment is only marked ready after this passes, so everything here runs against untrusted
output under a hard wall-clock budget. Two of those bounds are the reason this is its own module:
a user-supplied structured-output regex is evaluated with a timeout, and a user-supplied JSON
schema is compiled in a spawned process that is reaped whether or not it answers -- a catastrophic
pattern or a schema that hangs the validator must not wedge the serving route.

Split out of `flash.server.routes.serving` to keep that module under the file-size limit.
"""

from __future__ import annotations

import hashlib
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
from flash.serve.contract.errors import (
    RetryableServingUnavailable,
    ServingError,
)
from flash.serve.contract.protocol import LORA_REQUEST_ATTESTATION_CAPABILITY
from flash.serve.deployment.preflight import (
    SERVING_PROMPT_TOKEN_ALLOWANCE,
    ExternalSchemaReference,
    reject_external_schema_reference,
    resolve_smoke_completion_tokens,
    validate_local_json_schema,
    validate_structured_output_patterns,
)
from flash.server.asgi import app as _app


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
# three trusted 32x32 solid-colour squares, every pixel the stated colour. the smoke picks one
# per deployment, so a model that answers from a language prior rather than from the pixels is
# wrong two times in three. a single fixed colour would be guessable: "red" is the obvious answer
# to "what colour is the square", and a broken vision path -- wrong processor, dropped media,
# placeholders that never expanded -- would still emit it and pass.
_SMOKE_IMAGE_VARIANTS = (
    (
        "RED",
        (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAJ0lEQVR42u3NsQkAAAjAsP7/tF7h"
            "IASyp6lTCQQCgUAgEAgEgi/BAjLD/C5w/SM9AAAAAElFTkSuQmCC"
        ),
    ),
    (
        "BLUE",
        (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAJklEQVR42u3NsQkAAAjAsP7/tF7h"
            "IASyp5pjAoFAIBAIBAKB4EmwOkv8Lom8x/sAAAAASUVORK5CYII="
        ),
    ),
    (
        "GREEN",
        (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAJklEQVR42u3NsQkAAAjAsJ7u6V7h"
            "IASyp6ZbAoFAIBAIBAKB4EuwwgAAH2BCGKwAAAAASUVORK5CYII="
        ),
    ),
)


def _smoke_image_challenge(run_id: str) -> tuple[str, list[dict]]:
    """Pick this deployment's colour challenge and build its message.

    keyed on run_id so one deployment always gets the same colour: the smoke may be retried and a
    per-call random choice would make a flaky vision path look intermittently healthy. the answer
    still cannot be memorised across runs, which is what makes the check image-dependent.
    """
    expected, data_uri = _SMOKE_IMAGE_VARIANTS[
        int(hashlib.sha256(run_id.encode()).hexdigest(), 16) % len(_SMOKE_IMAGE_VARIANTS)
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SMOKE_IMAGE_PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]
    return expected, messages


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


def _smoke_provenance(result: dict, checkpoint_id: str, checkpoint: str) -> tuple[str, object]:
    choice = (result.get("choices") or [{}])[0]
    content = str((choice.get("message") or {}).get("content") or "")
    finish = choice.get("finish_reason")
    if not content.strip():
        raise ServingError(f"smoke generation returned no content (finish_reason={finish!r})")
    provenance = result.get("freesolo")
    if not isinstance(provenance, dict) or provenance.get("checkpoint_id") != checkpoint_id:
        raise ServingError("smoke response returned the wrong checkpoint identity")
    headers = result.get("_freesolo_headers")
    if headers != {"checkpoint_id": checkpoint_id}:
        raise ServingError("smoke response returned mismatched checkpoint headers")
    if checkpoint != checkpoint_id:
        raise ServingError("smoke expected checkpoint is not canonical")
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


def _lora_attestation_advertised(advertised: frozenset[str] | None) -> bool:
    """Whether the serving backend said it emits ``X-Freesolo-LoRA-Request-Adapter``.

    ``advertised`` is the capability set THIS deployment already gated on, captured once by
    ``deploy_adapter`` before registration and handed down. Asking ``/healthz`` a second time here
    would re-open a question the deployment has settled: a rolling serving deploy can answer from
    an older replica that does not advertise the capability, and a transient failure answers not
    at all -- either way the smoke would fail open on a deployment that WAS promised the header.

    ``None`` means no set was captured (a caller outside the deploy path), which is treated as not
    advertising so the attestation degrades rather than turning discovery into a deploy failure.
    """
    return advertised is not None and LORA_REQUEST_ATTESTATION_CAPABILITY in advertised


def _smoke_lora_request_adapter(
    result: dict, checkpoint_id: str, *, attestation_advertised: bool
) -> str | None:
    """Check which LoRA answered the smoke, when the serving backend can actually say.

    ``X-Freesolo-LoRA-Request-Adapter`` is emitted by the serving image, not by the run, so a
    missing header says nothing about the adapter under test -- exactly the shape of the bug that
    ``REVISION_PROVENANCE_CAPABILITY`` already had to fix once. Demanding it from a backend that
    never advertised it fails every deployment org-wide while proving nothing, so absence of the
    capability degrades the check instead of blocking the deploy.

    Where the capability IS advertised the check stays strict: a missing or mismatched header then
    means the serving backend broke a contract it agreed to, which is a real failure.
    """
    attested = result.get("_freesolo_lora_request_adapter")
    if not attestation_advertised:
        if attested and attested != checkpoint_id:
            # the backend volunteered an identity and it is the WRONG one. that is a genuine
            # mismatch regardless of what it advertises, so it still fails.
            raise ServingError("image deployment smoke returned the wrong LoRA request adapter")
        return str(attested) if attested else None
    if not attested:
        raise ServingError("image deployment smoke omitted LoRA request adapter attestation")
    if attested != checkpoint_id:
        raise ServingError("image deployment smoke returned the wrong LoRA request adapter")
    return str(attested)


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
    org_id: str,
    messages: list[dict] | None = None,
    structured_outputs: dict | None = None,
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
                    "org_id": org_id,
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
    org_id: str,
    # these facts were captured before the smoke; re-fetching inside the paid deadline can disagree
    # with the deployment that was actually registered or consume its verification budget.
    advertised_capabilities: frozenset[str] | None = None,
    adapter_targets_images: bool | None = None,
    budget_s: float = _SMOKE_BUDGET_SECONDS,
) -> dict:
    started = time.monotonic()
    deadline = started + budget_s
    constraint, max_tokens, stop_sequences = _smoke_request_settings(spec)
    # only an explicit multimodal targeting marker enables the image challenge. unavailable metadata
    # falls back to the weaker text smoke: it gives up vision-path verification, but cannot strand a
    # working text adapter by asking a task it was never trained for.
    use_image_challenge = adapter_targets_images is True
    attestation_advertised = _lora_attestation_advertised(advertised_capabilities)
    expected_colour, image_messages = _smoke_image_challenge(run_id)
    result = _bounded_smoke_chat(
        serving_model=serving_model,
        thinking=spec.thinking,
        expected_checkpoint=expected_checkpoint,
        org_id=org_id,
        messages=image_messages if use_image_challenge else None,
        structured_outputs={} if use_image_challenge and constraint is not None else None,
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
    attested_checkpoint_id: str | None = None
    if use_image_challenge:
        attested_checkpoint_id = _smoke_lora_request_adapter(
            result, serving_model, attestation_advertised=attestation_advertised
        )
        if answer.strip().upper() != expected_colour:
            raise ServingError(
                "image deployment smoke did not identify the trusted "
                f"{expected_colour.lower()} square"
            )
        if constraint:
            structured_result = _bounded_smoke_chat(
                serving_model=serving_model,
                thinking=spec.thinking,
                expected_checkpoint=expected_checkpoint,
                org_id=org_id,
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
            _smoke_lora_request_adapter(
                structured_result, serving_model, attestation_advertised=attestation_advertised
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
    smoke_result = {
        "verified_at": time.time(),
        "verify_kind": "fixed_image" if use_image_challenge else "fixed_prompt",
        "verify_turns": verify_turns,
        "verify_latency_s": time.monotonic() - started,
        "verify_finish_reason": finish,
        "thinking_tag": "<think>" in content or "</think>" in content,
        "verify_sample": answer[:160],
    }
    if attested_checkpoint_id is not None:
        smoke_result["verify_lora_request_adapter"] = attested_checkpoint_id
    return smoke_result
