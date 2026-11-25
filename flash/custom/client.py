"""Synchronous notebook/script API with explicit updates and safe ambiguous-request recovery."""

from __future__ import annotations

import json
import math
import numbers
import threading
from urllib.parse import urlsplit

from flash.client.http import ApiClient, ApiError
from flash.custom.types import MAX_BODY_BYTES, MAX_BODY_MIB, MAX_ENCODE_CHARACTERS, Datum


def pinned_context(certificate):
    """Trust exactly one self-signed session certificate (no public CA, no hostname).

    The worker's address is not known when its per-attempt identity is minted, so the
    certificate names no IP; pinning it as the only trust anchor authenticates the server.
    """
    import ssl

    # cadata replaces the system store: public CAs are never trusted for a session.
    context = ssl.create_default_context(cadata=certificate)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    return context


class TrainingClient:
    """Synchronous client for one managed custom-training GPU session.

    Every operation returns the operation's own fields merged with the session status
    (session_id, step, sequence, model_version, max_steps, vocab_size, max_context_tokens,
    max_batch, ...). Operation-specific results:

    * ``encode(text) -> list[int]``: plain tokenization, no chat template and no added special
      tokens; special-token strings in ``text`` (``<|im_start|>``...) become their ids.
      ``text`` holds at most 100,000 characters per call.
    * ``decode(tokens) -> str``, special tokens kept.
    * ``sample(prompt_ids, ...) -> {"sequences": [{"tokens", "text", "logprobs",
      "sampling_logprobs", "finish_reason"}, ...]}``: completion-only tokens and logprobs;
      ``finish_reason`` is "stop" or "length". Sampling is plain (no top-k/top-p; temperature
      0 is greedy) and stops at the tokenizer's EOS (<|endoftext|> for Qwen3.5), so text often
      ends "<|im_end|>\\n<|endoftext|>". No chat template, system prompt or structured-output
      constraint is applied: render the prompt exactly as serving does before ``encode``.
    * ``forward(data, top_k=0) -> {"outputs": [{"logprobs": [...], "top_logprobs": ...}]}``:
      one output per Datum; ``logprobs[i]`` scores ``tokens[i+1]``.
    * ``forward_backward(data) -> {"outputs": [...], "metrics": {"surrogate_loss": ...}}``.
      It minimizes -sum(weights * logprobs), so weights = -dL/dlogprob trains any loss L
      built from token logprobs (DPO, ranking, Brier, ...) without torch.
    * ``forward_backward_custom(data, loss_fn, gradients=False)``: forward, then
      forward_backward with loss_fn's -dL/dlogprob; torch autograd by default, or your
      own dL/dlogprob lists with gradients=True (no torch needed).
    * ``optim_step(...) -> {"metrics": {"grad_norm", "learning_rate", ...}}``.

    Limits: at most ``max_batch`` rows (or ``num_samples``) per call, 2..max_context_tokens
    tokens per datum and 32 MiB per request. A rejected call raises ApiError with code
    ``custom_rejected`` and leaves the session usable.
    """

    def __init__(self, url, *, api_key, timeout=300, tls_cert=None):
        parsed = urlsplit(url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("custom training requires HTTPS, except on localhost")
        if not api_key:
            raise ValueError("a dedicated custom-session API key is required")
        self._http = ApiClient(
            url,
            api_key,
            timeout=timeout,
            ssl_context=pinned_context(tls_cert) if tls_cert else None,
        )
        self._lock = threading.RLock()
        self._pending = None
        self._state = self.status()
        self.session_id = self._state["session_id"]

    @property
    def step(self):
        return self._state["step"]

    @property
    def restored_state(self):
        return self._state.get("restored_state", {})

    @property
    def model_version(self):
        return self._state["model_version"]

    @property
    def pending_request(self):
        """An exact retry envelope after a lost response; never replace it with a new operation."""
        return json.loads(json.dumps(self._pending)) if self._pending else None

    def status(self):
        return self._http._request("GET", "/v1/custom/status")

    def logs(self, after=0):
        return self._http._request("GET", f"/v1/custom/logs?after={int(after)}")

    def _call(self, operation, **arguments):
        with self._lock:
            if self._pending is not None:
                raise RuntimeError(
                    "previous outcome is unknown; call retry_pending() before more work"
                )
            envelope = {
                "session_id": self.session_id,
                "sequence": self._state["sequence"] + 1,
                "model_version": self.model_version,
                "operation": operation,
                "arguments": arguments,
            }
            # detach from mutable user lists; retries must send byte-equivalent intent.
            serialized = json.dumps(envelope, allow_nan=False)
            if len(serialized.encode()) > MAX_BODY_BYTES:
                raise ValueError(
                    f"custom operation exceeds {MAX_BODY_MIB} MiB; split the batch and try again"
                )
            self._pending = json.loads(serialized)
            return self.retry_pending()

    def retry_pending(self):
        with self._lock:
            if self._pending is None:
                raise ValueError("no unresolved request")
            try:
                result = self._http._request(
                    "POST", "/v1/custom/operation", self._pending, max_bytes=32 * 1024 * 1024
                )
            except ApiError as error:
                # structured worker rejections precede GPU execution; proxy errors are ambiguous.
                if error.status in {400, 409, 413} and error.code == "custom_rejected":
                    self._pending = None
                raise
            if (
                result.get("session_id") != self.session_id
                or result.get("sequence") != self._pending["sequence"]
            ):
                raise RuntimeError("invalid operation receipt; request remains unresolved")
            self._state = result
            self._pending = None
            return result

    def encode(self, text, *, images=None):
        """Token ids (list[int]) for at most 100,000 characters of text, with no template or
        added special tokens; with images (data URIs), each <|image_pad|> in text becomes the
        exact placeholder run its image needs, ready for Datum(tokens, images=images)."""
        if isinstance(text, str) and len(text) > MAX_ENCODE_CHARACTERS:
            raise ValueError(
                f"encode accepts at most {MAX_ENCODE_CHARACTERS:,} characters per call "
                f"(got {len(text):,}); split the text and concatenate the token ids"
            )
        extra = {"images": list(images)} if images is not None else {}
        return self._call("encode", text=text, **extra)["tokens"]

    def decode(self, tokens):
        return self._call("decode", tokens=tokens)["text"]

    def sample(self, prompt, *, max_tokens=64, temperature=1.0, num_samples=1, images=None):
        """Sample completions of a token-id prompt; returns {"sequences": [...]} (see class).

        prompt is a list of token ids, e.g. client.encode(rendered_prompt_text), never a string.
        """
        if isinstance(prompt, str):
            raise TypeError(
                "sample takes prompt token ids, not text: pass client.encode(prompt_text), "
                "with the prompt rendered exactly as serving's chat template renders it"
            )
        extra = {"images": list(images)} if images is not None else {}
        return self._call(
            "sample",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            num_samples=num_samples,
            **extra,
        )

    def forward(self, data, *, top_k=0):
        """Score Datum rows without gradients: {"outputs": [{"logprobs": [...]}, ...]}, one
        output per row (index result["outputs"][row], not result[row]); top_k > 0 adds
        "top_logprobs" per scored position."""
        return self._call("forward", data=[row.to_dict() for row in data], top_k=top_k)

    def forward_backward(self, data, *, metrics=None):
        """Accumulate gradients of -sum(weights * logprobs), without stepping the optimizer.

        Weights can be any -dL/dlogprob, so any loss built from token logprobs trains with no
        torch: forward() to get logprobs, compute dL/dlogprob yourself, then call this with
        weights = -dL/dlogprob (forward_backward_custom(..., gradients=True) does exactly
        that). The gradient matches L's as long as the model version has not changed between
        the two calls.
        """
        return self._call(
            "forward_backward", data=[row.to_dict() for row in data], metrics=metrics or {}
        )

    def forward_backward_custom(self, data, loss_fn, *, gradients=False):
        """Evaluate loss_fn(data, logprobs) locally, then send its token derivatives.

        Default (needs torch: pip install 'freesolo-flash[custom]'): logprobs is a list of
        float32 tensors with requires_grad, and loss_fn returns (scalar loss tensor, metrics).

        gradients=True (no torch needed): logprobs is a list of plain float lists, and loss_fn
        returns (loss, metrics, dloss_dlogprobs). The loss is a finite number.
        dloss_dlogprobs holds, per row, dL/dlogprob with that row's logprob shape (lists or
        numpy arrays), or None for a row the loss ignores.

        Either way the gradient pass is forward_backward with weights = -dL/dlogprob, because
        forward_backward minimizes -sum(weights * logprobs). Calling forward then
        forward_backward yourself with those weights is equivalent.

        No callable or pickle is sent to the GPU. Both passes use the same model version and
        dropout-free model. Cross-sequence objectives (DPO, PPO, ranking, etc.) are supported.
        Rows with a [T, K] target_tokens matrix receive [T, K] logprobs, so soft-target losses
        such as KL to a teacher's top-k read them directly. With target_positions, T is the
        number of listed positions; flash.custom.losses.option_ce_brier_loss is one such loss
        over option-token logprobs at an answer position.
        """
        torch = None if gradients else _require_torch()  # fail before spending a GPU pass.
        with self._lock:
            # Both passes must score identical targets even if loss_fn mutates the caller's data.
            snapshot = [
                Datum(
                    list(row.tokens),
                    target_tokens=[list(ids) for ids in row.target_tokens]
                    if row.target_tokens is not None
                    else None,
                    images=list(row.images) if row.images is not None else None,
                    target_positions=list(row.target_positions)
                    if row.target_positions is not None
                    else None,
                )
                for row in data
            ]
            scored = [row["logprobs"] for row in self.forward(snapshot)["outputs"]]
            if gradients:
                loss, metrics, weights = _manual_weights(loss_fn, data, scored)
            else:
                loss, metrics, weights = _autograd_weights(torch, loss_fn, data, scored)
            weighted = [
                Datum(row.tokens, weight, row.target_tokens, row.images, row.target_positions)
                for row, weight in zip(snapshot, weights, strict=True)
            ]
            return self.forward_backward(weighted, metrics={**metrics, "custom_loss": loss})

    def optim_step(self, *, learning_rate, max_grad_norm=1.0, algorithm_state=None):
        extra = {"algorithm_state": algorithm_state} if algorithm_state is not None else {}
        return self._call(
            "optim_step", learning_rate=learning_rate, max_grad_norm=max_grad_norm, **extra
        )

    def zero_grad(self):
        return self._call("zero_grad")

    def save_checkpoint(self, name, *, algorithm_state=None):
        extra = {"algorithm_state": algorithm_state} if algorithm_state is not None else {}
        return self._call("save_checkpoint", name=name, **extra)

    def load_checkpoint(self, name):
        return self._call("load_checkpoint", name=name)

    def close(self):
        if self._pending is not None:
            raise RuntimeError("resolve the pending operation with retry_pending() before closing")
        if not self._state.get("closed"):
            self._call("close")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            self.close()
        else:
            # retain the training exception, while still attempting GPU teardown.
            try:
                self.close()
            except Exception as cleanup_error:
                import warnings

                warnings.warn(
                    f"custom session cleanup failed: {type(cleanup_error).__name__}", stacklevel=2
                )


def _check_metrics(metrics):
    if not isinstance(metrics, dict):
        raise ValueError("custom loss metrics must be a JSON dictionary")
    json.dumps(metrics, allow_nan=False)


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "forward_backward_custom needs torch for autograd: pip install "
            "'freesolo-flash[custom]', or pass gradients=True and return dL/dlogprob yourself"
        ) from error
    return torch


def _autograd_weights(torch, loss_fn, data, scored):
    logprobs = [torch.tensor(row, dtype=torch.float32, requires_grad=True) for row in scored]
    loss, metrics = loss_fn(data, logprobs)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("custom loss must return a finite scalar torch tensor")
    _check_metrics(metrics)
    derivatives = torch.autograd.grad(loss, logprobs, allow_unused=True)
    weights = []
    for derivative, scores in zip(derivatives, logprobs, strict=True):
        if derivative is None:
            weights.append(torch.zeros_like(scores).tolist())
            continue
        if not torch.isfinite(derivative).all():
            raise ValueError("custom loss produced non-finite token gradients")
        weights.append((-derivative).tolist())
    return loss.item(), metrics, weights


def _manual_weights(loss_fn, data, scored):
    # loss_fn gets its own copy; shapes are checked against the server's untouched output.
    result = loss_fn(data, json.loads(json.dumps(scored)))
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError(
            "with gradients=True, custom loss must return (loss, metrics, dloss_dlogprobs)"
        )
    loss, metrics, derivatives = result
    if not _is_finite_number(loss):
        raise ValueError("custom loss must return a finite number")
    _check_metrics(metrics)
    if hasattr(derivatives, "tolist") and not isinstance(derivatives, list | tuple):
        derivatives = list(derivatives)
    if not isinstance(derivatives, list | tuple) or len(derivatives) != len(scored):
        raise ValueError("dloss_dlogprobs needs one entry per datum (None for an unused row)")
    weights = [
        _negate_like(derivative, scores, index)
        for index, (derivative, scores) in enumerate(zip(derivatives, scored, strict=True))
    ]
    return float(loss), metrics, weights


def _is_finite_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _negate_like(derivative, scores, index):
    """-derivative as plain floats, required to match one row's logprob shape exactly."""
    if derivative is None:
        return [[0.0] * len(cell) if isinstance(cell, list) else 0.0 for cell in scores]
    if hasattr(derivative, "tolist"):  # numpy arrays work; torch is never imported here.
        derivative = derivative.tolist()
    if not isinstance(derivative, list | tuple) or len(derivative) != len(scores):
        raise ValueError(
            f"dloss_dlogprobs[{index}] must have {len(scores)} entries, like its logprobs row"
        )
    weights = []
    for cell, reference in zip(derivative, scores, strict=True):
        if isinstance(reference, list):
            if not isinstance(cell, list | tuple) or len(cell) != len(reference):
                raise ValueError(
                    f"dloss_dlogprobs[{index}] must have the [T, {len(reference)}] shape "
                    "of its logprobs row"
                )
            if not all(_is_finite_number(value) for value in cell):
                raise ValueError("custom loss produced non-finite token gradients")
            weights.append([-float(value) for value in cell])
        else:
            if not _is_finite_number(cell):
                raise ValueError("custom loss produced non-finite token gradients")
            weights.append(-float(cell))
    return weights
