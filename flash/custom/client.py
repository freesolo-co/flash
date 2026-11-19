"""Synchronous notebook/script API with explicit updates and safe ambiguous-request recovery."""

from __future__ import annotations

import json
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
        """Accumulate gradients of -sum(weights * logprobs), without stepping the optimizer."""
        return self._call(
            "forward_backward", data=[row.to_dict() for row in data], metrics=metrics or {}
        )

    def forward_backward_custom(self, data, loss_fn):
        """Evaluate loss_fn(data, logprobs) locally, then send its token derivatives.

        loss_fn returns a scalar torch tensor and a JSON dictionary of metrics. No callable
        or pickle is sent to the GPU. Both passes use the same model version and dropout-free
        model. Cross-sequence objectives (DPO, PPO, ranking, etc.) are supported. Rows with a
        [T, K] target_tokens matrix receive [T, K] logprobs, so soft-target losses such as KL
        to a teacher's top-k read them directly. With target_positions, T is the number of
        listed positions; flash.custom.losses.option_ce_brier_loss is one such loss over
        option-token logprobs at an answer position.
        """
        import torch

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
            scored = self.forward(snapshot)
            logprobs = [
                torch.tensor(row["logprobs"], dtype=torch.float32, requires_grad=True)
                for row in scored["outputs"]
            ]
            loss, metrics = loss_fn(data, logprobs)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("custom loss must return a finite scalar torch tensor")
            if not isinstance(metrics, dict):
                raise ValueError("custom loss metrics must be a JSON dictionary")
            json.dumps(metrics, allow_nan=False)
            gradients = torch.autograd.grad(loss, logprobs, allow_unused=True)
            weights = []
            for row, gradient, scores in zip(snapshot, gradients, logprobs, strict=True):
                if gradient is None:
                    weight = torch.zeros_like(scores).tolist()
                else:
                    if not torch.isfinite(gradient).all():
                        raise ValueError("custom loss produced non-finite token gradients")
                    weight = (-gradient).tolist()
                weights.append(
                    Datum(row.tokens, weight, row.target_tokens, row.images, row.target_positions)
                )
            return self.forward_backward(weights, metrics={**metrics, "custom_loss": loss.item()})

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
