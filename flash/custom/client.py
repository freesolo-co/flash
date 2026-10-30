"""Synchronous notebook/script API with explicit updates and safe ambiguous-request recovery."""

from __future__ import annotations

import json
import threading
from urllib.parse import urlsplit

from flash.client.http import ApiClient, ApiError
from flash.custom.types import MAX_BODY_BYTES, Datum


class TrainingClient:
    def __init__(self, url, *, api_key, timeout=300):
        parsed = urlsplit(url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("custom training requires HTTPS, except on localhost")
        if not api_key:
            raise ValueError("a dedicated custom-session API key is required")
        self._http = ApiClient(url, api_key, timeout=timeout)
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
                raise ValueError("custom operation exceeds 4 MiB; split the batch and try again")
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

    def encode(self, text):
        return self._call("encode", text=text)["tokens"]

    def decode(self, tokens):
        return self._call("decode", tokens=tokens)["text"]

    def sample(self, prompt, *, max_tokens=64, temperature=1.0, num_samples=1):
        return self._call(
            "sample",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )

    def forward(self, data, *, top_k=0):
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
        to a teacher's top-k read them directly.
        """
        import torch

        with self._lock:
            snapshot = [Datum(list(row.tokens), target_tokens=row.target_tokens) for row in data]
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
                weights.append(Datum(row.tokens, weight, row.target_tokens))
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
