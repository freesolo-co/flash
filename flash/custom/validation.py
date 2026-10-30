"""Validate all custom-session arguments before touching model or optimizer state."""

from flash.custom.types import checkpoint_name, integer, number, parse_batch


def validate_operation(backend, operation, arguments):
    allowed = {
        "forward": {"data", "top_k"},
        "forward_backward": {"data", "metrics"},
        "optim_step": {"learning_rate", "max_grad_norm"},
        "zero_grad": set(),
        "sample": {"prompt", "max_tokens", "temperature", "num_samples"},
        "encode": {"text"},
        "decode": {"tokens"},
        "save_checkpoint": {"name"},
        "load_checkpoint": {"name"},
        "close": set(),
    }
    if not isinstance(operation, str) or operation not in allowed:
        raise ValueError("unknown custom training operation")
    if not isinstance(arguments, dict) or set(arguments) - allowed[operation]:
        raise ValueError(f"invalid arguments for {operation}")
    args = dict(arguments)
    if operation in {"forward", "forward_backward"}:
        args["data"] = parse_batch(
            args.get("data"),
            vocab_size=backend.vocab_size,
            max_tokens=backend.max_tokens,
            max_batch=backend.max_batch,
            weighted=operation == "forward_backward",
        )
        if operation == "forward":
            args["top_k"] = integer(args.get("top_k", 0), "top_k", 0, min(20, backend.vocab_size))
            if sum(len(row.tokens) - 1 for row in args["data"]) * args["top_k"] > 100_000:
                raise ValueError("top-k output exceeds 100000 entries; split the scoring batch")
        else:
            metrics = args.get("metrics", {})
            if not isinstance(metrics, dict) or len(metrics) > 64:
                raise ValueError("metrics must contain at most 64 finite numeric values")
            if any(not isinstance(key, str) or not 1 <= len(key) <= 128 for key in metrics):
                raise ValueError("metric names must contain 1-128 characters")
            args["metrics"] = {key: number(value, key) for key, value in metrics.items()}
    elif operation == "optim_step":
        args["learning_rate"] = number(args.get("learning_rate"), "learning_rate", 1e-12, 1)
        args["max_grad_norm"] = number(args.get("max_grad_norm", 1.0), "max_grad_norm", 1e-12, 1000)
        if not backend.pending_backward:
            raise ValueError("optim_step requires accumulated gradients")
    elif operation == "sample":
        prompt = args.get("prompt")
        if not isinstance(prompt, list) or not 1 <= len(prompt) < backend.max_tokens:
            raise ValueError("prompt must be a nonempty token list below max_context_tokens")
        args["prompt"] = [integer(t, "token", 0, backend.vocab_size - 1) for t in prompt]
        args["max_tokens"] = integer(
            args.get("max_tokens", 64), "max_tokens", 1, backend.max_tokens - len(prompt)
        )
        args["temperature"] = number(args.get("temperature", 1.0), "temperature", 0, 10)
        args["num_samples"] = integer(
            args.get("num_samples", 1), "num_samples", 1, backend.max_batch
        )
    elif operation == "encode":
        text = args.get("text")
        if not isinstance(text, str) or len(text) > 100_000:
            raise ValueError("text must be a string of at most 100000 characters")
    elif operation == "decode":
        tokens = args.get("tokens")
        if not isinstance(tokens, list) or len(tokens) > backend.max_tokens:
            raise ValueError("tokens must be a list within max_context_tokens")
        args["tokens"] = [integer(t, "token", 0, backend.vocab_size - 1) for t in tokens]
    elif operation in {"save_checkpoint", "load_checkpoint"}:
        args["name"] = checkpoint_name(args.get("name"))
        if backend.pending_backward:
            raise ValueError("apply or clear accumulated gradients before checkpoint operations")
        path = backend.checkpoint_dir / args["name"]
        if operation == "save_checkpoint" and path.exists():
            raise ValueError("checkpoint already exists; choose a new immutable name")
        if operation == "load_checkpoint" and not (path / "state.pt").is_file():
            raise ValueError("unknown checkpoint")
    return args
