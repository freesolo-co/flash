"""Separate upstream model access from private Azure artifact access."""


def _hydration_credential(value, kind):
    import json

    from flash.serve.app.materialize import MaterializationError

    try:
        credentials = json.loads(value)
    except (TypeError, ValueError):
        # Explicit Azure URLs remain accepted for public upstream models.
        credentials = {
            "azure": value,
            "upstream": None if str(value).startswith("https://") else value,
        }
    if not isinstance(credentials, dict):
        raise MaterializationError("invalid hydration credentials")
    selected = credentials.get(kind)
    if kind == "azure" and not selected:
        raise MaterializationError("Azure artifact credential is required")
    return selected
