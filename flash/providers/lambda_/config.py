"""Control-plane-only Lambda Cloud credentials and account ownership.

The API key never reaches a GPU host: launch, polling and teardown all run on the control plane.
"""

import hashlib
import os


def settings() -> dict[str, str]:
    return {
        "api_key": os.environ.get("FLASH_LAMBDA_API_KEY", "").strip(),
        "ssh_key_name": os.environ.get("FLASH_LAMBDA_SSH_KEY_NAME", "").strip(),
    }


def configured() -> bool:
    config = settings()
    return bool(config["api_key"] and config["ssh_key_name"])


def scope_fingerprint() -> str:
    """The owning account, bound into every persisted handle without storing the key itself."""
    config = settings()
    if not configured():
        raise RuntimeError("Configure FLASH_LAMBDA_API_KEY and FLASH_LAMBDA_SSH_KEY_NAME")
    # Namespaced so a persisted identity from another provider can never be adopted by Lambda.
    return hashlib.sha256(f"lambda:{config['api_key']}".encode()).hexdigest()
