"""Independent fleet lifecycle service; survives rollback of the inference router."""

import os

import modal

from .config import APP_NAME
from .modal_control import register_control

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx>=0.27", "modal==1.6.2")
    .add_local_python_source("flash")
)
environment = str(modal.config.config.get("environment") or "main")
if environment not in {"main", "dev"}:
    raise ValueError("Lambda fleet must deploy to main or dev")
secret = modal.Secret.from_dict(
    {
        "LAMBDA_API_KEY": os.environ.get("LAMBDA_API_KEY", ""),
        "FLASH_LAMBDA_ENVIRONMENT": environment,
    }
)
LambdaFleet, reap_lambda_workers = register_control(app, secret, image, __name__)
