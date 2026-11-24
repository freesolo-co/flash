"""Independent fleet lifecycle service; survives rollback of the inference router."""

import os

import modal

from . import reserved
from .config import APP_NAME
from .modal_control import register_control

app = modal.App(APP_NAME)
environment = str(modal.config.config.get("environment") or "main")
if environment not in {"main", "dev"}:
    raise ValueError("Lambda fleet must deploy to main or dev")
# Same decision as the serving deploy: refuse on-demand GPUs only while the reserved slots
# serve those models. Baked into the shared image, not just the secret, so every function on it
# (including the secret-less reaper) reuses it and a cold start never looks the secret up.
reserved_env = reserved.baked_env(environment)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx>=0.27", "modal==1.6.2")
    .env(reserved_env)
    .add_local_python_source("flash")
)
secret = modal.Secret.from_dict(
    {
        "LAMBDA_API_KEY": os.environ.get("LAMBDA_API_KEY", ""),
        "FLASH_LAMBDA_ENVIRONMENT": environment,
        "FREESOLO_DEPLOYMENT_SHA": os.environ.get("FREESOLO_DEPLOYMENT_SHA", ""),
        **reserved_env,
    }
)
LambdaFleet, reap_lambda_workers = register_control(app, secret, image, __name__)
