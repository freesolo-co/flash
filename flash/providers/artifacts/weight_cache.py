"""Provision and warm the shared Modal base-model Volume using a CPU sandbox."""

from __future__ import annotations

import argparse
import json
import os

from flash.providers.modal import api


def catalog_model_ids() -> list[str]:
    from flash.core.catalog import MODELS

    return sorted(MODELS)


def provision_weight_cache():
    async def provision(modal, client):
        volume = modal.Volume.from_name(
            api.CACHE_NAME,
            environment_name=api.environment(),
            create_if_missing=True,
            client=client,
        )
        await volume.hydrate.aio(client)
        return volume.object_id

    return api.call(provision)


def teardown_weight_cache():
    async def remove(modal, client):
        await modal.Volume.objects.delete.aio(
            api.CACHE_NAME, environment_name=api.environment(), client=client
        )

    api.call(remove)


def warm_weight_cache(*, models=None, timeout_s=3600):
    models = list(models if models is not None else catalog_model_ids())
    if not models or not set(models).issubset(catalog_model_ids()):
        raise ValueError("weight-cache warming accepts only trusted catalog base models")
    if not 1 <= timeout_s <= api.MAX_SANDBOX_SECONDS:
        raise ValueError("timeout must be between 1 and 86400 seconds")
    from flash.core.catalog import MODELS

    revisions = {model: MODELS[model].managed_revision or None for model in models}
    script = """import json, os
from huggingface_hub import snapshot_download
models = json.loads(os.environ['FLASH_CACHE_MODELS'])
for model, revision in models.items():
    snapshot_download(model, revision=revision, cache_dir='/weight-cache/hf-cache/hub',
                      token=os.environ.get('HF_TOKEN'), ignore_patterns=['*.bin', '*.pt', '*.pth'])
print(json.dumps({'status': 'ok', 'models': models}))
"""

    async def warm(modal, client):
        app = await modal.App.lookup.aio(
            api.APP_NAME, environment_name=api.environment(), create_if_missing=True, client=client
        )
        volume = modal.Volume.from_name(
            api.CACHE_NAME,
            environment_name=api.environment(),
            create_if_missing=True,
            client=client,
        )
        image = modal.Image.debian_slim(python_version="3.12").pip_install(
            "huggingface-hub==1.27.0"
        )
        sandbox = await modal.Sandbox.create.aio(
            "python",
            "-c",
            script,
            app=app,
            image=image,
            timeout=timeout_s,
            cpu=2,
            memory=4096,
            volumes={api.CACHE_MOUNT: volume},
            secrets=[
                modal.Secret.from_dict(
                    {
                        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                        "FLASH_CACHE_MODELS": json.dumps(revisions),
                    }
                )
            ],
            client=client,
        )
        try:
            await sandbox.wait.aio()
            if await sandbox.poll.aio() != 0:
                raise api.ModalApiError("Modal weight-cache warm failed")
            return {"status": "ok", "models": models, "volume": api.CACHE_NAME}
        finally:
            await sandbox.terminate.aio(wait=True)

    return api.call(warm, timeout=timeout_s + 120)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage the Modal base-model weight cache")
    parser.add_argument("--models", help="comma-separated catalog model ids")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--provision", action="store_true")
    actions.add_argument("--teardown", action="store_true")
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",")] if args.models else catalog_model_ids()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "volume": api.CACHE_NAME,
                    "action": "delete"
                    if args.teardown
                    else "provision"
                    if args.provision
                    else "warm",
                    "models": models,
                }
            )
        )
        return 0
    if args.teardown:
        teardown_weight_cache()
    elif args.provision:
        print(provision_weight_cache())
    else:
        print(json.dumps(warm_weight_cache(models=models, timeout_s=args.timeout_s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
