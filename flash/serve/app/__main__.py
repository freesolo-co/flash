"""hydrate or serve one externally bound immutable serving manifest."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path

from .bootstrap import bootstrap_serving
from .http import create_app
from .manifest import ServingManifest, load_serving_manifest
from .materialize import hydrate_manifest, read_artifact_token_fd

_MANIFEST_ENV = "FLASH_SERVING_MANIFEST"
_MANIFEST_ID_ENV = "FLASH_SERVING_MANIFEST_ID"
_IMAGE_DIGEST_ENV = "FLASH_SERVING_IMAGE_DIGEST"
_CACHE_ROOT_ENV = "FLASH_SERVING_CACHE_ROOT"
_INFERENCE_TOKEN_FD_ENV = "FLASH_INFERENCE_TOKEN_FD"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m flash.serve.app")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("hydrate", "serve"):
        command = subparsers.add_parser(mode)
        command.add_argument(
            "--manifest",
            default=os.environ.get(_MANIFEST_ENV, "/etc/flash/serving-manifest.json"),
        )
        command.add_argument(
            "--cache-root",
            default=os.environ.get(_CACHE_ROOT_ENV, "/var/lib/flash-serving"),
        )
    serve = subparsers.choices["serve"]
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _bound_manifest(path: str) -> ServingManifest:
    manifest = load_serving_manifest(Path(path).read_bytes())
    expected_manifest = os.environ.get(_MANIFEST_ID_ENV)
    expected_image = os.environ.get(_IMAGE_DIGEST_ENV)
    if expected_manifest != manifest.manifest_id:
        raise RuntimeError("serving manifest id does not match its external binding")
    if expected_image != manifest.expected_oci_digest:
        raise RuntimeError("serving image digest does not match its external binding")
    return manifest


def _read_inference_token() -> str:
    raw_fd = os.environ.get(_INFERENCE_TOKEN_FD_ENV)
    if raw_fd is None or not raw_fd.isdecimal():
        raise RuntimeError("inference token fd is not configured")
    return read_artifact_token_fd(int(raw_fd))


async def _serve(args: argparse.Namespace, manifest: ServingManifest) -> None:
    import uvicorn

    token = _read_inference_token()
    try:
        bearer_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    finally:
        token = ""
    owner = await bootstrap_serving(manifest, args.cache_root)
    try:
        app = create_app(owner, bearer_digest=bearer_digest)
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    finally:
        await owner.close()


def main() -> None:
    args = _parser().parse_args()
    manifest = _bound_manifest(args.manifest)
    if args.mode == "hydrate":
        hydrate_manifest(manifest, args.cache_root)
        return
    asyncio.run(_serve(args, manifest))


if __name__ == "__main__":
    main()
