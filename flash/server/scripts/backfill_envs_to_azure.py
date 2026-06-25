"""One-time backfill: migrate environments from the GitHub env-hub to Azure Blob + Postgres.

Reads every ``<namespace>/<name>/`` environment directory in a checkout of the old
``freesolo-co/environment-hub`` repo, packages each as a ``.tar.gz`` (``environment.py`` at the
root — the same shape ``flash env push`` uploads), uploads it to Azure Blob, and indexes the
slug -> blob pointer in Azure Postgres. Idempotent: an env already indexed in Postgres is skipped
unless ``--force``.

The Supabase UI-mirror rows are NOT touched here: their blob pointer is derived from the slug
(``flash-envs/<slug>/package.tar.gz``) by the freesolo migration, so no org mapping is needed.

Usage:
    # against a local checkout
    python -m flash.server.scripts.backfill_envs_to_azure --hub-path /path/to/environment-hub
    # or clone it first (needs a GITHUB token / gh auth in the environment)
    python -m flash.server.scripts.backfill_envs_to_azure --repo freesolo-co/environment-hub
    # preview only
    python -m flash.server.scripts.backfill_envs_to_azure --hub-path ./environment-hub --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from flash.server import azure_blob, environment_store
from flash.server.envs import blob_key_for

_ENVIRONMENT_FILE = "environment.py"
# Top-level dirs in the hub that are not env namespaces.
_SKIP_TOP_LEVEL = {".git", ".github"}


def _iter_env_dirs(hub_root: Path):
    """Yield (slug, namespace, name, env_dir) for every ``<namespace>/<name>/environment.py``."""
    for ns_dir in sorted(p for p in hub_root.iterdir() if p.is_dir()):
        if ns_dir.name in _SKIP_TOP_LEVEL:
            continue
        for env_dir in sorted(p for p in ns_dir.iterdir() if p.is_dir()):
            if (env_dir / _ENVIRONMENT_FILE).is_file():
                yield f"{ns_dir.name}/{env_dir.name}", ns_dir.name, env_dir.name, env_dir


def _package_dir(env_dir: Path) -> bytes:
    """Tar+gzip the env directory CONTENTS (environment.py at the archive root)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(env_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(env_dir)))
    return buf.getvalue()


def _clone_repo(repo: str) -> Path:
    """Shallow-clone ``owner/name`` from GitHub into a temp dir (uses ambient git/gh auth)."""
    dest = Path(tempfile.mkdtemp(prefix="flash-envhub-backfill-"))
    checkout = dest / "environment-hub"
    url = f"https://github.com/{repo}.git"
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    return checkout


def backfill(hub_root: Path, *, dry_run: bool = False, force: bool = False) -> int:
    """Backfill every env under ``hub_root``. Returns the number of envs uploaded."""
    if not hub_root.is_dir():
        raise SystemExit(f"hub path is not a directory: {hub_root}")
    uploaded = 0
    skipped = 0
    for slug, namespace, name, env_dir in _iter_env_dirs(hub_root):
        existing = None if force else environment_store.lookup(slug)
        if existing is not None:
            print(f"skip   {slug} (already indexed: {existing.blob_key})")
            skipped += 1
            continue
        data = _package_dir(env_dir)
        sha = hashlib.sha256(data).hexdigest()
        blob_key = blob_key_for(slug)
        if dry_run:
            print(f"DRY    {slug} -> {blob_key} ({len(data)} bytes, sha {sha[:12]})")
            uploaded += 1
            continue
        azure_blob.upload_package(blob_key, data)
        environment_store.upsert(
            slug=slug,
            namespace=namespace,
            name=name,
            blob_container=azure_blob.container_name(),
            blob_key=blob_key,
            package_sha256=sha,
            size_bytes=len(data),
        )
        print(f"upload {slug} -> {blob_key} ({len(data)} bytes, sha {sha[:12]})")
        uploaded += 1
    print(f"\ndone: {uploaded} uploaded, {skipped} skipped")
    return uploaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hub-path", help="path to a local checkout of the env-hub repo")
    src.add_argument("--repo", help="owner/name of the env-hub repo to shallow-clone")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be uploaded, write nothing"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-upload even if the slug is already indexed"
    )
    parser.add_argument(
        "--ensure-container",
        action="store_true",
        help="create the Azure Blob container first if it does not exist",
    )
    args = parser.parse_args(argv)

    if args.ensure_container and not args.dry_run:
        azure_blob.ensure_container()
    if not args.dry_run:
        environment_store.ensure_schema()

    hub_root = Path(args.hub_path) if args.hub_path else _clone_repo(args.repo)
    backfill(hub_root, dry_run=args.dry_run, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
