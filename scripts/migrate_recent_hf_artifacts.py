"""One-time, resumable import of the last 72 hours of Flash-owned Hub artifacts.

Plan first, review the JSON inventory, then apply that exact inventory. Runtime
Flash never imports this module and never falls back to Hugging Face storage.
The source is read-only: this tool does not delete Hugging Face data.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from flash.artifacts.manifest import (
    ArtifactNotFoundError,
    encode_manifest,
    repo_prefix,
    revision_for,
    validate_path,
    validate_revision,
)

WINDOW = timedelta(days=3)
PHASES = {"sft", "rl", "dpo", "grpo", "recomb", "custom"}


def state_references(directory, *, now):
    """Collect immutable dependencies from recent private run records without rewriting them."""
    references = set()

    def add(repo, revision, path):
        if repo and revision:
            repo_prefix(repo)
            validate_revision(revision)
            validate_path(path)
            references.add((repo, revision, path))

    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text())
        if not (now - WINDOW).timestamp() <= record.get("updated_at", 0) <= now.timestamp():
            continue
        spec = (record.get("effective_preparation") or {}).get("worker_spec") or record["spec"]
        train = spec.get("train", {})
        repo = train.get("hf_repo")
        # Completed runs are read through checkpoints, not restarted with old worker code.
        if record.get("state") in {"done", "deployed", "failed", "cancelled", "dry_run"}:
            continue
        source_snapshot = record.get("source_snapshot") or {}
        add(repo, source_snapshot.get("revision"), source_snapshot.get("archive_path"))
        package = spec.get("environment", {}).get("package") or {}
        if package:
            add(
                repo,
                package["artifact_revision"],
                f"environment-packages/archives/sha256/{package['archive_sha256']}/package.tar.gz",
            )
            add(
                repo,
                package["artifact_revision"],
                f"environment-packages/manifests/sha256/{package['manifest_sha256']}/manifest.json",
            )
        source = train.get("init_from_adapter", "")
        if ":" in source:
            add(
                source.split(":", 1)[0],
                train.get("init_from_adapter_revision"),
                source.split(":", 1)[1],
            )
        resume = (spec.get("custom") or {}).get("resolved_resume") or {}
        add(resume.get("repo"), resume.get("revision"), resume.get("path"))
        for deployment in (record.get("deployments") or {}).values():
            add(
                deployment.get("repo_id") or repo,
                deployment.get("artifact_revision"),
                deployment.get("subfolder"),
            )
    return [
        {"repo": repo, "revision": revision, "path": path}
        for repo, revision, path in sorted(references)
    ]


def checkpoint_group(path):
    """Select whole checkpoints by their newest file, never individual shards."""
    parts = validate_path(path).split("/")
    if len(parts) >= 5 and parts[0] in PHASES and parts[2] in {"checkpoints", "checkpoint"}:
        return "/".join(parts[:4])
    if len(parts) >= 4 and parts[0] in PHASES and parts[2] == "adapter":
        return "/".join(parts[:3])
    return path


def recent_files(entries, *, cutoff, end):
    groups = {}
    for entry in entries:
        if not hasattr(entry, "size"):
            continue
        stamp = getattr(getattr(entry, "last_commit", None), "date", None)
        if not isinstance(stamp, datetime) or stamp.tzinfo is None:
            raise ValueError("source file has no authoritative timezone-aware commit timestamp")
        identity = getattr(getattr(entry, "lfs", None), "sha256", None) or entry.blob_id
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", identity):
            raise ValueError("source file has no immutable content identity")
        if type(entry.size) is not int or entry.size < 0:
            raise ValueError("source file has invalid size")
        groups.setdefault(checkpoint_group(entry.path), []).append(
            {
                "path": entry.path,
                "size": entry.size,
                "source_blob_id": identity,
                "modified": stamp.timestamp(),
            }
        )
    selected = []
    for files in groups.values():
        newest = max(file["modified"] for file in files)
        if cutoff.timestamp() <= newest <= end.timestamp():
            selected.extend(files)
    return sorted(selected, key=lambda file: file["path"])


def make_plan(api, namespace, *, now, references=()):
    """Freeze heads and optional persisted checkpoint/source revisions into one plan."""
    validate_path(namespace)
    if "/" in namespace or now.tzinfo is None:
        raise ValueError("invalid namespace or planning time")
    requested = {}
    required_paths = {}
    for reference in references:
        repo, revision = reference["repo"], reference["revision"]
        repo_prefix(repo)
        validate_revision(revision)
        if not repo.startswith(f"{namespace}/flashrun-"):
            raise ValueError("reference is outside the Flash-owned artifact allowlist")
        requested.setdefault(repo, set()).add(revision)
        path = validate_path(reference["path"])
        required_paths.setdefault((repo, revision), set()).add(path)
    repositories = {item.id for item in api.list_datasets(author=namespace)} | requested.keys()
    snapshots = []
    for repo in sorted(repositories):
        if not repo.startswith(f"{namespace}/flashrun-"):
            continue
        info = api.repo_info(repo, repo_type="dataset")
        head = validate_revision(info.sha)
        for revision in sorted({head} | requested.get(repo, set())):
            entries = api.list_repo_tree(
                repo, repo_type="dataset", revision=revision, recursive=True, expand=True
            )
            entries = list(entries)
            selected = recent_files(entries, cutoff=now - WINDOW, end=now)
            selected_names = {file["path"] for file in selected}
            for required in required_paths.get((repo, revision), set()):
                source_names = {
                    entry.path
                    for entry in entries
                    if hasattr(entry, "size")
                    and (entry.path == required or entry.path.startswith(required + "/"))
                }
                if not source_names or not source_names <= selected_names:
                    raise ValueError(
                        "a required artifact path is missing or outside the three-day window"
                    )
            if selected:
                snapshots.append(
                    {
                        "repo": repo,
                        "revision": revision,
                        "head": revision == head,
                        "files": selected,
                    }
                )
            elif revision in requested.get(repo, set()):
                raise ValueError(
                    "a requested pinned revision has no artifacts within the three-day window"
                )
    return {
        "schema": 1,
        "namespace": namespace,
        "cutoff": (now - WINDOW).isoformat(),
        "end": now.isoformat(),
        "snapshots": snapshots,
    }


def validate_plan(plan):
    end = datetime.fromisoformat(plan["end"])
    cutoff = datetime.fromisoformat(plan["cutoff"])
    if (
        plan.get("schema") != 1
        or end.tzinfo is None
        or cutoff.tzinfo is None
        or end - cutoff != WINDOW
    ):
        raise ValueError("migration inventory must cover exactly three days")
    namespace = validate_path(plan["namespace"])
    if "/" in namespace:
        raise ValueError("invalid migration namespace")
    seen = set()
    for snapshot in plan["snapshots"]:
        repo = snapshot["repo"]
        repo_prefix(repo)
        revision = validate_revision(snapshot["revision"])
        if not repo.startswith(f"{namespace}/flashrun-") or (repo, revision) in seen:
            raise ValueError("invalid or duplicate migration repository revision")
        seen.add((repo, revision))
        groups = {}
        names = set()
        for file in snapshot["files"]:
            path = validate_path(file["path"])
            if path in names or type(file["size"]) is not int or file["size"] < 0:
                raise ValueError("invalid or duplicate migration file")
            names.add(path)
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", file["source_blob_id"]):
                raise ValueError("invalid source file digest")
            stamp = file["modified"]
            if type(stamp) not in {int, float} or not math.isfinite(stamp) or stamp <= 0:
                raise ValueError("invalid migration timestamp")
            groups.setdefault(checkpoint_group(path), []).append(stamp)
        if any(
            not cutoff.timestamp() <= max(stamps) <= end.timestamp() for stamps in groups.values()
        ):
            raise ValueError("migration inventory includes artifacts outside the three-day window")


def verify_source(path, entry):
    if path.stat().st_size != entry["size"]:
        raise ValueError("source artifact size changed")
    expected = entry["source_blob_id"]
    digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
    if len(expected) == 40:
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("source artifact does not match its pinned content identity")


def apply_plan(plan, store, download):
    """Publish imported aliases only after every selected byte passes read-back verification."""
    validate_plan(plan)
    results = []
    for snapshot in plan["snapshots"]:
        repo, source_revision = snapshot["repo"], snapshot["revision"]
        files = {}
        for entry in snapshot["files"]:
            with tempfile.TemporaryDirectory(prefix="flash-artifact-migration-") as temporary:
                path = Path(
                    download(
                        repo_id=repo,
                        filename=entry["path"],
                        revision=source_revision,
                        repo_type="dataset",
                        local_dir=Path(temporary) / "source",
                    )
                )
                verify_source(path, entry)
                stored = {
                    **store.put_file(repo, path),
                    "modified": entry["modified"],
                    "source_blob_id": entry["source_blob_id"],
                }
                store.download(repo, entry["path"], stored, Path(temporary) / "verify")
                files[entry["path"]] = stored
        raw = encode_manifest(repo, files)
        revision = revision_for(raw)
        prefix = repo_prefix(repo)
        store.immutable(f"{prefix}/revisions/{revision}.json", raw)
        store.immutable(f"{prefix}/aliases/{source_revision}", revision.encode())
        # An Azure writer or prior migration may already own HEAD; never overwrite it.
        if snapshot["head"]:
            from azure.core.exceptions import ResourceExistsError

            try:
                store.read(f"{prefix}/HEAD", 40)
            except ArtifactNotFoundError:
                with contextlib.suppress(ResourceExistsError):
                    store.container.get_blob_client(f"{prefix}/HEAD").upload_blob(
                        revision.encode(), overwrite=False
                    )
        results.append(
            {
                "repo": repo,
                "source_revision": source_revision,
                "azure_revision": revision,
                "files": len(files),
                "verified_bytes": sum(file["size"] for file in files.values()),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "apply"))
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--namespace", default="Freesolo-Co")
    parser.add_argument(
        "--run-state-directory",
        type=Path,
        help="Private Flash runs directory from a drained controller",
    )
    parser.add_argument(
        "--references",
        type=Path,
        help="JSON array of persisted {repo, revision, path} checkpoint/source references",
    )
    args = parser.parse_args()
    # This is the only remaining HF reader for Flash-owned storage, isolated to migration.
    from huggingface_hub import HfApi, hf_hub_download

    if args.mode == "plan":
        if args.run_state_directory is None or not args.run_state_directory.is_dir():
            parser.error(
                "plan requires --run-state-directory pointing to the controller's saved run records"
            )
        if args.references is None:
            parser.error(
                "plan requires --references with serving registry pins (use [] only after confirming none exist)"
            )
        now = datetime.now(UTC)
        references = json.loads(args.references.read_text()) if args.references else []
        references += state_references(args.run_state_directory, now=now)
        plan = make_plan(HfApi(), args.namespace, now=now, references=references)
        with args.inventory.open("x") as output:
            json.dump(plan, output, indent=2)
        print(
            f"Planned {len(plan['snapshots'])} snapshots from {plan['cutoff']} through {plan['end']}"
        )
    else:
        from flash.artifacts.config import open_container
        from flash.artifacts.storage import BlobArtifacts

        plan = json.loads(args.inventory.read_text())
        validate_plan(plan)
        with open_container() as container:
            if container.get_container_properties().public_access:
                raise ValueError("migration destination must be private")
            results = apply_plan(plan, BlobArtifacts(container), hf_hub_download)
        report = args.inventory.with_suffix(".verified.json")
        report.write_text(json.dumps(results, indent=2))
        print(f"Verified {sum(item['verified_bytes'] for item in results)} bytes; report: {report}")


if __name__ == "__main__":
    main()
