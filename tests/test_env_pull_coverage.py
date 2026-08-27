"""Hermetic rollback coverage for environment destination replacement helpers."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

import flash.envs.loading.pull as pull

_GENERIC_ERROR = "environment pull could not safely complete"


def _source(tmp_path: Path, files: dict[str, str]) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for name, text in files.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return source


def _staging_dirs(parent: Path) -> list[Path]:
    return sorted(parent.glob(".flash-env-pull-*"))


def _only_staging(parent: Path) -> Path:
    staging = _staging_dirs(parent)
    assert len(staging) == 1
    return staging[0]


def _probe_dirs(parent: Path) -> list[Path]:
    return sorted(parent.glob(".flash-env-rename-probe-*"))


def test_cwd_inside_returns_false_when_path_resolution_fails(monkeypatch, tmp_path) -> None:
    """destination probing returns false when filesystem resolution fails."""
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(Path, "resolve", lambda self: (_ for _ in ()).throw(OSError("gone")))

    assert pull._cwd_is_inside(tmp_path) is False


@pytest.mark.parametrize("destination_kind", ["missing", "empty", "directory", "file"])
def test_copy_environment_source_publishes_normally(tmp_path, destination_kind) -> None:
    """normal publication preserves the existing destination contract."""
    source = _source(tmp_path, {"environment.py": "new", "data/value.txt": "value"})
    dest = tmp_path / "dest"
    overwrite = destination_kind in {"directory", "file"}
    if destination_kind == "empty":
        dest.mkdir()
    elif destination_kind == "directory":
        dest.mkdir()
        (dest / "old.txt").write_text("old")
    elif destination_kind == "file":
        dest.write_text("old")

    pull._copy_environment_source(source, dest, overwrite=overwrite)

    assert (dest / "environment.py").read_text() == "new"
    assert (dest / "data/value.txt").read_text() == "value"
    assert _staging_dirs(tmp_path) == []


@pytest.mark.parametrize("backup_kind", ["directory", "file"])
def test_replace_with_tree_cleans_file_and_directory_backups(
    tmp_path, monkeypatch, backup_kind
) -> None:
    """successful replacement removes an owned file or directory backup."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    if backup_kind == "directory":
        dest.mkdir()
        (dest / "old.txt").write_text("old")
    else:
        dest.write_text("old")
    observed = []

    def hook(stage):
        if stage == "backup_cleanup_ready":
            observed.append(_only_staging(tmp_path).joinpath("old").is_dir())

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    pull._replace_with_tree(source, dest)

    assert observed == [backup_kind == "directory"]
    assert (dest / "environment.py").read_text() == "new"
    assert _staging_dirs(tmp_path) == []


def test_replace_with_tree_restores_original_after_publication_failure(
    tmp_path, monkeypatch
) -> None:
    """a failed publication restores the original and removes owned staging."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    real_rename = pull._rename_no_replace_at
    failed = False

    def rename(source_fd, source_name, target_fd, target_name):
        nonlocal failed
        if source_name == "new" and target_name == dest.name and not failed:
            failed = True
            raise OSError(errno.EPERM, "publication denied")
        real_rename(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(pull, "_rename_no_replace_at", rename)

    with pytest.raises(OSError, match="publication denied"):
        pull._replace_with_tree(source, dest)

    assert (dest / "old.txt").read_text() == "old"
    assert not (dest / "environment.py").exists()
    assert _staging_dirs(tmp_path) == []


def test_replace_with_tree_rejects_parent_substitution_after_publish(tmp_path, monkeypatch) -> None:
    """replacement cannot succeed after its captured parent leaves the pathname."""
    source = _source(tmp_path, {"environment.py": "new"})
    target_parent = tmp_path / "target"
    target_parent.mkdir()
    dest = target_parent / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    original_parent = tmp_path / "original-parent"

    def hook(stage):
        if stage != "backup_cleanup_ready":
            return
        os.replace(target_parent, original_parent)
        target_parent.mkdir()
        competing = target_parent / "dest"
        competing.mkdir()
        (competing / "competing.txt").write_text("keep")

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert (target_parent / "dest/competing.txt").read_text() == "keep"
    assert (original_parent / "dest/environment.py").read_text() == "new"
    staging = _only_staging(original_parent)
    assert (staging / "old/old.txt").read_text() == "old"


def test_replace_with_tree_retains_backup_on_parent_swap_at_cleanup(tmp_path, monkeypatch) -> None:
    """a parent swap at the final backup syscall retains the rollback artifact."""
    source = _source(tmp_path, {"environment.py": "new"})
    target_parent = tmp_path / "target"
    target_parent.mkdir()
    dest = target_parent / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    original_parent = tmp_path / "original-parent"
    swapped = False

    def hook(stage):
        nonlocal swapped
        if stage != "claimed_file_unlink_ready" or swapped or not _staging_dirs(target_parent):
            return
        swapped = True
        os.replace(target_parent, original_parent)
        target_parent.mkdir()
        competing = target_parent / "dest"
        competing.mkdir()
        (competing / "competing.txt").write_text("keep")

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert (target_parent / "dest/competing.txt").read_text() == "keep"
    assert (original_parent / "dest/environment.py").read_text() == "new"
    staging = _only_staging(original_parent)
    retained = [path for path in staging.rglob("*") if path.is_file() and path.read_text() == "old"]
    assert len(retained) == 1


def test_replace_with_tree_preserves_rollback_collision(tmp_path, monkeypatch) -> None:
    """a competing destination blocks rollback without being replaced or deleted."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")

    def hook(stage):
        if stage == "publish_ready":
            dest.mkdir()
            (dest / "competing.txt").write_text("keep")

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert (dest / "competing.txt").read_text() == "keep"
    staging = _only_staging(tmp_path)
    assert (staging / "old/old.txt").read_text() == "old"
    assert (staging / "new/environment.py").read_text() == "new"


@pytest.mark.parametrize(
    ("entry_kind", "stage"),
    [
        ("file", "claimed_file_unlink_ready"),
        ("directory", "claimed_directory_remove_ready"),
    ],
)
def test_backup_cleanup_preserves_child_substituted_at_removal(
    tmp_path, monkeypatch, entry_kind, stage
) -> None:
    """the final removal claim retains a syscall-adjacent child substitution."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    victim = dest / "victim"
    if entry_kind == "file":
        victim.write_text("old")
    else:
        victim.mkdir()
    retained = tmp_path / "original-child-retained"
    competing = None

    def hook(current_stage):
        nonlocal competing
        if current_stage != stage or competing is not None or not _staging_dirs(tmp_path):
            return
        staging = _only_staging(tmp_path)
        candidates = [
            path
            for path in staging.rglob(".flash-env-owned-*")
            if path != staging
            and (
                (entry_kind == "file" and path.is_file() and path.read_text() == "old")
                or (entry_kind == "directory" and path.is_dir() and not any(path.iterdir()))
            )
        ]
        assert len(candidates) == 1
        competing = candidates[0]
        os.replace(competing, retained)
        if entry_kind == "file":
            competing.write_text("competing")
        else:
            competing.mkdir()
            (competing / "competing.txt").write_text("keep")

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    if entry_kind == "file":
        assert retained.read_text() == "old"
        assert competing is not None
        assert competing.read_text() == "competing"
    else:
        assert retained.is_dir()
        assert competing is not None
        assert (competing / "competing.txt").read_text() == "keep"
    assert (dest / "environment.py").read_text() == "new"


def test_backup_cleanup_preserves_substituted_child(tmp_path, monkeypatch) -> None:
    """cleanup claims a child before deletion and retains a substituted child."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    retained = tmp_path / "original-child-retained"

    def hook(stage):
        if stage == "backup_cleanup_ready":
            old = _only_staging(tmp_path) / "old"
            os.replace(old / "old.txt", retained)
            (old / "old.txt").write_text("competing")

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert retained.read_text() == "old"
    assert (_only_staging(tmp_path) / "old/old.txt").read_text() == "competing"
    assert (dest / "environment.py").read_text() == "new"


def test_backup_cleanup_preserves_child_substituted_after_claim(tmp_path, monkeypatch) -> None:
    """cleanup revalidates a claimed child immediately before unlinking it."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    retained = tmp_path / "original-child-retained"
    real_claim = pull._claim_entry
    competing_claim = None

    def claim(parent_fd, name, snapshot):
        nonlocal competing_claim
        claimed_name = real_claim(parent_fd, name, snapshot)
        if name == "old.txt" and competing_claim is None:
            claimed_path = pull._fd_path(parent_fd) / claimed_name
            os.replace(claimed_path, retained)
            claimed_path.write_text("competing")
            competing_claim = claimed_name
        return claimed_name

    monkeypatch.setattr(pull, "_claim_entry", claim)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert retained.read_text() == "old"
    assert competing_claim is not None
    staging = _only_staging(tmp_path)
    competing = list(staging.rglob(competing_claim))
    assert len(competing) == 1
    assert competing[0].read_text() == "competing"
    assert (dest / "environment.py").read_text() == "new"


def test_backup_cleanup_preserves_substituted_root(tmp_path, monkeypatch) -> None:
    """cleanup atomically claims the backup root before recursive deletion."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    retained = tmp_path / "original-root-retained"

    def hook(stage):
        if stage == "backup_cleanup_ready":
            old = _only_staging(tmp_path) / "old"
            os.replace(old, retained)
            old.mkdir()
            (old / "competing.txt").write_text("keep")

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert (retained / "old.txt").read_text() == "old"
    assert (_only_staging(tmp_path) / "old/competing.txt").read_text() == "keep"
    assert (dest / "environment.py").read_text() == "new"


def test_backup_cleanup_preserves_substituted_non_directory(tmp_path, monkeypatch) -> None:
    """cleanup never unlinks a non-directory substituted for a directory backup."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    retained = tmp_path / "original-root-retained"

    def hook(stage):
        if stage == "backup_cleanup_ready":
            old = _only_staging(tmp_path) / "old"
            os.replace(old, retained)
            old.write_text("competing")

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert (retained / "old.txt").read_text() == "old"
    assert (_only_staging(tmp_path) / "old").read_text() == "competing"
    assert (dest / "environment.py").read_text() == "new"


def test_rollback_preserves_substituted_backup_and_original(tmp_path, monkeypatch) -> None:
    """rollback claims the verified tree before restoring it to the destination."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    retained = tmp_path / "original-backup-retained"
    real_rename = pull._rename_no_replace_at
    failed = False

    def rename(source_fd, source_name, target_fd, target_name):
        nonlocal failed
        if source_name == "new" and target_name == dest.name and not failed:
            failed = True
            raise OSError(errno.EPERM, "publication denied")
        real_rename(source_fd, source_name, target_fd, target_name)

    def hook(stage):
        if stage == "rollback_ready":
            old = _only_staging(tmp_path) / "old"
            os.replace(old, retained)
            old.mkdir()
            (old / "competing.txt").write_text("keep")

    monkeypatch.setattr(pull, "_rename_no_replace_at", rename)
    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert (retained / "old.txt").read_text() == "old"
    staging = _only_staging(tmp_path)
    assert (staging / "old/competing.txt").read_text() == "keep"
    assert (staging / "new/environment.py").read_text() == "new"
    assert not dest.exists()


def test_rollback_preserves_root_substituted_immediately_before_restore(
    tmp_path, monkeypatch
) -> None:
    """rollback retains a root replaced at the final restore boundary."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    retained = tmp_path / "original-backup-retained"
    real_rename = pull._rename_no_replace_at
    failed = False
    competing = None

    def rename(source_fd, source_name, target_fd, target_name):
        nonlocal failed
        if source_name == "new" and target_name == dest.name and not failed:
            failed = True
            raise OSError(errno.EPERM, "publication denied")
        real_rename(source_fd, source_name, target_fd, target_name)

    def hook(stage):
        nonlocal competing
        if stage != "rollback_restore_ready" or competing is not None:
            return
        staging = _only_staging(tmp_path)
        candidates = list(staging.glob(".flash-env-restore-*"))
        assert len(candidates) == 1
        competing = candidates[0]
        os.replace(competing, retained)
        competing.mkdir()
        (competing / "competing.txt").write_text("keep")

    monkeypatch.setattr(pull, "_rename_no_replace_at", rename)
    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert (retained / "old.txt").read_text() == "old"
    assert competing is not None
    assert (competing / "competing.txt").read_text() == "keep"
    assert (_only_staging(tmp_path) / "new/environment.py").read_text() == "new"
    assert not dest.exists()


def test_rollback_preserves_root_substituted_after_claim(tmp_path, monkeypatch) -> None:
    """rollback revalidates its claimed root before moving it to the destination."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    retained = tmp_path / "original-backup-retained"
    real_rename = pull._rename_no_replace_at
    real_claim = pull._claim_tree_for_move
    failed = False
    competing_claim = None

    def rename(source_fd, source_name, target_fd, target_name):
        nonlocal failed
        if source_name == "new" and target_name == dest.name and not failed:
            failed = True
            raise OSError(errno.EPERM, "publication denied")
        real_rename(source_fd, source_name, target_fd, target_name)

    def claim(parent_fd, name, snapshot):
        nonlocal competing_claim
        claimed_name = real_claim(parent_fd, name, snapshot)
        claimed_path = pull._fd_path(parent_fd) / claimed_name
        os.replace(claimed_path, retained)
        claimed_path.mkdir()
        (claimed_path / "competing.txt").write_text("keep")
        competing_claim = claimed_name
        return claimed_name

    monkeypatch.setattr(pull, "_rename_no_replace_at", rename)
    monkeypatch.setattr(pull, "_claim_tree_for_move", claim)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert (retained / "old.txt").read_text() == "old"
    assert competing_claim is not None
    staging = _only_staging(tmp_path)
    assert (staging / competing_claim / "competing.txt").read_text() == "keep"
    assert (staging / "new/environment.py").read_text() == "new"
    assert not dest.exists()


def test_empty_destination_binds_source_and_target_directories(tmp_path, monkeypatch) -> None:
    """publication uses captured descriptors when destination and staging paths are replaced."""
    source = _source(tmp_path, {"environment.py": "new"})
    target_parent = tmp_path / "target"
    target_parent.mkdir()
    dest = target_parent / "dest"
    dest.mkdir()
    original_parent = tmp_path / "original-parent"
    retained_staging = tmp_path / "original-staging-retained"

    def hook(stage):
        if stage != "empty_publish_ready":
            return
        staging = _only_staging(dest)
        os.replace(staging, retained_staging)
        replacement_staging = dest / staging.name
        replacement_staging.mkdir()
        (replacement_staging / "contents").mkdir()
        (replacement_staging / "contents/environment.py").write_text("competing")
        os.replace(target_parent, original_parent)
        target_parent.mkdir()
        replacement_dest = target_parent / "dest"
        replacement_dest.mkdir()
        (replacement_dest / "sentinel.txt").write_text("keep")

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._copy_environment_source(source, dest)

    assert (target_parent / "dest/sentinel.txt").read_text() == "keep"
    assert not (target_parent / "dest/environment.py").exists()
    assert (retained_staging / "contents/environment.py").read_text() == "new"
    assert (
        original_parent / f"dest/{retained_staging.name}/contents/environment.py"
    ).exists() is False
    replacement_staging = _only_staging(original_parent / "dest")
    assert (replacement_staging / "contents/environment.py").read_text() == "competing"


def test_empty_destination_rollback_preserves_substituted_entry(tmp_path, monkeypatch) -> None:
    """empty-directory rollback retains a substitution at its final move boundary."""
    source = _source(tmp_path, {"a.txt": "first", "b.txt": "second"})
    dest = tmp_path / "dest"
    dest.mkdir()
    retained = tmp_path / "published-entry-retained"
    real_rename = pull._rename_no_replace_at
    failed = False
    competing = None

    def rename(source_fd, source_name, target_fd, target_name):
        nonlocal failed
        if source_name == "b.txt" and target_name == "b.txt" and not failed:
            failed = True
            raise OSError(errno.EPERM, "publication denied")
        real_rename(source_fd, source_name, target_fd, target_name)

    def hook(stage):
        nonlocal competing
        if stage != "empty_rollback_move_ready" or competing is not None:
            return
        candidates = list(dest.glob(".flash-env-restore-*"))
        assert len(candidates) == 1
        competing = candidates[0]
        os.replace(competing, retained)
        competing.write_text("competing")

    monkeypatch.setattr(pull, "_rename_no_replace_at", rename)
    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._copy_environment_source(source, dest)

    assert retained.read_text() == "first"
    assert competing is not None
    assert competing.read_text() == "competing"
    staging = _only_staging(dest)
    assert (staging / "contents/b.txt").read_text() == "second"


def test_copy_environment_source_rejects_preflight_identity_swap(tmp_path, monkeypatch) -> None:
    """an identity captured before preflight cannot be replaced by another empty directory."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    real_ensure = pull.ensure_environment_pull_destination_available

    def ensure(path, *, overwrite=False):
        result = real_ensure(path, overwrite=overwrite)
        os.replace(dest, original)
        os.replace(replacement, dest)
        return result

    monkeypatch.setattr(pull, "ensure_environment_pull_destination_available", ensure)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._copy_environment_source(source, dest)

    assert list(original.iterdir()) == []
    assert list(dest.iterdir()) == []


def test_transaction_fails_before_mutation_when_capability_is_unsupported(
    tmp_path, monkeypatch
) -> None:
    """unsupported systems fail before moving or deleting the destination."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")

    def unsupported():
        raise RuntimeError("atomic environment publication is unsupported on this platform")

    monkeypatch.setattr(pull, "_require_transaction_support", unsupported)

    with pytest.raises(RuntimeError, match="unsupported on this platform"):
        pull._replace_with_tree(source, dest)

    assert (dest / "old.txt").read_text() == "old"
    assert _staging_dirs(tmp_path) == []


def test_capability_probe_cleanup_preserves_substituted_entry(tmp_path, monkeypatch) -> None:
    """probe cleanup retains a substituted entry rather than deleting it."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    retained = tmp_path / "probe-entry-retained"
    competing = None

    def hook(stage):
        nonlocal competing
        if stage != "probe_entry_cleanup_ready" or competing is not None:
            return
        probes = list(tmp_path.glob(".flash-env-rename-probe-*"))
        assert len(probes) == 1
        installed = probes[0] / "installed"
        assert installed.is_dir()
        os.replace(installed, retained)
        installed.mkdir()
        (installed / "competing.txt").write_text("keep")
        competing = installed

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert retained.is_dir()
    assert competing is not None
    assert (competing / "competing.txt").read_text() == "keep"
    assert (dest / "old.txt").read_text() == "old"


def test_transaction_retains_probe_when_native_no_replace_is_unsupported(
    tmp_path, monkeypatch
) -> None:
    """a missing no-replace primitive retains the probe instead of deleting by name."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")

    def unsupported(*args):
        raise RuntimeError("atomic environment publication is unsupported on this platform")

    monkeypatch.setattr(pull, "_rename_no_replace_at", unsupported)

    with pytest.raises(RuntimeError, match="unsupported on this platform"):
        pull._replace_with_tree(source, dest)

    assert (dest / "old.txt").read_text() == "old"
    probes = list(tmp_path.glob(".flash-env-rename-probe-*"))
    assert len(probes) == 1
    assert sorted(path.name for path in probes[0].iterdir()) == ["occupied", "source"]


def test_copy_environment_source_refuses_current_working_directory(tmp_path, monkeypatch) -> None:
    """the copy helper retains its current-directory guard after preflight."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    monkeypatch.setattr(pull, "ensure_environment_pull_destination_available", lambda *a, **k: dest)
    monkeypatch.setattr(pull, "_cwd_is_inside", lambda path: True)

    with pytest.raises(RuntimeError, match="current working directory"):
        pull._copy_environment_source(source, dest, overwrite=True)

    assert (dest / "old.txt").read_text() == "old"


def test_copy_environment_source_rejects_existing_destination_without_overwrite(
    tmp_path, monkeypatch
) -> None:
    """the copy helper requires overwrite for an occupied destination."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.write_text("old")
    monkeypatch.setattr(pull, "ensure_environment_pull_destination_available", lambda *a, **k: dest)

    with pytest.raises(FileExistsError, match="already exists"):
        pull._copy_environment_source(source, dest, overwrite=False)

    assert dest.read_text() == "old"


def test_publish_rejects_a_payload_substituted_after_the_final_sync(tmp_path, monkeypatch) -> None:
    """publication verifies the staged payload by inode, not by the name it was copied under."""
    source = _source(tmp_path, {"environment.py": "authentic"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")

    def hook(stage):
        if stage != "publish_ready":
            return
        staging = _only_staging(tmp_path)
        (staging / "new").rename(staging / "stashed")
        forged = staging / "new"
        forged.mkdir()
        (forged / "environment.py").write_text("forged")

    monkeypatch.setattr(pull, "_transaction_sync_hook", hook)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    # the swap is rejected and the backup is restored, so the destination is neither forged nor
    # vacant. rejecting outside the publish block would leave the destination missing entirely.
    assert (dest / "old.txt").read_text() == "old"
    assert not (dest / "environment.py").exists()
    assert _staging_dirs(tmp_path) == []


def test_rejected_backup_move_restores_the_destination(tmp_path, monkeypatch) -> None:
    """a backup whose identity fails its check is renamed back rather than left in staging."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")

    real_entry_identity = pull._entry_identity
    poisoned = False

    def fake_entry_identity(parent_fd, name):
        nonlocal poisoned
        if name == "old" and not poisoned:
            poisoned = True
            return (-1, -1, -1)
        return real_entry_identity(parent_fd, name)

    monkeypatch.setattr(pull, "_entry_identity", fake_entry_identity)

    with pytest.raises(RuntimeError, match=_GENERIC_ERROR):
        pull._replace_with_tree(source, dest)

    assert (dest / "old.txt").read_text() == "old"
    assert _staging_dirs(tmp_path) == []


def test_failed_copy_removes_its_own_staging_directory(tmp_path, monkeypatch) -> None:
    """a failure before the backup exists deletes the staged payload it created."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old")

    real_copy = pull._copy_tree_into_staging

    def failing_copy(source_path, staging, name):
        os.close(real_copy(source_path, staging, name))
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(pull, "_copy_tree_into_staging", failing_copy)

    with pytest.raises(OSError, match="No space left on device"):
        pull._replace_with_tree(source, dest)

    assert _staging_dirs(tmp_path) == []
    assert (dest / "old.txt").read_text() == "old"


def test_failed_probe_does_not_occupy_an_empty_destination(tmp_path, monkeypatch) -> None:
    """a probe that fails mid-way removes itself so a retry still sees an empty destination."""
    source = _source(tmp_path, {"environment.py": "new"})
    dest = tmp_path / "dest"
    dest.mkdir()

    real_rename = pull._rename_no_replace_at

    def failing_rename(source_fd, source_name, target_fd, target_name):
        if target_name == "installed":
            raise OSError(errno.EIO, "Input/output error")
        return real_rename(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(pull, "_rename_no_replace_at", failing_rename)

    with pytest.raises(OSError, match="Input/output error"):
        pull._copy_environment_source(source, dest)

    # probe residue would make this empty destination occupied, and every later pull without
    # overwrite would then fail with FileExistsError on evidence the probe created itself.
    assert _probe_dirs(dest) == []
    assert list(dest.iterdir()) == []

    monkeypatch.undo()
    pull._copy_environment_source(source, dest)
    assert (dest / "environment.py").read_text() == "new"


def test_copy_environment_source_accepts_the_current_directory(tmp_path, monkeypatch) -> None:
    """a '.' destination names a directory, not an entry, so it is resolved before the stat."""
    source = _source(tmp_path, {"environment.py": "new"})
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    pull._copy_environment_source(source, Path("."))

    assert (workdir / "environment.py").read_text() == "new"


def test_addressable_destination_resolves_a_parent_reference(tmp_path) -> None:
    """'..' is not normalized away by pathlib, so it would stat as the grandparent."""
    nested = tmp_path / "outer" / "inner"
    nested.mkdir(parents=True)

    resolved = pull._addressable_destination(nested / "..")

    assert resolved == (tmp_path / "outer").resolve()
    assert resolved.name == "outer"


def test_addressable_destination_refuses_the_root_directory(tmp_path, monkeypatch) -> None:
    """resolution that lands on the root leaves no entry name to publish over."""
    monkeypatch.chdir(Path(os.sep))

    with pytest.raises(RuntimeError, match="root directory"):
        pull._addressable_destination(Path("."))
