"""Extra coverage for `flash.envs.loading.pull` and `flash.envs.loading.loader`.

Focuses on branches the existing env-pull suites don't reach: path-safety
validation, destination-availability guards, oversized/malformed package
extraction, and the pure ref-parsing / dataset-probing helpers in the loader.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

from flash.envs.loading import loader, pull
from flash.envs.package.limits import LimitedArchiveReader
from flash.envs.package.unpack import extract_validated_archive_members


def _package_tarball(entries: dict[str, bytes]) -> bytes:
    """A flat managed-environment package tarball (files at the archive root)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


_ENV_ONLY = {"environment.py": b"# env\n", "datasets/train.jsonl": b'{"a":1}\n'}


def test_extract_validated_archive_members_public_boundary(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name, data in {
            "keep.txt": b"ok",
            "filtered.txt": b"too large for the content limit",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    archive.seek(0)
    reader = LimitedArchiveReader(
        archive,
        1 << 20,
        lambda: RuntimeError("unexpected archive limit"),
    )
    observed: list[list[str]] = []

    extract_validated_archive_members(
        reader,
        extract_base=tmp_path,
        content_byte_limit=2,
        extracted_member_limit=1,
        scanned_member_limit=2,
        member_filter=lambda segments: segments == ["keep.txt"],
        segment_observer=observed.append,
    )

    assert observed == [["keep.txt"], ["filtered.txt"]]
    assert (tmp_path / "keep.txt").read_bytes() == b"ok"
    assert not (tmp_path / "filtered.txt").exists()


# --- flash.envs.loading.pull ------------------------------------------------------------------


def test_safe_repo_relative_path_normalizes_and_rejects():
    # dot-segments are stripped and backslashes normalized to a clean relative path.
    assert pull._safe_repo_relative_path(".\\a\\./b") == "a/b"
    assert pull._safe_repo_relative_path("dataset/train.jsonl") == "dataset/train.jsonl"

    for bad in ("", "   ", "/etc/passwd", "a/../b", ".", "./.."):
        with pytest.raises(ValueError, match="invalid environment file path"):
            pull._safe_repo_relative_path(bad)


def test_ensure_destination_available_occupied_and_overwrite(tmp_path):
    occupied = tmp_path / "taken"
    occupied.write_text("keep")

    with pytest.raises(FileExistsError, match="already exists"):
        pull.ensure_environment_pull_destination_available(occupied)

    # overwrite=True on a plain file just returns the path (extraction handles replacement).
    assert pull.ensure_environment_pull_destination_available(occupied, overwrite=True) == occupied

    # a free path is always returned untouched.
    fresh = tmp_path / "fresh"
    assert pull.ensure_environment_pull_destination_available(fresh) == fresh


def test_ensure_destination_refuses_overwrite_containing_cwd(tmp_path, monkeypatch):
    dest = tmp_path / "here"
    dest.mkdir()
    monkeypatch.chdir(dest)

    with pytest.raises(RuntimeError, match="current working directory"):
        pull.ensure_environment_pull_destination_available(dest, overwrite=True)


def test_download_env_file_missing_raises_filenotfound():
    package = _package_tarball(_ENV_ONLY)

    with pytest.raises(FileNotFoundError, match="not found in package"):
        pull.download_environment_file_from_archive(package, "datasets/valid.jsonl")


def test_pull_package_missing_entrypoint_raises(tmp_path):
    package = _package_tarball({"README.md": b"# no entrypoint\n"})

    with pytest.raises(FileNotFoundError, match=r"entrypoint 'environment\.py' not found"):
        pull.pull_environment_package_from_archive(package, tmp_path / "out")


def test_extract_archive_rejects_oversized_compressed_package(monkeypatch):
    # The compressed-size guard fires before the archive is even opened.
    monkeypatch.setattr(loader, "_MAX_ARCHIVE_BYTES", 5)
    package = _package_tarball(_ENV_ONLY)
    assert len(package) > 5

    with pytest.raises(RuntimeError, match="too large compressed"):
        pull.download_environment_file_from_archive(package, "environment.py")


def test_pull_into_empty_dir_then_overwrite_nonempty_with_force(tmp_path):
    package = _package_tarball(_ENV_ONLY)

    # Existing but EMPTY destination: contents are populated in place.
    empty_dest = tmp_path / "empty"
    empty_dest.mkdir()
    out = pull.pull_environment_package_from_archive(package, empty_dest)
    assert out == empty_dest
    assert (empty_dest / "environment.py").read_bytes() == b"# env\n"
    assert (empty_dest / "datasets" / "train.jsonl").read_bytes() == b'{"a":1}\n'

    # Existing NON-empty destination: replaced wholesale only under overwrite=True.
    busy_dest = tmp_path / "busy"
    busy_dest.mkdir()
    (busy_dest / "stale.txt").write_text("old")
    pull.pull_environment_package_from_archive(package, busy_dest, overwrite=True)
    assert (busy_dest / "environment.py").read_bytes() == b"# env\n"
    assert not (busy_dest / "stale.txt").exists()


# --- flash.envs.loading.loader ----------------------------------------------------------------


def test_normalize_env_path_variants():
    assert loader._normalize_env_path(None) == "environment.py"
    assert loader._normalize_env_path("") == "environment.py"
    assert loader._normalize_env_path("   ") == "environment.py"
    assert loader._normalize_env_path("sub\\env.py") == "sub/env.py"
    assert loader._normalize_env_path("a//b/env.py") == "a/b/env.py"

    for bad in ("/abs/env.py", "a/../b", "x/./y"):
        with pytest.raises(ValueError, match="unsafe environment path"):
            loader._normalize_env_path(bad)


def test_slug_and_github_predicates_and_conversion():
    assert loader.is_managed_environment_slug("david-freesolo-co/my-project/stuff") is True
    assert loader.is_managed_environment_slug("has:colon") is False
    assert loader.is_managed_environment_slug("only-one-part") is False
    assert loader.is_managed_environment_slug("too/many/parts") is True
    assert loader.is_managed_environment_slug("too/many/parts/extra") is False

    assert loader.is_github_environment_ref("github:owner/repo@main:env/environment.py") is True
    assert loader.is_freesolo_environment_id("david-freesolo-co/my-project/stuff") is True
    assert loader.is_freesolo_environment_id("nonsense value") is False

    assert loader.managed_slug_to_github_ref("david-freesolo-co/my-project/stuff") == (
        "github:freesolo-co/environment-hub@main:david-freesolo-co/my-project/stuff/environment.py"
    )
    with pytest.raises(ValueError, match="not a Freesolo environment slug"):
        loader.managed_slug_to_github_ref("github:owner/repo@main:environment.py")


def test_parse_github_ref_from_https_urls():
    plain = loader._parse_github_environment_ref("https://github.com/owner/repo.git")
    assert plain == loader.GitHubEnvironmentRef("owner", "repo", "main", "environment.py")

    blob = loader._parse_github_environment_ref(
        "https://github.com/owner/repo/blob/dev/sub/environment.py"
    )
    assert blob == loader.GitHubEnvironmentRef("owner", "repo", "dev", "sub/environment.py")

    # A `tree` URL pointing at a directory appends the default entrypoint.
    tree = loader._parse_github_environment_ref("https://github.com/owner/repo/tree/main/mydir")
    assert tree == loader.GitHubEnvironmentRef("owner", "repo", "main", "mydir/environment.py")

    # Non-github hosts and too-short paths are not github refs.
    assert loader._parse_github_environment_ref("https://example.com/owner/repo") is None
    assert loader._parse_github_environment_ref("https://github.com/owner") is None


def test_managed_hub_package_root_is_one_environment_not_the_project():
    """The package root is `<org>/<project>/<name>`, the directory of a single environment.

    `<org>/<project>` is the project directory holding every environment the project has
    published, and the package root is what gets downloaded and copied into the cache, so
    stopping a segment short fetches all of a project's environments to import one.
    """
    ref = loader._parse_github_environment_ref(
        loader.managed_slug_to_github_ref("acme/checkout-bot/math")
    )
    assert ref.path == "acme/checkout-bot/math/environment.py"
    assert loader._managed_hub_package_root(ref) == "acme/checkout-bot/math"

    # A managed ref that stops at the project has no environment to root on.
    two_segment = loader._parse_github_environment_ref(
        "github:freesolo-co/environment-hub@main:acme/environment.py"
    )
    assert loader._managed_hub_package_root(two_segment) == ""

    # Outside the managed hub the layout is the user's own, so there is no root to derive.
    foreign = loader._parse_github_environment_ref("github:owner/repo@main:a/b/c/environment.py")
    assert loader._managed_hub_package_root(foreign) == ""


def test_resolve_path_arg_variants(tmp_path):
    # Non-string / empty values pass through unchanged.
    assert loader._resolve_path_arg(123, tmp_path) == 123
    assert loader._resolve_path_arg("", tmp_path) == ""

    # URLs and absolute paths are returned verbatim (never resolved against base_dir).
    assert loader._resolve_path_arg("https://x/y.jsonl", tmp_path) == "https://x/y.jsonl"
    assert loader._resolve_path_arg("/abs/data.jsonl", tmp_path) == "/abs/data.jsonl"

    # A relative path is rewritten only when it resolves to something that exists.
    (tmp_path / "data.jsonl").write_text("x")
    assert loader._resolve_path_arg("data.jsonl", tmp_path) == str(tmp_path / "data.jsonl")
    assert loader._resolve_path_arg("missing.jsonl", tmp_path) == "missing.jsonl"


def test_load_contract_text_variants(tmp_path):
    assert loader._load_contract_text(None) == ""
    assert loader._load_contract_text("") == ""
    assert loader._load_contract_text(str(tmp_path / "nope.md")) == ""

    good = tmp_path / "contract.md"
    good.write_text("be nice\n", encoding="utf-8")
    assert loader._load_contract_text(str(good)) == "be nice\n"

    # Invalid UTF-8 falls back to a lossy decode instead of raising.
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe bad bytes")
    recovered = loader._load_contract_text(str(bad))
    assert isinstance(recovered, str)
    assert "bad bytes" in recovered


def test_validate_split_and_packaged_dataset_file(tmp_path):
    assert loader._validate_packaged_dataset_split("train") == "train"
    assert loader._validate_packaged_dataset_split("eval-2.v1") == "eval-2.v1"
    for bad in ("has/slash", "..", "", "-leading"):
        with pytest.raises(ValueError, match="simple dataset name"):
            loader._validate_packaged_dataset_split(bad)

    # No packaged dataset present -> None.
    assert loader._packaged_dataset_file(tmp_path, "train") is None

    # dataset/<name>.jsonl takes priority over a top-level fallback.
    (tmp_path / "dataset").mkdir()
    canonical = tmp_path / "dataset" / "train.jsonl"
    canonical.write_text("row\n")
    (tmp_path / "train.jsonl").write_text("shadow\n")
    assert loader._packaged_dataset_file(tmp_path, "train") == canonical

    # Falls through to a top-level .json when no dataset/ file exists for the split.
    side = tmp_path / "eval.json"
    side.write_text("[]")
    assert loader._packaged_dataset_file(tmp_path, "eval") == side


def test_resolve_environment_reference_local_and_passthrough(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("# local env\n")
    assert loader._resolve_environment_reference(str(env_file)) == str(env_file)

    # A non-ref that doesn't exist on disk is returned unchanged.
    assert loader._resolve_environment_reference("not-a-real-thing-xyz") == "not-a-real-thing-xyz"


def test_github_response_message_and_safe_contents_path():
    assert loader._github_response_message({"message": "Not Found"}) == " (Not Found)"
    assert loader._github_response_message({"message": ""}) == ""
    assert loader._github_response_message({"other": 1}) == ""
    assert loader._github_response_message("not a dict") == ""

    assert loader._safe_contents_path("pkg/environment.py", ["pkg"]) == "pkg/environment.py"

    with pytest.raises(RuntimeError, match="did not include a path"):
        loader._safe_contents_path(123, ["pkg"])
    with pytest.raises(RuntimeError, match="unsafe path in environment contents"):
        loader._safe_contents_path("../escape.py", ["pkg"])
    with pytest.raises(RuntimeError, match="unexpected path in environment contents"):
        loader._safe_contents_path("other/environment.py", ["pkg"])


def test_freesolo_import_error_reports_the_real_failure(monkeypatch):
    """The wrapper must surface the underlying ImportError, not a fixed "install it" message.

    A fixed message is actively misleading when the SDK IS installed and something beneath it
    failed (missing transitive dep, version conflict, half-removed package after a tool upgrade):
    it sends the user to reinstall a package that is already there.
    """
    import builtins

    real_import = builtins.__import__

    def explode(name, *args, **kwargs):
        if name.startswith("freesolo"):
            raise ImportError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", explode)

    with pytest.raises(ImportError) as excinfo:
        loader._import_freesolo_environment_tools()

    message = str(excinfo.value)
    # the real cause, not a generic substitute
    assert "No module named 'pyarrow'" in message
    # which interpreter failed: `flash` runs from its own uv-tool env
    assert sys.executable in message
    # and the original exception stays chained for a traceback
    assert isinstance(excinfo.value.__cause__, ImportError)
