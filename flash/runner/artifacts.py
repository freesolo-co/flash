"""Artifact repository naming and environment pinning."""

from __future__ import annotations

import hashlib
import os
import re

import flash.runner as runner
from flash.core.spec import JobSpec


def artifact_namespace() -> str:
    """The HuggingFace namespace run artifacts are created under.

    Flash streams code, checkpoints and adapters through HF dataset repos that the control plane
    CREATES, so the namespace has to be one the operator's ``HF_TOKEN`` can write to. Hardcoding
    Freesolo's made self-hosting impossible: ``_assign_managed_hf_repo`` runs on every submit, and a
    self-hoster's token cannot create ``Freesolo-Co/flashrun-*``, so the run failed at upload before
    any training started.
    """
    return (
        os.environ.get("FLASH_HF_NAMESPACE") or ""
    ).strip() or runner._DEFAULT_ARTIFACT_NAMESPACE


def _environment_artifact_repo_name(env_id: str) -> str:
    """Stable HF dataset repo name for all runs of one environment."""
    raw = (env_id or "default-environment").strip() or "default-environment"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw.lower()).strip("-") or "environment"
    budget = runner._ARTIFACT_REPO_NAME_MAX - len(runner._ARTIFACT_REPO_PREFIX) - len(digest) - 1
    slug = slug[:budget].rstrip("-") or "environment"
    return f"{runner._ARTIFACT_REPO_PREFIX}{slug}-{digest}"


def managed_hf_repo_for_environment(env_id: str) -> str:
    """Private HF dataset repo shared by runs that use the same environment id."""
    return f"{runner.artifact_namespace()}/{runner._environment_artifact_repo_name(env_id)}"


def _file_digest(path: str, digest) -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)


def flash_code_prefix() -> str:
    """Content-addressed HF path for the current ``flash`` package snapshot."""
    import flash

    pkg_dir = os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    digest = hashlib.sha1()
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, pkg_dir).replace(os.sep, "/")
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            runner._file_digest(path, digest)
            digest.update(b"\0")
    return f"code/{digest.hexdigest()[:32]}/flash"


def _assign_managed_hf_repo(spec: JobSpec) -> JobSpec:
    """Assign the environment-scoped HF artifact repo (platform-managed, never user-set)."""
    if not spec.run_id or spec.run_id == "local":
        raise ValueError("run_id must be finalized before assigning the artifact repo")
    repo = runner.managed_hf_repo_for_environment(spec.environment.id)
    d = spec.to_internal_dict()
    d["train"] = {**d["train"], "hf_repo": repo}
    return runner.JobSpec.from_dict(d)


def _pin_env_sha_with_reason(spec: JobSpec) -> tuple[JobSpec, str]:
    """Pin env ref->SHA, returning the spec and why the pin failed (or "" when it did not).

    One resolve, two answers. The pin itself is best-effort and its caller only ever sees the
    ABSENCE of a sha, which is the same observation for four different causes -- a ref that does
    not exist, a rate limit, an outage, a private repo the token cannot read -- each needing a
    different fix. GitHub already answered with which one; this carries that answer out instead of
    discarding it.

    Returning the reason rather than re-resolving on the failure path matters for correctness, not
    just cost: a second call can succeed where the first failed (a rate-limit window that reset, a
    blip that cleared), and a caller that has already committed to rejecting would then report an
    empty reason for a ref that just resolved fine. The reason must describe the attempt whose
    result is actually being used.
    """
    import logging

    env_id = spec.environment.id
    if not env_id or spec.environment.resolved_sha:
        return spec, ""
    try:
        from flash.envs.loader import (
            _parse_github_environment_ref,
            _resolve_ref_sha,
            is_managed_environment_slug,
            managed_slug_to_github_ref,
        )

        ref_str = (
            managed_slug_to_github_ref(env_id) if is_managed_environment_slug(env_id) else env_id
        )
        parsed = _parse_github_environment_ref(ref_str)
        if parsed is None:
            return spec, ""
        sha = _resolve_ref_sha(parsed, timeout=10.0, max_rate_limit_retries=0)
    except Exception as e:
        logging.getLogger(runner.__name__).warning(
            "resolve-once: could not pin env ref->sha for %r (%s); worker will resolve", env_id, e
        )
        return spec, str(e).strip()
    if not sha:
        return spec, ""
    d = spec.to_internal_dict()
    d["environment"] = {**d["environment"], "resolved_sha": sha}
    return runner.JobSpec.from_dict(d), ""


def _assign_resolved_env_sha(spec: JobSpec) -> JobSpec:
    """Pin env ref->SHA once so N workers don't fan-out N GitHub API calls (secondary rate-limit). Best-effort."""
    return _pin_env_sha_with_reason(spec)[0]


def preflight_validate_environment_ref(spec: JobSpec) -> JobSpec:
    """Refuse an environment ref GitHub has permanently rejected, before a GPU is allocated.

    ``_assign_resolved_env_sha`` is best-effort by design: grpo and opd have no workload profile
    keyed on the pin, so a GitHub blip at submit must not fail the run -- the worker resolves the ref
    itself, and the lifecycle fallback pins it on recovery. That deferral is correct for a blip and
    wrong for a typo: a repo that does not exist gets the same "worker will resolve" treatment, so
    ``--dry-run`` reported ``state: dry_run, error: null`` and a real submit rented a GPU to discover
    a 404 the control plane already had in hand.

    This splits the two cases on the only thing that distinguishes them -- whether GitHub's answer
    can change on a retry. A permanent answer (404/422) is refused here; everything else, including
    both transient classes, still defers. Read-only and allocation-independent, so it runs in
    dry-run too: that is the mode whose whole purpose is to answer this question without paying.

    Requires a GitHub token to refuse anything. GitHub answers 404 for a repo that does not exist
    AND for a private one the caller may not see, so an unauthenticated 404 is not evidence of a
    typo: every private environment, including every managed-hub slug, looks exactly like one. Only
    an authenticated 404 means "not there", so a tokenless plane keeps the old deferral rather than
    refusing runs whose environment it simply cannot see.

    Two things it deliberately does not catch, because catching them costs a request each and the
    deferral is already correct for both. A ref that is itself a 40-hex commit sha short-circuits
    inside ``_resolve_ref_sha`` without contacting GitHub, so a fabricated sha still reaches the
    worker -- and it must, since the pinned-sha path is the fan-out this exists to keep cheap. And
    the ref's PATH is never fetched: ``/repos/{repo}/commits/{ref}`` proves the repo and the ref,
    not that ``environment.py`` sits where the id says. A typo'd path therefore fails on the worker
    with ``environment archive did not contain required entrypoint``, which names the file.

    sft never reaches this state -- its profile gate already fails closed on any unpinned
    environment -- but this runs for it as well, so the failure names the ref rather than the
    missing profile it caused.

    Returns the spec with the resolved sha pinned when the resolve succeeded. The gate has to make
    the call anyway, and ``_assign_resolved_env_sha`` would otherwise make the identical one a few
    lines later; keeping the answer removes that second round trip against the same secondary rate
    limit the pin exists to protect. Callers that do not want the pin can ignore the return value --
    it is the same spec whenever nothing was resolved.
    """
    env_id = spec.environment.id
    if not env_id or spec.environment.resolved_sha:
        return spec
    from flash.envs.identity import GitHubPermanentError
    from flash.envs.loader import (
        _github_token,
        _parse_github_environment_ref,
        _resolve_ref_sha,
        is_managed_environment_slug,
        managed_slug_to_github_ref,
    )

    if not _github_token():
        return spec

    try:
        ref_str = (
            managed_slug_to_github_ref(env_id) if is_managed_environment_slug(env_id) else env_id
        )
        parsed = _parse_github_environment_ref(ref_str)
    except Exception:
        # not a resolvable github ref (a local path, a malformed slug). the loader owns that
        # verdict and reports it far better than a half-parsed ref could here.
        return spec
    if parsed is None:
        return spec
    try:
        # 4s, not the pin's 10s: this call is on the API request thread for every submit (the server
        # always passes background=True, so the pin below it is off-thread and can afford to wait).
        # a GitHub incident that hangs every submit for 10s exhausts the plane's request workers,
        # and the run it would have blocked is one the deferral admits anyway -- a timeout is
        # transient, so this gate's answer for it is "defer" no matter how long it waits. the only
        # thing a longer budget buys is a named refusal for a 404 that GitHub is slow to give.
        sha = _resolve_ref_sha(parsed, timeout=4.0, max_rate_limit_retries=0)
    except GitHubPermanentError as exc:
        # what GitHub rejected is the repo or the ref -- ``/repos/{repo}/commits/{ref}`` is the only
        # call made -- so the message names those two and not the path, which is checked by nobody
        # here. see the note below on why it stays that way.
        raise runner.EnvironmentRefNotFound(
            f"environment {env_id!r} could not be resolved on GitHub: {exc}. Verify the repository "
            "and ref exist and that the plane's GitHub token can read them"
        ) from exc
    except Exception:
        # transient, or anything else unproven: keep the documented best-effort deferral. the pin
        # is an optimisation for these, and the worker resolves the ref on its own.
        return spec
    if not sha:
        return spec
    d = spec.to_internal_dict()
    d["environment"] = {**d["environment"], "resolved_sha": sha}
    return runner.JobSpec.from_dict(d)
