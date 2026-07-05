"""Provider-neutral HF-artifact reading + heartbeat provenance.

The control plane observes a remote worker purely through the artifacts it uploads to a private HF
dataset repo: rate-limited text/JSON readers for ``heartbeat.json`` / ``error_*.txt`` / console logs,
plus the provenance predicates that decide whether a heartbeat belongs to THIS attempt or a leftover
prior one. None of this is provider-specific — RunPod, Lambda, and Vast all poll the same artifact
shape — so it lives here in the shared kernel and no provider package imports another for it.

The worker-side bootstrap (``_instance_bootstrap``) cannot import flash, so it re-implements the
upload half; this module is the read half every poller shares.
"""

from __future__ import annotations

import json
import os
import time

from flash.providers._poll import _attempt_int


def make_hf_text_reader(hf_repo: str, path_in_repo: str, min_interval_s: float = 45.0):
    """Rate-limited reader for an HF artifact; returns None until it exists or on any error."""
    state = {"last": 0.0}

    def read(force: bool = False) -> str | None:
        if not hf_repo:
            return None
        if not force and time.time() - state["last"] < min_interval_s:
            return None
        state["last"] = time.time()
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(
                hf_repo,
                path_in_repo,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
                force_download=True,
            )
            with open(p) as f:
                return f.read()
        except Exception:
            return None

    return read


def make_hf_heartbeat_reader(hf_repo: str, prefix: str, min_interval_s: float = 30.0):
    """Rate-limited JSON reader for ``{prefix}/heartbeat.json`` on HF."""
    text_reader = make_hf_text_reader(hf_repo, f"{prefix}/heartbeat.json", min_interval_s)

    def read(force: bool = False) -> dict | None:
        raw = text_reader(force=force)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    return read


def heartbeat_reader_for(spec):
    """The HF heartbeat reader for a run's spec (None when the run has no hf_repo)."""
    hf_repo = spec.train.hf_repo
    return make_hf_heartbeat_reader(hf_repo, f"{spec.phase}/{spec.run_id}") if hf_repo else None


def error_artifact_name(phase: str, attempt) -> str:
    """Worker error-artifact filename for a phase+attempt (mirrors the worker's error_artifact_name)."""
    return f"error_{phase}_attempt{int(attempt or 0)}.txt"


def make_hf_failure_detail_reader(
    hf_repo: str,
    prefix: str,
    phase: str,
    min_interval_s: float = 45.0,
    attempt: int = 0,
):
    """Reader for worker-uploaded failure artifacts on HF (error/console txt); force-read after terminal failure."""
    # Attempt-scoped to match the worker's error_artifact_name(mode, attempt).
    err_name = error_artifact_name(phase, attempt)
    error_reader = make_hf_text_reader(hf_repo, f"{prefix}/{err_name}", min_interval_s)
    console_reader = make_hf_text_reader(
        hf_repo, f"{prefix}/console_{phase}.txt", min_interval_s
    )

    def read(force: bool = False) -> str | None:
        parts: list[str] = []
        error_text = error_reader(force=force)
        if error_text:
            parts.append(f"--- {err_name} ---\n{error_text[-4000:]}")
        console_text = console_reader(force=force)
        if console_text:
            parts.append(f"--- console_{phase}.txt ---\n{console_text[-4000:]}")
        return "\n".join(parts) if parts else None

    return read


def _heartbeat_is_prior_attempt(hb: dict, launch_ts: float | None, current_attempt) -> bool:
    """Positively attribute a heartbeat to a PRIOR (earlier) attempt: an explicit ``attempt`` differing
    from ``current_attempt`` (definitive provenance — a MATCH proves THIS launch, so a lagging worker-host
    clock cannot demote it), else a parseable ``ts`` predating this attempt's launch. Un-dateable (no
    attempt AND no usable ts) -> False: mere absence of proof is never treated as a leftover. Truthy
    ``launch_ts`` only (0.0 = unknown launch, uncomparable). Shared decision core of
    ``worker_flagged_retriable`` (honor the retriable flag iff NOT prior) and
    ``heartbeat_is_stale_prior_attempt`` (stale iff prior) — one edit site instead of two lockstep copies."""
    hb_attempt = _attempt_int(hb.get("attempt"))
    cur_attempt = _attempt_int(current_attempt)
    if hb_attempt is not None and cur_attempt is not None:
        return hb_attempt != cur_attempt
    if launch_ts:
        try:
            ts = float(hb.get("ts"))
        except (TypeError, ValueError):
            ts = None
        if ts is not None and ts < float(launch_ts):
            return True
    return False


def worker_flagged_retriable(
    heartbeat_reader, *, launch_ts: float | None = None, current_attempt: int | None = None
) -> bool:
    """True if the worker stamped ``retriable`` (a RetriableInfraError) in its last heartbeat — the
    structured worker<->poller contract that replaces failure-detail parsing: ``retriable`` means
    retry on a fresh worker. Forces a fresh read past the rate limit.

    ``launch_ts`` / ``current_attempt``, when supplied, gate the flag to THIS attempt. The seed
    heartbeat path is shared across retries, so a leftover ``retriable=True`` from attempt N-1 must
    NOT override attempt N's own (non-retriable) failure marker — otherwise a deterministic
    bootstrap/config error that fails BEFORE this attempt's worker emits any heartbeat would be
    reported job_preempted and burn GPUs on an endless retry instead of failing fast. Only positive
    prior-attempt evidence gates the flag: a ts that predates launch OR an explicit attempt mismatch.
    With NEITHER arg the flag is honored ungated (back-compat for callers that don't date heartbeats)."""
    if heartbeat_reader is None:
        return False
    hb = heartbeat_reader(force=True)
    if not isinstance(hb, dict):
        return False
    if not bool(hb.get("retriable")):
        return False
    if launch_ts is None and current_attempt is None:
        return True  # ungated: caller can't date the heartbeat -> preserve prior behavior
    # Honor the retriable flag unless the heartbeat provably belongs to a PRIOR attempt.
    return not _heartbeat_is_prior_attempt(hb, launch_ts, current_attempt)


def heartbeat_is_stale_prior_attempt(
    heartbeat_reader, *, launch_ts: float | None = None, current_attempt: int | None = None
) -> bool:
    """True ONLY when a heartbeat can be POSITIVELY attributed to a PRIOR (earlier) attempt — either it
    carries an explicit ``attempt`` that differs from ``current_attempt``, OR a parseable ``ts`` that
    predates THIS attempt's launch. Everything else returns False: no heartbeat, an empty/uninformative
    heartbeat (no ts AND no attempt — e.g. ``{}``), a heartbeat matching this attempt, or one that
    cannot be dated. The asymmetry is deliberate — we suppress a crash classification only on PROOF of
    a leftover, never on mere absence of proof (an un-dateable heartbeat is NOT evidence of a prior run
    and must not mask THIS attempt's deterministic crash).

    The seed heartbeat path AND the seed-scoped ``error_<phase>.txt`` crash artifact are BOTH shared
    across this seed's retries, so a prior attempt can leave either behind. When the latest heartbeat
    provably belongs to an earlier attempt, the co-located error file is presumed leftover too — so a
    dead-host poll on attempt N must NOT read that stale crash file as THIS attempt's DETERMINISTIC
    failure (which would fail-fast a genuine host LOSS instead of retrying it on a fresh host). Gating
    requires BOTH ``launch_ts`` and ``current_attempt``; without them a heartbeat cannot be dated, so
    it is never called stale (conservative — keep the caller's existing classification)."""
    if heartbeat_reader is None:
        return False
    hb = heartbeat_reader(force=True)
    if not isinstance(hb, dict) or launch_ts is None or current_attempt is None:
        return False
    return _heartbeat_is_prior_attempt(hb, launch_ts, current_attempt)
