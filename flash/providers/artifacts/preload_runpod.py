"""Warm and tear down account-scoped RunPod weight-cache volumes with short-lived Pods."""

from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from flash.providers._lifecycle.instances.instance import build_payload
from flash.providers._lifecycle.instances.poll import preload_instance_run_id
from flash.providers.artifacts.hf import make_hf_text_reader
from flash.providers.core.base import UnreconciledCreateError
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.client import auth as runpod_auth
from flash.providers.runpod.client import pods as runpod_pods
from flash.providers.runpod.execution import resources
from flash.providers.runpod.execution.hf_intent import (
    INTENT_LEASE_S,
    HfRunpodIntentStore,
    intent_lock,
    intent_path,
    new_intent_owner,
)
from flash.providers.runpod.execution.identity import RunpodCreateAbsent, RunpodPodHandle
from flash.providers.runpod.execution.pods import (
    launch_payload_pod,
    resolve_pending_handle,
    terminate_handle,
)

_HF_HOME = "/runpod-volume/hf-cache"
_PRELOAD_GPU = "RTX 4090"
_PRELOAD_TIMEOUT_S = 5400
_RESULT_POLL_MIN_S = 5.0
_DEAD_POD_POLLS = 2


def _preload():
    from flash.providers.artifacts import weight_cache

    return weight_cache


def catalog_model_ids() -> list[str]:
    """Return cache-fitting catalog models largest first."""
    from flash.core.catalog import MODELS
    from flash.runner.accounting.weight_cache import _fits_weight_cache

    fitting = [(model_id, info) for model_id, info in MODELS.items() if _fits_weight_cache(info)]
    fitting.sort(key=lambda pair: (-(pair[1].params_b or 0.0), pair[0]))
    return [model_id for model_id, _info in fitting]


def _preload_spec(gpu: str, run_id: str, timeout_s: int):
    from flash.core.spec import JobSpec
    from flash.runner.accounting.weight_cache import (
        WEIGHT_CACHE_VOLUME_GB,
        WEIGHT_CACHE_VOLUME_NAME,
    )

    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "run_id": run_id,
            "train": {
                "hf_repo": _preload()._preload_status_repo(),
                "credit_assignment": "per_episode",
            },
            "gpu": {
                "type": gpu,
                "max_wall_seconds": max(60, int(timeout_s)),
                "network_volume": WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": WEIGHT_CACHE_VOLUME_GB,
            },
        }
    )


def _preload_payload(spec, models: list[str], token: str | None, deadline_at: float) -> str:
    payload = build_payload(
        spec,
        spec.seed,
        0,
        arm="runpod",
        mode="preload",
        models=models,
        deadline_at=deadline_at,
        preserve_runpod_volume=True,
    )
    env = payload.setdefault("env", {})
    env["FLASH_WEIGHT_CACHE_DIR"] = f"{_HF_HOME}/hub"
    if token:
        env["HF_TOKEN"] = token
    else:
        env.pop("HF_TOKEN", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _read_json(reader) -> dict | None:
    text = reader(force=True)
    if not text:
        return None
    value = json.loads(text)
    if type(value) is not dict:
        raise RuntimeError("preload completion artifact is not a JSON object")
    return value


def _validate_preload_result(result: dict, models: list[str]) -> dict:
    """Require terminal evidence for every requested model before reporting a completed warm."""
    requested = set(models)
    preloaded = result.get("preloaded")
    already = result.get("already_cached")
    failed = result.get("failed")
    resolved = result.get("resolved_snapshots")
    if result.get("error"):
        if preloaded == [] and already == [] and failed == {}:
            return result
        raise RuntimeError("preload completion error evidence is malformed")
    if (
        type(preloaded) is not list
        or type(already) is not list
        or type(failed) is not dict
        or type(resolved) is not dict
        or any(type(item) is not str or not item for item in preloaded + already)
        or any(type(key) is not str or not key for key in failed)
        or any(
            type(key) is not str or type(value) is not str or not value
            for key, value in resolved.items()
        )
    ):
        raise RuntimeError("preload completion evidence is malformed")
    successful = set(preloaded) | set(already)
    failed_models = set(failed)
    if set(preloaded) & set(already) or successful & failed_models:
        raise RuntimeError("preload completion evidence contains conflicting model outcomes")
    if successful | failed_models != requested or set(resolved) != successful:
        raise RuntimeError("preload completion evidence does not cover the requested models")
    if any(os.path.isabs(path) or ".." in path.split("/") for path in resolved.values()):
        raise RuntimeError("preload completion snapshot evidence is not mount-relative")
    return result


def _poll_preload(
    handle,
    spec,
    models: list[str],
    timeout_s: int,
    poll_interval_s: float,
    *,
    renew_lease=None,
) -> dict:
    deadline_at = min(handle.started_ts + timeout_s, spec.gpu.max_wall_seconds + handle.started_ts)
    prefix = f"{spec.phase}/{spec.run_id}"
    interval = min(max(_RESULT_POLL_MIN_S, poll_interval_s), INTENT_LEASE_S / 3.0)
    result_reader = make_hf_text_reader(
        spec.train.hf_repo,
        f"{prefix}/preload_result.json",
        min_interval_s=interval,
    )
    attempt_reader = make_hf_text_reader(
        spec.train.hf_repo,
        f"{prefix}/runpod_attempt0.json",
        min_interval_s=interval,
    )
    missing_polls = 0
    successful_marker_seen = False
    next_renewal = time.time() + INTENT_LEASE_S / 3.0
    while time.time() < deadline_at:
        if renew_lease is not None and time.time() >= next_renewal:
            renew_lease()
            next_renewal = time.time() + INTENT_LEASE_S / 3.0
        result = _read_json(result_reader)
        if result is not None:
            return _validate_preload_result(result, models)
        attempt = _read_json(attempt_reader)
        if attempt is not None:
            if attempt.get("ok") is True:
                successful_marker_seen = True
            if attempt.get("ok") is False:
                raise RuntimeError(
                    f"preload Pod failed early: {attempt.get('error') or 'see worker artifacts'}"
                )
        pod = runpod_pods.get_pod_for_fingerprint(
            handle.pod_id,
            handle.key_fingerprint,
            deadline_at=deadline_at,
        )
        if pod is None or pod.desired_status in {"DEAD", "EXITED", "FAILED", "TERMINATED"}:
            missing_polls += 1
            if missing_polls >= _DEAD_POD_POLLS:
                result = _read_json(result_reader)
                if result is not None:
                    return _validate_preload_result(result, models)
                detail = " after a successful attempt marker" if successful_marker_seen else ""
                raise RuntimeError(
                    f"preload Pod terminated without validated completion evidence{detail}"
                )
        else:
            missing_polls = 0
        time.sleep(interval)
    if successful_marker_seen:
        raise RuntimeError("preload Pod reached its deadline without validated completion evidence")
    raise TimeoutError(f"preload Pod did not finish within {timeout_s}s")


def _target_result(account_index: int, dc_id: str, status: str, **fields) -> dict:
    return {"account": f"acct{account_index}", "datacenter": dc_id, "status": status, **fields}


def _intent_store(fingerprint: str, dc_id: str, token: str | None, owner: str):
    from huggingface_hub import HfApi

    repo = _preload()._preload_status_repo()
    identity = f"{fingerprint}:{dc_id}"
    return HfRunpodIntentStore(
        HfApi(token=token),
        repo,
        intent_path("preload", identity),
        token,
        "preload",
        identity,
        owner,
    )


def _cleanup_claimed_intent(store: HfRunpodIntentStore, record: dict, timeout_s: int) -> None:
    handle = RunpodPodHandle.from_dict(record["handle"])
    spec = _preload_spec(handle.gpu, record["run_id"], timeout_s)
    try:
        resolved = resolve_pending_handle(
            handle,
            spec,
            record["seed"],
            deadline_at=time.time() + 120.0,
        )
    except RunpodCreateAbsent:
        resolved = handle
    else:
        if resolved.to_dict() != handle.to_dict():
            store.publish_active(record["run_id"], record["seed"], resolved.to_dict())
    store.renew()
    terminate_handle(resolved, deadline_at=time.time() + 120.0)
    store.clear()


def _recover_target_intent(store: HfRunpodIntentStore, timeout_s: int) -> None:
    record = store.claim_expired()
    if record is not None:
        _cleanup_claimed_intent(store, record, timeout_s)


def _cleanup_owned_intent(store: HfRunpodIntentStore, timeout_s: int) -> None:
    if store.load() is None:
        return
    record = store.renew()
    _cleanup_claimed_intent(store, record, timeout_s)


def _classify_result(account_index: int, dc_id: str, result: dict) -> dict:
    if result.get("error"):
        return _target_result(account_index, dc_id, "error", error=result["error"], result=result)
    if result.get("failed"):
        return _target_result(account_index, dc_id, "partial", result=result)
    return _target_result(account_index, dc_id, "ok", result=result)


def _preload_one_dc(
    account_index: int,
    fingerprint: str,
    dc_id: str,
    models: list[str],
    token: str | None,
    gpu: str,
    timeout_s: int,
    poll_interval_s: float,
) -> dict:
    from flash.runner.accounting.weight_cache import (
        WEIGHT_CACHE_VOLUME_GB,
        WEIGHT_CACHE_VOLUME_NAME,
    )

    deadline_at = time.time() + max(60, timeout_s)
    suffix = f"a{account_index}-{uuid.uuid4().hex[:6]}"
    run_id = preload_instance_run_id("runpod", dc_id, int(deadline_at), suffix)
    spec = _preload_spec(gpu, run_id, timeout_s)
    payload = _preload_payload(spec, models, token, deadline_at)
    owner = new_intent_owner("preload")
    store = _intent_store(fingerprint, dc_id, token, owner)
    lock = intent_lock(store.repo, store.path)
    with lock:
        try:
            _recover_target_intent(store, timeout_s)
            volume = resources.ensure_account_volume(
                fingerprint,
                base=WEIGHT_CACHE_VOLUME_NAME,
                data_center_id=dc_id,
                size_gb=WEIGHT_CACHE_VOLUME_GB,
                deadline_at=deadline_at,
            )
            handle = launch_payload_pod(
                spec,
                spec.seed,
                serialized_payload=payload,
                fingerprint=fingerprint,
                data_center_id=dc_id,
                network_volume_id=volume.id,
                on_handle=lambda value: store.publish_active(spec.run_id, spec.seed, value),
                cleanup_guard=store.renew,
                deadline_at=deadline_at,
            )
            _preload().logger.info(
                "preload acct%d/%s: Pod %s launched for %d models",
                account_index,
                dc_id,
                handle.pod_id,
                len(models),
            )
            result = _poll_preload(
                handle,
                spec,
                models,
                timeout_s,
                poll_interval_s,
                renew_lease=store.renew,
            )
            outcome = _classify_result(account_index, dc_id, result)
        except runpod_pods.RunpodCapacityError as exc:
            error = str(exc)
            _preload().logger.warning(
                "preload acct%d/%s NO CAPACITY for %s: %s", account_index, dc_id, gpu, error
            )
            outcome = _target_result(account_index, dc_id, "no_capacity", gpu=gpu, error=error)
        except (runpod_pods.RunpodMutationAmbiguous, UnreconciledCreateError) as exc:
            outcome = _target_result(account_index, dc_id, "error", error=str(exc))
        except TimeoutError as exc:
            outcome = _target_result(account_index, dc_id, "timeout", error=str(exc))
        except Exception as exc:
            _preload().logger.warning("preload acct%d/%s FAILED: %s", account_index, dc_id, exc)
            outcome = _target_result(account_index, dc_id, "error", error=str(exc))
        try:
            _cleanup_owned_intent(store, timeout_s)
        except Exception as exc:
            _preload().logger.warning(
                "preload acct%d/%s cleanup unconfirmed: %s", account_index, dc_id, exc
            )
            outcome = _target_result(
                account_index, dc_id, "error", error=f"cleanup unconfirmed: {exc}"
            )
        return outcome


def _account_storage_targets() -> list[tuple[int, str, str]]:
    """Return every configured account/datacenter cache target after a complete catalog read."""
    targets = []
    failures = []
    seen_fingerprints = set()
    for account_index, key in enumerate(runpod_auth.ordered_keys()):
        fingerprint = runpod_api.key_fingerprint(key)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        try:
            datacenters = resources.weight_cache_datacenters(
                fingerprint, deadline_at=time.time() + 120.0
            )
        except Exception as exc:
            failures.append(f"acct{account_index}: {exc}")
            continue
        targets.extend((account_index, fingerprint, dc_id) for dc_id in datacenters)
    if failures:
        raise runpod_api.RunpodApiError(
            "could not completely discover account storage datacenters: " + "; ".join(failures)
        )
    return sorted(targets, key=lambda target: (target[0], target[2]))


def _account_storage_datacenters() -> list[str]:
    return sorted({dc_id for _account, _fingerprint, dc_id in _account_storage_targets()})


def warm_weight_cache(
    models: list[str] | None = None,
    datacenters: list[str] | None = None,
    gpu: str = _PRELOAD_GPU,
    timeout_s: int = _PRELOAD_TIMEOUT_S,
    max_workers: int = 4,
    poll_interval_s: float = 10.0,
    token: str | None = None,
) -> list[dict]:
    """Warm every selected account/datacenter target with one exact short-lived Pod."""
    models = models or catalog_model_ids()
    targets = _preload()._account_storage_targets()
    available = {dc_id for _account, _fingerprint, dc_id in targets}
    selected = set(datacenters) if datacenters is not None else available
    invalid = sorted(selected - available)
    if invalid:
        raise ValueError(f"RunPod storage datacenter id(s) unavailable: {', '.join(invalid)}")
    selected_targets = [target for target in targets if target[2] in selected]
    token = token or os.environ.get("HF_TOKEN")
    _preload()._ensure_status_repo(token)
    _preload().logger.info(
        "warming %d account/datacenter target(s) with %d model(s)",
        len(selected_targets),
        len(models),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _preload()._preload_one_dc,
                account_index,
                fingerprint,
                dc_id,
                models,
                token,
                gpu,
                timeout_s,
                poll_interval_s,
            )
            for account_index, fingerprint, dc_id in selected_targets
        ]
        results = [future.result() for future in as_completed(futures)]
    for result in results:
        if result.get("status") != "partial":
            continue
        for model_id, detail in sorted(((result.get("result") or {}).get("failed") or {}).items()):
            _preload().logger.warning(
                "preload %s: %s FAILED: %s", result["datacenter"], model_id, detail
            )
    return results


def teardown_weight_cache(datacenters: list[str] | None = None) -> list[str]:
    """Delete only the selected managed cache volumes across all configured accounts."""
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    if datacenters is not None and not datacenters:
        return []
    keys = runpod_auth.ordered_keys()
    if not keys:
        return []
    selected = set(datacenters) if datacenters is not None else set(_account_storage_datacenters())
    deleted = []
    multi = len(keys) > 1
    for index, key in enumerate(keys):
        fingerprint = runpod_api.key_fingerprint(key)
        deadline_at = time.time() + 120.0
        try:
            volumes = runpod_pods.list_network_volumes_for_fingerprint(
                fingerprint, deadline_at=deadline_at
            )
        except Exception as exc:
            _preload().logger.warning(
                "teardown: RunPod account %d volume listing failed: %s", index, exc
            )
            continue
        targets = [
            volume
            for volume in volumes
            if volume.data_center_id in selected
            and volume.name
            == resources.weight_cache_volume_name(WEIGHT_CACHE_VOLUME_NAME, volume.data_center_id)
        ]
        for volume in targets:
            try:
                runpod_pods.delete_network_volume_for_fingerprint(
                    fingerprint, volume.id, deadline_at=deadline_at
                )
            except Exception as exc:
                _preload().logger.warning(
                    "teardown: cache volume %s deletion failed: %s", volume.name, exc
                )
                continue
            deleted.append(f"acct{index}:{volume.name}" if multi else volume.name)
    return deleted


def teardown_lambda_filesystems(name: str | None = None) -> list[str]:
    """Delete Lambda weight-cache filesystems without changing the Lambda path."""
    from flash.providers.lambda_.client import api as lambda_api
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    target = name or WEIGHT_CACHE_VOLUME_NAME
    deleted = []
    try:
        filesystems = lambda_api.list_filesystems()
    except Exception as exc:
        _preload().logger.warning("teardown: lambda list_filesystems failed: %s", exc)
        return deleted
    for filesystem in filesystems:
        if (
            filesystem.get("name") == target
            and filesystem.get("id")
            and lambda_api.delete_filesystem(filesystem["id"])
        ):
            region = (filesystem.get("region") or {}).get("name") or "?"
            deleted.append(f"lambda:{region}/{target}")
    return deleted
