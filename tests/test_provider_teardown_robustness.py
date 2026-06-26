"""Regression locks for the teardown/robustness hardening from the adversarial review — the
"won't randomly die / no billing leak" guarantees:

- is_not_found keys off the HTTP status CODE, not a bare "404" substring (a transient 5xx on a
  resource whose id contains "404" must NOT be read as "gone").
- Lambda terminate_instances is PER-ID isolated: one bad/stale id can't abort teardown of the rest.
"""

from __future__ import annotations

import io
import urllib.error


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x/y", code, "msg", {}, io.BytesIO(b""))


def _chained(code: int, text: str) -> Exception:
    e = RuntimeError(text)
    e.__cause__ = _http_error(code)
    return e


def test_is_not_found_keys_off_status_code_not_bare_404():
    from flash.providers._http import is_not_found

    # genuine 404 (chained HTTPError) -> gone
    assert is_not_found(_chained(404, "GET /core/virtual-machines/7 -> HTTP 404: Not Found")) is True
    # a 5xx on a VM whose id CONTAINS 404 -> NOT gone (the old bare-substring bug)
    assert is_not_found(_chained(500, "GET /core/virtual-machines/1404 failed: HTTP 500")) is False
    assert is_not_found(_chained(403, "GET /x/404 -> HTTP 403: Forbidden")) is False
    # no chained cause: only an unambiguous "HTTP 404" token counts, never a bare "404"
    assert is_not_found(RuntimeError("GET /x -> HTTP 404: Not Found")) is True
    assert is_not_found(RuntimeError("GET /core/virtual-machines/1404 failed after 5 attempts")) is False
    # a STATUS-like number that merely begins 404 must NOT match (trailing-\b rejects HTTP 4040/4041)
    assert is_not_found(RuntimeError("GET /x -> HTTP 4040: weird")) is False
    assert is_not_found(RuntimeError("GET /x -> HTTP 4041")) is False
    assert is_not_found(RuntimeError("GET /x -> HTTP 404")) is True  # trailing token at end-of-string


def test_is_conflict_keys_off_status_code_not_bare_409():
    from flash.providers._http import is_conflict

    # genuine 409 (chained HTTPError) -> conflict
    assert is_conflict(_chained(409, "DELETE /core/virtual-machines/7 -> HTTP 409: Conflict")) is True
    # a non-409 failure whose text merely contains "409"/"conflict" (e.g. a "4090" GPU name) -> NOT
    # a conflict (the old bare-substring bug would swallow it as success and leak a billing VM)
    assert is_conflict(_chained(500, "launch(RTX 4090) failed: HTTP 500 conflict")) is False
    assert is_conflict(_chained(403, "DELETE /x/4090 -> HTTP 403: Forbidden conflict")) is False
    # no chained cause: only an unambiguous "HTTP 409" token counts, never a bare "409"/"conflict"
    assert is_conflict(RuntimeError("DELETE /x -> HTTP 409: Conflict")) is True
    assert is_conflict(RuntimeError("launch(RTX 4090) failed: conflict")) is False
    # a STATUS-like number that merely begins 409 must NOT match (trailing-\b rejects HTTP 4090/4091)
    assert is_conflict(RuntimeError("DELETE /x -> HTTP 4090: weird")) is False
    assert is_conflict(RuntimeError("DELETE /x -> HTTP 4091")) is False
    assert is_conflict(RuntimeError("DELETE /x -> HTTP 409")) is True  # trailing token at end-of-string


def test_lambda_terminate_is_per_id_isolated(monkeypatch):
    """One bad id must NOT abort teardown of the others (the crash-backstop sweep passes many)."""
    from flash.providers.lambdalabs import api as la_api

    calls = []

    def fake_req(path, method="GET", body=None, retries=4, base_delay=2.0):
        iid = body["instance_ids"][0]
        calls.append(iid)
        if iid == "bad":
            raise la_api.LambdaApiError("instance bad does not exist")
        return {}

    monkeypatch.setattr(la_api, "request_with_retries", fake_req)
    deleted = la_api.terminate_instances(["i-1", "bad", "i-3"])
    assert deleted == ["i-1", "i-3"]  # the bad id is skipped, the rest still terminate
    assert calls == ["i-1", "bad", "i-3"]  # each tried independently (per-id POST)


# --- review-fix regressions (bounded label, provider rates, fail-fast user_data) ---
def test_label_bounded_and_sweep_matches_for_long_run_id():
    """A run id long enough to overflow the provider name cap is shortened DETERMINISTICALLY, and the
    same bounded prefix is used by launch AND sweep matching (so a live run can't be mis-swept)."""
    from flash.providers._instance import instance_label, run_label_prefix

    long_id = "flash-" + "x" * 200
    name = instance_label(long_id, 0, 0)
    assert len(name) <= 60
    # the bounded prefix the sweep matches on is a prefix of the launched name
    assert name.startswith(run_label_prefix(long_id))
    assert name == run_label_prefix(long_id) + "-s0-a0"
    # deterministic + collision-resistant: two distinct long ids -> distinct bounded prefixes
    assert run_label_prefix(long_id) != run_label_prefix("flash-" + "y" * 200)
    # short ids pass through unchanged (the common case)
    assert run_label_prefix("flash-1700-abcd") == "flash-1700-abcd"


def test_provider_cost_uses_provider_rate():
    """A provider-specific quote prices through the provider's pricing, not the RunPod nominal."""
    from flash.cost.facts import gpu_hourly_usd

    runpod = gpu_hourly_usd("RTX A6000")  # nominal RunPod snapshot
    lam = gpu_hourly_usd("RTX A6000", provider="lambda")  # Lambda static fallback (no key in tests)
    assert runpod == 0.49
    assert lam == 1.09  # Lambda A6000
    # unknown-on-provider class falls back to nominal (e.g. a RunPod-only consumer card on lambda)
    assert gpu_hourly_usd("RTX 4090", provider="lambda") == gpu_hourly_usd("RTX 4090")


def test_user_data_fails_fast_and_throttles_bootlog():
    """The cloud-init writes a retriable failure marker on docker failure and throttles the host
    boot-log (no 30s-forever loop that would blow HF's commit budget)."""
    from flash.providers._instance import build_user_data

    payload = {
        "hf_repo": "org/repo", "hf_prefix": "sft/x/seed0", "flash_arm": "lambda", "attempt": 0,
        "job_spec_json": "{}", "phase": "sft", "seed": 0, "max_wall_s": 60, "extra_pip": [],
        "env": {"HF_TOKEN": "t"},
    }
    s = build_user_data(payload, image="img:tag")
    assert "failmark.py" in s  # host writes the attempt-failure marker
    assert 'fail "worker image pull failed' in s  # fail fast when the image can't pull
    assert 'fail "worker container did not start' in s
    assert "sleep 120" in s  # boot-log throttled to 120s
    assert "sleep 30;" not in s  # NOT the old 30s-forever loop
    # A fast CLEAN exit (code 0 — an already-complete retry restores metrics + writes its ok-marker
    # in <5s) is the success signal itself: the host inspects the exit code and writes the retriable
    # failmark ONLY on a non-zero exit, so it never clobbers the worker's just-written ok-marker.
    assert "{{.State.ExitCode}}" in s
    assert '[ "$EXIT" = "0" ] || fail' in s
    assert "donecheck" not in s  # no host-side HF check (it would race the worker's marker upload)


def test_instance_realized_cost_is_wall_times_rate():
    """Lambda has no billing API; realized COGS = wall(launch->run_end) x flat $/hr."""
    from flash.providers.realized import realized_cost_for_remote

    remote = {"provider": "lambda", "instance_id": "i-9", "hourly_usd": 1.20, "started_ts": 1000.0}
    rc = realized_cost_for_remote(remote, start=999.0, end=4600.0)  # 3600s = 1h after launch
    assert rc is not None
    assert rc.provider == "lambda"
    assert rc.realized_usd == 1.20  # 1h x $1.20
    assert rc.wall_seconds == 3600.0
    assert rc.by_resource == {"gpu": 1.20}
    # no rate, or no auditable resource id -> unattributable (stays unreconciled, never books cost)
    assert realized_cost_for_remote({"provider": "lambda", "instance_id": "i-9"}, start=0, end=10) is None
    assert realized_cost_for_remote({"provider": "lambda", "hourly_usd": 1.0}, start=0, end=10) is None


def test_instance_realized_cost_uses_run_end_not_settle_padded_end():
    """The instance wall must use the TRUE run end (run_end ~ teardown), never the settle-padded
    billing-query ``end`` — otherwise reconciliation over-bills by the settle hour (up to 1h x rate)."""
    from flash.providers.realized import realized_cost_for_remote

    remote = {"provider": "lambda", "instance_id": "i-1", "hourly_usd": 2.0, "started_ts": 0.0}
    # end is padded +3600 for the RunPod billing query; run_end is the real teardown at 1800s (0.5h).
    rc = realized_cost_for_remote(remote, start=0.0, end=1800.0 + 3600.0, run_end=1800.0)
    assert rc is not None
    assert rc.wall_seconds == 1800.0  # 0.5h, NOT 1.5h
    assert rc.realized_usd == 1.0  # 0.5h x $2.0 (not $3.0)
