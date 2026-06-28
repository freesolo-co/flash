"""CUDA-OOM -> escalate the retry to a strictly LARGER GPU (CPU-only; no GPU/network).

A training CUDA OOM means the card was too small for THIS run's peak. Before this change it was a
``job_failed`` and failed fast (no walk). Now the worker stamps a structured ``oom`` heartbeat flag,
the poller maps it to ``failure="oom"``, and the runner retries on a strictly larger class (a
same-size walk would just OOM again) — failing terminally only once the largest available card OOMs.
These cover the pure pieces + the wiring; the end-to-end escalation needs a live-GPU run.
"""

from __future__ import annotations

import ast
import pathlib


def test_is_cuda_oom_matches_the_real_traceback_not_generic_errors():
    from flash.engine.worker.perf.lifecycle import is_cuda_oom

    # the actual prod crash (fla/Triton raises a plain RuntimeError, not torch's typed OOM)
    assert is_cuda_oom(RuntimeError("Triton Error [CUDA]: out of memory")) is True
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")) is True
    # detection also works off the traceback text when the top exception is a wrapper
    tb = "...\n  File fla/modules/l2norm.py ...\nRuntimeError: Triton Error [CUDA]: out of memory\n"
    assert is_cuda_oom(None, tb) is True
    # genuine NON-oom training/config errors must NOT escalate (they'd waste a bigger GPU)
    assert is_cuda_oom(ValueError("bad config")) is False
    assert is_cuda_oom(RuntimeError("loss became nan")) is False
    # an OOM with NO CUDA/Triton context is a host-RAM / other-library OOM — must NOT trigger GPU
    # escalation just because the substring "out of memory" appears.
    assert is_cuda_oom(RuntimeError("DataLoader worker (pid 1) killed: out of memory")) is False
    assert is_cuda_oom(MemoryError("Unable to allocate array: out of memory")) is False


def _card(gpu, vram, provider="runpod", price=1.0):
    import types

    return types.SimpleNamespace(gpu=gpu, vram_gb=vram, provider=provider, hourly_usd=price)


def test_oom_escalated_keeps_only_strictly_larger_cards():
    from flash.runner.lifecycle import _oom_escalated

    cands = [
        _card("A100 PCIe", 80),
        _card("A100 SXM", 80),
        _card("RTX Pro 6000", 96),
        _card("H200", 141),
        _card("B200", 180),
    ]
    # no prior OOM -> list unchanged
    assert [c.gpu for c in _oom_escalated(cands, 0)] == [c.gpu for c in cands]
    # after an 80GB OOM -> only the >80GB classes (NOT the other 80GB A100)
    assert {c.gpu for c in _oom_escalated(cands, 80)} == {"RTX Pro 6000", "H200", "B200"}
    # OOM'd the biggest -> nowhere larger to escalate to (caller fails terminally)
    assert _oom_escalated(cands, 180) == []


def test_worker_flagged_oom_reads_the_structured_flag():
    from flash.providers.runpod.jobs import worker_flagged_oom

    assert worker_flagged_oom(lambda force=False: {"oom": True}) is True
    assert worker_flagged_oom(lambda force=False: {"oom": False}) is False
    assert worker_flagged_oom(lambda force=False: {"retriable": True}) is False  # not an OOM
    assert worker_flagged_oom(lambda force=False: "not a dict") is False
    assert worker_flagged_oom(None) is False


def test_once_forced_reads_underlying_heartbeat_only_once():
    # A terminal path surfaces the heartbeat AND extracts retriable+oom from it; _once_forced makes
    # those share ONE underlying force-read (fewer HF round-trips, no cross-snapshot race).
    from flash.providers.runpod.jobs import (
        _once_forced,
        worker_flagged_oom,
        worker_flagged_retriable,
    )

    calls = {"n": 0}

    def underlying(force=False):
        calls["n"] += 1
        return {"oom": True, "retriable": False}

    reader = _once_forced(underlying)
    # three logical consumers (surface + 2 flags) -> still one underlying read
    reader(force=True)
    assert worker_flagged_oom(reader) is True
    assert worker_flagged_retriable(reader) is False
    assert calls["n"] == 1
    assert _once_forced(None) is None


def test_lambda_poller_escalates_oom_for_cross_provider_parity():
    # The Lambda poller must also map a worker OOM flag to failure="oom" so a Lambda candidate's OOM
    # escalates to a larger GPU instead of failing fast as job_failed (parity with RunPod's poll_job).
    import re

    src, _ = _src("flash.providers.lambdalabs.jobs", "")
    compact = re.sub(r"\s+", "", src)  # drop all whitespace (call sites wrap across lines)
    # the oom flag is read in BOTH Lambda worker-fail paths (marker + dead-host error_*.txt crash),
    # BOTH gate on handle.attempt (5Wy: a prior attempt's lingering {"oom": true} can't misclassify),
    # and BOTH read off a single shared _once_forced snapshot hb_once (8bS: two separate forced reads
    # could disagree, the first missing a not-yet-visible heartbeat the second would see).
    assert compact.count("worker_flagged_oom(hb_once,handle.attempt)") == 2
    assert "worker_flagged_oom(heartbeat_reader" not in compact  # no direct-reader read survives
    assert "worker_flagged_retriable(hb_once)" in compact  # retriable shares the same snapshot
    assert "worker_flagged_retriable(heartbeat_reader)" not in compact
    # marker path: oom wins over the retriable/job_failed split
    assert 'failure="oom" if oom else ("job_preempted" if retriable else "job_failed")' in src
    # dead-host crash path: oom takes precedence before the crash/preempt split
    assert '"job_failed" if worker_crashed else "job_preempted"' in src


def test_is_cuda_oom_ignores_cuda_only_in_traceback_file_paths():
    # 8bM: the CUDA-context signal must come from the exception MESSAGE, not stack frame file paths.
    # A host-RAM OOM whose traceback merely traverses torch/cuda/* must NOT be treated as a GPU OOM
    # (more VRAM can't fix it), even though "cuda" appears in a `File ".../torch/cuda/..."` frame.
    from flash.engine.worker.perf.lifecycle import is_cuda_oom

    host_oom_tb = (
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.11/site-packages/torch/cuda/memory.py", line 1, in alloc\n'
        "    return _do()\n"
        "MemoryError: Unable to allocate 8.00 GiB: out of memory\n"
    )
    assert is_cuda_oom(MemoryError("Unable to allocate 8.00 GiB: out of memory"), host_oom_tb) is False
    # a DataLoader host OOM whose stack also passes through torch/cuda frames — still NOT a CUDA OOM
    dl_tb = (
        "Traceback (most recent call last):\n"
        '  File "/x/torch/cuda/streams.py", line 9, in run\n'
        "    loss.backward()\n"
        "RuntimeError: DataLoader worker (pid 7) is killed by signal: out of memory\n"
    )
    assert is_cuda_oom(RuntimeError("DataLoader worker (pid 7) is killed by signal: out of memory"), dl_tb) is False
    # a REAL CUDA OOM whose root cause is on the (un-indented) exception line of a wrapper traceback
    # is still detected (the message, not a file path, supplies the context).
    real_tb = (
        "Traceback (most recent call last):\n"
        '  File "/x/fla/modules/l2norm.py", line 3, in bwd\n'
        "    k()\n"
        "RuntimeError: Triton Error [CUDA]: out of memory\n"
    )
    assert is_cuda_oom(RuntimeError("boom"), real_tb) is True


def test_worker_does_not_stamp_oom_for_a_retriable_infra_exception():
    # 81I: wait_for_gpu raises RetriableInfraError carrying the last CUDA readiness error, which can
    # be "CUDA out of memory". The pollers give oom precedence over retriable, so the worker must NOT
    # stamp oom when the exception is already retriable — else a host-readiness flake escalates onto a
    # larger, costlier card instead of retrying on a fresh same-size GPU.
    src, _ = _src("flash.engine.worker", "")
    assert "oom = is_cuda_oom(e, tb) and not retriable" in src


def test_detail_flags_cuda_oom_is_a_gated_fallback():
    # 81K/81L: when the structured heartbeat oom flag is absent (best-effort upload lost) OR the OOM
    # is in a subprocess the worker didn't classify (vLLM EngineCore child, root cause only in the
    # appended stdout tail), the poller falls back to the SAME CUDA predicate on the assembled detail.
    from flash.providers.runpod.jobs import detail_flags_cuda_oom

    # a CUDA OOM surfaced only in a worker/subprocess stdout tail is caught
    assert detail_flags_cuda_oom("... vLLM EngineCore died: CUDA out of memory. Tried to allocate") is True
    # a host/library OOM in the text does NOT escalate (no CUDA/Triton context)
    assert detail_flags_cuda_oom("DataLoader worker killed: out of memory") is False
    # a RetriableInfraError that merely names a CUDA OOM must NOT escalate (retriable wins) — the
    # marker in the text suppresses the fallback so it retries on a fresh same-size GPU.
    assert detail_flags_cuda_oom("RETRIABLE_INFRA_GPU: cuda readiness: CUDA out of memory") is False
    assert detail_flags_cuda_oom(None) is False
    assert detail_flags_cuda_oom("") is False


def test_oom_fallback_scans_only_attempt_fresh_text_not_the_shared_artifact():
    # AVE/AVG/AVJ: the OOM fallback must scan only ATTEMPT-FRESH text — this attempt's live job output
    # / its attempt-scoped marker — NEVER the shared per-seed error_<phase>.txt artifact (a prior
    # attempt's lingering traceback would wrongly escalate). It is gated on ``not oom`` alone (NOT
    # ``not retriable``) so a STALE retriable heartbeat can't block a real OOM; retriable-wins is
    # preserved inside detail_flags_cuda_oom (it skips retriable-infra text).
    import re

    runpod = re.sub(
        r"\s+", " ", pathlib.Path(__import__("flash.providers.runpod.jobs", fromlist=["x"]).__file__).read_text()
    )
    lam = re.sub(
        r"\s+", " ", pathlib.Path(__import__("flash.providers.lambdalabs.jobs", fromlist=["x"]).__file__).read_text()
    )
    # RunPod: decode-error path scans the live handler error; TERMINAL_FAIL path scans the captured
    # live_detail (st error + worker stdout tail), both BEFORE the HF artifact is appended.
    assert "if not oom: oom = detail_flags_cuda_oom(str(e))" in runpod
    assert "live_detail = detail" in runpod
    assert "if not oom: oom = detail_flags_cuda_oom(live_detail)" in runpod
    # the fallback never scans the shared, artifact-appended `detail` and never gates on `not retriable`
    assert "detail_flags_cuda_oom(detail)" not in runpod
    assert "not oom and not retriable" not in runpod
    # Lambda: the fallback runs on the ATTEMPT-SCOPED marker's error text, and NOT on the shared
    # error_<phase>.txt artifact (`err`) of the dead-host path.
    assert 'if not oom and marker: oom = detail_flags_cuda_oom(str(marker.get("error") or ""))' in lam
    assert "detail_flags_cuda_oom(err)" not in lam


def test_worker_flagged_oom_ignores_a_prior_attempts_stale_flag():
    # The shared-prefix heartbeat outlives an attempt: after an OOM escalation, attempt 0's
    # {"oom": true} lingers until the bigger card's worker overwrites it. If that escalated endpoint
    # dies before writing its own heartbeat, the forced read must NOT re-classify the new (unrelated)
    # failure as an OOM and escalate AGAIN — it gates on the heartbeat's stamped attempt.
    from flash.providers.runpod.jobs import worker_flagged_oom

    stale = lambda force=False: {"oom": True, "attempt": "0"}  # noqa: E731
    fresh = lambda force=False: {"oom": True, "attempt": "1"}  # noqa: E731
    # polling attempt 1: attempt-0's lingering OOM flag is NOT trusted...
    assert worker_flagged_oom(stale, 1) is False
    # ...but attempt-1's own OOM flag still escalates.
    assert worker_flagged_oom(fresh, 1) is True
    # attempt stamped as int (not the worker's str env) also matches.
    assert worker_flagged_oom(lambda force=False: {"oom": True, "attempt": 1}, 1) is True
    # a missing/blank attempt with a known current_attempt can't be dated -> not trusted.
    assert worker_flagged_oom(lambda force=False: {"oom": True}, 1) is False
    assert worker_flagged_oom(lambda force=False: {"oom": True, "attempt": ""}, 1) is False
    # current_attempt=None (reattach poll, no expected attempt) keeps the historical trust-the-flag.
    assert worker_flagged_oom(stale, None) is True
    assert worker_flagged_oom(stale) is True


def _src(modfile_attr, name):
    import importlib

    mod = importlib.import_module(modfile_attr)
    return pathlib.Path(mod.__file__).read_text(), name


def test_worker_stamps_oom_flag_in_its_failure_heartbeat():
    # The worker classifies OOM off its LIVE exception and stamps it alongside `retriable`, so the
    # poller never has to parse the failure detail string.
    src, _ = _src("flash.engine.worker", "")
    assert "oom = is_cuda_oom(e, tb)" in src
    assert '{"retriable": retriable, "oom": oom}' in src


def test_poller_maps_oom_flag_to_the_oom_failure_category():
    src, _ = _src("flash.providers.runpod.jobs", "")
    # the oom flag wins over retriable/job_failed in BOTH worker-fail paths
    assert src.count('failure="oom" if oom else ("job_preempted" if retriable else "job_failed")') == 2
    assert "def worker_flagged_oom" in src


def test_runner_classifies_oom_as_retriable_and_escalates_vram():
    src = pathlib.Path(
        __import__("flash.runner.lifecycle", fromlist=["x"]).__file__
    ).read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_submit_seed_supervised"
    )
    body = ast.get_source_segment(src, fn)
    # OOM is a retry category (NOT fail-fast) AND it grows the escalation floor
    assert 'oom_shaped = res.failure == "oom"' in body
    assert "infra_shaped or oom_shaped" in body  # OOM joins the retry-shaped categories
    assert "if not retry_shaped:" in body  # the fail-fast break now honors oom too
    assert "oom_vram_floor = max(oom_vram_floor, int(chosen.vram_gb))" in body
    # the next attempt restricts to strictly-larger cards, and an OOM does NOT escape the provider
    assert "_oom_escalated(alloc.candidates, oom_vram_floor)" in body
    assert "if not oom_shaped:" in body  # guards failed_providers.add
    # an OOM on the largest class terminates immediately (no extra "retrying on a larger GPU" spin)
    assert "oom_no_larger" in body
    assert "retry_shaped = (infra_shaped or oom_shaped) and not oom_no_larger" in body


def test_oom_escalation_floor_survives_a_control_plane_restart():
    # 5W1: the GPU-escalation floor is in-process state. A control-plane restart that reattaches and
    # resumes the in-flight seed must NOT lose it, or recovery re-rolls the same too-small card and
    # OOMs again. The floor (+ in-flight card VRAM) is persisted into the run handle and threaded back
    # into the resumed seed loop.
    life = pathlib.Path(
        __import__("flash.runner.lifecycle", fromlist=["x"]).__file__
    ).read_text()
    deploy = pathlib.Path(
        __import__("flash.runner.deploy", fromlist=["x"]).__file__
    ).read_text()

    # the floor + the in-flight card's VRAM are persisted into the handle on every submit (on_handle)
    assert '"oom_vram_floor": int(oom_vram_floor)' in life
    assert '"allocated_vram_gb": current_gpu.get("vram_gb")' in life
    assert 'current_gpu["vram_gb"] = int(chosen.vram_gb)' in life
    # the seed supervisor accepts a starting floor and seeds oom_vram_floor from it (not a hard 0)
    assert "oom_vram_floor_start: int = 0" in life
    assert "oom_vram_floor = int(oom_vram_floor_start)" in life
    # _run_training (single fixed-seed adapter) forwards the resumed floor to the supervised submit
    assert "resume_oom_vram_floor: int = 0" in life
    assert "oom_vram_floor_start=resume_oom_vram_floor" in life
    # recovery reads the persisted floor, bumps it by the reattached card's VRAM on an OOM, and
    # threads it into the resumed training submit
    assert 'persisted_oom_floor = int(remote.pop("oom_vram_floor", 0) or 0)' in deploy
    assert 'allocated_vram_gb = int(remote.pop("allocated_vram_gb", 0) or 0)' in deploy
    assert "resumed_floor = max(resumed_floor, allocated_vram_gb)" in deploy
    assert "resume_oom_vram_floor=resumed_floor" in deploy


def test_runpod_reattach_poll_gates_oom_on_the_persisted_attempt():
    # -CF/-CH/-t5: the reattach poll (RunpodProvider.poll) must ALSO gate worker_flagged_oom, else a
    # control-plane restart trusts a PRIOR attempt's lingering OOM heartbeat and needlessly escalates
    # (and, via 5W1, raises resume_oom_vram_floor). The attempt is persisted into the run handle by
    # on_handle and read back here (same field path as on_last_gpu), then forwarded as current_attempt.
    life = pathlib.Path(
        __import__("flash.runner.lifecycle", fromlist=["x"]).__file__
    ).read_text()
    rp = pathlib.Path(
        __import__("flash.providers.runpod", fromlist=["x"]).__file__
    ).read_text()
    assert '"attempt": int(attempt)' in life  # persisted on every submit
    assert 'current_attempt = _attempt_int(handle.to_dict().get("attempt"))' in rp
    assert "current_attempt=current_attempt" in rp


def test_instance_worker_env_stamps_attempt_for_heartbeat_gating():
    # BRZ: the shared RunPod env builder does NOT set ATTEMPT (RunPod's submit adds it), so an instance
    # (Lambda) worker would heartbeat attempt="" and the poller's attempt-gated worker_flagged_oom
    # would then REJECT a real CUDA OOM. The instance bootstrap must stamp ATTEMPT from the payload.
    boot = pathlib.Path(
        __import__("flash.providers._instance_bootstrap", fromlist=["x"]).__file__
    ).read_text()
    assert 'env["ATTEMPT"] = str(int(payload.get("attempt", 0)))' in boot


def test_instance_marker_carries_console_classified_oom():
    # BRc: a CUDA OOM can surface ONLY in the training subprocess console (a vLLM EngineCore child),
    # which the worker's own exception classifier misses. The instance bootstrap classifies THIS
    # attempt's LOCAL console (attempt-safe) into the attempt-scoped marker's oom field. (Scanning the
    # SHARED console_<phase>.txt from the poller would reintroduce the cross-attempt staleness AVE
    # flagged, so it's done at the source instead.)
    boot = pathlib.Path(
        __import__("flash.providers._instance_bootstrap", fromlist=["x"]).__file__
    ).read_text()
    assert "def _console_flags_cuda_oom" in boot
    assert '"oom": bool(oom)' in boot  # marker carries the structured flag
    assert "oom = _console_flags_cuda_oom(payload.get(\"phase\"))" in boot
    assert "if not retriable:" in boot  # retriable wins (don't escalate an infra crash)
    assert "write_attempt_marker(payload, ok, error, retriable=retriable, oom=oom)" in boot


def test_oom_recovery_respects_the_user_retry_budget():
    # BRa: a control-plane restart that reattaches to an in-flight OOM must NOT re-grant a fresh
    # larger-GPU escalation budget past max_retries. The consumed attempt count rides into the resumed
    # in-flight seed: an entry guard fails fast when the budget is already spent (notably
    # max_retries=0), and the loop's budget check adds it so any remaining escalations are capped.
    life = pathlib.Path(
        __import__("flash.runner.lifecycle", fromlist=["x"]).__file__
    ).read_text()
    deploy = pathlib.Path(
        __import__("flash.runner.deploy", fromlist=["x"]).__file__
    ).read_text()
    # the supervisor takes the consumed-attempt count and enforces it
    assert "resume_oom_attempts: int = 0" in life
    assert "resume_oom_attempts > max_retries" in life  # entry guard (fail-fast when exhausted)
    assert "budget_spent = walk_attempt + (resume_oom_attempts if oom_shaped else 0)" in life
    assert "budget_spent < retry_budget" in life
    assert "if budget_spent >= retry_budget and not first_cache_drop:" in life
    # infra retries are unaffected (the offset only applies when oom_shaped)
    # recovery computes the consumed count from the persisted handle attempt and threads it in
    assert 'persisted_attempt = int(remote.get("attempt", 0) or 0)' in deploy
    assert "resumed_oom_attempts = persisted_attempt + 1" in deploy
    assert "resume_oom_attempts=resumed_oom_attempts" in deploy
