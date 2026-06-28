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


def test_is_cuda_oom_is_structured_not_string_based(monkeypatch):
    import torch

    from flash.engine.worker.perf import lifecycle as lc

    # 1) torch's TYPED OutOfMemoryError -> OOM, regardless of message text.
    assert lc.is_cuda_oom(torch.cuda.OutOfMemoryError("anything")) is True

    # 2) NO string matching: a RuntimeError whose message literally SAYS "out of memory" is NOT
    #    classified by text — with no allocator-counter advance it's False. (A driver/kernel-launch OOM
    #    behind such a RuntimeError is caught later by the poller's text fallback, not here.) Host-RAM
    #    OOMs (DataLoader kill, MemoryError) are likewise NOT GPU OOMs and stay False.
    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 0)
    assert lc.is_cuda_oom(RuntimeError("Triton Error [CUDA]: out of memory")) is False
    assert lc.is_cuda_oom(RuntimeError("DataLoader worker (pid 1) killed: out of memory")) is False
    assert lc.is_cuda_oom(MemoryError("Unable to allocate array: out of memory")) is False
    assert lc.is_cuda_oom(ValueError("bad config")) is False

    # 3) torch's num_ooms allocator counter advancing -> OOM (the structured "error code" signal).
    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 3)
    assert lc.is_cuda_oom(RuntimeError("whatever"), oom_count_before=0) is True
    assert lc.is_cuda_oom(RuntimeError("whatever"), oom_count_before=3) is False  # no advance
    # ...but a host MemoryError stays False even with the counter pinned >0 (short-circuited).
    assert lc.is_cuda_oom(MemoryError("host ram")) is False


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
    # could disagree, the first missing a not-yet-visible heartbeat the second would see). Match with a
    # regex that tolerates positional OR keyword (current_attempt=) form so a harmless refactor passes.
    oom_calls = re.findall(
        r"worker_flagged_oom\(hb_once,(?:current_attempt=)?handle\.attempt\)", compact
    )
    assert len(oom_calls) == 2
    assert "worker_flagged_oom(heartbeat_reader" not in compact  # no direct-reader read survives
    assert re.search(r"worker_flagged_retriable\(hb_once[),]", compact)  # retriable shares the snapshot
    assert "worker_flagged_retriable(heartbeat_reader)" not in compact
    # marker path: oom wins over the retriable/job_failed split
    assert 'failure="oom" if oom else ("job_preempted" if retriable else "job_failed")' in src
    # dead-host crash path: oom takes precedence before the crash/preempt split
    assert '"job_failed" if worker_crashed else "job_preempted"' in src


def test_text_flags_cuda_oom_ignores_cuda_only_in_traceback_file_paths():
    # 8bM: the control-plane text scanner's CUDA-context signal must come from the exception-type
    # (un-indented) lines, not stack frame file paths. A host-RAM OOM whose traceback merely traverses
    # torch/cuda/* must NOT be treated as a GPU OOM (more VRAM can't fix it), even though "cuda"
    # appears in a `File ".../torch/cuda/..."` frame. (The worker path is structural; this is the
    # control-plane fallback that DOES read text — so the disambiguation lives here now.)
    from flash.engine.worker.perf.lifecycle import text_flags_cuda_oom

    host_oom_tb = (
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.11/site-packages/torch/cuda/memory.py", line 1, in alloc\n'
        "    return _do()\n"
        "MemoryError: Unable to allocate 8.00 GiB: out of memory\n"
    )
    assert text_flags_cuda_oom(host_oom_tb) is False
    # a DataLoader host OOM whose stack also passes through torch/cuda frames — still NOT a CUDA OOM
    dl_tb = (
        "Traceback (most recent call last):\n"
        '  File "/x/torch/cuda/streams.py", line 9, in run\n'
        "    loss.backward()\n"
        "RuntimeError: DataLoader worker (pid 7) is killed by signal: out of memory\n"
    )
    assert text_flags_cuda_oom(dl_tb) is False
    # a REAL CUDA OOM whose root cause is on the (un-indented) exception line of a wrapper traceback
    # is still detected (the message line, not a file path, supplies the context).
    real_tb = (
        "Traceback (most recent call last):\n"
        '  File "/x/fla/modules/l2norm.py", line 3, in bwd\n'
        "    k()\n"
        "RuntimeError: Triton Error [CUDA]: out of memory\n"
    )
    assert text_flags_cuda_oom(real_tb) is True


def test_worker_does_not_stamp_oom_for_a_retriable_infra_exception():
    # 81I: wait_for_gpu raises RetriableInfraError carrying the last CUDA readiness error, which can
    # be "CUDA out of memory". The pollers give oom precedence over retriable, so the worker must NOT
    # stamp oom when the exception is already retriable — else a host-readiness flake escalates onto a
    # larger, costlier card instead of retrying on a fresh same-size GPU.
    src, _ = _src("flash.engine.worker", "")
    assert "oom = is_cuda_oom(e) and not retriable" in src


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


def test_detail_flags_cuda_oom_scans_only_the_terminal_exception_block():
    # 5Wq: is_cuda_oom ANDs an OOM marker with CUDA/Triton context across whatever text it's given.
    # Feeding the WHOLE tail lets the two come from UNRELATED lines (an early 'cuda' init log + a
    # later host-RAM/DataLoader OOM) -> a false escalate. The fallback must restrict the scan to the
    # LAST traceback so both signals come from the SAME terminal failure.
    from flash.providers.runpod.jobs import detail_flags_cuda_oom

    mixed = (
        "INFO: CUDA initialized on device 0 (NVIDIA A100)\n"  # unrelated early 'cuda'
        "... lots of training output ...\n"
        "Traceback (most recent call last):\n"
        '  File "/x/torch/utils/data/_utils.py", line 9, in _try_get\n'
        "RuntimeError: DataLoader worker (pid 7) killed by signal: out of memory\n"  # host-RAM OOM
    )
    # whole-tail scanning would see 'cuda' (line 1) + 'out of memory' (terminal) and falsely escalate;
    # restricting to the terminal traceback keeps it a non-CUDA (host) OOM -> no escalation.
    assert detail_flags_cuda_oom(mixed) is False
    # a REAL CUDA OOM in the terminal block is still caught (both signals on the terminal line)
    real = (
        "INFO: starting vLLM EngineCore\n"
        "Traceback (most recent call last):\n"
        '  File "/x/vllm/engine.py", line 3, in step\n'
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
    )
    assert detail_flags_cuda_oom(real) is True


def test_worker_flagged_crash_is_attempt_scoped_to_a_terminal_error_heartbeat():
    # The dead-host crash/preempt split keys on THIS attempt's own terminal error heartbeat, not the
    # SHARED per-seed error_<phase>.txt artifact (which a prior attempt's OOM-escalation can leave).
    from flash.providers.runpod.jobs import worker_flagged_crash

    err_hb = lambda force=False: {"stage": "error_rl", "attempt": "1"}  # noqa: E731
    # THIS attempt (1) crashed -> True; a prior attempt's (0) error heartbeat does NOT count for 1
    assert worker_flagged_crash(err_hb, 1) is True
    assert worker_flagged_crash(lambda force=False: {"stage": "error_rl", "attempt": "0"}, 1) is False
    # a NON-error heartbeat (mid-training) is not a crash even when it's this attempt's (host loss)
    assert worker_flagged_crash(lambda force=False: {"stage": "rl_step", "attempt": "1"}, 1) is False
    # empty / non-dict / missing reader
    assert worker_flagged_crash(lambda force=False: {}, 1) is False
    assert worker_flagged_crash(lambda force=False: "x", 1) is False
    assert worker_flagged_crash(None, 1) is False
    # current_attempt=None keeps the historical 'trust any error heartbeat' behavior
    assert worker_flagged_crash(err_hb, None) is True


def test_lambda_dead_host_stale_error_after_oom_escalation_stays_preempted():
    # The fix's behavioral core (5x-): on attempt > 0 a prior attempt's lingering SHARED error_*.txt
    # (e.g. an OOM that ESCALATED) must NOT flip a host loss to a fail-fast job_failed. The dead-host
    # crash verdict requires THIS attempt's own terminal error heartbeat (handle.attempt==0 exempt).
    src, _ = _src("flash.providers.lambdalabs.jobs", "")
    assert "crashed_this_attempt = handle.attempt == 0 or worker_flagged_crash(hb_once, handle.attempt)" in src
    assert "worker_crashed = bool(err and err.strip()) and crashed_this_attempt and not retriable" in src


def test_instance_bootstrap_oom_scanner_is_self_contained():
    # 6Mx: the bootstrap runs as /root/flash/bootstrap.py with flash NOT importable (only the child
    # worker gets PYTHONPATH=/runcode/code). The console OOM scanner must NOT import flash — an import
    # error under the broad except would silently drop a child-process CUDA OOM and mislabel the job.
    boot_path = pathlib.Path(
        __import__("flash.providers._instance_bootstrap", fromlist=["x"]).__file__
    )
    boot = boot_path.read_text()
    # No ACTUAL import statement (module- OR function-level) pulls in flash (the docstring/comments
    # mention the word "flash", so scan the parsed import nodes, not raw substrings).
    tree = ast.parse(boot)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "flash":
            raise AssertionError(f"bootstrap imports flash: from {node.module}")
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == "flash" for a in node.names)
    assert "def _text_flags_cuda_oom" in boot  # inlined predicate
    # the inlined predicate is applied to the TERMINAL exception block of the console tail, never the
    # whole tail — mirrors runpod.jobs.detail_flags_cuda_oom so an early 'cuda' init line + a later
    # host-RAM OOM can't be combined into a false escalating marker (and stays self-contained).
    assert "def _terminal_exception_block" in boot  # inlined narrower
    assert "if _text_flags_cuda_oom(_terminal_exception_block(tail)):" in boot
    # the inlined predicate matches the shared control-plane text scanner (text_flags_cuda_oom)
    import importlib

    boot_mod = importlib.import_module("flash.providers._instance_bootstrap")
    from flash.engine.worker.perf.lifecycle import text_flags_cuda_oom

    for text in (
        "RuntimeError: Triton Error [CUDA]: out of memory",
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "DataLoader worker (pid 1) killed: out of memory",  # host OOM -> False both
        "ValueError: bad config",
    ):
        assert boot_mod._text_flags_cuda_oom(text) == text_flags_cuda_oom(text)
    # the terminal-block narrower mirrors the worker: a mixed tail (early 'cuda' init log + a later
    # host-RAM OOM in the terminal traceback) is NOT a CUDA OOM, while a real CUDA OOM in the
    # terminal traceback still flags. This is the bootstrap analog of detail_flags_cuda_oom narrowing.
    mixed = (
        "INFO: CUDA initialized on device 0 (NVIDIA A100)\n"
        "Traceback (most recent call last):\n"
        '  File "/x/torch/utils/data/_utils.py", line 9, in _try_get\n'
        "RuntimeError: DataLoader worker (pid 7) killed by signal: out of memory\n"
    )
    assert boot_mod._text_flags_cuda_oom(boot_mod._terminal_exception_block(mixed)) is False
    real = (
        "INFO: starting vLLM EngineCore\n"
        "Traceback (most recent call last):\n"
        '  File "/x/vllm/engine.py", line 3, in step\n'
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
    )
    assert boot_mod._text_flags_cuda_oom(boot_mod._terminal_exception_block(real)) is True


def _src(modfile_attr, name):
    import importlib

    mod = importlib.import_module(modfile_attr)
    return pathlib.Path(mod.__file__).read_text(), name


def test_worker_stamps_oom_flag_in_its_failure_heartbeat():
    # The worker classifies OOM off its LIVE exception (structured signals, not the message text) and
    # stamps it alongside `retriable`, so the poller reads a flag rather than parsing the detail.
    src, _ = _src("flash.engine.worker", "")
    assert "oom = is_cuda_oom(e)" in src
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
    # persisted on every submit as the CUMULATIVE/monotonic id (local loop index + resume base) so a
    # recovered attempt's heartbeat can't collide with a prior physical attempt's stale OOM flag.
    assert '"attempt": int(submit_attempt)' in life
    assert "submit_attempt = attempt + resume_attempt_base" in life
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


def test_attempt_marker_upload_is_retried(monkeypatch):
    # Mxb8a: the attempt marker is the ONLY attempt-safe structured signal the poller reads for a
    # console-only Lambda OOM (the dead-host path intentionally does NOT scan the shared console), so a
    # silently-lost single-shot best-effort upload would misclassify a real OOM as fail/preempt. The
    # marker upload retries a few times (and logs loudly on final failure) instead of one best-effort shot.
    import importlib

    boot = importlib.import_module("flash.providers._instance_bootstrap")
    # the marker write asks hf_upload to retry; ordinary console/log artifacts keep the single shot
    src = pathlib.Path(boot.__file__).read_text()
    assert (
        "hf_upload(payload, p, f\"{_arm(payload)}_attempt{marker['attempt']}.json\", retries=3)" in src
    )
    assert "def hf_upload(payload: dict, local_path: str, repo_subpath: str, retries: int = 0)" in src

    # behavioral: a transient upload error is retried, not dropped on the first failure
    monkeypatch.setattr(boot.time, "sleep", lambda *_a, **_k: None)  # don't actually back off
    calls = {"n": 0}

    class _FlakyApi:
        def __init__(self, *a, **k):
            pass

        def upload_file(self, **k):
            calls["n"] += 1
            if calls["n"] < 3:  # fail twice, then succeed
                raise RuntimeError("transient HF 503")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FlakyApi)
    payload = {"hf_prefix": "p/r", "hf_repo": "o/d", "env": {"HF_TOKEN": "t"}, "attempt": 1}
    p = "/tmp/_test_marker_retry.txt"
    pathlib.Path(p).write_text("x")
    boot.hf_upload(payload, p, "instance_attempt1.json", retries=3)
    assert calls["n"] == 3  # retried past the two transient failures to a landed upload
    # default (retries=0) stays a single best-effort shot — one call, never raises
    calls["n"] = 0

    class _DeadApi(_FlakyApi):
        def upload_file(self, **k):
            calls["n"] += 1
            raise RuntimeError("down")

    monkeypatch.setattr(huggingface_hub, "HfApi", _DeadApi)
    boot.hf_upload(payload, p, "console_rl.txt")  # default retries=0
    assert calls["n"] == 1


def test_console_oom_scan_catches_a_buried_subprocess_oom(tmp_path, monkeypatch):
    # MxaQ4: when the 8000-byte console slice has NO "Traceback" header, the terminal-block fallback
    # only sees the last 15 lines, so a bare subprocess CUDA OOM dumped earlier in the slice (then
    # buried under teardown noise) was missed. A per-LINE backstop scans the whole slice but requires
    # cuda+oom CO-LOCATED on one un-indented line, catching the buried OOM WITHOUT re-opening the
    # cross-line borrow the terminal-block scan guards against.
    import importlib

    boot = importlib.import_module("flash.providers._instance_bootstrap")
    teardown = "\n".join(f"INFO shutdown step {i}" for i in range(40))  # > the 15-line fallback
    # real single-line CUDA OOM with no traceback framing, buried above the teardown noise
    buried = "ERROR EngineCore: torch.cuda.OutOfMemoryError: CUDA out of memory\n" + teardown
    assert boot._any_line_flags_cuda_oom(buried) is True
    # cross-line borrow still rejected: an early 'cuda' init line + a later UNRELATED host-RAM OOM on
    # separate lines must NOT combine into a false escalation
    cross = "INFO: CUDA initialized on device 0\n" + teardown + "\nDataLoader worker killed: out of memory\n"
    assert boot._any_line_flags_cuda_oom(cross) is False

    # end-to-end through _console_flags_cuda_oom (reads /tmp/console_<mode>.txt)
    monkeypatch.setattr(boot.os.path, "exists", lambda _p: True)
    console = tmp_path / "console.txt"

    def _read(mode_text):
        console.write_text(mode_text)
        real_open = open

        def fake_open(path, *a, **k):
            return real_open(console, *a, **k) if path == "/tmp/console_buried.txt" else real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", fake_open)
        try:
            return boot._console_flags_cuda_oom("buried")
        finally:
            monkeypatch.setattr("builtins.open", real_open)

    assert _read(buried) is True  # buried in-slice OOM now caught
    assert _read(cross) is False  # cross-line host OOM still not escalated


def test_console_oom_scan_catches_early_oom_before_an_unrelated_later_traceback(tmp_path, monkeypatch):
    # PRRT_kwDOS-63f86MxgUM: a real EARLY single-line CUDA OOM followed by a LATER, UNRELATED traceback
    # used to be missed — the terminal-block scan scopes to the last traceback (no OOM there) and the
    # per-line backstop was gated to the no-"Traceback" case, so neither fired. The backstop now runs
    # over the WHOLE slice unconditionally, catching the early OOM WITHOUT re-opening the cross-line
    # borrow (it still requires cuda+oom CO-LOCATED on one un-indented line).
    import importlib

    boot = importlib.import_module("flash.providers._instance_bootstrap")
    later_tb = (
        "Traceback (most recent call last):\n"
        '  File "/x/teardown.py", line 1, in close\n'
        "RuntimeError: connection reset by peer\n"
    )
    # early real CUDA OOM, then an unrelated later traceback whose terminal block carries NO OOM:
    # the terminal-block scan alone misses it, so the unconditional backstop is what catches it.
    early_oom = "ERROR EngineCore: torch.cuda.OutOfMemoryError: CUDA out of memory\n" + later_tb
    assert boot._text_flags_cuda_oom(boot._terminal_exception_block(early_oom)) is False
    # cross-line borrow must STILL be rejected even with a (benign) traceback in between: an early
    # 'cuda' init line + a later UNRELATED host-RAM OOM on separate lines must NOT combine.
    cross = "INFO: CUDA initialized on device 0\n" + later_tb + "DataLoader worker killed: out of memory\n"

    monkeypatch.setattr(boot.os.path, "exists", lambda _p: True)
    console = tmp_path / "console.txt"

    def _read(mode_text):
        console.write_text(mode_text)
        real_open = open

        def fake_open(path, *a, **k):
            return real_open(console, *a, **k) if path == "/tmp/console_x.txt" else real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", fake_open)
        try:
            return boot._console_flags_cuda_oom("x")
        finally:
            monkeypatch.setattr("builtins.open", real_open)

    assert _read(early_oom) is True  # early OOM caught despite the later unrelated traceback
    assert _read(cross) is False  # cross-line host OOM still not escalated


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
    assert "resume_oom_attempts=resumed_oom_attempts" in deploy
    # cache-drop bonus attempts (the free cache-less no_capacity/poll_error fallback the shared weight
    # cache grants) are EXCLUDED from the live walk budget (walk_attempt = attempt - cache_drop_consumed).
    # The same count is persisted into the handle and SUBTRACTED on an OOM-aware resume, else a restart
    # would seed the OOM budget off the PHYSICAL attempt id (which counts the bonus) and over-count past
    # max_retries — a max_retries=1 run doing cache attempt 0 -> cacheless OOM attempt 1 would resume as
    # 2 spent and trip the entry guard instead of taking its one allowed escalation.
    assert '"cache_drop_consumed": int(cache_drop_consumed)' in life  # persisted on every submit
    assert 'persisted_cache_drops = int(remote.pop("cache_drop_consumed", 0) or 0)' in deploy
    assert "resumed_oom_attempts = persisted_attempt + 1 - persisted_cache_drops" in deploy  # oom reattach
    # an INFRA-shaped reattach must STILL remember a spent OOM budget when a prior OOM created the
    # floor (else a subsequent OOM after a preempted larger attempt re-escalates past max_retries).
    assert "elif persisted_oom_floor > 0:" in deploy
    assert "resumed_oom_attempts = persisted_attempt - persisted_cache_drops" in deploy
    # the resumed attempts get a MONOTONIC physical-id base so a recovered attempt's heartbeat can't
    # collide with a prior physical attempt's lingering OOM flag (independent of the budget accounting)
    assert "resume_attempt_base=persisted_attempt + 1" in deploy
    assert "resume_attempt_base: int = 0" in life
    assert "submit_attempt = attempt + resume_attempt_base" in life
    assert '"attempt": submit_attempt' in life  # the cumulative id is what is submitted/heartbeated
