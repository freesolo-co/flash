"""RunPod-validation guards for serving and standalone evals.

A GPU class can have a RunPod enum member yet be validated only on Vast (e.g.
RTX 3090). Such a class must not be served or eval'd directly on the unvalidated
RunPod substrate — serving falls back to a validated class, and standalone eval
is rejected before spend.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm.flash.gpus import GPU_INFO, is_validated


def test_servable_gpu_falls_back_for_vast_only_validated_enum_class():
    from autoslm.serve.deploy import servable_gpu

    # RTX 3090 has a RunPod enum member but is validated only on Vast.
    assert GPU_INFO["RTX 3090"].enum_member is not None
    assert "runpod" not in GPU_INFO["RTX 3090"].validated_on

    served = servable_gpu("RTX 3090", "Qwen/Qwen3-0.6B")
    assert served != "RTX 3090", "must not serve directly on the unvalidated RunPod class"
    assert is_validated(served, "runpod"), "fallback must be RunPod-validated"
    assert GPU_INFO[served].vram_gb >= GPU_INFO["RTX 3090"].vram_gb

    # A genuinely RunPod-validated class is served directly.
    assert servable_gpu("RTX 4090", "Qwen/Qwen3-0.6B") == "RTX 4090"


def test_submit_eval_job_rejects_vast_only_validated_class(tmp_path, monkeypatch):
    import autoslm.orchestrator as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path))
    from autoslm.worker_spec import GpuSpec, JobSpec

    spec = JobSpec(
        run_id="eval-3090",
        model="Qwen/Qwen3-0.6B",
        algorithm="grpo",
        gpu=GpuSpec(type="RTX 3090"),
    )
    status = orch.submit_eval_job(spec, adapter_prefix="grpo/x/seed0", dry_run=False)
    assert status.state == "failed"
    assert "RunPod-validated" in (status.error or "")


def test_submit_eval_job_allows_vast_only_class_when_unvalidated_opt_in(tmp_path, monkeypatch):
    # gpu.allow_unvalidated opts past the validation gate; the run then reaches the
    # GPU submit (mocked here) instead of being rejected.
    import autoslm.flash.train as ftrain
    import autoslm.orchestrator as orch

    monkeypatch.setattr(orch, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(orch, "RESULTS_DIR", str(tmp_path / "results"))
    from autoslm.worker_spec import GpuSpec, JobSpec

    spec = JobSpec(
        run_id="eval-3090-ok",
        model="Qwen/Qwen3-0.6B",
        algorithm="grpo",
        gpu=GpuSpec(type="RTX 3090", allow_unvalidated=True),
    )
    monkeypatch.setattr(ftrain, "upload_code", lambda: "repo")
    monkeypatch.setattr(ftrain, "submit_eval", lambda *a, **k: {"cost_usd": 0.1})
    monkeypatch.setattr(orch, "_gc_run_endpoints", lambda s: None)

    status = orch.submit_eval_job(spec, adapter_prefix="grpo/x/seed0", dry_run=False)
    assert status.state == "done", "allow_unvalidated must pass the substrate gate"
