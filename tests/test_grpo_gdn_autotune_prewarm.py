"""GatedDeltaNet l2norm autotune pre-warm for colocated GRPO (CPU-only, no GPU/fla/network).

Prod run flash-1782588906 (9B GRPO on an 80 GB A100 PCIe) died with a CUDA OOM *inside* fla's
``l2norm_bwd`` Triton autotuner (``do_bench`` over 25 configs) on the FIRST training backward — not in
steady-state training. On a resident colocate step the rollout engine is held through the backward, so
that cold-key benchmark's transient tips the ~full card. ``prewarm_gdn_autotune`` runs the benchmark at
setup while VRAM is free; since the autotune key buckets by token count (``NB = cdiv(rows, 65536)``) it
sweeps every NB bucket the run can reach. These cover the pure dispatch logic; the GPU warm itself is
exercised by the live smokes.
"""

from __future__ import annotations

import sys
import types

import flash.engine.worker.kernel_warmup as kw


def _nb(seq, heads):
    rows = seq * heads
    return -(-rows // kw._L2NORM_NB_GRANULARITY)  # ceil


def test_nb_buckets_cover_each_bucket_once_and_increase():
    # The sweep must hit every NB bucket from 1..nb_max exactly once, in increasing order, so the
    # train-time backward (any seq up to the bound) finds its [D, NB] key cached.
    for heads in (8, 16, 32):
        max_rows = 8 * 3072 * heads  # group_size x max_length x heads
        seqs = kw.gdn_autotune_nb_buckets(heads, max_rows)
        nbs = [_nb(s, heads) for s in seqs]
        assert nbs == list(range(1, len(seqs) + 1)), (heads, seqs, nbs)


def test_nb_buckets_tiny_run_is_single_bucket():
    # A run whose worst-case rows fit one bucket warms only NB=1 (the cheap fast path).
    assert kw.gdn_autotune_nb_buckets(8, 1000) == [256]
    assert kw.gdn_autotune_nb_buckets(16, kw._L2NORM_NB_GRANULARITY) == [256]


def test_nb_buckets_no_overshoot_with_large_head_count():
    # Each warmed seq must land in EXACTLY its target bucket. With a large num_key_heads the old
    # max(256, ...) floor could push 256*h past the bucket boundary (e.g. NB=1 -> rows land in NB=2),
    # silently skipping that [D, NB] key. Every returned seq must map to its 1-based bucket index.
    for heads in (8, 16, 32, 256, 512):
        max_rows = 8 * 4096 * heads
        seqs = kw.gdn_autotune_nb_buckets(heads, max_rows)
        nbs = [_nb(s, heads) for s in seqs]
        assert nbs == list(range(1, len(seqs) + 1)), (heads, seqs, nbs)


def test_nb_buckets_capped():
    # A pathological (huge) row count can't explode the sweep into dozens of warms.
    seqs = kw.gdn_autotune_nb_buckets(16, 8 * 65536 * 64)
    assert len(seqs) == kw._MAX_NB_BUCKETS


def test_gdn_dims_none_for_non_gdn(monkeypatch):
    # A plain Llama/MiniCPM config has no linear_key_head_dim -> nothing to warm.
    cfg = types.SimpleNamespace(model_type="llama")  # no linear_* attrs
    fake_tf = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: cfg)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    assert kw._gdn_dims_from_config("some/llama") is None


def test_gdn_dims_reads_text_config(monkeypatch):
    # VL checkpoints nest the GDN dims under text_config; read head_dim + head count there. With no
    # value-head attr the count falls back to key heads.
    inner = types.SimpleNamespace(linear_key_head_dim=128, linear_num_key_heads=16)
    cfg = types.SimpleNamespace(text_config=inner)
    fake_tf = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: cfg)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    assert kw._gdn_dims_from_config("some/qwen35-vl") == (128, 16)


def test_gdn_dims_uses_value_heads_not_key_heads(monkeypatch):
    # GatedDeltaNet repeat_interleaves q/k up to the VALUE-head count before chunk_gated_delta_rule,
    # and l2norm flattens that repeated tensor — so the head count driving NB is num_value_heads
    # (>= num_key_heads). Sizing by key heads would under-count rows and stop the sweep early.
    cfg = types.SimpleNamespace(
        linear_key_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=32
    )
    fake_tf = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: cfg)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    assert kw._gdn_dims_from_config("some/qwen35") == (128, 32)  # value heads, not the 16 key heads


def _fake_cuda_torch(available=True):
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: available))


def test_prewarm_noop_for_non_gdn(monkeypatch):
    monkeypatch.setattr(kw, "_gdn_dims_from_config", lambda mid: None)
    called = []
    monkeypatch.setattr(kw, "warm_fla_gdn", lambda *a, **k: called.append(k) or True)
    assert kw.prewarm_gdn_autotune("x/llama", max_length=3072, group_size=8) is False
    assert called == []


def test_prewarm_skips_huge_head_dim(monkeypatch):
    # head_dim above fla's autotuned range -> the kernel doesn't benchmark, so there's nothing to warm.
    monkeypatch.setattr(kw, "_gdn_dims_from_config", lambda mid: (kw._L2NORM_AUTOTUNE_MAX_D + 1, 8))
    called = []
    monkeypatch.setattr(kw, "warm_fla_gdn", lambda *a, **k: called.append(k) or True)
    assert kw.prewarm_gdn_autotune("x/big-head", max_length=3072, group_size=8) is False
    assert called == []


def test_prewarm_noop_without_cuda(monkeypatch):
    monkeypatch.setattr(kw, "_gdn_dims_from_config", lambda mid: (128, 16))
    monkeypatch.setitem(sys.modules, "torch", _fake_cuda_torch(available=False))
    called = []
    monkeypatch.setattr(kw, "warm_fla_gdn", lambda *a, **k: called.append(k) or True)
    assert kw.prewarm_gdn_autotune("x/gdn", max_length=3072, group_size=8) is False
    assert called == []


def test_prewarm_warms_every_bucket_at_real_dims(monkeypatch):
    # GDN + CUDA: warm_fla_gdn is called once per NB bucket at the model's real head_dim/heads, and the
    # FIRST call also warms the varlen path.
    monkeypatch.setattr(kw, "_gdn_dims_from_config", lambda mid: (128, 16))
    monkeypatch.setitem(sys.modules, "torch", _fake_cuda_torch(available=True))
    calls = []
    monkeypatch.setattr(kw, "warm_fla_gdn", lambda torch, **k: calls.append(k) or True)
    assert kw.prewarm_gdn_autotune("x/gdn", max_length=3072, group_size=8) is True
    # one call per bucket, each at the real head_dim/num_heads
    assert calls, "expected at least one warm"
    assert all(c["head_dim"] == 128 and c["num_heads"] == 16 for c in calls)
    nbs = [_nb(c["seq_len"], 16) for c in calls]
    assert nbs == list(range(1, len(calls) + 1)), nbs
    # varlen only on the first (GRPO rollouts are unpacked)
    assert calls[0]["varlen"] is True
    assert all(c["varlen"] is False for c in calls[1:])


def test_prewarm_sizes_rows_by_per_device_comps_above_group_size(monkeypatch):
    # TRL GRPO's logp forward processes per_device_train_batch_size COMPLETIONS, which
    # rl_per_device_comps can grow above group_size. The bucket sweep must size rows by the larger of
    # the two, so the higher NB bucket the training forward reaches is pre-warmed (not benchmarked on
    # the memory-tight resident step).
    monkeypatch.setattr(kw, "_gdn_dims_from_config", lambda mid: (128, 16))
    monkeypatch.setitem(sys.modules, "torch", _fake_cuda_torch(available=True))
    small = []
    monkeypatch.setattr(kw, "warm_fla_gdn", lambda torch, **k: small.append(k) or True)
    kw.prewarm_gdn_autotune("x/gdn", max_length=3072, group_size=8, per_device_comps=8)
    big = []
    monkeypatch.setattr(kw, "warm_fla_gdn", lambda torch, **k: big.append(k) or True)
    kw.prewarm_gdn_autotune("x/gdn", max_length=3072, group_size=8, per_device_comps=16)
    # per_device 16 > group 8 doubles the row bound -> at least as many buckets, reaching a HIGHER NB
    assert len(big) >= len(small)
    assert _nb(big[-1]["seq_len"], 16) > _nb(small[-1]["seq_len"], 16)


def test_prewarm_defaults_to_group_size_when_per_device_unset(monkeypatch):
    # Default per_device_comps=0 must fall back to group_size (max(group_size, 0)) — unchanged sweep.
    monkeypatch.setattr(kw, "_gdn_dims_from_config", lambda mid: (128, 16))
    monkeypatch.setitem(sys.modules, "torch", _fake_cuda_torch(available=True))
    a = []
    monkeypatch.setattr(kw, "warm_fla_gdn", lambda torch, **k: a.append(k) or True)
    kw.prewarm_gdn_autotune("x/gdn", max_length=3072, group_size=8)
    b = []
    monkeypatch.setattr(kw, "warm_fla_gdn", lambda torch, **k: b.append(k) or True)
    kw.prewarm_gdn_autotune("x/gdn", max_length=3072, group_size=8, per_device_comps=8)
    assert [_nb(c["seq_len"], 16) for c in a] == [_nb(c["seq_len"], 16) for c in b]


def test_run_rl_prewarm_passes_per_device_ceiling():
    # The call must bound rows by the per-device completion ceiling (_RL_PER_DEVICE_MAX), not just
    # group_size — else a short non-thinking run whose per_device_train_batch_size exceeds group_size
    # leaves the higher NB bucket cold to benchmark (and OOM) on the resident colocate step.
    import ast
    import pathlib

    src = pathlib.Path(kw.__file__).with_name("rl.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_rl")
    call = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == "prewarm_gdn_autotune"
    )
    kwargs = {k.arg: k.value for k in call.keywords}
    assert "per_device_comps" in kwargs, "prewarm call must pass the per-device completion bound"
    assert "_RL_PER_DEVICE_MAX" in ast.dump(kwargs["per_device_comps"])


def test_warm_fla_gdn_signature_keeps_build_time_default():
    # The build-time bake calls warm_fla_gdn(torch) with no dims; the head-dim/seq params must default
    # so that path is unchanged.
    import inspect

    p = inspect.signature(kw.warm_fla_gdn).parameters
    assert p["head_dim"].default == 64
    assert p["num_heads"].default == 4
    assert p["seq_len"].default == 256
    assert p["varlen"].default is True


def test_run_rl_prewarms_before_trainer_under_heartbeat():
    # Wiring: run_rl imports prewarm_gdn_autotune and calls it BEFORE the model load / GRPOTrainer build,
    # wrapped in liveness_heartbeat so the CUDA-busy autotune sweep can't trip the stall watchdog.
    import ast
    import pathlib

    src = pathlib.Path(kw.__file__).with_name("rl.py").read_text()
    tree = ast.parse(src)
    assert any(
        isinstance(n, ast.ImportFrom)
        and (n.module or "").endswith("kernel_warmup")
        and any(a.name == "prewarm_gdn_autotune" for a in n.names)
        for n in ast.walk(tree)
    )
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_rl")

    def lineno(name):
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                f = n.func
                if (getattr(f, "id", None) or getattr(f, "attr", None)) == name:
                    return n.lineno
        return None

    pre, init, trainer = (
        lineno("prewarm_gdn_autotune"),
        lineno("_init_adapter_model"),
        lineno("GRPOTrainer"),
    )
    assert pre is not None
    assert init is not None
    assert trainer is not None
    assert pre < init < trainer, (pre, init, trainer)
    # heartbeat-wrapped
    assert any(
        isinstance(n, ast.With)
        and "liveness_heartbeat" in ast.dump(n)
        and "prewarm_gdn_autotune" in ast.dump(n)
        for n in ast.walk(fn)
    )
