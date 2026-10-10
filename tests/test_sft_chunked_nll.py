from __future__ import annotations

import inspect


def test_sft_worker_requests_the_fused_loss_its_sizing_assumes():
    """The SFT worker must enable a dense-logit-free loss, because vram.py already sized for it.

    `sft_chunked_nll_enabled` drops the dense [batch, seq, vocab] logits term from the estimate and
    raises the micro-batch cap for the listed families. trl supplied that property via chunked_nll;
    verl supplies it via fused kernels. Either is fine, but the worker must request one -- a
    plain-nll run against a fused-sized reservation materializes logits the estimate never budgeted
    and OOMs.

    Liger is deliberately NOT the provider here. verl disables liger's fused linear CE anyway
    (it conflicts with verl's forward patching), so liger never supplied the dense-logit-free
    property; `use_fused_kernels` + `impl_backend=torch` does. Liger additionally zeroed the
    lora gradient on the sft path (GRAD-001), so it is off.

    The rest of this module asserted trl's own chunked-CE kernel (loss parity vs plain nll, router
    aux loss, SFTTrainer token metrics) and flash's trl-only `_prepare_chunked_nll_model` surgery.
    Flash no longer requests `loss_type="chunked_nll"` anywhere and verl trains out-of-process, so
    there is no in-process model to patch and nothing left of ours for those to cover.
    """
    from flash.engine.plan.vram import sft_chunked_nll_enabled
    from flash.engine.worker.train.entry import sft_train

    # the fused-kernel override is rendered in train.sft.config, the liger choice still in
    # sft_train, so this guard spans both.
    from flash.engine.worker.train.sft.setup import config as sft_config

    source = inspect.getsource(sft_train) + inspect.getsource(sft_config)
    assert '"model.use_fused_kernels=true"' in source
    assert '"use_liger": False' in source
    # the discount is not universal, so the gate must still be a real allowlist, not a constant.
    assert sft_chunked_nll_enabled("Qwen/Qwen3.5-9B")
    assert not sft_chunked_nll_enabled("meta-llama/Llama-3.1-8B")
