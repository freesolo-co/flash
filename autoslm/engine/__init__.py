"""Substrate-neutral fine-tuning internals for the AutoSLM package.

This subpackage holds the shared recipe, data loaders, graders, run accounting,
and the on-GPU worker entrypoint. It has no dependency on any compute backend; the
RunPod Flash integration in ``autoslm.flash`` invokes ``autoslm.engine.worker`` on the
provisioned GPU.
"""
