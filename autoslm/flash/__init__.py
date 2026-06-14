"""RunPod Flash integration: managed, serverless GPUs (no Docker) for AutoSLM.

Fine-tuning runs on a dedicated RunPod GPU (RTX 4090 / RTX 5090) provisioned by Flash.
A decorated Python handler (``autoslm.flash.train``) executes the ``autoslm.engine.worker``
on the GPU; Flash handles provisioning, dependency install, execution, and scale-to-zero
teardown. Serving (``autoslm.flash.serve``) exposes an OpenAI-compatible endpoint for a
trained LoRA adapter.
"""
