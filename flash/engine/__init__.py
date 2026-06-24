"""Fine-tuning internals for the Flash package.

This subpackage holds the shared recipe, data loaders, graders, run accounting,
and the on-GPU worker entrypoint. The RunPod provider invokes ``flash.engine.worker``
on the provisioned GPU.
"""
