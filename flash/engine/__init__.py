"""Substrate-neutral fine-tuning internals for the Flash package.

This subpackage holds the shared recipe, data loaders, graders, run accounting,
and the on-GPU worker entrypoint. It has no dependency on any compute backend; each
provider (``flash.providers.runpod`` / ``flash.providers.vast``) invokes
``flash.engine.worker`` on the provisioned GPU.
"""
