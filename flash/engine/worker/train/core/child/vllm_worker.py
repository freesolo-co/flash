"""Explicit vLLM worker startup for image-baked kernels."""

from vllm.v1.worker.gpu_worker import Worker

from flash.engine.worker.train.core.child.flashinfer_cache import activate_current


class FlashGPUWorker(Worker):
    def init_device(self):
        result = super().init_device()
        activate_current()
        return result
