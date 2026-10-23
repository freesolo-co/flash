"""Explicit vLLM worker startup for image-baked kernels."""

from vllm.v1.worker.gpu_worker import Worker

from flash.engine.worker.runtime.phases import training_phase
from flash.engine.worker.train.core.child.flashinfer_cache import activate_current


class FlashGPUWorker(Worker):
    def init_device(self):
        result = super().init_device()
        activate_current()
        return result

    @training_phase("model_initialization")
    def load_model(self, *args, **kwargs):
        return super().load_model(*args, **kwargs)

    @training_phase("compilation_and_warmup")
    def compile_or_warm_up_model(self):
        return super().compile_or_warm_up_model()
