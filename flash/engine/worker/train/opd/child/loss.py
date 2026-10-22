"""register the remote OPD objective through Verl's loss registry."""

from flash.engine.worker.train.opd.child.policy import _register_flash_distillation_loss

_register_flash_distillation_loss()
