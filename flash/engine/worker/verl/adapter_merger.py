"""Export LoRA with upstream shard merging, without constructing or exporting the base model."""

from verl.model_merger.base_model_merger import generate_config_from_args, parse_args
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


class AdapterMerger(FSDPModelMerger):
    def save_hf_model_and_tokenizer(self, state_dict):
        if self.save_lora_adapter(state_dict) is None:
            raise RuntimeError("checkpoint contains no LoRA adapter tensors")


def main():
    from flash.engine.worker.train.core.child.upstream import validate_upstream

    validate_upstream()
    config = generate_config_from_args(parse_args())
    if config.backend != "fsdp" or config.operation != "merge" or config.hf_upload:
        raise ValueError("Flash adapter export requires a local FSDP merge")
    merger = AdapterMerger(config)
    try:
        merger.merge_and_save()
    finally:
        merger.cleanup()


if __name__ == "__main__":
    main()
