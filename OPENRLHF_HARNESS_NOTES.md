# OpenRLHF shared harness notes

- OpenRLHF is pinned to commit `bc71bb19464aca306b33080b2d2bb45d154e2f49` in the prebuilt `ghcr.io/freesolo-co/flash-openrlhf:cu13-vllm0221` image. RunPod pulls the private image with registry auth ID `cmqg4x3z6006i5k4uyzfl2ouv`.
- The normal Flash worker is CUDA 12.8. OpenRLHF requires the isolated CUDA 13 and torch 2.11 stack, so workers using this harness run on the OpenRLHF image and set `FLASH_OPENRLHF_PYTHON`. The harness never installs vLLM, FlashAttention, or OpenRLHF at runtime.
- OpenRLHF runs as `python -m openrlhf.cli.train_ppo_ray`. Later GRPO and OPD workers will keep Flash-specific reward and teacher logic in the parent and expose it to the subprocess over authenticated localhost bridges, matching the existing verl process boundary.
- OpenRLHF final model saves consolidate DeepSpeed ZeRO-3 state before PEFT export. ZeRO-3 PEFT saves use `adapter_model.bin`, which the harness converts to `adapter_model.safetensors` in the isolated interpreter. Raw DeepSpeed recovery shards require the matching `<tag>_hf` export produced by `--ckpt.save_hf`; full-model Hugging Face weights cannot be reinterpreted as a LoRA adapter.
- This PR does not select OpenRLHF for SFT, GRPO, or OPD and does not remove TRL.
