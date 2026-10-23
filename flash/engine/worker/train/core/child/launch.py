"""explicit, packaged startup shared by SFT, GRPO, and OPD entrypoints."""

import argparse
import json
import os
from pathlib import Path

from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.core.child.startup import initialize_process, validate_environment


def load_launch(algorithm, argv=None):
    parser = argparse.ArgumentParser(description=f"Flash {algorithm} trainer")
    parser.add_argument("--flash-config", required=True)
    arguments, overrides = parser.parse_known_args(argv)
    value = json.loads(Path(arguments.flash_config).read_text())
    options = RuntimeOptions.decode(dict(value, algorithm=algorithm))
    return options, overrides


def configure(options, overrides):
    from hydra import compose, initialize_config_module
    from omegaconf import open_dict
    from verl.utils.device import auto_set_device

    from flash.engine.worker.train.core.child.upstream import validate_upstream

    validate_upstream()
    name = "sft_trainer_engine" if options.algorithm == "sft" else "ppo_trainer"
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(config_name=name, overrides=overrides)
    with open_dict(config):
        model = config.model if options.algorithm == "sft" else config.actor_rollout_ref.model
        model._target_ = "flash.engine.worker.train.core.child.configuration.FlashModelConfig"
        model.runtime_options = options.encode()
        if options.algorithm == "grpo":
            model.runtime_options["rollout_template_kwargs"] = dict(
                config.data.get("apply_chat_template_kwargs", {})
            )
        if options.algorithm == "sft":
            engines = [config.engine]
        else:
            engines = [
                config.actor_rollout_ref.actor.fsdp_config,
                config.actor_rollout_ref.ref.fsdp_config,
            ]
            for actor in (config.actor_rollout_ref.actor, config.actor_rollout_ref.ref):
                actor._target_ = (
                    "flash.engine.worker.train.core.child.configuration.FlashActorConfig"
                )
            config.actor_rollout_ref.rollout.name = "flash_vllm"
            config.actor_rollout_ref.rollout.engine_kwargs.vllm.worker_cls = (
                "flash.engine.worker.train.core.child.vllm_worker.FlashGPUWorker"
            )
            config.trainer.use_v1 = True
            config.trainer.v1.trainer_mode = "sync"
            config.transfer_queue.enable = True
            module = "rl" if options.algorithm == "grpo" else "opd"
            name = "FlashGrpoManager" if module == "rl" else "FlashOpdManager"
            agent = config.actor_rollout_ref.rollout.agent
            agent.agent_loop_manager_class = (
                f"flash.engine.worker.train.{module}.child.rollout.{name}"
            )
            agent.agent_loop_config_path = str(Path(__file__).with_name("agents.yaml"))
            if options.algorithm == "grpo" and agent.default_agent_loop == "single_turn_agent":
                agent.default_agent_loop = "flash_grpo_single_turn"
        for engine in engines:
            engine._target_ = "flash.engine.worker.train.core.child.configuration.FlashEngineConfig"
            engine.strategy = "flash_" + str(engine.strategy)
    auto_set_device(config)
    return config


def run(algorithm, argv=None):
    options, overrides = load_launch(algorithm, argv)
    if algorithm == "opd" and options.structured_outputs:
        from flash.engine.worker.train.opd.child.policy import _require_structured_runtime_versions

        _require_structured_runtime_versions()
    validate_environment()
    initialize_process(options.seed if algorithm == "sft" else None)
    config = configure(options, overrides)
    if algorithm == "sft":
        return run_sft(config, options)
    if algorithm == "grpo":
        from flash.engine.worker.train.rl.child.metrics import stream_metrics

        with stream_metrics(os.environ["VERL_FILE_LOGGER_PATH"]):
            return run_rl(config, algorithm)
    return run_rl(config, algorithm)


def run_sft(config, options):
    import torch.distributed as distributed
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    from flash.engine.worker.train.core.child.readiness import publish_ready
    from flash.engine.worker.train.sft.child.trainer import FlashSFTTrainer

    initialize_global_process_group()
    try:
        trainer = FlashSFTTrainer(config)
        distributed.barrier()
        if distributed.get_rank() == 0:
            publish_ready(options)
        trainer.fit()
    finally:
        destroy_global_process_group()


def run_rl(config, algorithm):
    import ray
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    from flash.engine.worker.train.core.child.trainer import FlashTaskRunner

    runner = ray.remote(num_cpus=1)(FlashTaskRunner)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    try:
        run_ppo(config, task_runner_class=runner)
    finally:
        ray.shutdown()
