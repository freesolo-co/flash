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
    from verl.runtime_interfaces import API_VERSION
    from verl.utils.device import auto_set_device

    if API_VERSION != 2:
        raise RuntimeError("unsupported Verl trainer interface version")
    name = "sft_trainer_engine" if options.algorithm == "sft" else "ppo_trainer"
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(config_name=name, overrides=overrides)
    with open_dict(config):
        model = config.model if options.algorithm == "sft" else config.actor_rollout_ref.model
        model.custom_engine_class = "flash.engine.worker.train.core.child.engine.FlashFSDPEngine"
        model.runtime_options = options.encode()
        if options.algorithm != "sft":
            module = "rl" if options.algorithm == "grpo" else "opd"
            name = "FlashGrpoManager" if module == "rl" else "FlashOpdManager"
            config.actor_rollout_ref.rollout.agent.agent_loop_manager_class = (
                f"flash.engine.worker.train.{module}.child.rollout.{name}"
            )
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

    if algorithm == "grpo":
        from verl.trainer.main_ppo import migrate_legacy_reward_impl

        from flash.engine.worker.train.rl.child.trainer import FlashGrpoTaskRunner

        config = migrate_legacy_reward_impl(config)
        runner = ray.remote(num_cpus=1)(FlashGrpoTaskRunner)
    else:
        from flash.engine.worker.train.opd.child.trainer import FlashTaskRunner

        config.transfer_queue.enable = True
        runner = FlashTaskRunner
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    try:
        run_ppo(config, task_runner_class=runner)
    finally:
        ray.shutdown()
