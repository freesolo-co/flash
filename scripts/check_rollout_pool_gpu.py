"""Run inside a disposable three-GPU worker with the pinned Verl interpreter.

Creates a tiny random model locally, trains two steps with two rollout replicas,
then resumes for one more step. No model downloads or artifact-store writes.
This checks infrastructure correctness, not model quality or production scale.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_model(path):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(42)
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    vocab.update({f"word{i}": i + 3 for i in range(125)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}word0",
    )
    tokenizer.save_pretrained(path)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=256,
            tie_word_embeddings=False,
            eos_token_id=2,
            pad_token_id=1,
        )
    )
    model.to(torch.bfloat16).save_pretrained(path)


class RewardHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/identity/register":
            output = {"ok": True}
        elif self.path == "/score":
            # Deliberately alternating rewards ensure nonzero GRPO advantages.
            output = {"score": float(data["identity"]["rollout_ordinal"] % 2)}
        else:
            self.send_error(404)
            return
        payload = json.dumps(output).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def config(root, steps):
    return {
        "fused_ce_backend": "torch",
        "train_files": str(root / "train.parquet"),
        "val_files": str(root / "train.parquet"),
        "model_path": str(root / "model"),
        "lora_rank": 8,
        "lora_alpha": 16,
        "target_modules": "all-linear",
        "exclude_modules": None,
        "lr": 1e-3,
        "group_size": 2,
        "prompts_per_step": 2,
        "max_prompt_len": 64,
        "max_model_len": 128,
        "max_token_len_per_gpu": 128,
        "max_completion": 32,
        "max_response_len": 32,
        "multi_turn": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "kl_coef": 0.0,
        "stop_sequences": (),
        "structured_outputs": None,
        "thinking": False,
        "loss_agg_mode": "seq-mean-token-sum-norm",
        "seed": 42,
        "ppo_epochs": 1,
        "steps": steps,
        "gpu_mem_util": 0.3,
        "n_gpus": 1,
        "loggers": ["console", "file"],
        "fp8_kv": False,
        "enforce_eager": True,
        "warmstart_adapter": "",
        "reward_path": str(
            Path(__file__).resolve().parents[1] / "flash/engine/worker/train/rl/child/reward.py"
        ),
        "reward_name": "compute_score",
        "mask_truncated_completions": False,
        "total_epochs": 3,
        "save_freq": 1,
        "ckpt_to_keep": 2,
        "local_dir": str(root / "checkpoints"),
        "project_name": "rollout-pool-check",
        "experiment_name": "tiny-random",
        "rollout_pool": {
            "gpus": 2,
            "tensor_parallel": 1,
            "gpu_memory_utilization": 0.3,
        },
    }


def attempt(root, steps):
    from flash.engine.worker.train.core.child.options import RuntimeOptions
    from flash.engine.worker.train.rl.launch.verl_config import build_verl_overrides

    cfg = config(root, steps)
    ready = root / f"ready-{steps}"
    options = RuntimeOptions("grpo", steps, str(ready), rollout_pool=cfg["rollout_pool"])
    launch = root / f"launch-{steps}.json"
    launch.write_text(json.dumps(options.encode()))
    journal = root / f"metrics-{steps}.jsonl"
    env = dict(os.environ, VERL_FILE_LOGGER_PATH=str(journal))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "flash.engine.worker.train.rl.child.entry",
            "--flash-config",
            str(launch),
            *build_verl_overrides(cfg),
        ],
        env=env,
        check=True,
        timeout=900,
    )
    assert ready.read_text().strip() == "runtime-ready"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    records = [row for row in records if "actor/grad_norm" in row.get("data", {})]
    expected = [1, 2] if steps == 2 else [3]
    assert [row["step"] for row in records] == expected, records
    for row in records:
        data = row["data"]
        assert data["actor/grad_norm"] > 0, row
        assert data["rollout_pool/replicas"] == 2, row
        assert data["rollout_pool/published_step"] == row["step"], row
    return records


def main(root):
    from flash.engine.worker.train.rl.launch.verl_config import (
        build_verl_dataset_rows,
        write_verl_grpo_parquet,
    )

    root.mkdir(parents=True, exist_ok=False)
    make_model(root / "model")
    prompts = [[{"role": "user", "content": f"word{i} word8"}] for i in range(4)]
    write_verl_grpo_parquet(
        build_verl_dataset_rows(prompts, list(range(4)), [""] * 4),
        str(root / "train.parquet"),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), RewardHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ.update(
        FLASH_VERL_REWARD_URL=f"http://127.0.0.1:{server.server_port}",
        FLASH_CHECKPOINT_MODEL_ID="local/tiny-random-qwen3",
        FLASH_CHECKPOINT_MODEL_REVISION="0" * 40,
    )
    try:
        fresh = attempt(root, 2)
        resumed = attempt(root, 3)
        result = {"fresh_steps": len(fresh), "resumed_steps": len(resumed), "replicas": 2}
        (root / "result.json").write_text(json.dumps(result))
        print("ROLLOUT_POOL_GPU_OK " + json.dumps(result), flush=True)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.output)
