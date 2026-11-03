"""Build the actual Qwen3.5-9B-Base FP8 checkpoint; publishing is explicit.

Uses the same FP8_DYNAMIC scheme as Freesolo's Instruct checkpoint. Conversion
runs on CPUs so it does not compete with serving for a GPU. GPU serving and LoRA
qualification remain separate from this structural/short-text quality smoke.
Set QWEN_BENCH_IMAGE_ID to a built serving image before invoking with modal run.
"""

import json
import os
from pathlib import Path

import modal

from flash.serving.app.modal_app import hf_cache_volume as volume

SOURCE = "Qwen/Qwen3.5-9B-Base"
REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
TARGET = "Freesolo-Co/Qwen3.5-9B-Base-FP8"
IGNORE = ["re:.*lm_head", "re:model.visual.*"]
app = modal.App("flash-qwen-base-fp8-conversion")
image_id = os.environ.get("QWEN_BENCH_IMAGE_ID")
if not image_id:
    raise ValueError("Set QWEN_BENCH_IMAGE_ID to a built serving image for conversion")
serving_image = modal.Image.from_id(image_id)
quant_image = serving_image.pip_install("llmcompressor==0.12.0")
image = quant_image.add_local_python_source("flash")
TEXTS = [
    "The library opened early in the morning. A reader returned three books and borrowed two more.",
    "Water freezes at zero degrees Celsius under standard atmospheric pressure and boils at one hundred.",
    "A reliable software service checks its inputs, records failures, and retries only transient errors.",
    "The gardener planted seeds in moist soil. After several days, small green shoots began to appear.",
    "If a train travels sixty kilometers in one hour, its average speed is sixty kilometers per hour.",
    "A database transaction groups related changes together so that they either all succeed or all fail.",
    "The hikers followed the river upstream until they reached a wooden bridge near the forest.",
    "In Python, a function can accept arguments and return a value. Lists preserve the order of their items.",
]


def text_loss(model, tokenizer):
    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(TEXTS, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[:, :-1].float()
        labels = inputs["input_ids"][:, 1:]
        mask = inputs["attention_mask"][:, 1:]
        token_losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape(labels.shape)
        losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
        if not torch.isfinite(losses).all():
            raise RuntimeError("Non-finite short-text evaluation loss")
    return losses.tolist()


@app.function(
    image=image,
    cpu=8,
    memory=98304,
    timeout=3600,
    max_containers=1,
    volumes={"/vol/hosting-cache": volume},
)
def convert():
    import importlib.metadata

    import torch
    from huggingface_hub import snapshot_download
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.utils import load_context
    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(8)
    source = snapshot_download(SOURCE, revision=REVISION)
    output = Path("/vol/hosting-cache/quantized/qwen35-9b-base-fp8/" + REVISION)
    output.mkdir(parents=True, exist_ok=True)
    with load_context(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            source, dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
        ).eval()
    processor = AutoProcessor.from_pretrained(source)
    before = text_loss(model, processor.tokenizer)
    print("BASELINE_LOSS", before, flush=True)
    oneshot(
        model=model,
        recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE),
    )
    after = text_loss(model, processor.tokenizer)
    print("FP8_LOSS", after, flush=True)
    if sum(after) / len(after) > sum(before) / len(before) + 0.1:
        raise RuntimeError("FP8 short-text mean loss increased by more than 0.1 nats")
    model.save_pretrained(output, save_compressed=True)
    processor.save_pretrained(output)
    for license_file in Path(source).glob("LICENSE*"):
        (output / license_file.name).write_bytes(license_file.read_bytes())
    quantized = 0
    vision = 0
    for shard in output.glob("*.safetensors"):
        with safe_open(shard, framework="pt") as tensors:
            for key in tensors.keys():  # noqa: SIM118 - safe_open is not a mapping
                dtype = tensors.get_slice(key).get_dtype()
                if "F8" in dtype:
                    quantized += 1
                    if key.startswith(("model.visual.", "lm_head.")):
                        raise RuntimeError(f"Excluded tensor was quantized: {key}")
                if key.startswith("model.visual."):
                    vision += 1
    if quantized != 248 or vision == 0:
        raise RuntimeError(f"Incomplete checkpoint: {quantized=}, {vision=}")
    report = {
        "source": SOURCE,
        "source_revision": REVISION,
        "target": TARGET,
        "scheme": "FP8_DYNAMIC",
        "ignore": IGNORE,
        "fp8_tensors": quantized,
        "vision_tensors": vision,
        "bf16_losses": before,
        "fp8_losses": after,
        "quality_smoke_texts": TEXTS,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "llmcompressor", "compressed-tensors")
        },
    }
    (output / "quantization-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3.5-9B-Base\n"
        "library_name: transformers\ntags:\n- compressed-tensors\n- fp8\n---\n\n"
        "# Qwen3.5-9B-Base FP8\n\n"
        f"FP8_DYNAMIC conversion of `{SOURCE}` at revision `{REVISION}`. "
        "This is the pretrained Base checkpoint, not the instruction-tuned model.\n\n"
        "Linear weights use per-channel FP8 and activations use dynamic per-token FP8. "
        "Linear-attention projection layers are included. The vision tower, normalization "
        "layers, and language-model head retain their original precision. "
        "Tokenizer, processor, and context configuration come from the "
        "pinned source. No training or instruction tuning was performed.\n\n"
        "See `quantization-report.json` for exact versions, exclusions, and a short-text "
        "loss comparison. That smoke check is not a comprehensive quality evaluation.\n"
    )
    volume.commit()
    print("CONVERSION_RESULT", json.dumps(report), flush=True)
    return report


@app.function(
    image=quant_image.env(
        {"HF_HUB_DISABLE_XET": "0", "HF_XET_HIGH_PERFORMANCE": "1"}
    ).add_local_python_source("flash"),
    cpu=8,
    memory=16384,
    timeout=3600,
    max_containers=1,
    volumes={"/vol/hosting-cache": volume},
)
def publish_checkpoint():
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    output = Path("/vol/hosting-cache/quantized/qwen35-9b-base-fp8/" + REVISION)
    report = json.loads((output / "quantization-report.json").read_text())
    if report["source_revision"] != REVISION or report["fp8_tensors"] != 248:
        raise RuntimeError("Saved checkpoint does not match the qualified conversion recipe")
    api = HfApi()
    try:
        existing = api.model_info(TARGET)
    except RepositoryNotFoundError:
        existing = None
    if existing and any(s.rfilename != ".gitattributes" for s in existing.siblings):
        raise RuntimeError("Target already contains artifacts; refusing to replace them")
    api.create_repo(TARGET, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=TARGET,
        folder_path=str(output),
        commit_message="Add pinned Qwen3.5-9B-Base FP8_DYNAMIC conversion",
    )
    report["published_revision"] = commit.oid
    print("PUBLISH_RESULT", json.dumps(report), flush=True)
    return report


@app.local_entrypoint()
def main(
    publish: bool = False,
    publish_existing: bool = False,
    output: str = "/tmp/flash-base-fp8-conversion.json",
):
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise RuntimeError("A Hugging Face credential is required")
    if publish_existing and not publish:
        raise ValueError("--publish-existing requires --publish")
    secrets = [modal.Secret.from_dict({"HF_TOKEN": token})]
    report = None
    if not publish_existing:
        report = convert.with_options(secrets=secrets).remote()
    if publish:
        report = publish_checkpoint.with_options(secrets=secrets).remote()
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
