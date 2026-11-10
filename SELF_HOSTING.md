# Nebius Spot training

Set `[gpu] spot = true` to opt into Nebius preemptible VMs. Modal remains
the default; Spot never falls back to Modal or Nebius on-demand instances. Only
H100, H200 and B200 are offered. RTX Pro 6000 and L40S are excluded.

| GPU | Start below / fixed estimate | Keep running through |
| --- | ---: | ---: |
| H100 | $1.00 | $1.50 |
| H200 | $1.00 | $2.00 |
| B200 | $4.00 | $5.00 |

The table lives in `flash/providers/nebius/pricing.py`. Live regional prices
admit fitting rentals in H100, H200, B200 priority order, even when H200 is cheaper.
The entry thresholds price Flash's estimates; running prices can be higher. Estimates exclude disk, network and taxes and are not Nebius invoices.
Nebius presets are 1 or 8 GPUs. All rented cards contribute to the price.

Customer charges use the accepted Spot-priced quote: $1 per GPU-hour for H100/H200
or $4 for B200 in the work estimate. Completion charges that quote; early completion
or cancellation charges its completed-work fraction, capped at the accepted quote.
Recovery preserves Spot pricing even when a saved resource record lacks its provider.
These are fixed Flash Spot rates, not a pass-through of fluctuating Nebius invoices.

Flash reads Nebius's public spot-price API and queues jobs if prices are over the
cap, absent, stale, or unavailable. It checks again every 60 seconds, without
consuming training retries or creating a VM. Queueing respects cancellation and
the run's wall-clock deadline. Admission is rechecked before launch, and every VM
also carries a Nebius maximum-price policy so a later price rise cannot silently
switch to uncapped pricing. On preemption, Flash provisions another eligible Spot
VM and resumes the latest durable trainer checkpoint using its existing retry
budget. Spot uses a greedy scheduler based on the latest completed step duration and checkpoints before another similar
step would exceed **10 minutes** of unsaved training (for example, every 2 five-minute
steps or 10 one-minute steps). It also saves the first and final completed step.
Steps lasting 10 minutes or longer checkpoint after every step. This is a recovery target: indivisible steps and asynchronous uploads can exceed it.
Do not set `train.save_every` for Spot. Optional `train.save_at_steps` exports
cannot suppress automatic recovery saves.
Work since the latest fully uploaded checkpoint may be repeated, including any
checkpoint whose upload was still in progress. Portal is not used.

On the control-plane host:

1. Install and authenticate the current Nebius CLI with access to the intended
   USD-billed project, compute resources and billing pricing policies. Freesolo's
   published control-plane images already contain a pinned CLI (`INSTALL_NEBIUS=true`;
   a plain `docker build .` omits it). Flash calls it non-interactively with
   `--no-browser`, so give the container a service-account profile, not a browser
   login: for example, mount a root-only directory holding `config.yaml` and the
   authorized-key PEM at `/root/.nebius`.
2. Set `FLASH_NEBIUS_PROJECT_ID`, `FLASH_NEBIUS_REGION`,
   `FLASH_NEBIUS_SUBNET_ID`, and `FLASH_NEBIUS_IMAGE_ID`. The region must match
   the project and subnet. The configured GPU host image must have Docker,
   a suitable NVIDIA driver and NVIDIA container toolkit installed, plus outbound
   access to the worker container registry and Azure Blob Storage. B200 requires the
   driver/CUDA support of Flash's configured Blackwell worker image.
3. Retain the existing worker image, artifact and runtime-secret configuration.
   The VM launches the same Flash worker capsule and training container as Modal.
4. Start the control plane and submit with `spot = true`. `flash gpus`
   lists the fixed Spot estimates alongside Modal rates. Set `spot = false` for regular
   Modal training. Legacy `provider = "spot"` is also accepted.

Supported platform IDs are `gpu-h100-sxm`, `gpu-h200-sxm`, and `gpu-b200-sxm`;
only platforms offered in the configured region can be admitted. Managed boot
disks use fast ephemeral Network SSD Non-replicated storage, rounded to 93-GiB
allocation units with a 279-GiB minimum for the image, model and checkpoints.
They are deleted with the VM; fully uploaded Azure Blob Storage checkpoints are the
durable recovery source. Runtime secrets are in root-only cloud-init files;
restrict access to VM metadata and the image accordingly. Use a dedicated project
for Flash training resources. The worker has a deadline shutdown, and normal
cancellation, completion and orphan reconciliation delete the VM and its disk.

References: [Nebius spot prices](https://docs.nebius.com/signup-billing/spot-prices),
[maximum-price policies](https://docs.nebius.com/signup-billing/manage-pricing-policy),
and [preemptible VM lifecycle](https://docs.nebius.com/compute/virtual-machines/preemptible).

To qualify a configured Nebius account, create a disposable private Azure artifact repository, configure Azure artifact storage locally, and run:

```sh
python scripts/smoke_nebius_sft.py --run-paid-test --repo owner/private-test-dataset
```

This rents one H100 at a time, interrupts a two-step SFT job after checkpoint 1
is fully uploaded, then verifies that a replacement VM resumes and finishes step 2. Each attempt is limited to 45 minutes to allow cold image downloads and full-state checkpoint transfers.
The harness deletes its VMs and managed disks; test artifacts remain in the private
repository. GPU quota and active billing are required. Price thresholds, GPU
priority, cancellation and adaptive checkpoint timing are covered by offline tests.

# Hosted model capacity and providers

Edit `MODEL_POLICIES` in `flash/serving/app/inference_provider.py`. Each model has
one row with `provider`, `min_capacity`, and `max_capacity`; redeploy after editing.
For example, to keep one 9B Base replica warm and allow up to three:

```python
"Qwen/Qwen3.5-9B-Base": ModelPolicy(
    "lambda", min_capacity=1, max_capacity=3, lambda_gpus=("H100",),
),
```

The shipping settings are:

| Model | Provider | GPU pin | Minimum | Maximum |
| --- | --- | --- | ---: | ---: |
| Qwen3.5-0.8B | Lambda | A10 | 1 | 2 |
| Qwen3.5-9B-Base | Lambda | H100 | 1 | 2 |
| Other Qwen models | Modal | Existing per-model GPU options | 0 | 2 |
| DeepSeek | Modal | Existing four-B200 group | 0 | 4 main / 1 dev |

Capacity counts model replicas, not GPU cards, and applies independently in each
environment. A positive minimum starts the parameterized pool during router startup and keeps
it warm; GPU availability and worker startup still apply. Lambda's CPU proxy owns one VM
per replica. Its lease guard enforces the same maximum per model and source
revision; rolling releases can temporarily overlap old and new replicas.

`lambda_gpus` restricts the model's qualified hardware. A10 is the cheapest
currently [listed Lambda shape](https://lambda.com/pricing) supported by the 0.8B
serving profile (24 GB, listed at $1.29/hour on 2026-11-09). An explicit
pin waits for capacity instead of falling back to another GPU. An empty tuple
permits the existing qualified alternatives, choosing the cheapest available US
instance from Lambda's live catalog.

For Qwen, `provider="all"` enables both providers: the maximum is split across
them, with the odd slot and warm minimum assigned to Lambda first. Routing checks
capacity before dispatch; an already dispatched generation is never retried on
the other provider. Use `modal` or `lambda` to select just one provider. DeepSeek
supports Modal only; its optional `dev_max_capacity` preserves the smaller dev
pool and must be at least its minimum. Its main rolling deployment can temporarily
double the maximum for a replacement pool.

The deployment workflows require `LAMBDA_API_KEY` whenever any model uses Lambda
and deploy the fleet controller before serving. `/healthz` exposes
`provider_by_model`, `capacity_by_model` (including the optional dev override), and
`gpu_by_model` so the deployed configuration can be inspected. These settings
control hosted inference; training and customer-owned deployments are separate.
