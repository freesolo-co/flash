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

## Custom algorithms on Spot

`algorithm = "custom"` runs accept `[gpu] spot = true` on exactly one GPU (custom layer
sharding stops at two cards and Spot presets are 1 or 8). Without a provider tunnel, the local
controller connects straight to the VM:

- The worker serves its session over HTTPS on port 8000, published from the container. Allow
  inbound TCP 8000 to Flash's Spot VMs from wherever controllers run (Nebius's default security
  group allows all ingress).
- Each attempt has its own Ed25519 TLS identity and session token, both derived from
  `FREESOLO_INTERNAL_KEY`, so nothing new is stored. `GET /v1/runs/{id}/custom` returns the VM's
  public address, the token and the certificate; the controller trusts only that certificate.
- Recovery saves are automatic and time-based, as for other Spot jobs: the first step, every
  `save_at_steps` entry, the final step, and before another typical step would exceed 10 minutes
  of unsaved work. `train.save_every` must be omitted.
- After preemption, the replacement attempt restores the run's newest complete custom
  checkpoint (adapter, optimizer state and `algorithm_state`). The controller notices its
  session failed, waits for the new attempt's session, and calls `main(client)` again; write
  `main` to continue from `client.step` and `client.restored_state`. A failure while the same
  session is still alive is the algorithm's own error and cancels the run as before.
- Billing is session time across all attempts, as a fraction of the accepted quote.

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
`min_capacity`, `max_capacity`, and an optional `lambda_gpus` allowlist; redeploy after editing.
All Qwen models prefer Lambda and use Modal when Lambda capacity is unavailable.
For example, to keep one production 9B Base replica warm and allow up to three:

```python
"Qwen/Qwen3.5-9B-Base": ModelPolicy(
    min_capacity=1, max_capacity=3, lambda_gpus=("A100-40GB", "H100"),
),
```

The shipping settings are:

| Model | Preferred backend | Main minimum | Dev minimum | Maximum |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | Lambda A10 | 1 | 0 | 2 |
| Qwen3.5-9B-Base | Lambda A100-40GB / A100-80GB / H100 | 2 | 0 | 4 |
| Other Qwen models | Lambda with per-model GPU options | 0 | 0 | 2 |
| DeepSeek | Modal four-B200 group | 0 | 0 | 4 main / 1 dev |

Capacity counts model replicas, not GPU cards, and applies independently in each
environment. Every dev GPU pool scales to zero when idle, regardless of the configured
production minimum. Requests still start workers and trigger demand-based scaling;
idle workers drain after the demand and scale-down windows expire. CPU routing and
fleet control services remain available to handle cold requests.

Qwen maximums cover Lambda and Modal together, with one temporary extra replica
allowed during handoff. Modal fallback always has a zero idle minimum. The fleet
controller reconciles demand every minute and moves traffic to Lambda once its
workers are ready. Production retains the configured minimums.

`lambda_gpus` restricts the model's qualified hardware. An empty tuple permits the
existing qualified alternatives, choosing the cheapest available US instance from
Lambda's live catalog. DeepSeek retains its Modal backend and smaller dev maximum;
its main rolling deployment can temporarily double the maximum for a replacement pool.

The deployment workflows require `LAMBDA_API_KEY` and deploy the fleet controller
before serving. `/healthz` exposes `provider_by_model`, `capacity_by_model`
(including `dev_min: 0` and the optional `dev_max`), and `gpu_by_model`.
These settings control hosted inference; training and customer-owned deployments
are separate. Lambda capacity waiting and retry intervals are unchanged.
