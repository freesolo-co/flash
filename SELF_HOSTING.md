# Mithril Spot training

Set `[gpu] spot = true` to use Mithril's Compute API. Modal remains the default;
Spot never falls back to Modal or on-demand reservations.

## Operator setup

Set these secrets/configuration on the control plane:

- `FLASH_MITHRIL_API_KEY`: Mithril API key with access to the intended project.
- `FLASH_MITHRIL_PROJECT_ID`: project FID from `GET /v2/projects`.
- Optional `FLASH_MITHRIL_REGION`: pin one auction region from `GET /v2/spot/availability`.
  Leave it unset to admit offers from every region: Flash bids in the region of the cheapest
  qualifying offer, and each bid keeps its own region. Ownership is then the project; attempts
  launched while a region was pinned stay recognized after unpinning (checked against their bid).
- Optional `FLASH_MITHRIL_IMAGE_VERSION`: GPU host image version FID. Otherwise,
  Mithril selects the region's default image. Use an image with Docker, a compatible
  NVIDIA driver, NVIDIA container toolkit, Python 3 and outbound registry/Azure access.
- Optional `FLASH_MITHRIL_SSH_KEY_IDS`: comma-separated existing SSH key FIDs for diagnostics.

Host images are selected through the Mithril API; no provider CLI is needed.
Keep the existing worker image, Azure artifact and runtime-secret configuration.
The Mithril API key stays on the control plane; workers receive only their runtime
credentials. Use a dedicated project for Flash training.

**Migration:** finish or cancel all existing legacy Spot runs using the previous
release before deploying this change. Persisted legacy resource IDs and ownership
fingerprints are deliberately rejected by the Mithril provider. Installing this
release does not migrate or delete existing legacy VMs.

## Admission, prices and recovery

| GPU | Fixed estimate / charge rate | Start below | Bid cap per GPU-hour |
| --- | ---: | ---: | ---: |
| H100 | $1.00 | $1.01 | $2.01 |
| H200 | $1.00 | $1.01 | $2.51 |
| B200 | $4.00 | $4.01 | $5.01 |

The policy lives in `flash/providers/mithril/pricing.py`. Flash joins the authenticated
instance-type catalog to live auctions (every region, or only `FLASH_MITHRIL_REGION` when set), admitting only advertised SXM
shapes with sufficient VRAM, capacity and local scratch. Supported counts are 1 and 8;
a listed eight-GPU auction never implies single-GPU availability. B200 remains an
eligible policy class but cannot be selected unless Mithril actually offers it.
Fitting rentals retain H100, H200, B200 priority order.

Mithril reports prices and accepts bids **per instance-hour**. Flash divides the
auction price by the actual GPU count for admission and multiplies the hold cap by
that count for the bid. It requests one instance, not one instance per GPU.
Each cap includes a one-cent per-GPU-hour offset. Admission uses strict `<`
thresholds: an eight-H200 price of $8.00 passes the $1.01/GPU-hour entry threshold,
while $8.08 does not. Current market prices may leave all jobs queued.
Malformed or unavailable prices fail closed. Jobs recheck every 60 seconds without
creating bids or consuming training retries; waiting respects cancellation and the
run deadline. The selected GPU count and disk requirement are rechecked before launch.
Project Spot quotas are checked for the exact instance type, including usage by
other jobs. Exhausted or unreadable quotas keep the job queued. Missing shape
quota rows defer to server enforcement. Definitively refused creates with confirmed
quota exhaustion requeue only after reconciliation finds no bid; unknown POST
outcomes remain fail-closed. Organization-level limits remain server-enforced.

Customer charges retain the accepted Spot quote ($1 per GPU-hour for H100/H200 or
$4 for B200), with the existing completed-work/session-time settlement rules.
These estimates are not Mithril invoices and exclude network, taxes and other charges.
The bid cap is higher than the entry threshold and bounds subsequent compute pricing.

Each attempt owns one bid. Its durable resource handle stores the **bid FID**, so
cancellation also stops unallocated bids and automatic reallocation. Cleanup confirms
both bid termination and stopped instances. An ambiguous create is reconciled by
attempt name and ownership; it is never blindly retried. Preemption or relocation
ends the attempt, cancels the bid, and lets Flash resume a fresh attempt from its
latest fully uploaded trainer checkpoint. SIGTERM interruption is treated as Spot
preemption, not a budgeted worker fault, including custom jobs with zero retries. Workers atomically claim their attempt
in private Azure storage, preventing a duplicate or replacement host from running
the same attempt twice. Flash does not install a service to restart training after
Mithril reboots a preserved boot disk.

Spot saves the first and final completed step, explicit `train.save_at_steps`, and
before another step of similar duration would exceed ten minutes of unsaved work.
Steps lasting ten minutes or longer save after every step. This is a recovery target;
indivisible steps and asynchronous uploads may exceed it. Do not set `train.save_every`
for Spot. Work since the latest complete upload may be repeated after preemption.

## Worker startup and custom algorithms

The startup script downloads checksum-verified launch data and the Flash capsule
from private Azure storage. Large payloads stay outside Mithril's 10,000-character
startup-script limit. Runtime files are root-only; restrict access to bid metadata,
startup scripts and the artifact container because they contain scoped credentials.
Docker image layers and training scratch use the host's ephemeral `/mnt/local`
storage. Flash creates no persistent Mithril volumes. Uploaded Azure checkpoints
remain the durable recovery source.

Workers stop at their absolute deadline and shut down the host on exit. The control
plane must remain available to cancel the bid and confirm resource cleanup; host
shutdown alone does not cancel a Mithril bid. Normal completion, cancellation and
orphan reconciliation all cancel the owned bid.

`algorithm = "custom"` supports exactly one Spot GPU. The worker publishes port 8000
through Mithril's `foundrypf` utility and serves the existing per-attempt pinned TLS
session. The controller receives the running instance's public IPv4 address, session
token and certificate from `GET /v1/runs/{id}/custom`. After preemption, it reconnects
to a new attempt and invokes `main(client)` again; continue from `client.step` and
`client.restored_state`. Custom billing still accumulates session time across attempts.

References: [Compute API](https://docs.mithril.ai/compute-api/compute-api-reference/spot),
[Spot lifecycle](https://docs.mithril.ai/compute-and-storage/spot-bids),
[startup scripts](https://docs.mithril.ai/compute-and-storage/startup-scripts), and
[port forwarding](https://docs.mithril.ai/compute-and-storage/managing-open-ports).

To qualify a configured account with paid resources, prepare a disposable private
Azure artifact repository and run:

```sh
python scripts/smoke_mithril_sft.py --run-paid-test --repo owner/private-test-dataset
```

This rents one H100 at a time and requests cancellation after checkpoint 1 is durable.
The small first run may finish step 2 before cancellation takes effect. A new bid
restores the latest checkpoint with the target extended to step 4, proving that
recovery executes additional optimizer steps in either case. Each attempt is limited to
45 minutes. The harness cancels its bids; test artifacts remain in private storage.
It will stay queued if the regional H100 price is not below the unchanged entry cap.

For an explicitly paid qualification when current prices exceed the normal entry
threshold, select `--gpu H200 --entry-price-cap 2`. This changes admission only in
the smoke-test process and refuses caps above that GPU's existing hold cap. It does
not change production pricing. `--model` selects the model to qualify.

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
