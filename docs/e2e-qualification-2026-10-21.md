# Main-branch E2E qualification — October 21, 2026

This audit exercised real Modal GPUs, upstream model weights, private Hugging Face artifacts, Flash's standalone control plane, and customer-owned serving deployments. The authorized Modal budget was $400. This is smoke qualification, not a model-quality evaluation, load test, or certification of every hardware/model combination.

## Source and runtime identities

- Release tested: `main` at `eac8e6fb6f296562ddaf8976f1d04a72525389ed` (1.3.8).
- Fix branch starts from `dev` at `46b6c623`; the runtime change pins the training image, and the package version advances to 1.3.9.
- Required trainer: official `verl-project/verl@ddf700725c34bb8966685c8ecf5ca36a2cc54a35`.
- Fixed training image: `ghcr.io/freesolo-co/flash-worker@sha256:34c84092f87b9ae43457620a8e5c0dc2e3dc30fdd5de5b4300c1eeb384e9a066`.
- Serving image: `ghcr.io/freesolo-co/freesolo-flash-serve@sha256:fc8ed231211562b3c67ca80ffbf7fb3e992aad6a42087a16c0dd26c1e249a758`.
- Base-runtime tests overlaid the tested `main` serving source onto that serving image. Customer-owned HTTPS deployments used the image itself, with a newly trained adapter and immutable artifact provenance.

Model revisions were fixed for the runs:

| Model | Training revision | Serving revision |
| --- | --- | --- |
| Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Freesolo-Co FP8 `878d83edd1b920e1206ab324b121e3a5561f7c26` |
| Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Qwen FP8 `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Qwen3.6-35B-A3B | `995ad96eacd98c81ed38be0c5b274b04031597b0` | Same BF16 revision |

## Confirmed release failure and fix

Unmodified `main` selected `flash-worker:cu128-sm90` on H100/H200. That image contained the retired Freesolo Verl fork at `32d6200de81dc484893baf8b9cf30297ebe7fa49`. Flash correctly rejected it because it requires the pinned, unmodified upstream trainer. Real 9B text SFT, 9B image SFT, 9B GRPO, and 27B SFT all failed before the first optimizer update. The failure occurred after paid model downloads in several attempts.

The PR pins the immutable upstream-compatible image already documented in [upstream qualification](upstream-verl-qualification.md). Mutable per-architecture tags are no longer selected merely because an architecture has a cache-bake entry. Native JIT remains available. Restoring architecture-specific images requires new qualified digests; the pin is a correctness fix, not a startup performance improvement. Hardware outside the live matrix below remains covered only by selection tests and existing qualification evidence.

Two qualification-test defects also surfaced only with the GPU dependencies installed:

- The capability probe test assumed the pytest interpreter lacked FlashInfer and W&B. It now probes a disposable standard-library-only virtual environment, so the expected missing dependencies are explicit.
- The real-Qwen XGrammar replay test downloaded Hub assets without an opt-in and used a mutable model revision. It now requires `FLASH_LIVE=1`, pins the tested model revision, and loads the downloaded tokenizer/config locally, matching worker prefetch and avoiding interference from the offline metadata fixture. The choice, regex, and JSON replay cases passed with real XGrammar.

## Offline validation

| Source | Interpreter / suite | Passed | Skipped | Failed |
| --- | --- | ---: | ---: | ---: |
| main | Python 3.12, non-wallclock | 10,548 | 181 | 0 |
| main | Python 3.11, non-wallclock | 10,548 | 181 | 0 |
| main | Python 3.12, serial wallclock | 25 | 2 | 0 |
| fix | Python 3.12, non-wallclock | 10,553 | 181 | 0 |
| fix | Python 3.12, serial wallclock | 25 | 2 | 0 |
| fix | Linux, actual pinned PyTorch/Verl stack, selected integration suites | 1,157 | 1 | 0 |

Ruff lint/format, file-size and function-size gates, code-layout gate, and the upstream Verl commit reachability check were also run. Local skips included 52 missing-PyTorch cases, 44 pinned-Verl cases, 28 Linux process-group cases, and opt-in live/conformance tests. Additional worker-image validation is reported separately. Skips are retained explicitly; the test count is not a claim that every optional integration ran.

## Live matrix

| Path | Model / allocation | Result |
| --- | --- | --- |
| Text SFT + final CPU publication + fresh-process recovery | Qwen3.5-9B, H100 | Passed; durable step 2 restored after removing DONE |
| Image SFT + final CPU publication | Qwen3.5-9B, H100 | Passed; finite changed language and vision LoRA weights |
| Text SFT + final CPU publication | Qwen3.8-27B, H200 | Passed |
| CLI → standalone server → SFT → artifacts → terminal status | Qwen3.6-35B-A3B, H200 | Passed; run reached `done` |
| GRPO + final CPU publication | Qwen3.5-9B, H100 | Passed; two updates and changed adapters |
| Distributed GRPO + final CPU publication | Qwen3.5-9B, two H100s | Passed; two active training ranks, finite gradient norms 0.28125 and 0.283203125 |
| Explicit two-card SFT allocation | Qwen3.5-9B, two H100s | Passed with documented one-rank fallback for the tiny unpacked batch; not distributed SFT qualification |
| Real-teacher multimodal, multi-turn OPD | Qwen3.5-9B, two H100s | Passed; teacher and student on separate devices |
| HTTPS serving, trained text adapter | Qwen3.5-9B, L40S | 16/16 checks passed; teardown `absent` |
| HTTPS serving, trained vision adapter | Qwen3.5-9B, L40S | 16/16 checks passed; teardown `absent` |
| HTTPS serving, trained text adapter | Qwen3.8-27B, H100 | 15/15 checks passed, including unsupported-tool rejection; teardown `absent` |
| HTTPS serving, trained text adapter | Qwen3.6-35B-A3B, H200 | 15/15 checks passed, including unsupported-tool rejection; teardown `absent` |
| Base serving runtime | 9B / 27B / 35B | Positive feature cases passed; see harness corrections below |

All four customer-owned HTTPS deployments passed: **62 checks total**, with `absent` teardown receipts for each. These checks include exact checkpoint provenance, authenticated inventory, unauthorized rejection, completion usage, raw SSE termination/usage, JSON schema, multiple choices, logprobs, red PNG and blue JPEG semantics, tool lifecycle where supported, and invalid-request rejection.

### Serving features

Real vLLM base-runtime inference covered Qwen3.5-9B (FP8 serving base on L40S), Qwen3.8-27B (FP8 on H100), and Qwen3.6-35B-A3B (BF16 on H200):

- Text generation and streaming, explicit thinking mode and thinking streaming.
- JSON schema and streamed JSON, regex and choice constraints.
- Multiple choices, streamed multiple choices, token log probabilities.
- PNG, JPEG and WEBP data-URL images; two images in one request; image streaming and image JSON output. GIF is outside the supported image contract and must be rejected; its fixture is negative coverage, not evidence of GIF support.
- Concurrent requests, runtime health, and explicit engine shutdown.
- Real 9B generated tool calls, streamed tool calls, tool-result replay, and `tool_choice="none"`.
- Invalid/corrupt images and image-count limits; unsupported feature combinations.

The documented tool contract supports the 9B family in this matrix. Unsupported tools on the other models must be rejected rather than counted as a missing supported feature.

Two initial negative-test assertions checked an exact exception name and mistakenly classified the correct `MultimodalRequestError` rejections as failures. The corrected harness accepts the documented error hierarchy. An initial internal-runtime tool fixture also omitted defaults supplied by the public HTTP parser; its corrected rerun passed. These were harness mistakes, not product fixes, and the original logs are retained.

### Training and lifecycle

Training smokes use two optimizer steps, rank-8 LoRA, explicit saves at steps 1 and 2, and small deterministic fixtures. Image SFT uses red, blue, green, and white images. These fixtures establish execution and artifact behavior, not visual reasoning accuracy.

A local-teacher OPD test used two H100 GPUs: one real 9B teacher and one student. Multimodal, multi-turn training completed two updates, with gradient norms 2.921875 and 2.59375, 43 teacher bridge requests, and different step-1/step-2 adapter hashes. The maximum observed adapter delta was 0.0212588. This validates the trainer/teacher bridge path; it does not validate the hosted Parasail teacher service or its billing.

The Modal infrastructure smoke passed CUDA/NCCL across two L4 GPUs, duplicate sandbox adoption, persistent cache reuse, and owned-resource termination. Standalone CLI login, identity, dry-run, cost estimate, submission, running status, and cancellation were exercised against a local server with real Modal workers.

## Documentation reviewed and coverage boundaries

The audit read README, CONTRIBUTING, SELF_HOSTING, the serving contract, trainer runtime, upstream Verl qualification, checkpoint persistence, CPU checkpoint handoff, weight prefetch, and background task lifecycle documentation. Those documents informed the image/tool restrictions, checkpoint tests, and cleanup requirements.

Not established by this run: hosted production authentication and billing, Parasail availability, every catalog model, every GPU architecture, large-dataset convergence, long-context limits, warm-start quality, structured-output training beyond replay contracts, sustained throughput, long-running preemption/soak behavior, and every tenant-isolation failure mode. The CPU suite covers many failure paths, but mocked coverage is not live evidence. No existing production app was redeployed or stopped.

## Budget and cleanup

The resource model totals approximately **$52**, including conservative time/resource allowances for serving and incomplete failed-attempt timestamps, plus $10 for infrastructure smoke, CPU uploads/tests, image builds, and short-lived storage. Allowing additional headroom for unmeasured bursts and region premiums keeps the working estimate under $80 of the $400 budget. This is not the Modal invoice: CPU/RAM bursts, placement premiums, and storage were not metered through a billing API. The CLI customer quote is a different quantity and was not used as provider spend.

Training cleanup confirmed zero active owned training sandboxes and no remaining per-run checkpoint handoff volumes. The four audit cache/test volumes were deleted, the audit application was stopped, and the local control plane was stopped. Serving cleanup receipts are listed with each result above. Shared production apps and the default shared weight cache were not removed. Private Hugging Face artifacts were retained for inspection.

## Reproduction and evidence

Baseline commands:

```bash
uv sync --extra server --extra serving --dev --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q -m 'not wallclock'
uv run pytest -q -m wallclock
uv run python scripts/check_file_size.py
uv run python scripts/check_function_size.py
uv run python scripts/check_source_layout.py
uv run python scripts/check_verl_pin_exists.py
```

Paid infrastructure and fresh-process checkpoint recovery use the repository scripts:

```bash
uv run python scripts/smoke_modal_training.py --run-paid-test
uv run python scripts/smoke_modal_sft.py --run-paid-test \
  --repo <disposable-private-dataset> --gpu H100
```

Additional matrix drivers, JUnit XML, raw worker logs, runtime JSON results, and cleanup receipts are retained in `flash-e2e-artifacts-20261021/` alongside the worktrees; its manifest records file hashes. The launch drivers are audit-specific and require new disposable resource names/repositories before reuse. Credentials and serving bearer-token files must not be published with those artifacts. Private model artifacts remain in the audit's private dataset repositories for inspection.

Proposed follow-ups and measurements are in [optimization opportunities](optimization-opportunities-2026-10-21.md).

The subsequent [October 22 kernel-cache requalification](kernel-cache-requalification-2026-10-22.md) records live verification and immutable image selection for architecture-specific caches.
