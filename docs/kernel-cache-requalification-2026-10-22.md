# Kernel-cache requalification — October 22, 2026

This follow-up to [the October 21 qualification](e2e-qualification-2026-10-21.md) restores architecture-specific worker caches using qualified immutable digests. PR #1454 merged into `dev` as `034746f5`; its compatible base digest remains the fallback for workers without a GPU or an architecture without a qualified image.

## Image identity and acceptance gates

The candidate images were already published by the [automatic re-layer job](https://github.com/freesolo-co/flash/actions/runs/39725927119). This audit independently ran them on real Modal GPUs. A successful re-layer is not itself a GPU qualification.

All six images declare source `eac8e6fb6f296562ddaf8976f1d04a72525389ed`, cache fingerprint `b7c2efcc70f0f87c`, and base fingerprint `b79b5a07c8e30e5b`. Both fingerprints match this checkout. Every live probe ran `validate_upstream()` in `/opt/verl-venv/bin/python` and verified official `verl-project/verl@ddf700725c34bb8966685c8ecf5ca36a2cc54a35`, Python 3.12, PyTorch 2.10.0+cu128, and CUDA 12.8.

The trainer must accept the baked cache metadata, then complete `kernel_warmup.warmup(arch=..., verify_cache=True)` from a fresh process. The guard rejects compilation through the pinned Triton, PyTorch C++ extension, Hopper TileLang, and FlashInfer entry points. All forward/backward and variable-length GDN groups must succeed; Hopper also checks the FlashInfer GDN prefill AOT library. A compilation attempt fails the probe even if a backend catches its exception.

## Fresh-process GPU results

| Architecture | Live GPU | Reload seconds | Result |
| --- | --- | ---: | --- |
| sm80 | A100-80GB | 9.9 | Passed, no compilation |
| sm86 | A10 | 8.3 | Passed, no compilation |
| sm89 | L40S | 14.8 | Passed, no compilation |
| sm90 | H100 | 16.5 | Passed, no compilation |
| sm100 | B200 | 12.1 | Passed, no compilation |
| sm120 | RTX-PRO-6000 | 6.6 | Passed, no compilation |
| sm90 | H200 | 17.8 | Passed, no compilation |

These are kernel replay times after interpreter startup, not end-to-end job startup benchmarks. The cache covers the warmup's kernel shapes; new model shapes and other compiler paths can still JIT. H100 and H200 were tested separately. L4 and A100 40 GB share qualified architectures but were not individually exercised.

## Immutable image inventory

All references use repository `ghcr.io/freesolo-co/flash-worker`:

| Architecture | SHA-256 digest |
| --- | --- |
| sm80 | `sha256:10ddd08a8c3fa4794f5370a143b433d38597e3fa15860313c1b2eb0340d62e2e` |
| sm86 | `sha256:14252986499b9f8ca36273d28400171cf77bebc27b633084a5efa8d386be70ef` |
| sm89 | `sha256:5a435995295d942fffcafc0805fa199451ad24a77c8e4059477230005acd1a84` |
| sm90 | `sha256:c794abb352bb99c8a90e58f4f1f8f4b56012a1c00fcfe607dd8a037d0f119fb8` |
| sm100 | `sha256:341e0724f1f1730ff8f8ea0492ee56b16df3e374bf3464528a59c55cc67025d3` |
| sm120 | `sha256:c5bacd1780becbf383f3e72b5abf9e27b133e4b49cab6817be8d13e6b399ac85` |

## Training and artifact checks

All runs used `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, two optimizer steps, rank-8 LoRA, and explicit saves at steps 1 and 2. The standard worker launch path accepted the baked caches.

| Path | GPU | Updates | Checkpoint publication | Adapter checks |
| --- | --- | ---: | --- | --- |
| GRPO | H100 | 2 | Passed | Finite and changed |
| SFT | A100 SXM | 2 | Passed | Finite and changed |
| SFT | B200 | 2 | Passed | Finite and changed |
| SFT | H200 | 2 | Passed | Finite and changed |
| SFT | L40S | 2 | Passed | Finite and changed |
| SFT | RTX Pro 6000 | 2 | Passed | Finite and changed |

All 496 adapter tensors changed between the two saved steps in each run; all tensors were finite, and all 248 LoRA-B tensors were nonzero. The largest observed step-to-step change was `6.103515625e-05` for each SFT run and `1.0132789611816406e-05` for GRPO. Checkpoint inspection used recorded immutable Hugging Face dataset revisions. These assertions establish real optimizer and artifact behavior, not model quality.

## Selection and offline verification

`worker_image_for_gpu` selects only the explicit qualified digest map. The bake inventory alone cannot enable an image. GPU validation remains enforced; the existing upstream-compatible base remains the fallback. Environment variables cannot override the selected image. Both package version fields advance to 1.3.10.

Python 3.12 passed **10,599 non-wallclock tests** (181 skips) and **25 serial wallclock tests** (2 skips). The 88 selected image/cache tests also passed. Ruff lint/format, file/function size gates, source layout, and upstream commit reachability passed.

Use the same offline environment as CI:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run pytest -q -n 2 --dist loadfile -m 'not wallclock'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run pytest -q -m wallclock
```

The first local suite omitted these CI environment settings. Four dataset tests attempted download-count telemetry and were correctly blocked by the offline HTTP guard. That incomplete run was stopped; the full rerun with the CI environment passed without product changes.

## Scope, evidence, and cleanup

This is cache and training smoke qualification, not a repeat of the previous serving matrix or a model-quality evaluation. A10's 24 GB memory is below the catalog's single-GPU training floor, so its qualification covers provenance and real forward/backward kernel replay; no full-model single-A10 claim is made.

Two initial source-packaging attempts (A100 and A10) failed before GPU launch because local bytecode changed while Modal was building the audit image. Retrying with a frozen source tree excluding bytecode succeeded. Original failure logs are retained.

Raw probe logs, immutable image metadata, training specifications, worker logs, checkpoint checks, JUnit XML, and cleanup receipts are retained beside the worktree in `flash-requalification-artifacts-20261022/`. The manifest hashes each artifact. Audit launchers use disposable resource names and operator credentials; adapt those names before reuse.

Cleanup confirmed zero active audit training or probe sandboxes, removed the owned weight-cache volume and all six runs' checkpoint handoff volumes, and stopped the audit app. Private model artifacts remain for inspection. Existing production applications and shared caches were not changed.

Modeled incremental provider spend is approximately **$17**, using [Modal pricing](https://modal.com/pricing) checked October 22. This includes conservative probe durations (even image-import time before GPU allocation), GPU and CPU/RAM reservations, and a $10 allowance for image builds, CPU checkpoint uploads, storage, and unmetered bursts. It is an estimate, not an invoice, and remains below the prior $400 audit budget including the previous approximately $52 estimate.
