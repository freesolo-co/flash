# Shared training qualification — 2026-10-22

This qualifies the experimental token API described in [shared-training.md](shared-training.md).
It does not qualify the existing managed `flash train` path against this backend.

## Real tensor and transport checks

The 29 shared-training checks passed in each of these environments:

| Environment | PyTorch / PEFT | Transformers | Result |
| --- | --- | --- | --- |
| macOS arm64, Python 3.12.8, source checkout | 2.10.0 / 0.19.0 | 5.10.2 | 29 passed |
| macOS arm64, Python 3.11.13, installed 1.3.24 wheel | 2.10.0 / 0.19.0 | 5.12.1 | 29 passed |
| NVIDIA L4, Python 3.12, qualified worker image | 2.10.0 / 0.19.0 | 5.12.1 | 29 passed, 24.63 seconds |

The Python 3.11 run used a fresh environment and copied test files outside the
repository, ensuring imports came from the installed wheel. The L4 run included
the actual standard-library HTTP client over loopback, ASGI recipe calls with a
resident model, exact isolated-versus-interleaved tensor comparisons, checkpoint
continuation, and admission/accounting/recovery tests. The L4 source snapshot was
commit `45909fee882ea6f1f3749a36815c791ec47c4e51`; subsequent changes update dependency
contract tests, synchronize fault-test setup, and record these results.

## Qwen 9B on an H100

Model: `Qwen/Qwen3.5-9B`, immutable revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Device: NVIDIA H100 80GB HBM3. PyTorch 2.10.0, PEFT 0.19.0,
Transformers 5.12.1, bfloat16 base weights, LoRA rank 8 / alpha 16 on q/v projections.

For each objective, the script created two independent resident adapters, trained
each, checked the peer adapter and optimizer remained unchanged, captured a
checkpoint, then restored it into a third adapter. The restored next update
matched the uninterrupted update exactly, including weights, loss, and gradient
norm. Frozen base parameters remained unchanged and had no gradients.

| Objective | First loss | Next loss | Peer/optimizer isolation | Exact resume | Base unchanged |
| --- | ---: | ---: | --- | --- | --- |
| SFT | 1.1349017620 | 1.1221765280 | Pass | Pass | Pass |
| GRPO | -0.0009765625 | -0.0009854923 | Pass | Pass | Pass |
| OPD | -0.3344352841 | -0.1955014169 | Pass | Pass | Pass |

These are tiny correctness updates, not quality metrics. The Qwen GRPO update uses
a synthetic advantage and OPD uses synthetic aligned teacher log-probabilities.
Group-reward calculation and client teacher callbacks are separately exercised by
the tiny-model API tests. No hosted teacher or user environment was involved.

The final H100 run also passed the 28 fixture checks present in its snapshot.
The subsequent L4 run covers the final 29-check suite and HTTP admission fencing.
The model/backend and loss implementation in both snapshots are identical:

- `backend.py` SHA-256: `f29a5aded443fb9171cdd55ac940141d62b02636ae3b6cf325456f262b137637`
- `losses.py` SHA-256: `80af2f77f9ab73cd1a3417e1f7cdca26fda38517e0b053105c3b3347c568e01f`

The H100 run took 196.38 seconds from provisioning through sandbox termination;
model loading and checks took 92.72 seconds. The final L4 run took 141.63 seconds
including provisioning. Both sandbox reports recorded exit code 0 and successful
termination. These durations are not provider invoice estimates or throughput
benchmarks.

## Reproduction and limits

Use `scripts/qualify_shared_training.py` as documented in the operator guide. It
requires an explicit paid-test flag, snapshots source before building the image,
uses bounded sandbox timeouts, and terminates the sandbox in `finally`.

The dedicated CI workflow installs real CPU tensors on Python 3.11 and 3.12.
Ruff lint/format, module/function size, package layout, new-package mypy, lockfile
consistency, and source/wheel builds passed locally. Installing real PyTorch also
exposed two stale existing OPD/GPU-guard test stubs; those failures were reproduced
on unchanged `dev` and the stubs were corrected without changing production code.
Dependency contract tests now include the additional isolated training extra.

The full offline regression run on the rebased branch passed **10,813 tests** with
141 skips (`-n 2 --dist loadfile -m 'not wallclock'`). The separate serial timing
run passed **25 tests** with 2 skips (`-m wallclock`). Existing opt-in/dependency
skips remain; they are not represented as passed live integration tests.

Unqualified: managed run routing, hosted credentials and billing, deployable
adapter export, multi-turn/multimodal/structured recipes, external teacher transport,
large contexts, long-running training quality, throughput, automatic recovery,
retention, and production failure SLOs. Keep the backend opt-in and experimental.
