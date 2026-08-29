# B200 / B300 hosted serving capacity tuning

Source: Flash `dev` @ `94210a323f9beaa713241e305f178b364848446d` (re-verified against `origin/dev`
immediately before the first paid launch; unchanged).

Budget authorized: **$50**, Modal private-source upload authorized. Scope per the user: sweep B200
and B300; L40S and H200 explicitly out of scope for this campaign.

## Safety decisions

- **Production is untouched.** `APP_NAME = "freesolo-lora-serving"` is hardcoded with no deploy-time
  knob (`modal_app.py:47`) and that app is live with a running task in both `main` and `dev`. The
  benchmark therefore deploys as a separate app, `flash-serving-capacity-bench`, which cannot
  overwrite it. No change was made to production's app-name logic.
- **A peer is running B200 viability work** in the same workspace (`flash-27b-h…`, created 01:26 UTC
  2026-08-27). Distinct app name, distinct functions; no shared mutable state.
- `HF_TOKEN` is passed by reference into a Modal secret. It is never printed, logged, or committed.

## Stage 0 — image and harness (no GPU billed)

The cutlass repair (`docker/repair_cutlass_dsl.py`) is wired into the serving image, following the
existing `patch_vllm_moe_lora.py` pattern: copy in, run, `--verify`, delete. Both wheels are the
exact SHA-256-pinned artifacts authorized in CLAUDE.md, verified before install.

Two non-obvious things it gets right, both from prior recorded evidence:

1. **The version is pinned, not discovered.** An unpinned `--force-reinstall --no-deps` takes the
   newest release; the resulting skew (`-libs-cu13` newer than its 4.5.2 siblings) still _passes_
   the intactness check, because the check re-hashes the package against its own RECORD. The check
   is structurally blind to the defect the naive fix introduces.
2. **`-libs-base` installs first, `-libs-cu13` second**, so cu13 content wins the shared paths.

**Build-time result, from vLLM's own predicate:**

    cutlass-dsl cu13 repaired to pinned 4.5.2
    cutlass-dsl cu13 intact at pinned 4.5.2

That is the third condition of the Blackwell FlashInfer gate passing. It is evidence about the build
container; the on-card assertion is a separate reading the probe takes.

### Harness defects found and fixed, and what each one cost

- `_is_libs_cu13_install_intact` is **not** in `vllm.utils.flashinfer_utils` — no such module exists
  in 0.23.0. It is defined in the GDN module that uses it,
  `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`. The wrong import failed the image
  build _after_ the repair had already succeeded, which reads misleadingly like a repair failure.
  Located by reading the module source in a CPU-only container, not by guessing.
- `engine_args_for(base_model, overrides, cfg)` takes the **settings** module as `cfg`, not
  `model_config`. Passing the wrong one raised `AttributeError: no attribute 'MAX_LORAS'` ~~75s into
  a B200 boot (~~$0.13). Now validated offline before every launch.
- `engine.engine.cache_config` is the **v0** shape. Under 0.23.0 the top-level
  `vllm.AsyncLLMEngine` name is bound to `vllm.v1.engine.async_llm.AsyncLLM`, which has no `.engine`
  at all. Reading `AsyncLLM.__init__` source (CPU-only) shows `self.vllm_config = vllm_config`
  directly on the instance — invisible to `dir(cls)`, which is why the earlier attribute enumeration
  missed it. This one cost a full 960s boot (~$1.67) because it failed only _after_ the engine was
  up.

**The lesson applied to the rewrite:** every reading now tries several candidate paths and records
which one answered, instead of committing to one guess. A wrong guess costs a dict entry rather than
another cold boot. The same probe also pinned two other things needed to avoid a third wasted boot:
`AsyncLLM` builds its asyncio mp client in `__init__`, so the engine must be constructed inside the
loop that later drives it (a separate `asyncio.run` per load cell would fail against a client bound
to the dead loop), and `shutdown` exists for explicit teardown.

### The boot is the bill, so one boot runs the whole sweep

At ~960s of the ~1000s spent per cell, the engine boot — not the load — dominates cost. The probe
therefore takes a **plan** of load cells and runs all of them against a single engine. A six-cell
sweep costs one boot instead of six, turning ~$10 of sequential probing into ~$1.9.

Each cell reports aggregate container throughput _and_ per-request median together, because that
pair is what distinguishes a real capacity knee from a card that is merely busy: under a cap, the
aggregate goes flat while per-request rate falls in proportion to offered load.

### `n=4` cells sample rather than run greedy

The fan-out cells cannot use `temperature=0`. The mirror parity contract rejects `n > 1` at
temperature 0 outright, so a greedy fan-out cell would benchmark a request shape the product never
serves. The `n=1` cells stay greedy; the `n=4` cells use `temperature=0.7`.

## The gate the sweep enforces

`_resolve_gdn_prefill_backend` (read verbatim from the installed 0.23.0) selects FlashInfer GDN
prefill on SM90 with **no further constraints**, but on SM10.x only when
`linear_key_head_dim == 128` AND `cuda_runtime_major >= 13` AND `_is_libs_cu13_install_intact()`.
The failure mode is `warning_once` then `return backend, "triton"` — no raise. So an unrepaired
Blackwell boot succeeds, serves, and bills the Blackwell rate while running the slower kernel.

A third path exists that the earlier notes missed: **`cutedsl`**, available on SM10.x with
`head_k_dim == 128` and _not_ gated on the cutlass install, but opt-in only via
`additional_config["gdn_prefill_backend"]` — `auto` never selects it.

**The sweep asserts the backend from vLLM's own decision rather than from boot success**, and
publishes no capacity number for a kernel that would not ship.

## Spend

| item                                         | GPU-seconds | approx |
| -------------------------------------------- | ----------- | ------ |
| B200 probe attempt 1 (harness `cfg` bug)     | ~75         | $0.13  |
| B200 probe attempt 2 (9B, `max_num_seqs=16`) | in flight   | —      |

Modal B200 $6.25/hr, B300 ~$7.10/hr, per-second billing. Region and non-preemptible multipliers are
not applied.

## Not claimed

- No production tier has been moved to Blackwell. This campaign measures; it does not migrate.
- No SFT/GRPO/OPD claim. Customer-owned `flash/serve` is untouched.
- `SCALEDOWN_WINDOW_SECONDS_BY_GPU` deliberately still has **no** Blackwell entry. The break-even
  window is a tier's own measured cold boot, so it gets added with the tier, from measurement --
  never guessed ahead of it.
