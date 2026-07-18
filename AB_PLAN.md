# packer setup-time A/B plan

## invariant

Replace only the linear first-fit bin lookup in `pack_token_ids` with a logarithmic lookup. The treatment must produce exactly the same packed rows as `origin/dev` at `8fdfc858`, including block order, example order within each block, truncation, `seq_lengths`, and completion-mask alignment.

## arms

- control: `origin/dev` at `8fdfc858`
- treatment: `perf/packer-bfd` at its sealed experiment commit

Use the same pretokenized dataset artifact, SFT configuration, seed, machine type, CPU allocation, storage, Python environment, and dependency lock for both arms. Do not run the arms concurrently on the same host. Alternate arm order across repetitions to control for filesystem cache and host load.

## workload

Use a pure-attention SFT configuration with packing enabled, `max_length=4096`, completion-only loss, and a frozen dataset expanded deterministically to both 50,000 and 100,000 non-empty examples. Preserve the exact example order and tokenized `input_ids` artifact across arms. The length distribution should include enough short and medium examples to create many partially filled bins, because that is where the control's linear first-fit scan is visible.

Run the CPU preparation benchmark first. No GPU is required for this stage.

## setup-time measurement

For each dataset size:

1. warm the Python environment once without recording the result;
2. run five measured repetitions per arm, alternating control and treatment;
3. time only the frozen pretokenized-input call to `pack_token_ids` using `time.perf_counter` outside the implementation;
4. record wall time, peak RSS, block count, and a stable checksum of every row's `input_ids`, `seq_lengths`, and `completion_mask`;
5. separately record end-to-end SFT process time from worker start to the existing first training-step log, using the same SFT config and host.

Primary metric: median `pack_token_ids` wall time at 100,000 examples.

Secondary metrics: median wall time at 50,000 examples, end-to-end time to first training step, and peak RSS.

Expected result: the treatment removes the quadratic growth and saves seconds to minutes at 50,000 to 100,000 examples. Treatment and control checksums and block counts must match exactly.

## training parity

After the CPU setup-time gate passes, run matched seeded SFT arms on identical GPU hardware and infrastructure. Both FA-path arms must retain TRL's `bfd` final packer; the treatment changes only the preceding block-count estimate. Use the same immutable tokenized dataset, initialization, seed, batch settings, token budget, and checkpoint cadence.

Compare:

- packed block count and derived update horizon;
- training example and token counts;
- loss at every shared step;
- final validation loss and task metric;
- time to first training step and total wall time.

Because final packing and order are unchanged, exact packed-data checksums are required. Training losses should be numerically identical when the backend is deterministic. If backend nondeterminism prevents exact equality, require overlapping per-seed loss curves with no consistent directional shift and equivalent final validation metrics across at least three paired seeds.

## gates

Accept the optimization only if all of the following hold:

- exact packed-row checksum parity at both dataset sizes;
- exact token, completion-mask, `seq_lengths`, block-count, and order parity;
- no increase greater than 5% in median peak RSS;
- at least 5x lower median direct packing time at 100,000 examples;
- no training-parity regression under the criteria above.

This experiment measures CPU setup and preparation time. It must not be presented as a GPU throughput improvement.
