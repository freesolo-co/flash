# SFT tokenization A/B plan

## Invariant

The treatment must produce exactly the same ordered `input_ids` and `completion_mask` rows as the serial control for the same resolved environment, selected dataset prefix, seed, model revision, tokenizer, chat template, thinking mode, and maximum context length.

## Arms

- control: `origin/dev` at `8fdfc8580b9058d8c40289d4fa1a2e6e3b7bced2`
- treatment: `perf/sft-tokenize-parallel` at the final implementation commit

Run both arms on identical hardware and from clean per-arm process state. Keep the environment SHA, model revision, configuration, CPU allocation, storage mount, and package lock fixed.

## Representative configuration

Use an SFT configuration with a moderate-large selected prefix, targeting 10,000 to 25,000 rows. Use a tokenizer and chat template representative of current Qwen SFT runs, a fixed seed, fixed `max_examples`, fixed thinking mode, and fixed `max_context_tokens`. The dataset must be large enough that template rendering and tokenization dominate process startup noise.

## Procedure

1. Materialize the exact selected dataset prefix once and record its digest.
2. Run one cold-cache control preparation and one cold-cache treatment preparation in randomized order.
3. Compare every ordered `input_ids` and `completion_mask` row byte-for-byte before training.
4. Run a second treatment process against the completed cache to model a worker retry.
5. Repeat the cold-cache comparison at least three times per arm, clearing only the SFT preparation cache between cold runs.
6. If tokenized parity passes, run a short matched training smoke with the same initialization, data order, batch settings, and update count. Compare step losses and final adapter outputs within normal deterministic tolerance.

## Metrics and gates

Primary metric:

- wall time from selected-row availability through completed tokenized dataset availability

Secondary metrics:

- cold-cache treatment speedup versus control, expected 2x to 4x on the moderate-large dataset
- warm-cache retry speedup versus control, expected to exceed the cold-cache speedup
- exact ordered tokenized output equality
- short-run training loss parity and matching completed update count

Pass gates:

- zero token or completion-mask mismatches
- no partial cache is readable after an interrupted preparation
- cold-cache treatment is materially faster, with a target of at least 2x
- warm-cache retry loads the completed Arrow dataset instead of rendering or tokenizing again
- no training-parity regression

This optimization targets CPU data-preparation time only. GPU throughput, model-step latency, and tokens per second are explicitly out of scope.
