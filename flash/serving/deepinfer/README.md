# DeepSeek V4.1 Flash 1017 on Modal

This backend serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` through native
DeepInfer, with four B200s cooperating inside each replica. The existing Flash
CPU router retains Freesolo API-key authorization, organization attribution,
OpenAI response formatting, and durable usage settlement. Qwen/LoRA engines
keep their separate vLLM images and capacity.

## Capacity and price

| Setting | Value |
| --- | --- |
| GPU allocation | `B200:4` per replica, EP=4 / DP=4 |
| Replica pool | 0–4, at most 16 B200s |
| Admission | 32 requests per replica; scale-out target 24 |
| Native scheduler | 8 sequences per rank, 512-token prefill chunks |
| Idle shutdown | 600 seconds, no warm floor or spare buffer |
| Host allocation | 16 CPUs / 256 GiB RAM per replica |
| Initial context limit | 32,768 tokens including output and 64 reserved tokens |
| Startup / generation budgets | 45 minutes / 10 minutes |
| Input / output / cached input | $0.15 / $0.60 / $0.003 per million tokens |

Modal measures active HTTP inputs and scales whole four-GPU replicas. Saturated
replicas have a bounded native admission queue; excess traffic queues in Modal
or receives a retryable capacity error. Each GPU group exits if one rank dies or
stalls. Closing a request aborts its native sequence and waits for acknowledgment
before releasing the slot; it does not terminate the other requests in that group.

Prices are captured as exact per-token decimals with version
`2026-10-23.deepseek-1017.1`. Completion usage includes native reasoning and EOS
tokens. The bridge currently reports zero cached tokens with
`cached_tokens_reported=False`; verified prefix reuse and its per-request billing
counts still need implementation in this serving path.

## Deployment and qualification

The native bridge passed a **four-B200 smoke and performance run**; see the
[measured results and limits in PR #1484](https://github.com/freesolo-co/flash/pull/1484).
Its validated default configuration peaked at approximately 1,597 aggregate
output tokens/s on short prompts. Separate warm benchmarks exceeded 12,000
output tokens/s on both long-context and short-chat workloads at approximately
92% input-token reuse. Those experiments used retained prefix states, 128
sequences per rank, speculative decoding, and FP8 execution of losslessly
expanded FP4 expert values. The deployable bridge retains the configuration in
the capacity table above. Promoting the experimental path requires bounded,
tenant-aware cache management, verified cache billing counts, and service
qualification at the larger batch sizes. A 92% hit rate also depends on the
request traffic and cache warmth.

This is not complete customer rollout qualification. CPU tests cover the HTTP/native
bridge, router authorization, pricing and settlement, thinking/non-thinking
responses, native errors, and cancellation. Keep customer traffic disabled until
the remaining service and accounting checks below are completed.

Deploy the private backend in the intended Modal environment:

```sh
modal deploy --env dev -m flash.serving.deepinfer.modal_app
```

The GPU endpoint requires [Modal Proxy Tokens](https://modal.com/docs/guide/webhook-proxy-auth),
so unauthenticated requests are rejected before provisioning GPUs. The GPU backend exposes only `/health` and `/internal/generate`; OpenAI routes,
stream formatting, usage responses, and model listing use the existing Flash
hosted router. Run chat qualification through that router too, so probes exercise
the same authorization, response, and accounting path as customer requests.

After qualification, supply these connection credentials when deploying the
existing serving router, following its development wiring instructions:

- `DEEPINFER_URL`: the HTTPS root URL printed by the backend deployment.
- `DEEPINFER_PROXY_KEY` and `DEEPINFER_PROXY_SECRET`: an environment-scoped Modal
  Proxy Token pair, **not** ordinary Modal API credentials.

The router forwards these through its existing runtime secret. All three must be
present together; partial configuration fails closed. Without them, the native
model is absent from the router's advertised base models. No vLLM class is
created for DeepSeek, and LoRA registration is unsupported.

The bridge follows Modal's same-origin 303 result redirects during long cold
starts without submitting a second generation. It rejects cross-origin redirects
before forwarding proxy credentials. Direct qualification clients must also
follow result redirects (for example, `curl -L`). See
[Modal request timeouts](https://modal.com/docs/guide/webhook-timeouts).

**Control-plane dependency:** the separately deployed Freesolo backend must also
recognize this exact model and the same prices in `src/billing/token_pricing.py`.
Its `/api/serving/authorize` uses that catalog to authorize base-model requests.
This Flash PR alone does not change that repository or enable customer access.
Keep the router connection unset until that catalog and settlement path are ready.

Qualification must exercise cold boot and coherent output against the official
encoder, native FP8/FP4/Engram parity, thinking and streaming, cancellation during
prefill and decoding alongside another request, rank failure, load above the
24-input target, the four-replica cap, and return to zero after idle timeout.
Verify durable accounting through the real backend at the configured rates.

## Pinned runtime

- [Model and encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017):
  `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- [DeepInfer](https://github.com/deepseek-ai/DeepInfer):
  `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.
- Compiled CUDA 13.2 / Python 3.12 image: `im-01M5GXCE5NCMK4TBY608BEPZ1Y`,
  a workspace-local immutable image from the four-B200 benchmark. It contains
  compiled DeepInfer under `/opt/DeepInfer` and Python under `/opt/deepinfer-venv`.
  Other workspaces must build the pinned upstream runtime and replace this image
  identity; the image is not a portable public registry artifact.

Image preparation checks the upstream Git revision and the exact indexer source
hash before applying the short-context candidate-bound repair. The local Engram
transport substitutes mapped shared host RAM/CUDA IPC for RDMA, preserving native
FP8 embedding values and packed scales. It does not widen weights or substitute
benchmark prompts. Native model weights are downloaded at the pinned revision
into the persistent `freesolo-deepseek-1017-weights` volume.

The inherited build also selects `sm_100a` in FlashMLA and permits CUDA
cross-compilation in the DeepSelect/FlashMLA platform probes. The runtime's
`PATH` and `VIRTUAL_ENV` match the earlier successful four-B200 campaign.
Each initialized GPU worker checks its engine source and native extension paths
against the pinned source tree and reports the native binary SHA-256, commit,
GPU model, and PyTorch/CUDA versions in the private `/health` response's
`loaded_ranks` field. All four ranks must initialize before health succeeds.
These loaded-worker records are distinct from inspecting the image on CPU.

The initial API supports text-only system/user/assistant messages, one completion,
temperature, top-p, seed, max tokens, and `chat_template_kwargs.enable_thinking`.
It uses the checkpoint's `encoding/encoding.py`, because the model does not ship
a Jinja chat template. Tool calls, images, raw prompts, structured outputs,
logprobs, penalties, and custom stop sequences are rejected explicitly. The
model's advertised million-token context is not enabled in this initial capacity
configuration.
