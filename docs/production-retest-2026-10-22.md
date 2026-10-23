# Hosted production retest: October 22, 2026

Release tested: Flash 1.3.23, commit `3c7511575c8e3632a4ba4d1b7a960bda38ca3d1d`, serving deployment `39859712055-1`.

The deployment blocker was a VM Docker credential that could pull the backend image but received 403 for Flash and agent-worker. Reusing an existing authorized service credential on that VM allowed the normal deployment timer to install the release. Package visibility and permissions were unchanged. All six release workflows and the serving canary passed; readiness was monitored every five minutes during qualification.

## Coverage and outcomes

These were short correctness runs against hosted production, not model-quality or load benchmarks.

- 28 training submissions: 25 completed, one intentional checkpoint cancellation, one zero-gradient fixture rejection, and one harness cancellation following a laptop sleep/read timeout. Corrected replacements for the last two completed successfully.
- SFT, GRPO and OPD on Qwen3.5-9B, Qwen3.8-27B and Qwen3.6-35B-A3B; additional two-GPU, image, multi-turn, structured-output, recovery and warm-start cases.
- OPD teachers: DeepSeek v4.1 Flash 1017, Kimi K3, GLM 5.3, Qwen3 VL 235B, Qwen3.8-27B and Qwen3.5-397B-A17B, including image cases on supported teachers.
- All 25 completed adapters passed checkpoint tensor checks and hosted deploy/chat/stream/CLI/undeploy. Checkpoints had finite learned weights; separate runs had isolated artifact repositories and matching GPU/CPU source attestations. The intentionally cancelled run's saved checkpoint was separately served successfully.
- All three hosted bases covered buffered/SSE generation, structured JSON/schema/choice/regex options, thinking, PNG/JPEG/WEBP and multiple images, logprobs, sampling, input rejection and three concurrent requests. Supported tool calls, streamed tool assembly and tool history/replay passed; unsupported combinations were rejected.
- Environment packaging, hosted evaluation, private adapter export, deployment disable/redeploy and retained-checkpoint lifecycle were exercised. The temporary export was deleted.
- All 192 captured successful API requests reconciled tokens and immutable prices with single delivered usage records. An overlapping audit of all 184 requests to owned checkpoint IDs covered CLI/evaluation traffic too. Do not add these overlapping counts.
- The 25 successful training charges matched accepted quotes. Total owned training ledger charge was $7.29, including a three-cent partial cancellation charge. This customer ledger amount is separate from Modal infrastructure cost.
- Cleanup verified 28 terminal runs, 27 undeployed checkpoint IDs, no owned active workers and no residual handoff volumes. Existing shared applications, three pre-existing adapters and audit/billing history were preserved.

## Production defect: disconnect can kill a shared replica

The HTTP disconnect checks correctly finalized usage as `client_disconnected`, with no billed usage. Server logs nevertheless showed both affected 9B replicas receiving a cancellation signal and then being killed by Modal's 30-second cancellation watchdog. Both were replaced. A successful HTTP cleanup/billing assertion alone therefore missed a production reliability defect.

Hosted dispatch used `stream_generate.remote_gen.aio()`. The installed Modal runtime drains generator output when a generator invocation exits; the affected logs included a pending generator-output sender during shutdown. The runtime drain is the identified failure path; the precise upstream condition that stalled that sender was not independently reproduced.

The fix connects the existing leased queue channel to a regular async Modal method. It owns an exact function-call handle, bounds cancellation/cleanup, aborts the corresponding vLLM request, and validates terminal delivery. The engine rechecks the invocation lease after adapter/image preparation and immediately before GPU submission, preventing late preparation from starting abandoned work.

Validation of the proposed fix:

- Offline tests cover channel cancellation, expired leases, cold admission, late spawn handles, bounded cleanup, terminal manifests and real hosted-entrypoint wiring.
- A CPU-only diagnostic on real Modal infrastructure completed cancellation in 0.384 seconds, allowed a concurrent 60-delta sibling stream to complete, and confirmed the same replica survived another 35 seconds. Exactly the cancelled generation was aborted.
- Reproduce the CPU diagnostic with `modal run scripts/qualify_stream_cancellation.py`. It creates a temporary CPU app with at most one container; it does not touch production serving or load a model.
- **The fix is not yet qualified on production GPUs.** After deployment, repeat disconnect before/after first content with a concurrent sibling, verify the original replica survives past the watchdog, and recheck usage and streamed structured/tool output. The CPU diagnostic does not substitute for vLLM validation.

## Other observations and limits

Greedy thinking on 9B repeated reasoning until both 768- and 4096-token limits. The exact prompt completed on two seeds using temperature 1, top_p 0.95 and presence_penalty 1.5, consistent with the supported subset of [Qwen's guidance](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices). This does not fix arbitrary model-quality failures; clients must handle `finish_reason=length`.

The first evaluation fixture omitted its required `evaluations.py`; adding the sidecar produced a completed, uploaded 2/2 SFT evaluation. A synthetic two-step image-GRPO adapter scored 1/2 on two arithmetic attempts because one answer exhausted its constrained-generation budget. Grading, uploads and failing CLI exit status were correct. Those low scores are retained as model-quality limitations.

The first structured GRPO fixture sampled identical answers and had no reward variance. Its zero-gradient rejection was correct. A varied-answer replacement trained successfully. A separate 35B OPD attempt was cancelled by the harness after local macOS sleep interrupted polling; the always-awake replacement completed.

Both live disconnects had already generated server-side tokens (6 and 105); exact pre-first-token cancellation is only covered by deterministic offline tests. Repeated-prefix requests did not expose a cache hit. Cache-hit performance, every GPU topology, customer-owned serving, external W&B delivery, long soaks and destructive outage recovery are not qualified by this audit.

## Cost estimate

The original cumulative Modal budget was $400. The prior conservative estimate of $297.23 was carried forward; this retest estimates $35.96 training and $48.55 serving, plus a full $10 ancillary allowance: **$391.74 cumulative estimated resources**.

The initial harness budget guard incorrectly assumed a five-minute serving idle tail. Production uses 35 minutes and permits four containers per model. Corrected accounting used task enqueue/finish timestamps for eight observed B200 containers, including the two killed replicas, and reserved the full tail for the six live replicas through final cleanup. No additional production generation was started after reconciliation. These are resource estimates, not provider invoices or a verified hard spending cap.

See [optimization opportunities](optimization-opportunities-2026-10-22.md) for follow-up proposals.
