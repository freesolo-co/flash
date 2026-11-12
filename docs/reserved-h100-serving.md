# Reserved 9B Base serving

Production `Qwen/Qwen3.5-9B-Base` uses four separately reserved Lambda H100 workers. Each has a dedicated CPU-only Modal proxy with a one-container ceiling. The router rotates requests across these four proxies and broadcasts adapter registration/removal to every worker. Other models retain their existing shared-capacity behavior.

Worker addresses, instance identities, source revisions, certificate pins and bearer tokens live in the main-environment Modal Secret `flash-reserved-h100-workers-20261112`. Never commit those values. Proxies validate worker model, revision and H100 identity before accepting calls. Proxy shutdown closes its client without terminating the reserved VM.

9B Base has no Modal GPU fallback in main or dev. Main forbids automatic Lambda allocation for this model. Development retains a scale-to-zero Lambda-only path, isolated from production's reserved workers and credentials.

The separate `flash-h100-reservation-20261112` operation owns the four instances' week-long retention. Its authoritative state and `hold_until` are in the same-named Modal Dict. Explicitly transfer lifecycle responsibility before stopping that scheduler. The reservation inventory is separate from ephemeral `flash-main-*` workers and their idle reaper.

Qualify all four workers before cutover: pinned-runtime lock verification, GPU/model identity, buffered and streamed generation, cancellation, and an authenticated public-endpoint canary. Drain prior 9B Base requests before terminating old dynamic workers. Reverting the routing code restores the prior dynamic paths; review capacity and live instances before rollback.
