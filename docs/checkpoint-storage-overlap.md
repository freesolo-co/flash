# Checkpoint storage overlap

SFT, GRPO and OPD now persist native recovery checkpoints on a separate, serial
worker thread. Adapter export and publication can proceed while an earlier
recovery upload is in flight. This targets a concrete serialization point: SFT
previously withheld its publication acknowledgement until the recovery upload
finished, making the next required save wait on that upload.

The execution order is:

1. The trainer captures consistent native state using its existing synchronous
   save. GPU-to-CPU transfer and local serialization still take time here.
2. The parent pins the completed files outside trainer retention. Same-filesystem
   staging uses hardlinks; cross-filesystem staging copies files. Failed staging
   never falls back to a path that retention can delete during an upload.
3. The parent prepares and publishes adapters under the existing algorithm rules.
   SFT releases its next-save gate after the adapter commit, while the pinned native
   checkpoint remains owned by persistence.
4. One recovery worker uploads snapshots in order and deletes each staging tree
   after its operation finishes. The parent records the returned commit facts.

At most two native snapshots are admitted, including the active upload. Admission
waits before pinning another snapshot, so sustained slow storage applies
backpressure instead of accumulating more snapshots. This is a count limit, not
a byte limit: disk provisioning must still cover two native checkpoint sizes plus
the trainer's existing retention and adapter exports. It adds no in-memory model
snapshot queue. Existing GRPO/OPD retained adapter stores are separate.

GRPO still requires metric evidence and durable required-adapter references before
persisting resume state. Its terminal validation gate still controls deployable
publication. OPD stages its accounting/retry contract before submitting native
state. Missing recovery commits retain their existing failure policy; staging is
never counted as durable recovery. Required deployable failures remain fatal.

Shutdown drains accepted persistence work, even after adapter publication fails.
The existing run deadline bounds draining. The final Modal checkpoint still uses
the CPU handoff described in [CPU checkpoint uploads](cpu-checkpoint-uploads.md);
the staging result is not credited as a durable recovery commit.

This change concerns artifact publication. The synchronous trainer-to-sampler
weight-update dependency remains in place; it does not introduce policy lag or
asynchronous training.

## Measurement and validation

Worker logs now include per-step `capture_s`, `snapshot_pin_s`,
`adapter_publication_s`, and `persistence_s`, plus `backpressure_s` when admission
waits. SFT also logs `retention_wait_s`. SFT/OPD adapter timing includes export and
preparation; GRPO's adapter timing measures publication of its already staged
export. Capture logs may be emitted by multiple distributed ranks; do not add
rank durations as wall time.

Offline tests block the first recovery upload and verify that the next adapter
publishes for all three algorithms. They delete the trainer's original checkpoint
during the upload, verify metadata and snapshot contents, check ordered draining,
bound admission, propagate failures, and distinguish final CPU staging from a
durable commit. Existing checkpoint lifecycle and recovery tests cover the
algorithm-specific publication gates.

A [single-H100 SFT benchmark](checkpoint-storage-benchmark-2026-10-22.md) measured
17.8% lower median GPU-worker time and 20.0% lower median training time across
two baseline and two candidate runs. This six-step workload saves every step;
it demonstrates the checkpoint-heavy case, not typical sustained throughput.
Matched runs produced identical final adapter tensors. An attempted slow-storage
comparison was blocked by the artifact host's commit rate limit.

Before production-wide rollout, measure longer representative workloads, disk
high-water usage and adapter availability latency. Resume a persisted intermediate
checkpoint and compare its continuation with an uninterrupted run. Include GRPO,
OPD and a multi-GPU checkpoint, and complete the slow-storage comparison. These
remain unmeasured by the SFT benchmark; no learning-quality gain is claimed.
