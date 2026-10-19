# Training background work

The FastAPI application lifespan constructs one `Supervisor` from
`flash._internal.background`. Run submission receives it explicitly through application state.
Callbacks run with a task-local owner and stop signal, allowing nested recovery work to use the
same supervisor. There is no process-global executor. Direct runner integrations that request
background work must supply a supervisor or execute inside `owner.bind()` and close it afterward.

## Admission and scheduling

| Lane | Maximum running tasks | Maximum outstanding tasks |
| --- | ---: | ---: |
| Training / provider monitoring | 64 | 192 |
| Recovery observations | 4 | 132 |
| Resource cleanup | 4 | 132 |

Outstanding tasks include reservations, queued callbacks, scheduled retries, and running work.
The limits belong to the supervisor constructor. A scheduler starts only eligible work; waiting
for a retry does not occupy a worker thread. A callback returns `Retry(delay)` to retain its key
and schedule its next observation. Thread creation failures retain admitted work for another
start attempt.

Run submission reserves capacity before publishing source or writing initial status. A full
lane returns a retryable HTTP 503. Recovery overflow stays in persisted run records and is
revisited every 30 seconds. The scanner reserves its own slot before startup recovery can fill
the recovery lane. Terminal runs only receive cleanup tasks when they have outstanding resource
records. Cleanup tasks retry until those records are removed after confirmed deletion.

Task keys deduplicate each operation within this application. Durable launch claims and remote
identity comparisons remain authoritative across processes. A queued replacement rechecks
provider clearance before acquiring its launch claim. Recovery observations hand replacement
training to the training lane so long-running monitors cannot consume recovery capacity.

## Cancellation and restart

Shutdown closes admission, discards pending callbacks, and signals active tasks. Provider polling
and runner waits cooperate with the signal. Existing blocking network calls are allowed to
return; Python threads are not forcibly interrupted. The application bounds its final training
wait at five seconds and logs when a blocking call remains.

Stopping local monitoring preserves a durably recorded remote worker for restart recovery.
Lambda and Vast still tear down unrecorded resources and preserve their normal failure cleanup.
OPD teacher capabilities retain their existing expiration when a monitor detaches. User
cancellation performs its existing durable cancellation, capability revocation, and resource
teardown before cancelling local follow-up tasks; cleanup tasks remain eligible.

Queued jobs that require request-only runtime secrets persist a private boolean requirement,
never the secrets themselves. After a restart they cannot be relaunched without those inputs;
they fail with a resubmission instruction. Already-running workers can still be attached, but a
replacement requiring the lost secrets is refused.

The supervisor owns execution and lifetime. The runner's reconciliation code continues to own
completion, retry eligibility, billing, and resource decisions.
