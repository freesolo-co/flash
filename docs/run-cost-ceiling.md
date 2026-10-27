# Training launch cost ceilings

`POST /v1/runs` accepts an optional finite, nonnegative numeric `max_cost_usd`.
The server compares it against the freshly prepared, whole-cent customer quote
before affordability checks, ownership recording, submission, or GPU allocation.
A quote above the ceiling returns HTTP 400 with
`X-Flash-Submission-Outcome: not-created`. No paid run was created by that request.

Successful responses, including dry runs, advertise
`"budget_contract": "max-cost-usd-v1"`. Clients can preview, reserve the quoted
amount in their own atomic ledger, and submit the identical spec with that ceiling.
A changed quote cannot silently exceed the reservation. Existing callers without
a ceiling retain their existing behavior. The accepted quote remains the billing
basis; existing completion/cancellation accounting never increases it.

This is a per-training-run admission ceiling, not a shared account spending limit
or a serving-cost cap. Clients must reserve concurrent jobs before submission,
retain ambiguous reservations, and reconcile authoritative terminal receipts.
