# Artifact commit isolation

New runs use a private Hugging Face dataset repository derived from their environment and finalized run identity. Retries of one run keep that repository. Existing runs retain the repository in their persisted preparation, including the older repositories shared by environment. There is no migration or deletion of existing artifacts.

The hosted production audit hit a repository-specific commit limit while several training runs shared an environment. The observed response was HTTP 429 with a limit of 320 commits per hour. That number is evidence from the audit, not a guaranteed provider policy: Hugging Face documents [separate granular limits for repository commits](https://huggingface.co/docs/hub/en/rate-limits#granular-user-action-rate-limits), whose values can change.

Isolating new runs prevents unrelated runs of an environment from consuming the same repository commit quota. It does not remove account-level API, repository-creation, storage, or a single run's own commit limits. Repository counts and duplicated small source/environment archives increase; checkpoint paths, ownership validation, and lifecycle cleanup remain unchanged.

If artifact uploads fail, Modal's ordinary nonzero process exit remains failure evidence even when the worker cannot publish its failure marker. The supervisor preserves bounded artifact rereads and explicit retryable worker evidence. Unknown or signal exits keep their host-loss recovery behavior. An ordinary process error no longer causes a paid preemption retry solely because its failure artifact is missing.
