# Security policy

## Reporting a vulnerability

Do not open a public GitHub issue for a security vulnerability.

Report it privately through either channel:

- GitHub private vulnerability reporting: use the **Security** tab on this repository,
  then **Report a vulnerability**.
- Email: security@freesolo.co

Please include:

- what the issue is and the impact you believe it has,
- the affected version or commit,
- steps to reproduce (a minimal config or request is ideal),
- any logs or output, with credentials redacted.

We aim to acknowledge a report within 3 business days and to give you a remediation
timeline once we have confirmed the issue. Please give us a reasonable window to ship a
fix before disclosing publicly.

## Supported versions

Fixes land on the latest released `freesolo-flash` version. Older versions are not
patched; upgrade to pick up security fixes.

## Scope

Flash runs training jobs on rented GPUs and holds infrastructure credentials on the
control plane. Reports that are in scope include:

- authentication or authorization flaws in the control plane (`flash/server/`), such as
  one organization reaching another organization's runs, adapters, or logs,
- credential leakage through logs, API responses, error messages, or uploaded artifacts,
- remote code execution reachable from a training config, environment payload, or API
  request,
- privilege escalation on the GPU worker (`flash/engine/`).

Out of scope:

- vulnerabilities in third-party dependencies with no exploitable path in Flash (report
  those upstream),
- findings that require an operator to have already leaked their own credentials,
- denial of service caused by legitimately submitting expensive training runs.

## Handling credentials

Never paste real credentials into an issue, pull request, or test fixture. The control
plane reads operator credentials from the environment (see `.env.example`); user
authentication uses freesolo API keys. If you believe a credential has been committed to
this repository, report it privately using the process above rather than opening a public
issue.
