# Infisical secret injection

Flash reads its secrets from the **process environment**. That is the whole contract, and it is
all a deployment needs:

```bash
docker run -p 8080:8080 --env-file .env -v flash-state:/root/.flash \
  ghcr.io/freesolo-co/freesolo-flash:main
```

Kubernetes Secrets, systemd `EnvironmentFile`, Docker secrets mounted into env, and every
orchestrator secret store already produce that environment. **If that covers you, you are done**:
nothing below is required.

This directory covers the other case: fetching secrets from [Infisical](https://infisical.com) at
container start.

## Using it

Set `INFISICAL_CLIENT_ID` on a published image. There is nothing to build:

```bash
docker run -p 8080:8080 -v flash-state:/root/.flash \
  -e INFISICAL_CLIENT_ID=... \
  -e INFISICAL_CLIENT_SECRET=... \
  -e INFISICAL_PROJECT_ID=... \
  -e INFISICAL_PATH=/flash \
  ghcr.io/freesolo-co/freesolo-flash:main
```

The entrypoint logs in, then execs `infisical run -- <the image's CMD>`, so the server sees the
injected secrets as ordinary environment variables. It changes nothing about how Flash itself
resolves configuration.

**The switch is the environment, not the image.** One image does both: with
`INFISICAL_CLIENT_ID` unset the entrypoint is a straight passthrough and the `--env-file` run
above behaves exactly as it would with no entrypoint at all. A deployment can move between the
two without a rebuild.

## Getting the CLI into the image

Injection needs the `infisical` binary present. The published images
(`ghcr.io/freesolo-co/freesolo-flash:main`, `:latest`, `:dev`) already have it.

Building from a checkout does **not**, by default. A plain `docker build .` produces a
vendor-free image, which is the right default for an open-source consumer who is not an
Infisical user. Ask for it explicitly:

```bash
docker build --build-arg INSTALL_INFISICAL=true -t flash-control-plane .
```

If `INFISICAL_CLIENT_ID` is set in an image without the CLI, the container **refuses to start**
and names this build argument. It does not silently fall back to the ambient environment:
booting a control plane without the secrets it was told to fetch would surface as an
authentication failure much later, far from the cause.

The CLI is installed from the vendor's published `.deb`, pinned by version and verified against
its SHA-256 before `dpkg` sees it. This repository deliberately does not pipe a setup script into
a shell, because that executes whatever the vendor serves at request time, which is neither reviewable
nor reproducible. Bump `INFISICAL_VERSION` and both digests together; a stale digest fails the
build rather than installing an unverified binary.

## Another secret manager

`entrypoint.sh` is ~80 lines of POSIX shell and is a reasonable starting point for Vault, AWS
Secrets Manager, or anything else with a CLI that can exec a command with secrets in its
environment. The parts worth keeping are the passthrough when the switch is unset, the refusal
when the switch is on but unusable, and the `INFISICAL_KEEP` handling described below.

## Configuration

| Variable                  | Required | Default | Purpose                                                                                                                                                                                                                                                                                                                            |
| ------------------------- | -------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INFISICAL_CLIENT_ID`     | yes      | unset   | Universal-auth machine identity. **Unset makes the wrapper a no-op passthrough**, so the container behaves exactly like the base image. Set but _empty_ aborts startup: that is a broken `${VAR}` interpolation, not a request for the passthrough.                                                                               |
| `INFISICAL_CLIENT_SECRET` | yes      | unset   | Machine identity secret.                                                                                                                                                                                                                                                                                                           |
| `INFISICAL_PROJECT_ID`    | yes      | unset   | Infisical project (workspace) id.                                                                                                                                                                                                                                                                                                  |
| `INFISICAL_PATH`          | yes      | unset   | Secret folder path for this service, e.g. `/flash`.                                                                                                                                                                                                                                                                                |
| `INFISICAL_ENV`           | no       | `prod`  | Environment slug. **A dev deployment that omits this loads production secrets.**                                                                                                                                                                                                                                                   |
| `INFISICAL_KEEP`          | no       | empty   | Space-separated variable _names_ whose existing container values must win over the injected ones. `infisical run` overrides existing env, so these are re-applied after injection. Use it for values that are a property of where the container runs, like a `FREESOLO_BASE_URL` pointing at a service on the same Docker network. |

`INFISICAL_TOKEN` is minted by the wrapper at startup. Do not set it yourself.

A name listed in `INFISICAL_KEEP` that the container has **not** set is skipped, so the vault's
value survives — a typo there costs you an override, not a credential. Setting a name to an empty
string is treated as a deliberate choice and still wins over the vault. Entries must be shell
identifiers; anything else aborts startup rather than being evaluated.

## Failure behaviour

The wrapper runs under `set -eu`: a failed login or a missing `INFISICAL_PROJECT_ID` /
`INFISICAL_PATH` aborts container start rather than silently booting a control plane with no
credentials. Flash's own startup preflight would reject that state anyway, but failing at the
wrapper keeps the cause in the first line of the container log.

Only `INFISICAL_CLIENT_ID` is checked for presence, because it is the switch. With it set and the
rest missing, you get a hard failure — which is the intended outcome, not a fallback to unset
secrets. The same applies when the switch is on in an image with no `infisical` binary: the
container stops and names the build argument instead of falling back.
