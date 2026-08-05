# Infisical secret injection (opt-in)

Flash reads its secrets from the **process environment**. That is the whole contract, and it is
all a deployment needs:

```bash
docker run -p 8080:8080 --env-file .env -v flash-state:/root/.flash \
  ghcr.io/freesolo-co/freesolo-flash:main
```

Kubernetes Secrets, systemd `EnvironmentFile`, Docker secrets mounted into env, and every
orchestrator secret store already produce that environment. **If that covers you, ignore this
directory** — the published image has no Infisical CLI in it and does not need one.

This overlay is for deployments that instead fetch secrets from [Infisical](https://infisical.com)
at container start.

## Using it

Build the overlay on top of a Flash image:

```bash
docker build -f deploy/infisical/Dockerfile \
  --build-arg FLASH_IMAGE=ghcr.io/freesolo-co/freesolo-flash:main \
  -t flash-control-plane-infisical .
```

Run it with a universal-auth machine identity:

```bash
docker run -p 8080:8080 -v flash-state:/root/.flash \
  -e INFISICAL_CLIENT_ID=... \
  -e INFISICAL_CLIENT_SECRET=... \
  -e INFISICAL_PROJECT_ID=... \
  -e INFISICAL_PATH=/flash \
  flash-control-plane-infisical
```

The wrapper logs in, then execs `infisical run -- <the base image's CMD>`, so the server sees the
injected secrets as ordinary environment variables. It changes nothing about how Flash itself
resolves configuration.

You can also skip the overlay build and mount the script into the base image, provided the
`infisical` binary is available inside the container.

## Configuration

| Variable                  | Required | Default | Purpose                                                                                                                                                                                                                                                                                                                            |
| ------------------------- | -------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INFISICAL_CLIENT_ID`     | yes      | unset   | Universal-auth machine identity. **Unset makes the wrapper a no-op passthrough**, so the container behaves exactly like the base image.                                                                                                                                                                                            |
| `INFISICAL_CLIENT_SECRET` | yes      | unset   | Machine identity secret.                                                                                                                                                                                                                                                                                                           |
| `INFISICAL_PROJECT_ID`    | yes      | unset   | Infisical project (workspace) id.                                                                                                                                                                                                                                                                                                  |
| `INFISICAL_PATH`          | yes      | unset   | Secret folder path for this service, e.g. `/flash`.                                                                                                                                                                                                                                                                                |
| `INFISICAL_ENV`           | no       | `prod`  | Environment slug. **A dev deployment that omits this loads production secrets.**                                                                                                                                                                                                                                                   |
| `INFISICAL_KEEP`          | no       | empty   | Space-separated variable _names_ whose existing container values must win over the injected ones. `infisical run` overrides existing env, so these are re-applied after injection. Use it for values that are a property of where the container runs, like a `FREESOLO_BASE_URL` pointing at a service on the same Docker network. |

`INFISICAL_TOKEN` is minted by the wrapper at startup. Do not set it yourself.

## Failure behaviour

The wrapper runs under `set -eu`: a failed login or a missing `INFISICAL_PROJECT_ID` /
`INFISICAL_PATH` aborts container start rather than silently booting a control plane with no
credentials. Flash's own startup preflight would reject that state anyway, but failing at the
wrapper keeps the cause in the first line of the container log.

Only `INFISICAL_CLIENT_ID` is checked for presence, because it is the switch. With it set and the
rest missing, you get a hard failure — which is the intended outcome, not a fallback to unset
secrets.
