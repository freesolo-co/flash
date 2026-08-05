#!/bin/sh
# OPT-IN: wraps the flash control-plane command with infisical secret injection.
#
# The published image does NOT use this. Flash reads secrets from its process environment,
# so `docker run -e ... / --env-file`, a kubernetes Secret, or any orchestrator's secret
# store works with no wrapper at all. This file exists for deployments that pull secrets
# from Infisical at container start; see deploy/infisical/README.md to opt in.
#
# fallback-safe: if INFISICAL_CLIENT_ID is unset this is a no-op passthrough and the
# container behaves exactly as it does without the wrapper (reading its env_file / docker
# env). Once the bootstrap machine-identity creds are present, secrets are pulled from
# infisical at startup.
#
# expected env:
#   INFISICAL_CLIENT_ID / INFISICAL_CLIENT_SECRET  universal-auth machine identity creds
#   INFISICAL_PROJECT_ID                           the infisical project (workspace) id
#   INFISICAL_ENV                                  env slug (prod | dev); default prod
#   INFISICAL_PATH                                 folder path for this service (e.g. /flash)
#   INFISICAL_KEEP                                 space-separated var names that must WIN over
#                                                  infisical (docker-network overrides). infisical
#                                                  run overrides existing env, so we re-apply these
#                                                  AFTER injection via `env K=V`.
set -eu

# `exec "$@"` with ZERO arguments is a no-op that FALLS THROUGH to the injection path below,
# where `set -u` then blames INFISICAL_CLIENT_ID for what is really a missing command. Setting
# ENTRYPOINT in a derived image resets the inherited CMD to null, so an overlay that forgets to
# restate CMD lands here. Fail with the actual cause instead.
if [ "$#" -eq 0 ]; then
  echo "flash infisical entrypoint: no command given (the image's CMD is empty)" >&2
  exit 2
fi

if [ -z "${INFISICAL_CLIENT_ID:-}" ]; then
  exec "$@"
fi

INFISICAL_TOKEN="$(infisical login --method=universal-auth \
  --client-id="$INFISICAL_CLIENT_ID" \
  --client-secret="$INFISICAL_CLIENT_SECRET" \
  --silent --plain)"
export INFISICAL_TOKEN

# Re-apply the docker-network overrides AFTER infisical injection so they win. infisical
# run overrides existing env, so we hand each KEEP var to `env` as its own quoted `K=V`
# argument — values that contain spaces or shell metacharacters survive intact. (Building
# an unquoted string and word-splitting it would corrupt such values, and a leading space
# would make `env` choke.) INFISICAL_KEEP itself is a whitespace-separated list of variable
# NAMES, so splitting *it* is intentional.
# shellcheck disable=SC2086
for k in ${INFISICAL_KEEP:-}; do
  # These names are expanded inside `eval` below, so refuse anything that is not a shell
  # identifier instead of executing it.
  case $k in
    [!A-Za-z_]* | *[!A-Za-z0-9_]*)
      echo "flash infisical entrypoint: INFISICAL_KEEP entry is not a variable name: $k" >&2
      exit 2
      ;;
  esac
  # Only re-apply names the container actually SET. An unset name expands to nothing, and
  # handing `env` a bare `K=` would overwrite the injected secret with an empty string --
  # so a typo'd or absent KEEP entry would silently WIPE a credential rather than leave the
  # vault's value alone. `${K+set}` distinguishes unset from set-but-empty, so an explicitly
  # empty container value still wins (that is a deliberate choice by whoever wrote it).
  eval "keep_is_set=\${$k+set}"
  [ "${keep_is_set:-}" = set ] || continue
  set -- "$k=$(eval "printf '%s' \"\${$k}\"")" "$@"
done
unset keep_is_set

exec infisical run \
  --projectId "$INFISICAL_PROJECT_ID" \
  --env "${INFISICAL_ENV:-prod}" \
  --path "$INFISICAL_PATH" \
  --silent \
  -- env "$@"
