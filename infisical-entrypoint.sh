#!/bin/sh
# wraps the flash control-plane command with infisical secret injection.
#
# fallback-safe: if INFISICAL_CLIENT_ID is unset this is a no-op passthrough and the
# container behaves exactly as before (reading its env_file / docker env). once the
# bootstrap machine-identity creds are present, secrets are pulled from infisical at startup.
#
# expected env (set by docker-compose in the freesolo repo):
#   INFISICAL_CLIENT_ID / INFISICAL_CLIENT_SECRET  universal-auth machine identity creds
#   INFISICAL_PROJECT_ID                           the infisical project (workspace) id
#   INFISICAL_ENV                                  env slug (prod | dev); default prod
#   INFISICAL_PATH                                 folder path for this service (/flash)
#   INFISICAL_KEEP                                 space-separated var names that must WIN over
#                                                  infisical (docker-network overrides). infisical
#                                                  run overrides existing env, so we re-apply these
#                                                  AFTER injection via `env K=V`.
set -eu

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
  set -- "$k=$(eval "printf '%s' \"\${$k:-}\"")" "$@"
done

exec infisical run \
  --projectId "$INFISICAL_PROJECT_ID" \
  --env "${INFISICAL_ENV:-prod}" \
  --path "$INFISICAL_PATH" \
  --silent \
  -- env "$@"
