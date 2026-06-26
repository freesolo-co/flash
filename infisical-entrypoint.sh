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

# snapshot the docker-network overrides so they survive infisical injection
keep=""
for k in ${INFISICAL_KEEP:-}; do
  v=$(eval "printf '%s' \"\${$k:-}\"")
  keep="$keep $k=$v"
done

# shellcheck disable=SC2086
exec infisical run \
  --projectId "$INFISICAL_PROJECT_ID" \
  --env "${INFISICAL_ENV:-prod}" \
  --path "$INFISICAL_PATH" \
  --silent \
  -- env $keep "$@"
