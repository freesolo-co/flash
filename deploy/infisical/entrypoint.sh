#!/bin/sh
# Entrypoint for the flash control-plane image: run the server, optionally injecting secrets
# from Infisical first.
#
# Flash reads its secrets from the process environment, and that stays the only contract.
# This wrapper decides WHERE that environment comes from, using one variable:
#
#   INFISICAL_CLIENT_ID unset  -> exec the command unchanged. The container behaves exactly
#                                 as it would with no entrypoint: `--env-file`, a kubernetes
#                                 Secret, or any orchestrator's secret store, as before.
#   INFISICAL_CLIENT_ID set    -> `infisical login` (universal-auth), then `infisical run`
#                                 injects the vault's secrets before exec'ing the command.
#   ...set but EMPTY           -> refuse to start. That is what a broken interpolation looks
#                                 like, not a request for the passthrough.
#
# Both paths are supported and tested; neither is a fallback for the other. The image is the
# same either way -- only the environment differs -- so a deployment can move between them
# without a rebuild. The CLI itself is an opt-in build argument (INSTALL_INFISICAL), so an
# image built without it supports the first path only and says so rather than failing obscurely.
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
# ENTRYPOINT in a derived image resets the inherited CMD to null, so an image built FROM ours
# that forgets to restate CMD lands here. Fail with the actual cause instead.
if [ "$#" -eq 0 ]; then
  echo "flash infisical entrypoint: no command given (the image's CMD is empty)" >&2
  exit 2
fi

# Presence, not truthiness: `${VAR+set}` is non-empty only when VAR is SET, empty value included.
# This is the same distinction the INFISICAL_KEEP loop below makes, for the same reason.
if [ -z "${INFISICAL_CLIENT_ID+set}" ]; then
  exec "$@"
fi

# Set-but-empty is not how you turn the switch off. It is what a compose file's `${VAR}` with
# nothing behind it, or a kubernetes secretKeyRef to an absent key, produces -- an accident, not
# a choice. Reading it as "unset" would boot the control plane on whatever ambient environment
# happened to be there, which is precisely the silent-credentials failure this switch exists to
# prevent. Unset the variable to take the passthrough deliberately.
if [ -z "$INFISICAL_CLIENT_ID" ]; then
  echo "flash infisical entrypoint: INFISICAL_CLIENT_ID is set but empty -- refusing to start" \
       "on ambient credentials. Give it a value to use the vault, or unset it entirely to take" \
       "secrets from the container environment" >&2
  exit 2
fi

# The switch is on but the binary is absent: the image was built without INSTALL_INFISICAL=true
# (or the wrapper was mounted into an image that has no CLI). Say THAT, because the alternative
# is `infisical: not found` under `set -e` -- a message that names a missing command without
# naming the build argument that would have provided it. Refusing here is also the safe outcome:
# continuing would exec the server with the vault's secrets missing, and a control plane that
# boots with no credentials fails later and further from the cause.
if ! command -v infisical >/dev/null 2>&1; then
  echo "flash infisical entrypoint: INFISICAL_CLIENT_ID is set but the infisical CLI is not" \
       "installed in this image -- rebuild with --build-arg INSTALL_INFISICAL=true, or unset" \
       "INFISICAL_CLIENT_ID to take secrets from the container environment instead" >&2
  exit 2
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
  # Assign through eval rather than `$(...)`: command substitution strips ALL trailing newlines,
  # so a kept multiline value (a PEM private key, a certificate chain) would arrive one or more
  # bytes shorter than the container set it -- silently, and only for the values most likely to
  # break something downstream. The expansion sits inside double quotes, so the value's own
  # content is never re-parsed: `$(...)`, backticks, quotes, and backslashes in it stay literal.
  eval "keep_value=\"\${$k}\""
  set -- "$k=$keep_value" "$@"
done
unset keep_is_set keep_value

exec infisical run \
  --projectId "$INFISICAL_PROJECT_ID" \
  --env "${INFISICAL_ENV:-prod}" \
  --path "$INFISICAL_PATH" \
  --silent \
  -- env "$@"
