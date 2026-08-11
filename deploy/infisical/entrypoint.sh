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
#
# The loop runs in a SUBSHELL that prints shell code, instead of iterating here. A `for` in
# this shell has to iterate with a variable NAME, and POSIX `for` assigns to an existing name
# IN PLACE -- inheriting its export mark -- so a container that exported a variable of that
# name has its value replaced in the child. Any name can appear in INFISICAL_KEEP, so no
# choice of iterator is safe, and there is no way to make one local in POSIX sh. Patching the
# damage afterwards is what the previous revision did, by re-applying the captured value as an
# `env` operand: that restored the value but also made it beat the vault, which only names
# listed in INFISICAL_KEEP are allowed to do. A subshell removes the damage instead of
# compensating for it -- the assignment cannot outlive the loop, so the environment `exec`
# hands to the child is never touched, and a name nobody asked to keep is simply left alone
# for infisical to override.
#
# The emitted code carries only NAMES and lets this shell expand the values, so no value ever
# passes through the generated text: nothing to quote wrongly, and a multiline value (a PEM
# key, a certificate chain) arrives byte for byte. Names reaching that text are checked to be
# shell identifiers first.
eval "$(
  # Positional parameters, not a scratch variable, hold the one fact the loop destroys. Any
  # name used here would be an environment variable, and a container that set THAT name would
  # be read wrongly -- the same defect one rename further along. `$@` inside this subshell is
  # the command, which the loop does not need, and no child is exec'd from here.
  set -- "${_infisical_keep_name+set}"
  # shellcheck disable=SC2086
  for _infisical_keep_name in ${INFISICAL_KEEP:-}; do
    case $_infisical_keep_name in
      # The leading paren is optional POSIX syntax everywhere else and REQUIRED here: bash before
      # 4.0 (macOS ships 3.2 as both /bin/sh and /bin/bash) mis-parses a case pattern inside a
      # command substitution without it, dying with a syntax error AT RUNTIME that sh -n cannot
      # see, misreading the loop, and leaking this branch into the evaluated stream. The container
      # runs dash and is unaffected; this keeps the shebang honest on contributor shells too.
      ([!A-Za-z_]* | *[!A-Za-z0-9_]*)
        echo "flash infisical entrypoint: INFISICAL_KEEP entry is not a variable name: $_infisical_keep_name" >&2
        # The parent evaluates what this prints, so its exit status is what stops the startup.
        echo 'exit 2'
        exit 0
        ;;
    esac
    # Only re-apply names the container actually SET. An unset name expands to nothing, and
    # handing `env` a bare `K=` would overwrite the injected secret with an empty string --
    # so a typo'd or absent KEEP entry would silently WIPE a credential rather than leave the
    # vault's value alone. `${K+set}` distinguishes unset from set-but-empty, so an explicitly
    # empty container value still wins (that is a deliberate choice by whoever wrote it).
    if [ "$_infisical_keep_name" = _infisical_keep_name ]; then
      # Asking to keep the loop's own name: the check below would resolve THROUGH the iterator,
      # which by now holds that very name, and report a variable nobody set as present.
      [ "$1" = set ] || continue
    else
      eval "[ \"\${$_infisical_keep_name+set}\" = set ]" || continue
    fi
    echo "set -- \"$_infisical_keep_name=\${$_infisical_keep_name}\" \"\$@\""
  done
)"

exec infisical run \
  --projectId "$INFISICAL_PROJECT_ID" \
  --env "${INFISICAL_ENV:-prod}" \
  --path "$INFISICAL_PATH" \
  --silent \
  -- env "$@"
