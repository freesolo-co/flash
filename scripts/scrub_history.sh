#!/usr/bin/env bash
#
# One-time history scrub, to run BEFORE flipping this repository to public.
#
# What it removes from every commit on every ref:
#   - "Co-Authored-By: Claude ..." trailers
#   - "Claude-Session: https://claude.ai/..." trailers
#   - "Generated with ... Claude Code" attribution lines
#   - the build-box identity (a *.internal.cloudapp.net address) leaked into the
#     author/committer fields and into "Co-authored-by:" trailers
#
# This rewrites every commit sha. Run it once, on a fresh mirror clone, while the
# repository is still private. Rewriting after publication is pointless: the old
# shas stay reachable through forks, caches, and the GitHub API.
#
# The script re-reports the leak counts before and after, and REFUSES to print
# publication instructions unless every after-count is zero. A half-scrubbed
# repository that looks finished is the one outcome worth failing loudly over.
#
# Usage:
#   ./scripts/scrub_history.sh /tmp/flash-scrub
#
# It clones into that directory, rewrites, and stops. It never pushes. Review the
# result, then push manually (see the instructions it prints at the end).

set -euo pipefail

REMOTE="${FLASH_REMOTE:-https://github.com/freesolo-co/flash.git}"
WORKDIR="${1:-}"

if [ -z "$WORKDIR" ]; then
  echo "usage: $0 <empty-working-directory>" >&2
  exit 2
fi
if [ -e "$WORKDIR" ]; then
  echo "error: $WORKDIR already exists; pass a path that does not exist yet" >&2
  exit 2
fi

# --- identities to rewrite -------------------------------------------------
# Two problems in the author/committer fields:
#
#  1. A build box committed as "Ubuntu <...@...internal.cloudapp.net>", leaking an
#     internal hostname into the identity.
#  2. Some commits are authored by an AI assistant identity outright.
#
# Both are the maintainer's own work, so both map onto CANONICAL_IDENTITY.
#
# Matching is deliberately NOT a "^Claude" prefix test: that would also match a real
# contributor whose first name happens to be Claude and silently rewrite their
# authorship. Instead an identity is rewritten only when it is an exact known
# assistant name, when its EMAIL is on an assistant-owned domain, or when its EMAIL
# carries the leaked internal hostname.
#
# The name list alone is not enough: history also carries versioned identities like
# "Claude Opus 4.8 (1M context)", which no fixed name list can enumerate (the version
# and the parenthetical change per release). The email test catches those, and it is
# the SAFE half of the pair -- a human contributor named Claude commits under their own
# address, never under the assistant's noreply domain.
ASSISTANT_NAMES_RE='^(Claude|Claude Code|Claude Bot|claude)$'
ASSISTANT_EMAIL_RE='@anthropic[.]com$'
# "." as a wildcard is harmless here (it only widens the match to a near-identical
# string) and keeps the pattern usable verbatim in both awk and git --grep, which
# disagree about how to escape a literal dot inside a shell-passed variable.
LEAKED_EMAIL_RE='internal[.]cloudapp[.]net'

CANONICAL_IDENTITY="${FLASH_CANONICAL_IDENTITY:-David Shan <78061174+DavidBShan@users.noreply.github.com>}"

# A malformed override silently produces a mailmap git reads differently, which can
# leave the leak in place while reporting success. Require the "Name <email>" shape.
if ! printf '%s' "$CANONICAL_IDENTITY" | grep -qE '^[^<>]+ <[^<>[:space:]]+@[^<>[:space:]]+>$'; then
  echo "error: FLASH_CANONICAL_IDENTITY must look like 'Name <email@example.com>'" >&2
  echo "       got: $CANONICAL_IDENTITY" >&2
  exit 2
fi

# git-filter-repo is a single python script; uv fetches it without a global install.
FILTER_REPO=(uv run --with git-filter-repo git-filter-repo)

cat <<'EOF'
==> FREEZE THE REPOSITORY FIRST
    The rewrite is computed from a snapshot taken now. Anything pushed after this
    point is absent from the snapshot, and the publishing push would delete it.
    Tell every collaborator to stop pushing before you continue.
EOF

echo "==> cloning a fresh mirror of $REMOTE"
# --mirror: filter-repo must see every ref (main, dev, tags, and any other branch).
# Rewriting a single-branch clone would leave the other branches on the old shas.
git clone --mirror "$REMOTE" "$WORKDIR"
cd "$WORKDIR"
# absolute from here on: WORKDIR may have been relative, and we have just cd'd into it
WORKDIR="$PWD"

# Record what the remote looked like at snapshot time. The publish step re-reads the
# remote and diffs against this file: a rewrite pushed over a remote that moved in the
# meantime silently destroys whatever was pushed during the run, and --force gives no
# warning. Keeping the map lets the operator verify instead of hoping.
REFMAP="$WORKDIR/../flash-scrub-refmap-at-clone.txt"
git for-each-ref --format='%(objectname) %(refname)' > "$REFMAP"
REFMAP="$(cd "$(dirname "$REFMAP")" && printf '%s/%s' "$PWD" "$(basename "$REFMAP")")"
echo "    snapshot ref map: $REFMAP"

count_pattern() {
  # commits (not lines) whose message matches $1, counted by git itself so the
  # number means commits on every platform.
  git rev-list --all --count --grep="$1" --regexp-ignore-case --extended-regexp
}

count_identities() {
  # commits (not fields) whose author OR committer identity still matches the rewrite
  # rule. A commit where both fields match must count once, so dedupe per commit.
  #
  # This applies exactly the same name/email test as the mailmap builder. Counting a
  # looser thing (say, any identity containing "claude") would flag a real contributor
  # named Claude as an unscrubbed leak and block publication over nothing.
  git log --all --format='%H%x09%an <%ae>%x09%cn <%ce>' \
    | awk -F'\t' -v names_re="$ASSISTANT_NAMES_RE" -v mail_re="$LEAKED_EMAIL_RE" \
          -v bot_mail_re="$ASSISTANT_EMAIL_RE" '
        function leaks(ident,   name, email) {
          name = ident; sub(/ *<.*/, "", name)
          email = ident; sub(/^[^<]*</, "", email); sub(/>.*$/, "", email)
          return (name ~ names_re || email ~ bot_mail_re || email ~ mail_re)
        }
        leaks($2) || leaks($3) { n++ }
        END { print n + 0 }
      '
}

report_counts() {
  echo "    total commits:         $(git rev-list --count --all)"
  echo "    Co-Authored-By Claude: $(count_pattern 'Co-Authored-By:[ \t]*Claude')"
  echo "    Claude-Session:        $(count_pattern 'Claude-Session:')"
  echo "    Generated with Claude: $(count_pattern 'Generated with .*Claude Code')"
  echo "    leaked hostname:       $(count_pattern 'internal[.]cloudapp[.]net')"
  echo "    leaking identities:    $(count_identities)"
}

echo "==> before:"
echo "    branches:              $(git for-each-ref --format='%(refname)' refs/heads | wc -l)"
report_counts

echo "==> rewriting commit messages and identities"

# The mailmap is GENERATED from the repository rather than hardcoded: writing the
# leaked hostname into this file would reintroduce, in the published tree, the exact
# string the scrub exists to remove.
#
# Built with awk rather than sed so CANONICAL_IDENTITY is data, never part of a
# substitution program: an "&" in the name would otherwise be eaten as a
# backreference, and a "|" would collide with the sed delimiter.
MAILMAP="$WORKDIR/flash-scrub-mailmap"
git log --all --format='%an <%ae>%n%cn <%ce>' \
  | sort -u \
  | awk -v canon="$CANONICAL_IDENTITY" -v names_re="$ASSISTANT_NAMES_RE" -v mail_re="$LEAKED_EMAIL_RE" \
        -v bot_mail_re="$ASSISTANT_EMAIL_RE" '
      {
        name = $0
        sub(/ *<.*/, "", name)          # identity name, minus the address
        email = $0
        sub(/^[^<]*</, "", email); sub(/>.*$/, "", email)
        if (name ~ names_re || email ~ bot_mail_re || email ~ mail_re) print canon " " $0
      }
    ' > "$MAILMAP"

# No matches is a legitimate state (a repo already scrubbed), so it must not be an
# error. The awk pipeline above exits 0 on no matches, unlike grep.
if [ ! -s "$MAILMAP" ]; then
  echo "    (no identities needed remapping)"
else
  echo "    remapping $(wc -l < "$MAILMAP") identit(ies) onto $CANONICAL_IDENTITY"
fi

# Drop whole trailer lines. Matching is line-anchored and trailer-shaped, so a commit
# body that merely mentions Claude in prose is left alone.
"${FILTER_REPO[@]}" \
  --mailmap "$MAILMAP" \
  --commit-callback '
import re

patterns = [
    rb"(?im)^[ \t]*Co-Authored-By:[ \t]*Claude[^\n]*\n?",
    rb"(?im)^[ \t]*Co-authored-by:[^\n]*internal\.cloudapp\.net[^\n]*\n?",
    rb"(?im)^[ \t]*Claude-Session:[^\n]*\n?",
    # the generated-with footer, anchored to its own line and to "Claude Code", so
    # ordinary prose mentioning both words survives. the leading class allows any
    # non-ascii bytes (the footer is usually emoji-prefixed) plus whitespace.
    rb"(?im)^[ \t]*(?:[^\x00-\x7f][ \t]*)*Generated with[ \t]+[^\n]*Claude Code[^\n]*\n?",
]
message = commit.message
scrubbed = message
for pattern in patterns:
    scrubbed = re.sub(pattern, b"", scrubbed)

# Only reformat commits we actually touched: collapsing blank lines or trimming
# trailing newlines on every commit would rewrite unrelated message formatting.
if scrubbed != message:
    scrubbed = re.sub(rb"\n{3,}", b"\n\n", scrubbed)
    scrubbed = scrubbed.rstrip(b"\n")
    # a message that was nothing but trailers would otherwise become empty
    commit.message = (scrubbed + b"\n") if scrubbed.strip() else b"(no message)\n"
' \
  --force

echo "==> after:"
report_counts
echo
echo "    remaining author/committer identities:"
git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u | sed 's/^/      /'

rm -f "$MAILMAP"

# --- postcondition gate ----------------------------------------------------
# The counts above are printed for a human, but a human reading past a stray "2" is
# exactly how a half-scrubbed history gets published. Re-check them mechanically and
# refuse to print publication instructions unless every one is zero.
#
# Each count is taken in its own CHECKED assignment, never inline in `echo "$(...)"`
# or as a `for` word. `set -e` does not fire on a command substitution that fails
# inside a successful enclosing command, so an inline count whose git/awk died would
# expand to the empty string, arithmetic would read it as 0, and the gate would report
# a clean history it never actually verified. Assignment makes the failure fatal, and
# the integer test catches a command that "succeeded" while printing something else.
checked_count() {
  local label="$1" value
  shift
  if ! value="$("$@")"; then
    echo "error: leak check '$label' FAILED to run; cannot certify this history" >&2
    exit 1
  fi
  if ! printf '%s' "$value" | grep -qE '^[0-9]+$'; then
    echo "error: leak check '$label' returned a non-count: ${value:-<empty>}" >&2
    exit 1
  fi
  printf '%s' "$value"
}

residual=0
for spec in \
  "co-authored-by:Co-Authored-By:[ \t]*Claude" \
  "claude-session:Claude-Session:" \
  "generated-with:Generated with .*Claude Code" \
  "leaked-hostname:internal[.]cloudapp[.]net"
do
  n="$(checked_count "${spec%%:*}" count_pattern "${spec#*:}")"
  residual=$((residual + n))
done
n="$(checked_count identities count_identities)"
residual=$((residual + n))

if [ "$residual" -ne 0 ]; then
  cat >&2 <<EOF

==> SCRUB INCOMPLETE - DO NOT PUBLISH.

$residual leak(s) survived the rewrite (see the nonzero counts above). The mirror in
$WORKDIR is only partially scrubbed. Investigate before going any further; publishing
it now would expose exactly what this script exists to remove.
EOF
  exit 1
fi

cat <<EOF

==> rewrite complete, all leak counts are zero, nothing pushed.

Spot-check a few commits (git log -n 20 --format=full) before continuing.

BEFORE PUBLISHING, prune stale branches. This mirror carries every branch the
remote has (the count is printed above; it was ~407 at the time of writing, of
which only ~28 are merged into dev). Going public publishes all of them, including
abandoned experiment branches nobody has reviewed for internal detail. Delete the
ones you do not want public from the mirror before pushing:

  git branch -D <branch>          # repeat, or script it against a keep-list

Keeping just main, dev, and any live release branches is the safe default.

To publish the rewritten history:

  1. Confirm the freeze held. The rewrite was computed from a snapshot taken at
     clone time; anything pushed since is NOT in it and the push below would
     delete it. Diff the remote against the snapshot and expect no output:

       git ls-remote $REMOTE 'refs/heads/*' 'refs/tags/*' \\
         | awk '{print \$1, \$2}' | sort > /tmp/flash-remote-now.txt
       grep -E ' refs/(heads|tags)/' $REFMAP \\
         | sort > /tmp/flash-remote-at-clone.txt
       diff /tmp/flash-remote-at-clone.txt /tmp/flash-remote-now.txt

     Any difference means someone pushed during the run. STOP: re-run the scrub
     from a fresh clone rather than overwriting their work.
  2. Tell every collaborator to re-clone afterwards. Their existing clones point
     at shas that will no longer exist.
  3. Temporarily disable branch protection on main and dev (a rewrite is a
     force-push; protection will otherwise reject it).
  4. From $WORKDIR, push branches and tags EXPLICITLY. Not --mirror: a mirror
     clone also holds GitHub's read-only refs/pull/* (727 of them here), which
     --mirror tries to push and the server rejects.

       git remote add origin $REMOTE
       git push --force --atomic origin 'refs/heads/*:refs/heads/*' \\
                                       'refs/tags/*:refs/tags/*'

     --atomic so a rejected ref fails the whole push, rather than leaving the
     remote half-rewritten.
  5. Re-enable branch protection, then flip the repository to public.
  6. Re-clone locally. Do not reuse an old clone.

Open pull requests will show as rewritten and are best closed and reopened from
fresh branches.
EOF
