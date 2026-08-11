#!/usr/bin/env bash
#
# One-time history scrub, to run BEFORE flipping this repository to public.
#
# What it removes from every commit on every ref:
#   - "Co-Authored-By: Claude ..." trailers
#   - "Claude-Session: https://claude.ai/..." trailers
#   - "Generated with ... Claude Code" attribution lines
#   - a leaked internal build-box hostname, in the author/committer fields and in
#     "Co-authored-by:" trailers
#   - stray local git identities, folded onto the canonical account: some of them
#     resolve to UNRELATED THIRD PARTIES' accounts once the repo is public
#   - the same addresses and hostname where they appear in FILE CONTENT: earlier
#     revisions of this script hardcoded them, and a rewrite that only touches
#     identities and messages leaves those blobs serving the leak from every branch
#   - the same addresses and hostname in ANNOTATED TAG messages, which are neither a
#     commit message nor a blob and so survive a rewrite that only covers those two
#
# This rewrites every commit sha. Run it once, on a fresh mirror clone, while the
# repository is still private. Rewriting after publication is pointless: the old
# shas stay reachable through forks, caches, and the GitHub API.
#
# The script re-reports the leak counts before and after, and REFUSES to print
# publication instructions unless every after-count is zero. A half-scrubbed
# repository that looks finished is the one outcome worth failing loudly over.
#
# The identities and the hostname to scrub are themselves the data being removed, so
# they are NOT stored here: see the identity-file block below.
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

# --- operator-supplied identity data ---------------------------------------
# The addresses and the internal hostname this scrub targets ARE the leak. Hardcoding
# them here would republish, in the tree, the exact strings the rewrite exists to
# remove, the same reasoning that keeps the mailmap generated rather than committed.
# So they live in an untracked file the operator writes locally (gitignored):
#
#   scripts/scrub_identities.env      (override the path with FLASH_SCRUB_IDENTITIES)
#
# It is sourced as shell, and must set all three of:
#
#   # the identity every leaking author/committer is folded onto, "Name <email>"
#   FLASH_CANONICAL_IDENTITY='Some Name <id+user@users.noreply.github.com>'
#   # space-separated LOWERCASE addresses to fold onto it, never a name prefix
#   FLASH_ALIAS_EMAILS='one@example.com two@example.test'
#   # the leaked internal hostname, as a regex fragment
#   FLASH_LEAKED_HOST_RE='build[.]internal[.]example[.]net'
#
# Write literal dots as "[.]": that spelling is read identically by awk, by git's
# --extended-regexp, and by python's re, so the one fragment drives every counter and
# the rewrite callback without per-engine escaping.
IDENTITIES_FILE="${FLASH_SCRUB_IDENTITIES:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scrub_identities.env}"
if [ ! -f "$IDENTITIES_FILE" ]; then
  cat >&2 <<EOF
error: identity file not found: $IDENTITIES_FILE

  This script deliberately ships with no identities in it. Create that file (it is
  gitignored, and must stay untracked) defining FLASH_CANONICAL_IDENTITY,
  FLASH_ALIAS_EMAILS and FLASH_LEAKED_HOST_RE. See the comment at the top of
  $0 for the exact shape.
EOF
  exit 2
fi

# An address already exported wins over the file's, keeping FLASH_CANONICAL_IDENTITY
# usable as a one-off override.
CANONICAL_OVERRIDE="${FLASH_CANONICAL_IDENTITY:-}"
# The other two have no override semantics, so an inherited value must not be able to
# stand in for a missing assignment: if the operator's shell already exports one and the
# file forgets it, sourcing leaves the stale value in place, the nonempty gate below
# accepts the incomplete file, and a stale alias list or hostname silently drives both the
# rewrite and the residual check -- certifying the wrong set of identities as clean.
unset FLASH_ALIAS_EMAILS FLASH_LEAKED_HOST_RE
# shellcheck source=/dev/null
. "$IDENTITIES_FILE"
CANONICAL_IDENTITY="${CANONICAL_OVERRIDE:-${FLASH_CANONICAL_IDENTITY:-}}"
# Normalized at LOAD, so the "list entries in LOWERCASE" contract documented below is
# enforced rather than merely requested. Every consumer today already copes with a mixed-case
# entry on its own -- the two awk lookups key on tolower(a[i]), and the counters and rewrites
# are all case-insensitive -- so this changes no behaviour now. It removes the standing
# requirement that the NEXT consumer remember to, which is how a mixed-case alias would
# eventually reach a case-sensitive test and end up neither remapped nor counted.
MAINTAINER_ALIAS_EMAILS="$(printf '%s' "${FLASH_ALIAS_EMAILS:-}" | tr '[:upper:]' '[:lower:]')"
# Lowercased, because every identity test downcases the address before comparing and awk's
# "~" is case-sensitive. Hostnames are case-insensitive and git preserves whatever case was
# committed, so a mixed-case fragment would leave the identity counter and the mailmap
# builder blind to exactly the identities the case-insensitive trailer counter and message
# rewrite do catch -- the two halves would disagree and the gate would certify a history
# whose author/committer fields still carry the hostname.
LEAKED_HOST_RE="$(printf '%s' "${FLASH_LEAKED_HOST_RE:-}" | tr '[:upper:]' '[:lower:]')"

# A missing value must be fatal, never an empty pattern: an empty alias list would
# match nothing and an empty hostname fragment would match EVERY identity, and either
# way the residual gate would certify a history it never checked.
missing=""
[ -n "$CANONICAL_IDENTITY" ] || missing="$missing FLASH_CANONICAL_IDENTITY"
[ -n "$MAINTAINER_ALIAS_EMAILS" ] || missing="$missing FLASH_ALIAS_EMAILS"
[ -n "$LEAKED_HOST_RE" ] || missing="$missing FLASH_LEAKED_HOST_RE"
if [ -n "$missing" ]; then
  echo "error: $IDENTITIES_FILE does not set:$missing" >&2
  exit 2
fi

# --- identities to rewrite -------------------------------------------------
# Two problems in the author/committer fields:
#
#  1. A build box committed under an address on an internal hostname, leaking that
#     hostname into the identity.
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

# Notes on FLASH_ALIAS_EMAILS, the alias list loaded above.
#
# Folding stray local identities onto the canonical account is not cosmetic: GitHub
# resolves a commit's author from its EMAIL, so on a public repo an alias can credit
# an unrelated account that merely happens to have registered that address.
#
# Check what an address renders as with:
#   gh api repos/<owner>/<repo>/commits/<sha> --jq .author.login
#
# It must be an EXPLICIT ADDRESS LIST, never a name-prefix test. A name match is both
# too narrow (it misses shortened and lowercased spellings) and far too dangerous: the
# same rewrite applied by name would sweep up any future contributor whose name
# collides. Addresses are unambiguous and reviewable, and every identity NOT on the
# list belongs to a real contributor whose attribution must survive untouched.
#
# The canonical address itself belongs OFF the list: it is already correct, and listing
# it would make the mailmap map an identity onto itself.
#
# List entries in LOWERCASE; every comparison against the list lowercases its input
# first. The domain part of an address is case-insensitive and git preserves whatever
# case was committed, so a mixed-case spelling is the same GitHub account as its
# lowercase entry. An exact-case test would leave that identity unremapped AND
# uncounted, so the residual gate would report a clean history while GitHub still
# misattributed the commits, which is the false-clean certificate this script exists to
# prevent. The trailer counter (git --grep --regexp-ignore-case) and the message rewrite
# ((?i) below) are already case-insensitive; the identity tests must agree with them or
# the gate and the rewrite disagree.
#
# The aliases are LITERAL ADDRESSES, not patterns, so every regex metacharacter in them has
# to be escaped before they are joined into an alternation. Escaping only dots is not
# enough: a perfectly ordinary address like "name+old@example.com" carries a "+", which an
# alternation reads as a quantifier on the preceding character. The awk lookup compares
# whole strings and still remaps the identity FIELDS, but the trailer rewrite and the
# residual gate both miss the trailer -- so the script publishes a history whose
# Co-authored-by lines still credit the alias while reporting it clean, the false-clean
# certificate this whole file is arranged to prevent. Backslash escaping is read
# identically by ERE (git --extended-regexp, grep -E) and by python's re.
alias_alternation() {
  printf '%s' "$MAINTAINER_ALIAS_EMAILS" | tr ' ' '\n' | grep -v '^$' \
    | sed 's/[][^$.|?*+(){}\]/\\&/g' | paste -sd'|' -
}

# Message patterns, shared by the counters and (in spirit) the rewrite callback below.
# Each is LINE-ANCHORED and trailer-shaped so that counting and removing agree: a commit
# that merely mentions a trailer in prose is neither rewritten nor counted as a leak.
#
# "^" must be followed by what the line actually starts with. Writing '^[[:space:]]*.*X'
# makes the anchor a NO-OP -- ".*" re-admits any prefix -- so a prose mention counts as a
# leak, residual never reaches zero, and the gate blocks publication forever. The
# generated-with footer is usually emoji-prefixed, so allow leading NON-SPACE punctuation
# explicitly rather than with a blanket ".*".
# "Claude" is a HUMAN NAME as well as the assistant's. A bare prefix match would delete
# the attribution of a contributor named e.g. "Claude Dupont" -- stripping a real person's
# credit is worse than leaving an assistant trailer in. So require "Claude" to be either
# the COMPLETE name (next non-space is "<") or followed by a model family. Matching on the
# assistant's EMAIL instead would be wrong here: a handful of trailers in this history were
# committed under an operator address rather than the assistant's noreply domain, and because
# this same pattern drives the residual gate, an email test would report a CLEAN history while
# those survived -- a false clean certificate, the worst failure this script has.
CO_AUTHORED_RE='^[[:space:]]*Co-Authored-By:[[:space:]]*Claude([[:space:]]+(Code|Bot|Opus|Sonnet|Haiku|Fable)[^<]*)?[[:space:]]*<'
SESSION_RE='^[[:space:]]*Claude-Session:'
GENERATED_RE='^[[:space:]]*[^[:alnum:]]*[[:space:]]*Generated with .*Claude Code'
# The hostname leaks through a Co-authored-by TRAILER, so anchor on the trailer, not on
# the bare hostname: a commit discussing the hostname in prose is not a trailer leak.
# (The same hostname in an author/committer FIELD is caught by count_identities, which
# reads the identity fields directly and is unaffected by this message-level pattern.)
HOSTNAME_RE="^[[:space:]]*Co-authored-by:.*$LEAKED_HOST_RE"

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
  #
  # Callers pass a LINE-ANCHORED expression, matching what the rewrite callback
  # actually removes. An unanchored count would deadlock the script: a commit whose
  # message legitimately DISCUSSES a trailer ("Document the Claude-Session: trailer
  # format") is correctly left alone by the callback, but an unanchored counter still
  # counts it, so residual never reaches zero and the gate refuses to publish forever.
  # git's --grep is line-oriented, so "^" anchors to a line, not to the message.
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
    | awk -F'\t' -v names_re="$ASSISTANT_NAMES_RE" -v mail_re="$LEAKED_HOST_RE" \
          -v bot_mail_re="$ASSISTANT_EMAIL_RE" -v alias_list="$MAINTAINER_ALIAS_EMAILS" '
        BEGIN { split(alias_list, a, " "); for (i in a) if (a[i] != "") alias[tolower(a[i])] = 1 }
        function leaks(ident,   name, email) {
          name = ident; sub(/ *<.*/, "", name)
          email = ident; sub(/^[^<]*</, "", email); sub(/>.*$/, "", email)
          email = tolower(email)
          return (name ~ names_re || email ~ bot_mail_re || email ~ mail_re || (email in alias))
        }
        leaks($2) || leaks($3) { n++ }
        END { print n + 0 }
      '
}

# Co-authored-by trailers still carrying a maintainer alias address. Counted separately
# from count_identities, which reads the identity FIELDS: the mailmap fixes the fields but
# not the message, so without this count the gate would certify a history whose trailers
# still credit the wrong GitHub accounts.
ALIAS_ALT="$(alias_alternation)"
ALIAS_TRAILER_RE="^[[:space:]]*Co-authored-by:.*<($ALIAS_ALT)>[[:space:]]*$"

# The hostname as it can appear in PROSE and in FILE CONTENT, which is not the same thing
# as the identity fragment. LEAKED_HOST_RE spells its dots "[.]", so it matches the hostname
# itself -- but earlier revisions of this script embedded that fragment IN SOURCE, once as a
# shell/awk pattern ("internal[.]example[.]net") and once as a python one
# ("internal\.example\.net"). A scan built from the fragment alone reads straight past its
# own previous text, so the very blob this pass exists to remove would survive and be
# counted clean. All three spellings are therefore matched.
HOST_PLAIN="$(printf '%s' "$LEAKED_HOST_RE" | sed 's/\[\.\]/./g')"
HOST_CONTENT_ALT="$(printf '%s' "$HOST_PLAIN" | sed 's/[.]/[.]/g')"
HOST_CONTENT_ALT="$HOST_CONTENT_ALT|$(printf '%s' "$HOST_PLAIN" | sed 's/[.]/\\[\\.\\]/g')"
HOST_CONTENT_ALT="$HOST_CONTENT_ALT|$(printf '%s' "$HOST_PLAIN" | sed 's/[.]/\\\\[.]/g')"

# Every scrubbed string, matched ANYWHERE rather than in a trailer. Two places need this and
# neither is covered by the trailer-anchored patterns above:
#
#   - file content. --mailmap rewrites identity fields and the callback rewrites messages;
#     neither touches a blob, so without the --replace-text pass below every branch holding
#     an old revision of this script keeps serving the leak from the FILES.
#   - message PROSE. anchoring is right for the Claude trailers (a message that merely
#     discusses one is not a leak, and counting it would deadlock the gate forever) but
#     wrong for these: an address or an internal hostname written into a commit message IS
#     the leak, wherever in the message it sits.
#
# "Wherever it sits" still means the COMPLETE address, though. An alias may be very short
# (the test fixtures use "d@d"), and an unanchored alternation finds it inside unrelated
# third-party addresses -- "todd@dell.com" contains "d@d". Both the blob rewrite and the
# prose rewrite would then redact a stranger's identity out of historical content, which is
# a worse outcome than the leak: the scrub would be silently corrupting history it was
# never pointed at. So every alias match is bounded by characters that cannot continue an
# address on either side. The bound is spelled twice because the two engines differ: ERE
# (git --extended-regexp, grep -E) has no lookaround, so the counters consume a boundary
# character or a line edge, while the Python rewrites use zero-width lookaround so `sub`
# cannot eat the neighbours. The CLASS is shared, so the two agree on what a complete
# address is -- and they must, since the counters are the gate over the rewrites: a
# counter that matched more than the rewrite fixes would block publication forever.
# every RFC 5322 atext character plus dot and @: any of these adjacent to an alias means
# the alias is a fragment of a longer address, not a complete one. the class feeds ERE and
# python brackets directly; sed consumers must use a delimiter outside this set (comma) and
# escape & in replacements.
#
# "'" and the backtick are deliberately LEFT OUT of that otherwise-complete set. Both are
# legal in a local part, but prose and shell snippets QUOTE addresses constantly while an
# address containing a quote is vanishingly rare -- and with "'" in the class a quoted
# "'d@d'" reads as one longer address, so it matches neither the rewrite nor the gate and
# the leak survives certified clean. Practical quoting beats RFC completeness here.
ALIAS_EDGE_CLASS="A-Za-z0-9._%+!#\$&*/=?^{|}~@-"
ALIAS_WORD_ERE="(^|[^$ALIAS_EDGE_CLASS])($ALIAS_ALT)([^$ALIAS_EDGE_CLASS]|\$)"
LEAK_STRINGS_RE="$ALIAS_WORD_ERE|$HOST_CONTENT_ALT"

count_blob_leaks() {
  # Lines of file content, across every reachable blob, that still carry a scrubbed string.
  #
  # Deduped by object id and scanned once: a blob is shared by every commit that kept the
  # file unchanged, so walking commits instead would re-read the same content thousands of
  # times. Reachable objects only -- --batch-all-objects would also pick up the pre-rewrite
  # originals that filter-repo leaves unreferenced, and the gate would never clear.
  #
  # Every failure here must PROPAGATE. checked_count calls this inside an `if !` condition,
  # which disables `set -e` for the whole call, so a `|| true` anywhere in the function turns
  # a scan that died halfway into a confident "0" and the gate certifies a history it never
  # read. Only grep's exit 1 ("no match") may be swallowed, and only after it has printed the
  # zero we want.
  local blobs out matches statuses
  if ! blobs="$(git rev-list --objects --all \
    | awk '{print $1}' \
    | git cat-file --batch-check='%(objectname) %(objecttype)' \
    | awk '$2 == "blob" { print $1 }' \
    | sort -u)"; then
    return 1
  fi
  if [ -z "$blobs" ]; then
    echo 0
    return 0
  fi
  # PIPESTATUS is read INSIDE the subshell: the parent's copy would describe the assignment,
  # not the pipeline. -a because blobs are arbitrary bytes and grep would otherwise answer
  # "binary file matches" instead of a number.
  out="$(
    set +o pipefail
    printf '%s\n' "$blobs" | git cat-file --batch | grep -ac -iE -e "$LEAK_STRINGS_RE"
    printf 'status=%s' "${PIPESTATUS[*]}"
  )"
  matches="${out%%status=*}"
  statuses="${out#*status=}"
  case "$statuses" in
    # grep 0 = matched, 1 = no match; anything else, from any stage, is a broken scan.
    "0 0 0" | "0 0 1") printf '%s' "${matches//[$'\n']/}" ;;
    *) return 1 ;;
  esac
}

count_tag_leaks() {
  # Annotated TAG OBJECTS still carrying a scrubbed string. An annotated tag is neither a
  # commit nor a blob, so neither count_pattern (which walks commit messages) nor
  # count_blob_leaks sees one, and a tag message quoting an alias or the hostname would be
  # certified clean and then force-pushed by the publish step.
  #
  # The whole raw object is scanned, headers included, so the TAGGER field is gated too:
  # --mailmap rewrites it exactly as it rewrites author/committer, and a residual there is
  # as real a leak as one in the message. Lightweight tags are skipped -- they are just a
  # ref pointing at a commit that count_pattern already covers.
  #
  # Failures propagate for the same reason they do in count_blob_leaks: checked_count calls
  # this inside an `if !`, so a swallowed error becomes a confident "0".
  local tags out matches statuses
  if ! tags="$(git for-each-ref --format='%(objectname) %(objecttype)' \
    | awk '$2 == "tag" { print $1 }' \
    | sort -u)"; then
    return 1
  fi
  if [ -z "$tags" ]; then
    echo 0
    return 0
  fi
  out="$(
    set +o pipefail
    printf '%s\n' "$tags" | git cat-file --batch | grep -ac -iE -e "$LEAK_STRINGS_RE"
    printf 'status=%s' "${PIPESTATUS[*]}"
  )"
  matches="${out%%status=*}"
  statuses="${out#*status=}"
  case "$statuses" in
    "0 0 0" | "0 0 1") printf '%s' "${matches//[$'\n']/}" ;;
    *) return 1 ;;
  esac
}

report_counts() {
  echo "    total commits:         $(git rev-list --count --all)"
  echo "    Co-Authored-By Claude: $(count_pattern "$CO_AUTHORED_RE")"
  echo "    Claude-Session:        $(count_pattern "$SESSION_RE")"
  echo "    Generated with Claude: $(count_pattern "$GENERATED_RE")"
  echo "    leaked hostname:       $(count_pattern "$HOSTNAME_RE")"
  echo "    alias co-author lines: $(count_pattern "$ALIAS_TRAILER_RE")"
  echo "    leaking identities:    $(count_identities)"
  echo "    alias/host in prose:   $(count_pattern "$LEAK_STRINGS_RE")"
  echo "    leaks in file content: $(count_blob_leaks)"
  echo "    leaks in tag objects:  $(count_tag_leaks)"
}

echo "==> before:"
echo "    branches:              $(git for-each-ref --format='%(refname)' refs/heads | wc -l)"
report_counts

echo "==> rewriting commit messages, identities and file content"

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
  | awk -v canon="$CANONICAL_IDENTITY" -v names_re="$ASSISTANT_NAMES_RE" -v mail_re="$LEAKED_HOST_RE" \
        -v bot_mail_re="$ASSISTANT_EMAIL_RE" -v alias_list="$MAINTAINER_ALIAS_EMAILS" '
      BEGIN { split(alias_list, a, " "); for (i in a) if (a[i] != "") alias[tolower(a[i])] = 1 }
      {
        name = $0
        sub(/ *<.*/, "", name)          # identity name, minus the address
        email = $0
        sub(/^[^<]*</, "", email); sub(/>.*$/, "", email)
        email = tolower(email)
        if (name ~ names_re || email ~ bot_mail_re || email ~ mail_re || (email in alias)) print canon " " $0
      }
    ' > "$MAILMAP"

# No matches is a legitimate state (a repo already scrubbed), so it must not be an
# error. The awk pipeline above exits 0 on no matches, unlike grep.
if [ ! -s "$MAILMAP" ]; then
  echo "    (no identities needed remapping)"
else
  echo "    remapping $(wc -l < "$MAILMAP") identit(ies) onto $CANONICAL_IDENTITY"
fi

# --mailmap rewrites author/committer/tagger FIELDS only; it does not touch message
# trailers. The stray addresses also appear inside "Co-authored-by:" lines, so without the
# rewrite below those trailers keep crediting the wrong accounts on a public repo.
#
# These trailers are REWRITTEN onto the canonical identity, not deleted. Someone
# co-authoring with their own second identity is still a real co-authorship record;
# correcting the address fixes the misattribution without discarding history. (The Claude
# trailers above are deleted instead because there is no correct identity to point them at.)
#
# Passed through the ENVIRONMENT rather than interpolated into the callback source: the
# canonical identity is operator-supplied and may legitimately contain quotes, which would
# otherwise terminate the Python literal and inject arbitrary code into the rewrite.
export FLASH_SCRUB_ALIAS_ALT="$ALIAS_ALT"
export FLASH_SCRUB_CANON="$CANONICAL_IDENTITY"
# same reason: the hostname fragment is operator-supplied data, not source.
export FLASH_SCRUB_HOST_RE="$LEAKED_HOST_RE"
# the strings that must not survive ANYWHERE in a message, trailer or not. see
# LEAK_STRINGS_RE: unlike the Claude patterns, these are secrets rather than a common word,
# so a prose mention is a leak and the gate counts it as one.
#
# The two halves are exported SEPARATELY rather than as the assembled LEAK_STRINGS_RE,
# because that one is the ERE spelling: its alias bound consumes a neighbouring character,
# which is harmless for a counter but would make `sub` delete the neighbour along with the
# address. The callback re-assembles the same meaning with lookaround. See ALIAS_WORD_ERE.
export FLASH_SCRUB_HOST_CONTENT_ALT="$HOST_CONTENT_ALT"
export FLASH_SCRUB_ALIAS_EDGE_CLASS="$ALIAS_EDGE_CLASS"

# Blob-level redaction, for the copies of these strings that live in FILE CONTENT rather
# than in an identity or a message. It is generated at runtime for the same reason the
# mailmap is: a committed replacements file would be one more tracked copy of the leak.
#
# Every entry is "regex:", never a literal, so the pattern is case-insensitive: git
# preserves whatever case a file was written with, and every other test in this script
# lowercases before comparing. A literal lowercase entry would leave a mixed-case spelling
# in the tree AND uncounted by the gate below.
#
# The alias entries carry the same complete-address bound as everywhere else, spelled with
# lookaround because filter-repo compiles "regex:" entries with python's re and substitutes
# them: a consuming bound would delete the neighbouring character out of the blob. Without
# it a short alias redacts the middle of unrelated addresses in every reachable file.
REPLACEMENTS="$WORKDIR/flash-scrub-replacements"
{
  # comma delimiter: the edge class now contains '|'. the class's own '&' must not expand
  # to the sed match, so a replacement-safe copy escapes it.
  #
  # Escaped with sed rather than "${ALIAS_EDGE_CLASS//&/\&}". Bash 5.2 made an unquoted '&'
  # in a pattern-substitution REPLACEMENT expand to the matched text, which looks like it
  # would break that spelling. It does not: the "\\" is consumed as an escaped backslash
  # first, and the bare '&' left behind expands to the match, which here IS '&' -- so both
  # forms emit "\&" (checked on 3.2 and on 5.3 with patsub_replacement on). It survives by
  # coincidence, though, and reads like the bug it keeps being reported as. sed's
  # replacement rules are the same on every version, so spell it there instead.
  _edge_sed="$(printf '%s' "$ALIAS_EDGE_CLASS" | sed 's/&/\\&/g')"
  printf '%s' "$MAINTAINER_ALIAS_EMAILS" | tr ' ' '\n' | grep -v '^$' \
    | sed 's/[][^$.|?*+(){}\]/\\&/g' \
    | sed "s,^,regex:(?i)(?<![$_edge_sed]),; s,\$,(?![$_edge_sed])==>REDACTED,"
  # all three source spellings of the hostname, not just the fragment; see HOST_CONTENT_ALT.
  printf '%s' "$HOST_CONTENT_ALT" | tr '|' '\n' \
    | sed 's|^|regex:(?i)|; s|$|==>REDACTED|'
} > "$REPLACEMENTS"

# Drop whole trailer lines. Matching is line-anchored and trailer-shaped, so a commit
# body that merely mentions Claude in prose is left alone.
#
# --message-callback, not --commit-callback: filter-repo runs this one over ANNOTATED TAG
# messages as well as commit messages. --replace-text rewrites blobs only and --mailmap
# rewrites the tagger field only, so a tag whose message carries an alias or the hostname
# would otherwise sail through untouched and get force-pushed as part of the publish step.
"${FILTER_REPO[@]}" \
  --mailmap "$MAILMAP" \
  --replace-text "$REPLACEMENTS" \
  --message-callback '
import os
import re

_alias_alt = os.environ["FLASH_SCRUB_ALIAS_ALT"].encode()
_canon = os.environ["FLASH_SCRUB_CANON"].encode()
_host_re = os.environ["FLASH_SCRUB_HOST_RE"].encode()
# every alias and every source spelling of the hostname, matched anywhere in the message.
# applied LAST, so the trailer rewrite below still gets to fold a Co-authored-by line onto
# the canonical identity rather than having its address redacted out from under it.
#
# The alias half is bounded to a COMPLETE address: a short alias like "d@d" otherwise
# matches inside an unrelated "todd@dell.com" and this sub rewrites a third party out of
# the message. Lookaround rather than the ERE bound the counters use, so the neighbouring
# characters survive the substitution and so that adjacent aliases both match; the
# character class is the same one the counters use, passed in, so gate and rewrite
# cannot drift apart.
_edge = os.environ["FLASH_SCRUB_ALIAS_EDGE_CLASS"].encode()
_host_content_alt = os.environ["FLASH_SCRUB_HOST_CONTENT_ALT"].encode()
_leak_strings = re.compile(
    rb"(?i)(?:(?<![" + _edge + rb"])(?:" + _alias_alt + rb")(?![" + _edge + rb"])"
    rb"|(?:" + _host_content_alt + rb"))"
    )
# the address must be the WHOLE bracketed value (anchored by "<" and ">"), so a longer
# address that merely ends with an alias cannot match.
_alias_trailer = re.compile(
    rb"(?im)^([ \t]*Co-authored-by:)[ \t]*[^\n<]*<(?:" + _alias_alt + rb")>[ \t]*$",
    )

patterns = [
    # "Claude" must be the complete name or carry a model family -- see CO_AUTHORED_RE
    # above. Kept in sync with it: if the strip is wider than the gate, a human named
    # Claude loses credit; if narrower, the gate blocks publication forever.
    rb"(?im)^[ \t]*Co-Authored-By:[ \t]*Claude(?:[ \t]+(?:Code|Bot|Opus|Sonnet|Haiku|Fable)[^<\n]*)?[ \t]*<[^\n]*\n?",
    rb"(?im)^[ \t]*Co-authored-by:[^\n]*" + _host_re + rb"[^\n]*\n?",
    rb"(?im)^[ \t]*Claude-Session:[^\n]*\n?",
    # the generated-with footer, anchored to its own line and to "Claude Code", so
    # ordinary prose mentioning both words survives. the leading class allows any
    # non-ascii bytes (the footer is usually emoji-prefixed) plus whitespace.
    rb"(?im)^[ \t]*(?:[^\x00-\x7f][ \t]*)*Generated with[ \t]+[^\n]*Claude Code[^\n]*\n?",
]
scrubbed = message
for pattern in patterns:
    scrubbed = re.sub(pattern, b"", scrubbed)
# rewrite (not drop) the maintainer alias trailers onto the canonical identity.
# The replacement is a FUNCTION, so the canonical identity is returned as literal bytes
# and is never parsed as a replacement template. The previous form doubled the backslashes
# instead, which was also correct (every backreference begins with a backslash, so doubling
# neutralizes "\1" and "\g<1>" alike) -- this is the same behaviour without the escaping
# argument, since the identity is operator-editable and the next reader should not have to
# re-derive that proof.
scrubbed = _alias_trailer.sub(lambda m: m.group(1) + b" " + _canon, scrubbed)
# folding several aliases onto one identity can leave the same trailer twice
_seen, _kept = set(), []
for _line in scrubbed.split(b"\n"):
    _key = _line.strip().lower()
    if _key.startswith(b"co-authored-by:") and _key in _seen:
        continue
    if _key.startswith(b"co-authored-by:"):
        _seen.add(_key)
    _kept.append(_line)
scrubbed = b"\n".join(_kept)
# whatever is LEFT: an alias address or the internal hostname written into ordinary prose.
# the trailer patterns above are anchored deliberately, and rightly so for the Claude ones,
# but an address in a message body is the leak itself and the residual gate counts it, so
# leaving it would block publication forever rather than merely look untidy.
scrubbed = _leak_strings.sub(b"REDACTED", scrubbed)

# Only reformat messages we actually touched: collapsing blank lines or trimming
# trailing newlines on every commit would rewrite unrelated message formatting.
if scrubbed == message:
    return message
scrubbed = re.sub(rb"\n{3,}", b"\n\n", scrubbed)
scrubbed = scrubbed.rstrip(b"\n")
# a message that was nothing but trailers would otherwise become empty
return (scrubbed + b"\n") if scrubbed.strip() else b"(no message)\n"
' \
  --force

echo "==> after:"
report_counts
echo
echo "    remaining author/committer identities:"
git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u | sed 's/^/      /'

rm -f "$MAILMAP" "$REPLACEMENTS"

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
for label_and_re in \
  "co-authored-by|$CO_AUTHORED_RE" \
  "claude-session|$SESSION_RE" \
  "generated-with|$GENERATED_RE" \
  "leaked-hostname|$HOSTNAME_RE" \
  "alias-co-author|$ALIAS_TRAILER_RE" \
  "alias-host-prose|$LEAK_STRINGS_RE"
do
  n="$(checked_count "${label_and_re%%|*}" count_pattern "${label_and_re#*|}")"
  residual=$((residual + n))
done
n="$(checked_count identities count_identities)"
residual=$((residual + n))
# blobs are gated too, not just reported: a rewrite that fixed every identity and message
# while leaving the addresses in a historical copy of a tracked file is exactly the
# half-scrubbed history this gate exists to catch.
n="$(checked_count blob-content count_blob_leaks)"
residual=$((residual + n))
# annotated tags are gated for the same reason blobs are: they are pushed by the publish
# step's 'refs/tags/*' refspec, so a tag message the rewrite missed is published verbatim.
n="$(checked_count tag-objects count_tag_leaks)"
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

Keeping just main, dev, and any live release branches is the safe default. Deleting
here only removes the branch from THIS mirror; the push in step 4 needs --prune to
delete it from the remote as well (it is there, and it is spelled out below).

To publish the rewritten history:

  1. Confirm the freeze held. The rewrite was computed from a snapshot taken at
     clone time; anything pushed since is NOT in it and the push below would
     delete it. Diff the remote against the snapshot and expect no output:

       git ls-remote --refs $REMOTE 'refs/heads/*' 'refs/tags/*' \\
         | awk '{print \$1, \$2}' | sort > /tmp/flash-remote-now.txt
       grep -E ' refs/(heads|tags)/' $REFMAP \\
         | sort > /tmp/flash-remote-at-clone.txt
       diff /tmp/flash-remote-at-clone.txt /tmp/flash-remote-now.txt

     --refs matters: without it an ANNOTATED tag also lists its peeled "v1^{}" entry,
     which the saved map (built with for-each-ref) does not have, so the diff would
     report a difference on every run and you could never get past this step.

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
       git push --force --atomic --prune origin 'refs/heads/*:refs/heads/*' \\
                                               'refs/tags/*:refs/tags/*'

     --prune is what makes the branch pruning above real. A wildcard refspec only UPDATES
     refs that still exist locally; without --prune, every branch you deleted above
     stays on the remote carrying its ORIGINAL UNSCRUBBED history, which defeats the
     entire scrub. Verify afterwards that the remote holds only what you kept:

       git ls-remote --heads origin | wc -l

     --atomic so a rejected ref fails the whole push, rather than leaving the
     remote half-rewritten.
  5. Re-enable branch protection, then flip the repository to public.
  6. Re-clone locally. Do not reuse an old clone.

Open pull requests will show as rewritten and are best closed and reopened from
fresh branches.
EOF
