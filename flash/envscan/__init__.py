"""Credential scanning for `flash env push`.

An environment is published to a repository shared across an organization, whose git history is
permanent. Anything committed there is committed for good, so the scan runs on the staged package
before upload and again on the server before the hub commit.

The public surface is `flash.envscan.secrets`; everything else is a format helper it dispatches to.
The split exists because each format needs a real parser -- a deflated zip member, a PDF stream, a
PNG text chunk and a base64 run each hide a credential in a different way -- and one module holding
all of them would be unreadable and unreviewable.
"""
