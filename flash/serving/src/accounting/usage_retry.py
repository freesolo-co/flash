"""Retry classification for durable serving accounting RPCs."""

import re


def is_transient_rpc_code(code: str) -> bool:
    if code == "supabase_transport_failure":
        return True
    match = re.fullmatch(r"supabase_rpc_([0-9]{3})", code)
    if match is None:
        return False
    status_code = int(match.group(1))
    return status_code in {408, 429} or status_code >= 500
