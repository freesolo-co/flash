"""Project Spot quotas are instance counts, including resources owned by other jobs."""

from flash.providers.mithril.client import MithrilApiError, call
from flash.providers.mithril.config import settings


def quota_available(instance_type, *, deadline_at=None):
    project = settings()["project_id"]
    rows = call("GET", "/quotas", params={"project": project}, deadline_at=deadline_at)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise MithrilApiError("Mithril returned invalid quotas")
    for row in rows:
        if row.get("product_type") != "Spot" or row.get("instance_type") != instance_type:
            continue
        total, used = row.get("total_quantity"), row.get("used_quantity")
        if (
            row.get("project") != project
            or row.get("units") != "Instances"
            or type(total) is not int
            or type(used) is not int
            or min(total, used) < 0
        ):
            raise MithrilApiError("Mithril returned an invalid Spot quota")
        if used >= total:
            return False
    # Not all shapes have a project quota row; absence is not a zero quota.
    # The server still enforces shared organization limits and concurrent bid races.
    return True
