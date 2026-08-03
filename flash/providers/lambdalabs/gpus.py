"""Lambda GPU classes and friendly-name -> instance-type translation."""

from __future__ import annotations

import re

from flash.providers.base import UnsupportedGpuError, get_gpu_info, providers_for

__all__ = ["instance_type_for"]

# Lambda encodes cards-per-box in the instance-type NAME (``gpu_8x_h100_sxm5``), not as a launch
# parameter, so an N-card box is reached by rewriting the count segment of the 1x name.
_COUNT_PREFIX = re.compile(r"^gpu_(\d+)x_")


def instance_type_for(name: str, gpu_count: int = 1) -> str:
    """Lambda instance-type name (e.g. 'gpu_1x_a10') for a friendly GPU class name.

    ``gpu_count`` > 1 rewrites the count segment (``gpu_1x_h100_pcie`` -> ``gpu_4x_h100_pcie``).
    Whether that type EXISTS is deliberately not asserted here: only the live ``/instance-types``
    catalog knows which counts Lambda sells for a class, so callers needing a rentable type resolve
    it through the catalog (``usable_instances``) instead of trusting this string.
    """
    info = get_gpu_info(name)
    if not info.lambda_name:
        raise UnsupportedGpuError(
            f"{info.name} is not available on Lambda (providers: {', '.join(providers_for(name))})"
        )
    count = int(gpu_count)
    if count <= 1:
        return info.lambda_name
    if not _COUNT_PREFIX.match(info.lambda_name):
        raise UnsupportedGpuError(
            f"lambda instance type {info.lambda_name!r} for {info.name} carries no card-count "
            f"prefix, so a {count}-card variant cannot be named"
        )
    return _COUNT_PREFIX.sub(f"gpu_{count}x_", info.lambda_name, count=1)
