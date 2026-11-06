"""Read public regional spot prices; never admit using missing or stale data."""

from datetime import UTC, datetime

from flash.providers.core.base import CapacityLookupError
from flash.providers.nebius.pricing import SPOT_GPUS, accepts_price

URL = "https://api.nebius.cloud/billing/v1alpha1/spot-price"


def affordable_gpus(region: str) -> dict[str, float]:
    import httpx

    admitted = {}
    token = ""
    seen = set()
    try:
        with httpx.Client(timeout=10) as client:
            for _ in range(20):
                response = client.get(URL, params={"pageToken": token, "pageSize": 100})
                response.raise_for_status()
                data = response.json()
                snapshot = datetime.fromisoformat(data["snapshotTime"].replace("Z", "+00:00"))
                age = (datetime.now(UTC) - snapshot).total_seconds()
                if not 0 <= age <= 300:
                    raise ValueError("stale spot prices")
                for item in data["items"]:
                    if item["region"] != region:
                        continue
                    for gpu, shape in SPOT_GPUS.items():
                        if item["computeInstance"]["platform"] == shape.platform:
                            price = item["price"]
                            if accepts_price(
                                gpu,
                                price["value"],
                                currency=price["currency"],
                                unit=price["unitTitle"],
                            ):
                                admitted[gpu] = float(price["value"])
                token = data.get("nextPageToken", "")
                if not token:
                    return admitted
                if token in seen:
                    raise ValueError("repeated spot price page")
                seen.add(token)
            raise ValueError("too many spot price pages")
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise CapacityLookupError("Nebius spot prices unavailable; keeping job queued") from None
