"""Serving token prices for Flash deployments (per 1M tokens)."""

from __future__ import annotations

from dataclasses import dataclass

SERVING_MARKUP = 1.20


@dataclass(frozen=True)
class ServingPrice:
    model_id: str
    typical_input_usd_per_mtok: float
    typical_output_usd_per_mtok: float
    typical_cached_input_usd_per_mtok: float

    @property
    def billed_input_usd_per_mtok(self) -> float:
        return self.typical_input_usd_per_mtok * SERVING_MARKUP

    @property
    def billed_output_usd_per_mtok(self) -> float:
        return self.typical_output_usd_per_mtok * SERVING_MARKUP

    @property
    def billed_cached_input_usd_per_mtok(self) -> float:
        return self.typical_cached_input_usd_per_mtok * SERVING_MARKUP


SERVING_PRICES: dict[str, ServingPrice] = {
    "Qwen/Qwen3.5-0.8B": ServingPrice(
        model_id="Qwen/Qwen3.5-0.8B",
        typical_input_usd_per_mtok=0.01,
        typical_output_usd_per_mtok=0.05,
        typical_cached_input_usd_per_mtok=0.002,
    ),
    "Qwen/Qwen3.5-2B": ServingPrice(
        model_id="Qwen/Qwen3.5-2B",
        typical_input_usd_per_mtok=0.02,
        typical_output_usd_per_mtok=0.10,
        typical_cached_input_usd_per_mtok=0.004,
    ),
    "Qwen/Qwen3.5-4B": ServingPrice(
        model_id="Qwen/Qwen3.5-4B",
        typical_input_usd_per_mtok=0.03,
        typical_output_usd_per_mtok=0.15,
        typical_cached_input_usd_per_mtok=0.006,
    ),
    # Typical rates are the unweighted mean of active serverless providers' per-1M rates
    # (SERVING_MARKUP adds the platform's 20%). Small models (<=4B) already matched the
    # provider mean; 9B, 27B and 35B are set/updated from it (see the 27B provider survey).
    "Qwen/Qwen3.5-9B": ServingPrice(
        model_id="Qwen/Qwen3.5-9B",
        typical_input_usd_per_mtok=0.114,
        typical_output_usd_per_mtok=0.19,
        typical_cached_input_usd_per_mtok=0.023,
    ),
    "Qwen/Qwen3.6-27B": ServingPrice(
        model_id="Qwen/Qwen3.6-27B",
        typical_input_usd_per_mtok=0.4254,
        typical_output_usd_per_mtok=3.055,
        typical_cached_input_usd_per_mtok=0.14,
    ),
    "Qwen/Qwen3.6-35B-A3B": ServingPrice(
        model_id="Qwen/Qwen3.6-35B-A3B",
        typical_input_usd_per_mtok=0.198,
        typical_output_usd_per_mtok=1.265,
        typical_cached_input_usd_per_mtok=0.066,
    ),
}
