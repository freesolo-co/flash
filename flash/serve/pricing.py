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
    "openbmb/MiniCPM5-1B": ServingPrice(
        model_id="openbmb/MiniCPM5-1B",
        typical_input_usd_per_mtok=0.01,
        typical_output_usd_per_mtok=0.05,
        typical_cached_input_usd_per_mtok=0.002,
    ),
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
    "Qwen/Qwen3.5-9B": ServingPrice(
        model_id="Qwen/Qwen3.5-9B",
        typical_input_usd_per_mtok=0.10,
        typical_output_usd_per_mtok=0.15,
        typical_cached_input_usd_per_mtok=0.020,
    ),
    "Qwen/Qwen3.6-35B-A3B": ServingPrice(
        model_id="Qwen/Qwen3.6-35B-A3B",
        typical_input_usd_per_mtok=0.15,
        typical_output_usd_per_mtok=1.00,
        typical_cached_input_usd_per_mtok=0.050,
    ),
}


def serving_price(model_id: str) -> ServingPrice:
    try:
        return SERVING_PRICES[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown serving model {model_id!r}") from exc
