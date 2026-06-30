"""Serving token prices for Flash deployments (per 1M tokens)."""

from __future__ import annotations

from dataclasses import dataclass

from flash.catalog import MODELS


@dataclass(frozen=True)
class ServingPrice:
    model_id: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cached_input_usd_per_mtok: float

    @property
    def billed_input_usd_per_mtok(self) -> float:
        return self.input_usd_per_mtok

    @property
    def billed_output_usd_per_mtok(self) -> float:
        return self.output_usd_per_mtok

    @property
    def billed_cached_input_usd_per_mtok(self) -> float:
        return self.cached_input_usd_per_mtok


SERVING_PRICES: dict[str, ServingPrice] = {
    "openbmb/MiniCPM5-1B": ServingPrice(model_id="openbmb/MiniCPM5-1B", input_usd_per_mtok=0.01, output_usd_per_mtok=0.05, cached_input_usd_per_mtok=0.002),
    "Qwen/Qwen3.5-0.8B": ServingPrice(model_id="Qwen/Qwen3.5-0.8B", input_usd_per_mtok=0.01, output_usd_per_mtok=0.05, cached_input_usd_per_mtok=0.002),
    "Qwen/Qwen3.5-2B": ServingPrice(model_id="Qwen/Qwen3.5-2B", input_usd_per_mtok=0.02, output_usd_per_mtok=0.10, cached_input_usd_per_mtok=0.004),
    "Qwen/Qwen3.5-4B": ServingPrice(model_id="Qwen/Qwen3.5-4B", input_usd_per_mtok=0.03, output_usd_per_mtok=0.15, cached_input_usd_per_mtok=0.006),
    "Qwen/Qwen3.5-9B": ServingPrice(model_id="Qwen/Qwen3.5-9B", input_usd_per_mtok=0.10, output_usd_per_mtok=0.15, cached_input_usd_per_mtok=0.020),
    "Qwen/Qwen3.6-35B-A3B": ServingPrice(model_id="Qwen/Qwen3.6-35B-A3B", input_usd_per_mtok=0.15, output_usd_per_mtok=1.00, cached_input_usd_per_mtok=0.050),
}


def serving_price(model_id: str) -> ServingPrice:
    try:
        return SERVING_PRICES[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown serving model {model_id!r}") from exc


def serving_price_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for model_id in sorted(MODELS):
        price = serving_price(model_id)
        rows.append(
            {
                "model_id": model_id,
                "billed_input_usd_per_mtok": price.billed_input_usd_per_mtok,
                "billed_output_usd_per_mtok": price.billed_output_usd_per_mtok,
                "billed_cached_input_usd_per_mtok": price.billed_cached_input_usd_per_mtok,
            }
        )
    return rows
