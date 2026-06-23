"""Serving token prices for Flash deployments.

Prices are listed per 1M tokens. Flash bills the base model serverless list price
plus the fixed serving markup.
"""

from __future__ import annotations

from dataclasses import dataclass

from flash.catalog import MODELS

SERVING_MARKUP = 1.20


@dataclass(frozen=True)
class ServingPrice:
    model_id: str
    base_input_usd_per_mtok: float
    base_output_usd_per_mtok: float

    @property
    def billed_input_usd_per_mtok(self) -> float:
        return self.base_input_usd_per_mtok * SERVING_MARKUP

    @property
    def billed_output_usd_per_mtok(self) -> float:
        return self.base_output_usd_per_mtok * SERVING_MARKUP


SERVING_PRICES: dict[str, ServingPrice] = {
    "openbmb/MiniCPM5-1B": ServingPrice("openbmb/MiniCPM5-1B", 0.05, 0.10),
    "Qwen/Qwen3.5-0.8B": ServingPrice("Qwen/Qwen3.5-0.8B", 0.05, 0.10),
    "Qwen/Qwen3.5-2B": ServingPrice("Qwen/Qwen3.5-2B", 0.10, 0.20),
    "Qwen/Qwen3.5-4B": ServingPrice("Qwen/Qwen3.5-4B", 0.20, 0.40),
    "Qwen/Qwen3.5-9B": ServingPrice("Qwen/Qwen3.5-9B", 0.50, 1.00),
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
                "base_input_usd_per_mtok": price.base_input_usd_per_mtok,
                "base_output_usd_per_mtok": price.base_output_usd_per_mtok,
                "billed_input_usd_per_mtok": price.billed_input_usd_per_mtok,
                "billed_output_usd_per_mtok": price.billed_output_usd_per_mtok,
            }
        )
    return rows
