"""Scoring and ranking logic for option screening."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .analytics import apply_scoring


@dataclass(slots=True)
class FilterConfig:
    """Filter configuration used to prune option chains."""

    min_dte: int = 30
    max_dte: int = 90
    min_delta: float = 0.25
    max_delta: float = 0.40
    min_put_delta: float = -0.40
    max_put_delta: float = -0.25
    max_spread_pct: float = 0.10
    max_spread_absolute: float = 0.10
    min_oi: float = 300
    min_volume: float = 1
    max_premium: float | None = None
    strategy: str = "both"  # calls, puts, both


DEFAULT_WEIGHTS = {
    "w1": 1.0,
    "w2": 0.5,
    "w3": 0.5,
    "w4": 1.0,
    "w5": 1.0,
    "w6": 1.0,
    "w7": 1.0,
}


def apply_filters(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    """Filter the dataframe according to the provided configuration."""

    filtered = df.copy()
    filtered = filtered[(filtered["dte"] >= config.min_dte) & (filtered["dte"] <= config.max_dte)]
    if config.strategy.lower() == "calls":
        filtered = filtered[filtered["type"].str.lower() == "call"]
    elif config.strategy.lower() == "puts":
        filtered = filtered[filtered["type"].str.lower() == "put"]
    # Delta filters
    delta_series = pd.to_numeric(filtered["delta"], errors="coerce")
    filtered = filtered.assign(delta_numeric=delta_series)
    is_call = filtered["type"].str.lower() == "call"
    is_put = filtered["type"].str.lower() == "put"
    call_mask = is_call & filtered["delta_numeric"].between(
        config.min_delta, config.max_delta, inclusive="both"
    )
    put_mask = is_put & filtered["delta_numeric"].between(
        config.max_put_delta, config.min_put_delta, inclusive="both"
    )
    filtered = filtered[call_mask | put_mask]
    # Liquidity filters
    filtered = filtered[
        (filtered["open_interest"].fillna(0) >= config.min_oi)
        & (filtered["volume"].fillna(0) >= config.min_volume)
    ]
    spread_pct_ok = filtered["spread_pct"].fillna(1.0) <= config.max_spread_pct
    spread_abs_ok = (
        filtered["spread"].fillna(config.max_spread_absolute)
        <= config.max_spread_absolute
    ) | (filtered["mid"].fillna(0) >= 2)
    filtered = filtered[spread_pct_ok | spread_abs_ok]
    if config.max_premium is not None:
        filtered = filtered[filtered["mid"].fillna(config.max_premium + 1) <= config.max_premium]
    return filtered.drop(columns=["delta_numeric"])


def rank_contracts(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Apply scoring and return a ranked dataframe sorted by total score."""

    scored = apply_scoring(df, weights)
    scored = scored.sort_values("TotalScore", ascending=False)
    return scored.reset_index(drop=True)


__all__ = ["FilterConfig", "DEFAULT_WEIGHTS", "apply_filters", "rank_contracts"]
