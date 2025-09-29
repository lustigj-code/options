"""Analytical routines for option screening."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from .models import OptionContract, OptionType
from .utils import safe_div

TRADING_DAYS_PER_YEAR = 252


@dataclass(slots=True)
class ContractAnalytics:
    """Holds derived analytics for a single option contract."""

    contract: OptionContract
    dte: int
    spread: float | None
    spread_pct: float | None
    iv_percentile: float | None
    iv_rank: float | None
    hv20: float | None
    hv60: float | None
    iv_to_hv: float | None
    expected_move_dollar: float | None
    expected_move_pct: float | None
    breakeven_price: float | None
    breakeven_gap_dollar: float | None
    breakeven_gap_pct: float | None
    breakeven_vs_em_pct: float | None
    theo_edge_pct: float | None


def compute_historical_volatility(close_series: pd.Series, window: int) -> float | None:
    """Return annualised historical volatility over *window* trading days."""

    close_series = close_series.dropna()
    if len(close_series) <= 1:
        return None
    returns = close_series.pct_change().dropna()
    if returns.empty:
        return None
    windowed = returns.iloc[-window:]
    if len(windowed) <= 1:
        return None
    hv = np.sqrt(TRADING_DAYS_PER_YEAR) * windowed.std(ddof=1)
    return float(hv)


def compute_iv_percentile(iv_history: pd.Series | None, current_iv: float | None) -> tuple[float, float]:
    """Return IV percentile and IV rank for the current IV value."""

    if iv_history is None or current_iv is None:
        return (np.nan, np.nan)
    history = iv_history.dropna()
    if history.empty:
        return (np.nan, np.nan)
    percentile = float((history <= current_iv).mean())
    min_iv = float(history.min())
    max_iv = float(history.max())
    if max_iv == min_iv:
        iv_rank = 0.5
    else:
        iv_rank = float((current_iv - min_iv) / (max_iv - min_iv))
    return (percentile, iv_rank)


def expected_move_from_straddle(contracts: Iterable[OptionContract], spot: float | None) -> dict[datetime, tuple[float | None, float | None]]:
    """Compute expected move per expiry based on ATM straddle mid prices."""

    if spot is None:
        return {}
    grouped: dict[tuple[datetime, float], dict[OptionType, OptionContract]] = {}
    for contract in contracts:
        if contract.mid is None:
            continue
        key = (contract.expiry, contract.strike)
        grouped.setdefault(key, {})[contract.option_type] = contract
    exp_to_candidates: defaultdict[datetime, list[tuple[float, float, float | None]]] = defaultdict(list)
    for (expiry, strike), pair in grouped.items():
        call = pair.get(OptionType.CALL)
        put = pair.get(OptionType.PUT)
        if not call or not put:
            continue
        call_mid = call.mid
        put_mid = put.mid
        if call_mid is None or put_mid is None:
            continue
        distance = abs(strike - spot)
        em_dollar = float(call_mid + put_mid)
        em_pct = safe_div(em_dollar, spot)
        exp_to_candidates[expiry].append((distance, em_dollar, float(em_pct) if em_pct is not None else None))
    result: dict[datetime, tuple[float | None, float | None]] = {}
    for expiry, candidates in exp_to_candidates.items():
        if not candidates:
            continue
        _, em_dollar, em_pct = min(candidates, key=lambda item: item[0])
        result[expiry] = (em_dollar, em_pct)
    return result


def breakeven_price(contract: OptionContract) -> float | None:
    """Return the breakeven price for the contract based on mid premium."""

    premium = contract.premium
    if premium is None:
        return None
    if contract.option_type is OptionType.CALL:
        return float(contract.strike + premium)
    return float(contract.strike - premium)


def breakeven_gap(contract: OptionContract, breakeven: float | None) -> tuple[float | None, float | None]:
    """Return the dollar and percent gap between spot and breakeven."""

    if breakeven is None or contract.underlying_price is None:
        return (None, None)
    spot = float(contract.underlying_price)
    gap_dollar = abs(breakeven - spot)
    gap_pct = safe_div(gap_dollar, spot)
    return (float(gap_dollar), float(gap_pct) if gap_pct is not None else None)


def _contract_key(contract: OptionContract) -> str:
    return f"{contract.ticker}-{contract.expiry.isoformat()}-{contract.option_type.value}-{contract.strike:.2f}"


def compute_contract_rows(
    contracts: list[OptionContract],
    hv20: float | None,
    hv60: float | None,
    iv_history: pd.Series | None,
    asof: datetime,
    expected_moves: dict[datetime, tuple[float | None, float | None]],
    theo_prices: dict[str, float | None],
) -> pd.DataFrame:
    """Return a dataframe containing analytics per contract."""

    rows: list[dict[str, object]] = []
    for contract in contracts:
        spot = contract.underlying_price
        current_iv = contract.mark_iv
        iv_percentile, iv_rank = compute_iv_percentile(iv_history, current_iv)
        iv_to_hv = None
        if hv20 is not None and current_iv is not None:
            hv_base = max(hv20, 1e-6)
            iv_to_hv = float(current_iv / hv_base)
        em_dollar, em_pct = expected_moves.get(contract.expiry, (None, None))
        be_price = breakeven_price(contract)
        gap_dollar, gap_pct = breakeven_gap(contract, be_price)
        be_vs_em_pct = None
        if gap_dollar is not None and em_dollar not in (None, 0):
            be_vs_em_pct = float(gap_dollar / em_dollar)
        theo_price = theo_prices.get(_contract_key(contract))
        theo_edge_pct = None
        if theo_price is not None and contract.mid not in (None, 0):
            theo_edge_pct = float((theo_price - contract.mid) / contract.mid)
        spread = None
        spread_pct = None
        if contract.ask is not None and contract.bid is not None:
            spread = float(contract.ask - contract.bid)
            if contract.mid not in (None, 0):
                spread_pct = float(spread / contract.mid)
        dte = (contract.expiry.date() - asof.date()).days
        rows.append(
            {
                "ticker": contract.ticker,
                "expiry": contract.expiry,
                "dte": dte,
                "type": contract.option_type.value,
                "strike": contract.strike,
                "bid": contract.bid,
                "ask": contract.ask,
                "mid": contract.mid,
                "mark_iv": current_iv,
                "delta": contract.delta,
                "gamma": contract.gamma,
                "vega": contract.vega,
                "theta": contract.theta,
                "volume": contract.volume,
                "open_interest": contract.open_interest,
                "spot": spot,
                "spread": spread,
                "spread_pct": spread_pct,
                "iv_percentile": iv_percentile,
                "iv_rank": iv_rank,
                "hv20": hv20,
                "hv60": hv60,
                "iv_to_hv": iv_to_hv,
                "expected_move_dollar": em_dollar,
                "expected_move_pct": em_pct,
                "breakeven_price": be_price,
                "breakeven_gap_dollar": gap_dollar,
                "breakeven_gap_pct": gap_pct,
                "breakeven_vs_em_pct": be_vs_em_pct,
                "theo_edge_pct": theo_edge_pct,
            }
        )
    return pd.DataFrame(rows)


def normalize_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized liquidity metrics to the dataframe."""

    df = df.copy()
    if "spread_pct" in df.columns:
        df["spread_score_component"] = 1 - df["spread_pct"].clip(lower=0, upper=0.20) / 0.20
    else:
        df["spread_score_component"] = 0.0
    for column, score_column in (("open_interest", "oi_z"), ("volume", "volume_z")):
        if column not in df.columns:
            df[score_column] = np.nan
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        std = float(values.std(ddof=0)) if values.notna().any() else 0.0
        if std == 0:
            df[score_column] = np.where(values.notna(), 0.0, np.nan)
        else:
            df[score_column] = (values - values.mean()) / std
    return df


def apply_scoring(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Compute weighted scores for each contract."""

    df = normalize_liquidity(df)
    w1 = weights.get("w1", 1.0)
    w2 = weights.get("w2", 0.5)
    w3 = weights.get("w3", 0.5)
    w4 = weights.get("w4", 1.0)
    w5 = weights.get("w5", 1.0)
    w6 = weights.get("w6", 1.0)
    w7 = weights.get("w7", 1.0)

    df["LiquidityScore"] = (
        w1 * df["spread_score_component"].fillna(0)
        + w2 * df["oi_z"].fillna(0)
        + w3 * df["volume_z"].fillna(0)
    )
    df["IVCheapScore"] = (
        w4 * (1 - df["iv_percentile"].fillna(1.0))
        + w5 * (1 - df["iv_to_hv"].fillna(1.0))
    )
    df["EdgeScore"] = w6 * df["theo_edge_pct"].fillna(0)

    def _em_component(row: pd.Series) -> float:
        em = row.get("expected_move_dollar")
        gap = row.get("breakeven_gap_dollar")
        if em in (None, 0) or gap is None:
            return 0.0
        return float(np.clip((em - gap) / em, 0, 1))

    df["EMScore"] = w7 * df.apply(_em_component, axis=1)
    df["TotalScore"] = df[["LiquidityScore", "IVCheapScore", "EdgeScore", "EMScore"]].sum(axis=1)
    return df


__all__ = [
    "ContractAnalytics",
    "apply_scoring",
    "breakeven_gap",
    "breakeven_price",
    "compute_contract_rows",
    "compute_historical_volatility",
    "compute_iv_percentile",
    "expected_move_from_straddle",
]
