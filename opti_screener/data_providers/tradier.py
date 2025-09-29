"""Tradier REST data provider."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from ..models import OptionContract, OptionType
from ..utils import LOGGER, require_env
from .base import BaseProvider


class TradierProvider(BaseProvider):
    """Data provider that retrieves information from Tradier."""

    name = "tradier"
    BASE_URL = "https://api.tradier.com"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.api_key = require_env("TRADIER_API_KEY")
        self.session = session or requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.BASE_URL}{path}"
        params = params or {}
        backoff = 1.0
        for attempt in range(5):
            response = self.session.request(method, url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504}:
                LOGGER.warning(
                    "Tradier request %s failed with status %s; retrying in %.1fs",
                    path,
                    response.status_code,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            try:
                payload = response.json()
            except Exception:  # pragma: no cover
                response.raise_for_status()
            message = payload.get("fault", {}).get("faultstring") or payload.get("message") or str(payload)
            raise RuntimeError(f"Tradier API error ({response.status_code}): {message}")
        raise RuntimeError(f"Tradier API request {path} failed after retries")

    def get_chain(
        self, ticker: str, asof: datetime | None = None, max_contracts: int | None = None
    ) -> list[OptionContract]:
        params = {
            "symbol": ticker.upper(),
            "greeks": "true",
        }
        if asof is not None:
            params["expiration"] = asof.strftime("%Y-%m-%d")
        data = self._request("GET", "/v1/markets/options/chains", params=params)
        options = data.get("options", {}).get("option", [])
        if isinstance(options, dict):
            options = [options]
        contracts: list[OptionContract] = []
        for item in options:
            option_type = OptionType.CALL if item.get("option_type") == "call" else OptionType.PUT
            expiry_str = item.get("expiration_date") or item.get("expiration")
            if not expiry_str:
                continue
            expiry = datetime.fromisoformat(expiry_str)
            greeks = item.get("greeks") or {}
            bid = _to_float(item.get("bid"))
            ask = _to_float(item.get("ask"))
            mid = _mid_from_quotes(bid, ask)
            contracts.append(
                OptionContract(
                    ticker=ticker.upper(),
                    expiry=expiry,
                    strike=float(item.get("strike") or 0.0),
                    option_type=option_type,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    mark_iv=_to_float(greeks.get("mid_iv")) or _to_float(item.get("implied_volatility")),
                    delta=_to_float(greeks.get("delta")),
                    gamma=_to_float(greeks.get("gamma")),
                    vega=_to_float(greeks.get("vega")),
                    theta=_to_float(greeks.get("theta")),
                    volume=_to_float(item.get("volume")),
                    open_interest=_to_float(item.get("open_interest")),
                    underlying_price=_to_float(item.get("underlying_price")),
                    provider_payload=item,
                )
            )
            if max_contracts and len(contracts) >= max_contracts:
                break
        return contracts

    def get_underlying_ohlc(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days * 2)
        params = {
            "symbol": ticker.upper(),
            "interval": "daily",
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        }
        data = self._request("GET", "/v1/markets/history", params=params)
        history = data.get("history", {}).get("day", [])
        if isinstance(history, dict):
            history = [history]
        if not history:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(history)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
        frame = frame.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        return frame[["open", "high", "low", "close", "volume"]].astype(float)

    def get_iv_history(self, ticker: str, lookback_days: int) -> pd.Series | None:
        params = {
            "symbol": ticker.upper(),
            "interval": "daily",
        }
        data = self._request("GET", "/v1/markets/options/iv", params=params)
        series = data.get("implied_volatility", {}).get("data")
        if not series:
            return None
        frame = pd.DataFrame(series)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
        frame = frame.sort_index().tail(lookback_days)
        if "iv30" in frame.columns:
            return frame["iv30"].astype(float)
        if "value" in frame.columns:
            return frame["value"].astype(float)
        return None

    def get_theo_price(self, option: OptionContract) -> float | None:
        if option.provider_payload and "theoretical" in option.provider_payload:
            return _to_float(option.provider_payload.get("theoretical"))
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover
        return None


def _mid_from_quotes(bid: float | None, ask: float | None) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return bid or ask


__all__ = ["TradierProvider"]
