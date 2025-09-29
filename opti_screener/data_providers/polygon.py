"""Polygon.io data provider implementation."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from ..models import OptionContract, OptionType
from ..utils import LOGGER, require_env
from .base import BaseProvider


class PolygonProvider(BaseProvider):
    """Fetch data from the Polygon.io REST API."""

    name = "polygon"
    BASE_URL = "https://api.polygon.io"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.api_key = require_env("POLYGON_API_KEY")
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.BASE_URL}{path}"
        params = params.copy() if params else {}
        params.setdefault("apiKey", self.api_key)
        backoff = 1.0
        for attempt in range(5):
            response = self.session.request(method, url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504}:
                LOGGER.warning(
                    "Polygon request %s failed with status %s; retrying in %.1fs",
                    path,
                    response.status_code,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            try:
                payload = response.json()
            except Exception:  # pragma: no cover - defensive
                response.raise_for_status()
            message = payload.get("error") or payload.get("message") or str(payload)
            raise RuntimeError(f"Polygon API error ({response.status_code}): {message}")
        raise RuntimeError(f"Polygon API request {path} failed after retries")

    # ------------------------------------------------------------------
    # API surface
    # ------------------------------------------------------------------
    def get_chain(
        self, ticker: str, asof: datetime | None = None, max_contracts: int | None = None
    ) -> list[OptionContract]:
        params = {"underlying_ticker": ticker.upper()}
        if asof is not None:
            params["as_of"] = asof.strftime("%Y-%m-%d")
        data = self._request("GET", "/v3/snapshot/options", params=params)
        results = data.get("results", [])
        contracts: list[OptionContract] = []
        for item in results:
            option_data = item.get("details", {})
            greeks = item.get("greeks", {})
            quote = item.get("last_quote", {})
            bid = quote.get("bid_price")
            ask = quote.get("ask_price")
            mid = item.get("day") or {}
            mid_price = mid.get("mid")
            if mid_price is None and bid is not None and ask is not None:
                mid_price = (bid + ask) / 2
            ticker_symbol = option_data.get("ticker", "")
            if not ticker_symbol:
                continue
            option_type = OptionType.CALL if option_data.get("contract_type") == "call" else OptionType.PUT
            expiry_str = option_data.get("expiration_date") or option_data.get("expiration")
            if not expiry_str:
                continue
            expiry = datetime.fromisoformat(expiry_str)
            contracts.append(
                OptionContract(
                    ticker=ticker_symbol.split(":")[0],
                    expiry=expiry,
                    strike=float(option_data.get("strike_price") or 0.0),
                    option_type=option_type,
                    bid=_to_float(bid),
                    ask=_to_float(ask),
                    mid=_to_float(mid_price),
                    mark_iv=_to_float(greeks.get("iv")),
                    delta=_to_float(greeks.get("delta")),
                    gamma=_to_float(greeks.get("gamma")),
                    vega=_to_float(greeks.get("vega")),
                    theta=_to_float(greeks.get("theta")),
                    volume=_to_float(item.get("volume")),
                    open_interest=_to_float(option_data.get("open_interest")),
                    underlying_price=_to_float(item.get("underlying_asset", {}).get("price")),
                    provider_payload=item,
                )
            )
            if max_contracts and len(contracts) >= max_contracts:
                break
        return contracts

    def get_underlying_ohlc(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days * 2)
        path = f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
        data = self._request("GET", path)
        results = data.get("results", [])
        if not results:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(results)
        frame["t"] = pd.to_datetime(frame["t"], unit="ms")
        frame = frame.set_index("t")
        frame = frame.rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        return frame[["open", "high", "low", "close", "volume"]]

    def get_iv_history(self, ticker: str, lookback_days: int) -> pd.Series | None:
        params = {
            "ticker": ticker.upper(),
            "timespan": "day",
            "adjusted": "true",
            "window": 1,
            "limit": lookback_days,
        }
        data = self._request("GET", "/v1/indicators/iv", params=params)
        results = data.get("results", {})
        values = results.get("values") or []
        if not values:
            return None
        frame = pd.DataFrame(values)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
        frame = frame.set_index("timestamp")
        if "value" in frame.columns:
            series = frame["value"].astype(float)
        elif "iv" in frame.columns:
            series = frame["iv"].astype(float)
        else:
            return None
        return series

    def get_theo_price(self, option: OptionContract) -> float | None:
        details = option.provider_payload.get("details") if option.provider_payload else None
        if details and "theoretical_price" in details:
            return _to_float(details.get("theoretical_price"))
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


__all__ = ["PolygonProvider"]
