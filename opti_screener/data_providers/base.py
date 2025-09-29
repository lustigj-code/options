"""Base data provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import pandas as pd

from ..models import OptionContract


class BaseProvider(ABC):
    """Abstract interface for option data providers."""

    name: str = "base"

    @abstractmethod
    def get_chain(
        self, ticker: str, asof: datetime | None = None, max_contracts: int | None = None
    ) -> List[OptionContract]:
        """Return a list of :class:`OptionContract` for the given ticker."""

    @abstractmethod
    def get_underlying_ohlc(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """Return a OHLC dataframe indexed by datetime."""

    @abstractmethod
    def get_iv_history(self, ticker: str, lookback_days: int) -> pd.Series | None:
        """Return a series of IV30 values or ``None`` if unavailable."""

    @abstractmethod
    def get_theo_price(self, option: OptionContract) -> float | None:
        """Return the theoretical price for the contract or ``None``."""


__all__ = ["BaseProvider"]
