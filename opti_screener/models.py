"""Domain models for the opti_screener package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OptionType(str, Enum):
    """Enumerates supported option contract types."""

    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> int:
        """Return +1 for calls and -1 for puts."""

        return 1 if self is OptionType.CALL else -1


@dataclass(slots=True)
class OptionContract:
    """Represents an option contract and associated market data."""

    ticker: str
    expiry: datetime
    strike: float
    option_type: OptionType
    bid: float | None
    ask: float | None
    mid: float | None
    mark_iv: float | None
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    volume: float | None
    open_interest: float | None
    underlying_price: float | None
    provider_payload: dict[str, Any] = field(default_factory=dict)

    def copy_with(self, **updates: Any) -> "OptionContract":
        """Return a shallow copy with the given fields updated."""

        data = self.__dict__.copy()
        data.update(updates)
        return OptionContract(**data)

    @property
    def premium(self) -> float | None:
        """Return the tradable premium using the mid price if available."""

        if self.mid is not None:
            return self.mid
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.bid or self.ask


__all__ = ["OptionContract", "OptionType"]
