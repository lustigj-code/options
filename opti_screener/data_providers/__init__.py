"""Data provider factory."""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseProvider
from .polygon import PolygonProvider
from .tradier import TradierProvider

PROVIDERS: Dict[str, Type[BaseProvider]] = {
    PolygonProvider.name: PolygonProvider,
    TradierProvider.name: TradierProvider,
}


def get_provider(name: str) -> BaseProvider:
    """Instantiate a provider by name."""

    normalized = name.lower()
    if normalized not in PROVIDERS:
        raise KeyError(f"Unknown provider {name!r}")
    return PROVIDERS[normalized]()


__all__ = ["BaseProvider", "PolygonProvider", "TradierProvider", "get_provider", "PROVIDERS"]
