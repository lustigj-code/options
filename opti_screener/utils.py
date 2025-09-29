"""Utility helpers for opti_screener."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger("opti_screener")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_list(value: str | None) -> list[str]:
    """Parse a comma-separated list string into a list of stripped values."""

    if not value:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def parse_weights(weights: str | None, defaults: dict[str, float]) -> dict[str, float]:
    """Parse a weight override string into a dictionary."""

    result = defaults.copy()
    if not weights:
        return result
    for part in weights.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid weight segment: {part!r}")
        key, raw_value = (item.strip() for item in part.split("=", 1))
        if key not in result:
            raise KeyError(f"Unknown weight key {key!r}")
        try:
            result[key] = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric weight for {key!r}: {raw_value!r}") from exc
    return result


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp *value* to the inclusive range [minimum, maximum]."""

    return max(minimum, min(maximum, value))


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Safely divide two numbers returning ``None`` if not possible."""

    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def ensure_directory(path: Path) -> None:
    """Ensure that the parent directory of ``path`` exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    """Write metadata JSON next to ``path``."""

    ensure_directory(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=_json_default)


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return value.__dict__
    if hasattr(value, "_asdict"):
        return value._asdict()
    if hasattr(value, "__slots__"):
        return {slot: getattr(value, slot) for slot in value.__slots__}
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return list(value)  # type: ignore[return-value]
    return value


def require_env(name: str) -> str:
    """Return the environment variable ``name`` or raise a helpful error."""

    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is required for the selected provider."
        )
    return value


def daterange_days(start: datetime, end: datetime) -> int:
    """Return the number of days between two dates."""

    return (end.date() - start.date()).days


def zscore(series: Iterable[float | None]) -> list[float | None]:
    """Return z-score normalized values handling ``None`` entries."""

    values = [value for value in series if value is not None]
    if not values:
        return [None for _ in series]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    std = math.sqrt(variance)
    if std == 0:
        return [0.0 if value is not None else None for value in series]
    return [((value - mean) / std) if value is not None else None for value in series]


def asdict_sans_none(obj: object) -> dict[str, object]:
    """Return a dictionary representation without ``None`` values."""

    data = asdict(obj)
    return {key: value for key, value in data.items() if value is not None}
