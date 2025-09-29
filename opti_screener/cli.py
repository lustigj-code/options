"""Command line interface for the opti_screener application."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .analytics import (
    compute_contract_rows,
    compute_historical_volatility,
    expected_move_from_straddle,
)
from .data_providers import PROVIDERS, BaseProvider, get_provider
from .ranking import DEFAULT_WEIGHTS, FilterConfig, apply_filters, rank_contracts
from .utils import LOGGER, parse_list, parse_weights, setup_logging, write_metadata

app = typer.Typer(help="Screen for inexpensive long-premium option candidates.")
console = Console()


def _load_yaml_config(path: Path | None) -> dict[str, object]:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyyaml is required for --config support") from exc
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of options")
    return data


def _load_tickers(tickers: str | None, tickers_file: Path | None) -> list[str]:
    items = set(parse_list(tickers))
    if tickers_file:
        file_content = tickers_file.read_text(encoding="utf-8")
        items.update(parse_list(file_content))
    if not items:
        raise typer.BadParameter("At least one ticker is required")
    return sorted(items)


def _instantiate_provider(name: str) -> BaseProvider:
    try:
        return get_provider(name)
    except Exception as exc:  # pragma: no cover - defensive
        raise typer.BadParameter(str(exc))


def _fallback_providers(primary: str) -> list[str]:
    providers = list(PROVIDERS)
    normalized = primary.lower()
    if normalized not in providers:
        raise typer.BadParameter(f"Unknown provider {primary!r}")
    providers.remove(normalized)
    return [normalized, *providers]


def _format_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


@app.command()
def run(
    tickers: str = typer.Option(..., help="Comma separated tickers."),
    provider: str = typer.Option("polygon", help="Primary data provider to use."),
    asof: str | None = typer.Option(None, help="Market date (YYYY-MM-DD)."),
    min_dte: int = typer.Option(30, help="Minimum days to expiry."),
    max_dte: int = typer.Option(90, help="Maximum days to expiry."),
    min_delta: float = typer.Option(0.25, help="Minimum delta for calls."),
    max_delta: float = typer.Option(0.40, help="Maximum delta for calls."),
    min_put_delta: float = typer.Option(-0.40, help="Minimum delta for puts."),
    max_put_delta: float = typer.Option(-0.25, help="Maximum delta for puts."),
    max_spread_pct: float = typer.Option(0.10, help="Maximum bid/ask spread percentage."),
    min_oi: int = typer.Option(300, help="Minimum open interest."),
    min_volume: int = typer.Option(1, help="Minimum same-day volume."),
    max_premium: float | None = typer.Option(None, help="Maximum option premium."),
    strategy: str = typer.Option("both", help="Limit to calls, puts, or both."),
    weights: str | None = typer.Option(None, help="Override scoring weights (e.g. w1=1,w2=0.5)."),
    output: Path = typer.Option(Path("results.csv"), help="CSV output path."),
    verbose: bool = typer.Option(False, help="Enable debug logging."),
    tickers_file: Path | None = typer.Option(None, help="Optional file with tickers."),
    config: Path | None = typer.Option(None, help="YAML config file overriding defaults."),
) -> None:
    """Run the option screener for the specified tickers."""

    setup_logging(verbose=verbose)
    config_overrides = _load_yaml_config(config)

    tickers = config_overrides.get("tickers", tickers)  # type: ignore[assignment]
    tickers_file = config_overrides.get("tickers_file", tickers_file)  # type: ignore[assignment]
    provider = config_overrides.get("provider", provider)  # type: ignore[assignment]
    asof = config_overrides.get("asof", asof)  # type: ignore[assignment]
    weights = config_overrides.get("weights", weights)  # type: ignore[assignment]
    max_premium = config_overrides.get("max_premium", max_premium)  # type: ignore[assignment]

    ticker_list = _load_tickers(tickers, Path(tickers_file) if tickers_file else None)
    weights_dict = parse_weights(weights, DEFAULT_WEIGHTS)
    asof_dt = _format_date(asof) or datetime.utcnow()

    filter_config = FilterConfig(
        min_dte=config_overrides.get("min_dte", min_dte),
        max_dte=config_overrides.get("max_dte", max_dte),
        min_delta=config_overrides.get("min_delta", min_delta),
        max_delta=config_overrides.get("max_delta", max_delta),
        min_put_delta=config_overrides.get("min_put_delta", min_put_delta),
        max_put_delta=config_overrides.get("max_put_delta", max_put_delta),
        max_spread_pct=config_overrides.get("max_spread_pct", max_spread_pct),
        min_oi=config_overrides.get("min_oi", min_oi),
        min_volume=config_overrides.get("min_volume", min_volume),
        max_premium=config_overrides.get("max_premium", max_premium),
        strategy=config_overrides.get("strategy", strategy),
    )

    provider_sequence = _fallback_providers(provider)

    frames: list[pd.DataFrame] = []
    used_provider = provider_sequence[0]

    for ticker in ticker_list:
        contract_frame = None
        for provider_name in provider_sequence:
            try:
                provider_instance = _instantiate_provider(provider_name)
                frame = _process_ticker(
                    provider_instance,
                    ticker,
                    asof_dt,
                    filter_config,
                )
                if frame is not None and not frame.empty:
                    contract_frame = frame
                    used_provider = provider_name
                    break
            except Exception as exc:
                LOGGER.error("Provider %s failed for %s: %s", provider_name, ticker, exc)
                continue
        if contract_frame is None or contract_frame.empty:
            LOGGER.warning("No contracts returned for %s", ticker)
            continue
        frames.append(contract_frame)

    if not frames:
        raise typer.Exit(code=1)

    combined = pd.concat(frames, ignore_index=True)
    filtered = apply_filters(combined, filter_config)
    if filtered.empty:
        LOGGER.warning("No contracts passed the filter criteria.")
        raise typer.Exit(code=1)

    ranked = rank_contracts(filtered, weights_dict)
    write_outputs(ranked, output, ticker_list, weights_dict, filter_config, used_provider, asof_dt)
    display_table(ranked)


def _process_ticker(
    provider: BaseProvider,
    ticker: str,
    asof: datetime,
    filter_config: FilterConfig,
) -> pd.DataFrame | None:
    lookback = max(filter_config.max_dte + 60, 252)
    chain = provider.get_chain(ticker, asof=asof)
    if not chain:
        LOGGER.warning("%s returned empty chain", provider.name)
        return None
    ohlc = provider.get_underlying_ohlc(ticker, lookback)
    hv20 = compute_historical_volatility(ohlc.get("close", pd.Series(dtype=float)), 20)
    hv60 = compute_historical_volatility(ohlc.get("close", pd.Series(dtype=float)), 60)
    iv_history = provider.get_iv_history(ticker, 252)

    spot = None
    if not chain[0].underlying_price and not ohlc.empty:
        spot = float(ohlc["close"].iloc[-1])
    for contract in chain:
        if contract.underlying_price is None and spot is not None:
            contract.provider_payload.setdefault("underlying_asset", {})
            contract.underlying_price = spot
    if spot is None:
        spot = chain[0].underlying_price

    expected_moves = expected_move_from_straddle(chain, spot)
    theo_prices = {
        f"{contract.ticker}-{contract.expiry.isoformat()}-{contract.option_type.value}-{contract.strike:.2f}": provider.get_theo_price(contract)
        for contract in chain
    }
    frame = compute_contract_rows(
        contracts=chain,
        hv20=hv20,
        hv60=hv60,
        iv_history=iv_history,
        asof=asof,
        expected_moves=expected_moves,
        theo_prices=theo_prices,
    )
    return frame


def write_outputs(
    ranked: pd.DataFrame,
    output: Path,
    tickers: Iterable[str],
    weights: dict[str, float],
    filters: FilterConfig,
    provider: str,
    asof: datetime,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False)
    metadata = {
        "tickers": list(tickers),
        "weights": weights,
        "filters": filters.__dict__,
        "provider": provider,
        "timestamp": datetime.utcnow().isoformat(),
        "asof": asof.isoformat(),
    }
    write_metadata(output, metadata)
    console.print(f"Saved results to {output}")


def display_table(df: pd.DataFrame, limit: int = 15) -> None:
    table = Table(show_lines=False)
    columns = [
        "ticker",
        "expiry",
        "dte",
        "type",
        "strike",
        "delta",
        "mid",
        "spread_pct",
        "open_interest",
        "volume",
        "mark_iv",
        "iv_percentile",
        "hv20",
        "iv_to_hv",
        "expected_move_pct",
        "breakeven_vs_em_pct",
        "theo_edge_pct",
        "TotalScore",
    ]
    headers = {
        "ticker": "Ticker",
        "expiry": "Expiry",
        "dte": "DTE",
        "type": "Type",
        "strike": "Strike",
        "delta": "Delta",
        "mid": "Mid",
        "spread_pct": "Bid-Ask%",
        "open_interest": "OI",
        "volume": "Vol",
        "mark_iv": "IV30",
        "iv_percentile": "IVP",
        "hv20": "HV20",
        "iv_to_hv": "IV/HV",
        "expected_move_pct": "EM%",
        "breakeven_vs_em_pct": "Breakeven% vs EM",
        "theo_edge_pct": "TheoEdge%",
        "TotalScore": "TotalScore",
    }
    for column in columns:
        table.add_column(headers[column])
    subset = df.head(limit)
    for _, row in subset.iterrows():
        table.add_row(
            str(row.get("ticker")),
            row.get("expiry").strftime("%Y-%m-%d") if isinstance(row.get("expiry"), datetime) else str(row.get("expiry")),
            f"{row.get('dte')}",
            str(row.get("type")),
            f"{row.get('strike'):.2f}" if pd.notna(row.get("strike")) else "",
            _format_float(row.get("delta")),
            _format_float(row.get("mid")),
            _format_percent(row.get("spread_pct")),
            _format_int(row.get("open_interest")),
            _format_int(row.get("volume")),
            _format_percent(row.get("mark_iv")),
            _format_percent(row.get("iv_percentile")),
            _format_percent(row.get("hv20")),
            _format_ratio(row.get("iv_to_hv")),
            _format_percent(row.get("expected_move_pct")),
            _format_percent(row.get("breakeven_vs_em_pct")),
            _format_percent(row.get("theo_edge_pct")),
            _format_float(row.get("TotalScore")),
        )
    console.print(table)


def _format_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.2f}"


def _format_int(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):,}"


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value * 100:.1f}%" if abs(value) < 5 else f"{value:.2f}"


def _format_ratio(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.2f}"


__all__ = ["app"]
