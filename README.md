# opti_screener

`opti_screener` is a Python 3.11 CLI that surfaces long-premium option ideas that look "cheap to buy" based on implied volatility, liquidity and theoretical edge metrics. It integrates Polygon.io (primary) and Tradier (fallback) market data feeds, computes historical volatility analytics, and ranks contracts with a transparent scoring model.

## Features

- Adapter-based data providers with automatic fallback between Polygon.io and Tradier.
- Computes HV20/HV60, IV percentile & rank, IV/HV ratios, expected move via ATM straddles and breakeven gaps.
- Configurable filters for DTE, delta, spreads, open interest, premium caps and strategy (calls/puts/both).
- Weighted scoring model covering liquidity, IV cheapness, theoretical edge and expected-move alignment.
- Outputs both a Rich-powered console table and a full CSV file with companion metadata JSON.
- Optional YAML configuration files and ticker lists.

## Project Structure

```
opti_screener/
  __init__.py
  analytics.py
  cli.py
  models.py
  ranking.py
  utils.py
  data_providers/
    __init__.py
    base.py
    polygon.py
    tradier.py
requirements.txt
README.md
tests/
```

## Getting Started

### 1. Clone and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file (or export in your shell) containing the vendor API keys:

```
POLYGON_API_KEY=your_polygon_api_key
TRADIER_API_KEY=your_tradier_api_key
```

### 3. Example usage

```bash
opti-screener \
  --tickers "AAPL,MSFT,AMD" \
  --provider polygon \
  --asof "2025-09-29" \
  --min-dte 30 --max-dte 90 \
  --min-delta 0.25 --max-delta 0.40 \
  --max-spread-pct 0.10 \
  --min-oi 300 \
  --max-premium 150 \
  --output results.csv \
  --weights "w1=1,w2=0.5,w3=0.5,w4=1,w5=1,w6=1,w7=1"
```

Additional options:

- `--tickers-file`: provide a CSV/TXT file with comma or newline separated tickers.
- `--config config.yaml`: use a YAML config to override CLI defaults.
- `--strategy`: choose `calls`, `puts`, or `both` (default).

### 4. Output

- `results.csv`: full dataset of ranked option contracts.
- `results.csv.meta.json`: metadata including timestamp, tickers, weights, filters and provider used.
- Console table: top 15 ranked contracts with key metrics and total score.

## Testing

```bash
pytest
```

## Scoring Model Overview

```
LiquidityScore  = w1*(1 - SpreadPctNorm) + w2*OI_z + w3*Vol_z
IVCheapScore    = w4*(1 - IVPercentile) + w5*(1 - IV_to_HV)
EdgeScore       = w6*((Theo - Mid)/Mid)
EMScore         = w7*((EM_$ - BreakevenGap_$)/EM_$)
TotalScore      = LiquidityScore + IVCheapScore + EdgeScore + EMScore
```

Weights are tunable with the `--weights` flag or YAML configuration.

## FAQ

**Why does IV often exceed HV?**
: Implied volatility embeds market expectations of future variance plus risk premium, so it typically prices higher than realized volatility.

**What if a provider lacks theoretical prices?**
: The screener treats missing theo data as zero edge. You can prioritise liquidity or IV cheapness through weights.

**How do I interpret breakeven vs expected move?**
: A positive EM score indicates the breakeven point lies inside the expected move, suggesting a favourable probability of profit if the move materialises.

## License

This project is provided as-is without warranty. Consult your broker's API terms before use.
