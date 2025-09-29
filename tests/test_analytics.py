from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from opti_screener.analytics import (
    apply_scoring,
    compute_contract_rows,
    compute_historical_volatility,
    expected_move_from_straddle,
)
from opti_screener.models import OptionContract, OptionType


def make_contract(**kwargs):
    defaults = dict(
        ticker="TEST",
        expiry=datetime.utcnow() + timedelta(days=45),
        strike=100.0,
        option_type=OptionType.CALL,
        bid=5.0,
        ask=6.0,
        mid=5.5,
        mark_iv=0.25,
        delta=0.3,
        gamma=0.05,
        vega=0.1,
        theta=-0.02,
        volume=500,
        open_interest=1000,
        underlying_price=100.0,
        provider_payload={},
    )
    defaults.update(kwargs)
    return OptionContract(**defaults)


def test_historical_volatility_matches_manual_calculation():
    prices = pd.Series([100 * (1.01) ** i for i in range(0, 30)])
    hv = compute_historical_volatility(prices, 20)
    returns = prices.pct_change().dropna().iloc[-20:]
    expected = np.sqrt(252) * returns.std(ddof=1)
    assert hv == pytest.approx(expected)


def test_expected_move_from_straddle():
    expiry = datetime.utcnow() + timedelta(days=45)
    call = make_contract(option_type=OptionType.CALL, expiry=expiry, strike=100, mid=5.0)
    put = make_contract(option_type=OptionType.PUT, expiry=expiry, strike=100, mid=4.0)
    data = expected_move_from_straddle([call, put], spot=100)
    em_dollar, em_pct = data[expiry]
    assert em_dollar == pytest.approx(9.0)
    assert em_pct == pytest.approx(0.09)


def test_apply_scoring_handles_missing_values():
    expiry = datetime.utcnow() + timedelta(days=45)
    contracts = [
        make_contract(expiry=expiry, mid=10.0, mark_iv=0.3, volume=100, open_interest=2000),
        make_contract(
            option_type=OptionType.PUT,
            expiry=expiry,
            strike=95,
            mid=8.0,
            mark_iv=0.28,
            volume=None,
            open_interest=None,
        ),
    ]
    expected_moves = {expiry: (10.0, 0.1)}
    frame = compute_contract_rows(
        contracts=contracts,
        hv20=0.2,
        hv60=0.25,
        iv_history=pd.Series([0.2, 0.25, 0.3]),
        asof=datetime.utcnow(),
        expected_moves=expected_moves,
        theo_prices={f"{c.ticker}-{c.expiry.isoformat()}-{c.option_type.value}-{c.strike:.2f}": None for c in contracts},
    )
    scored = apply_scoring(frame, {"w1": 1, "w2": 1, "w3": 1, "w4": 1, "w5": 1, "w6": 1, "w7": 1})
    assert "TotalScore" in scored.columns
    assert not scored["TotalScore"].isna().any()
