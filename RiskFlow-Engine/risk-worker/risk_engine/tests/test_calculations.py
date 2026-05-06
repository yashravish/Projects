from decimal import Decimal
from risk_engine.equity_math import market_value, pnl, position
from risk_engine.bond_math import bond_price, dv01
from risk_engine.calculations import var95
from risk_engine.stress import equity_down, rates_up


def test_equity_pnl_calculation():
    assert position([Decimal("100")], [Decimal("25")]) == Decimal("75")
    assert market_value(Decimal("75"), Decimal("10")) == Decimal("750")
    assert pnl(Decimal("75"), Decimal("10"), Decimal("8")) == Decimal("150")


def test_bond_price_calculation():
    price = bond_price(Decimal("1000"), Decimal("0.05"), 5, Decimal("0.04"))
    assert price > Decimal("1000")


def test_dv01_calculation():
    value = dv01(Decimal("1000"), Decimal("0.05"), 5, Decimal("0.04"))
    assert value < Decimal("0")


def test_var_fallback_calculation():
    assert var95([], Decimal("1000000")) == Decimal("20000.00")


def test_stress_calculation():
    assert equity_down(Decimal("1000"), Decimal("0.05")) == Decimal("-50.00")
    assert rates_up(Decimal("-3"), 25) == Decimal("-75")
