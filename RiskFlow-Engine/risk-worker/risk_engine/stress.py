from decimal import Decimal

def equity_down(exposure: Decimal, pct: Decimal) -> Decimal:
    return exposure * -abs(pct)

def rates_up(dv01_value: Decimal, basis_points: int) -> Decimal:
    return dv01_value * Decimal(basis_points)
