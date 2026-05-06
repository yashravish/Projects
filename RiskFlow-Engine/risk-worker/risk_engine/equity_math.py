from decimal import Decimal

def position(buys: list[Decimal], sells: list[Decimal]) -> Decimal:
    return sum(buys, Decimal("0")) - sum(sells, Decimal("0"))

def market_value(quantity: Decimal, latest_price: Decimal) -> Decimal:
    return quantity * latest_price

def pnl(quantity: Decimal, latest_price: Decimal, average_trade_price: Decimal) -> Decimal:
    return quantity * (latest_price - average_trade_price)
