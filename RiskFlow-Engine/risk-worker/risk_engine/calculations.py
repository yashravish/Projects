from decimal import Decimal


def var95(values: list[Decimal], total_market_value: Decimal) -> Decimal:
    if len(values) < 2:
        return abs(total_market_value * Decimal("0.02"))
    try:
        import numpy as np
        import pandas as pd
        series = pd.Series([float(v) for v in values]).pct_change().dropna()
        if series.empty:
            return abs(total_market_value * Decimal("0.02"))
        quantile = np.percentile(series, 5)
        return abs(Decimal(str(quantile)) * total_market_value)
    except ImportError:
        returns = []
        previous = values[0]
        for value in values[1:]:
            if previous != 0:
                returns.append((value - previous) / previous)
            previous = value
        if not returns:
            return abs(total_market_value * Decimal("0.02"))
        returns.sort()
        index = max(0, int(len(returns) * Decimal("0.05")) - 1)
        return abs(returns[index] * total_market_value)


def portfolio_summary(equity_exposure: Decimal, fixed_income_exposure: Decimal, pnl: Decimal, dv01: Decimal) -> dict[str, Decimal]:
    total = equity_exposure + fixed_income_exposure
    return {"totalMarketValue": total, "totalPnL": pnl, "var95": var95([], total), "stressEquityDown5": equity_exposure * Decimal("-0.05"), "stressRatesUp25bps": dv01 * Decimal(25)}
