from decimal import Decimal, getcontext
getcontext().prec = 28

def bond_price(face_value: Decimal, coupon_rate: Decimal, years_to_maturity: int, market_yield: Decimal) -> Decimal:
    years = max(1, years_to_maturity)
    coupon = face_value * coupon_rate
    discount = Decimal("1") + market_yield
    pv_coupons = sum(coupon / (discount ** t) for t in range(1, years + 1))
    pv_face = face_value / (discount ** years)
    return pv_coupons + pv_face

def dv01(face_value: Decimal, coupon_rate: Decimal, years_to_maturity: int, market_yield: Decimal) -> Decimal:
    base = bond_price(face_value, coupon_rate, years_to_maturity, market_yield)
    shocked = bond_price(face_value, coupon_rate, years_to_maturity, market_yield + Decimal("0.0001"))
    return shocked - base
