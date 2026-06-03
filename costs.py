"""
costs.py  —  COST + FINANCING layer of the STR investment model.

Pipeline position:
    data -> revenue -> [ COST ] -> returns -> risk -> rank
                         ^^^^^^ this file

JOB OF THIS FILE
----------------
Given a purchase price and a revenue figure, produce:
    (1) the operating-cost breakdown   -- what it costs to RUN the property
    (2) the financing breakdown        -- how the purchase is FUNDED (leverage)

These are two different things and we keep them in two dataclasses:
    CostAssumptions       -> operating costs (mgmt, cleaning, maintenance, tax...)
    FinancingAssumptions  -> leverage (down payment, rate, term, upfront cash)

THE ONE BIG IDEA (operating costs)
----------------------------------
Every operating cost is exactly ONE of three types. Type it, and the math is
automatic:
    (A) FIXED DOLLAR    flat $/yr        insurance, utilities, HOA
    (B) % OF PRICE      scales w/ price  property tax
    (C) % OF REVENUE    scales w/ income management, cleaning, maintenance, fees

LEVERAGE (your high-leverage ask)
---------------------------------
"High leverage" = you do NOT put the full price down. You borrow most of it.
    down_payment_pct = 0.25  -> 25% down, 75% borrowed   (typical, leveraged)
    down_payment_pct = 1.00  -> 100% down, all-cash      (no leverage)
    down_payment_pct = 0.10  -> 10% down                 (very high leverage)
Leverage is a multiplier on returns: the less cash you put in, the higher your
cash-on-cash % swings -- in BOTH directions. Low down also means a bigger loan,
bigger debt service, and a thinner break-even cushion. This file reports the
leverage explicitly (LTV) so that trade-off is never hidden.

All defaults are PLACEHOLDERS. Real numbers are location-specific. Not advice.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. OPERATING-COST ASSUMPTIONS  -- the fee knobs you wanted to play with.
# ---------------------------------------------------------------------------


@dataclass
class CostAssumptions:
    """Operating-cost knobs, grouped by the three cost TYPES."""

    # --- (A) FIXED DOLLAR ($/year) ---
    insurance_annual: float = 2_400.0
    utilities_annual: float = 3_600.0     # power/water/gas/internet (~$300/mo)
    hoa_annual: float = 0.0

    # --- (B) % OF PURCHASE PRICE ---
    property_tax_rate: float = 0.011      # 1.1%/yr (varies hugely by state)

    # --- (C) % OF REVENUE  (the fees you wanted to tune) ---
    management_pct: float = 0.20          # full-service STR mgmt; 0 if self-managed
    cleaning_pct: float = 0.0             # NET cleaning cost as % of revenue.
    #   Cleaning is usually charged to the guest and paid to the cleaner -- often a
    #   near-wash (so default 0). Set this to the NET you eat, or compute it from
    #   per-stay economics with cleaning_pct_from_per_stay() below.
    maintenance_pct: float = 0.05         # repairs / wear-and-tear
    supplies_pct: float = 0.04            # consumables, restocking
    capex_reserve_pct: float = 0.05       # reserve for roof/HVAC/big replacements
    platform_fee_pct: float = 0.03        # Airbnb host service fee

    def revenue_cost_pct(self) -> float:
        """Total fraction of revenue eaten by all type-(C) costs."""
        return (self.management_pct + self.cleaning_pct + self.maintenance_pct
                + self.supplies_pct + self.capex_reserve_pct + self.platform_fee_pct)


def cleaning_pct_from_per_stay(
    adr: float, avg_length_of_stay: float, net_cost_per_clean: float
) -> float:
    """Helper to convert per-turnover cleaning economics into a % of revenue,
    so you can tune cleaning either way.

        revenue per stay   = adr * avg_length_of_stay
        cleaning_pct       = net_cost_per_clean / revenue_per_stay

    `net_cost_per_clean` = what you PAY the cleaner minus what the GUEST pays you.
    If guests fully cover cleaning, net is ~0; if you eat $40/turnover, put 40.
    """
    revenue_per_stay = adr * avg_length_of_stay
    if revenue_per_stay <= 0:
        return 0.0
    return net_cost_per_clean / revenue_per_stay


# ---------------------------------------------------------------------------
# 2. FINANCING ASSUMPTIONS  -- leverage + upfront cash.
# ---------------------------------------------------------------------------


@dataclass
class FinancingAssumptions:
    """How the purchase is funded. This is where leverage lives."""

    down_payment_pct: float = 0.25        # <-- THE leverage knob. 1.0 = all-cash.
    interest_rate: float = 0.07           # annual mortgage rate
    loan_term_years: int = 30
    interest_only: bool = False           # high-leverage deals sometimes use I/O

    # Upfront cash beyond the down payment (easy to forget, STR-specific):
    closing_cost_pct: float = 0.03        # ~3% of price
    furnishing_cost: float = 25_000.0     # an STR must be furnished before night 1


# ---------------------------------------------------------------------------
# 3. THE COST BREAKDOWNS  -- pure functions: assumptions in, dollars out.
# ---------------------------------------------------------------------------


@dataclass
class OperatingCosts:
    """Annual operating-cost breakdown, by type."""
    fixed_dollar: float
    price_based: float
    revenue_based: float
    total: float


def operating_costs(
    purchase_price: float, annual_revenue: float, costs: CostAssumptions
) -> OperatingCosts:
    """Classify and total the operating costs (the three-type engine)."""
    fixed_dollar = costs.insurance_annual + costs.utilities_annual + costs.hoa_annual
    price_based = purchase_price * costs.property_tax_rate
    revenue_based = annual_revenue * costs.revenue_cost_pct()
    return OperatingCosts(
        fixed_dollar=fixed_dollar,
        price_based=price_based,
        revenue_based=revenue_based,
        total=fixed_dollar + price_based + revenue_based,
    )


@dataclass
class Financing:
    """Financing breakdown: the loan, the leverage, the upfront cash."""
    down_payment: float
    loan_amount: float
    ltv: float                 # loan-to-value = leverage (0.75 = 75% borrowed)
    closing_costs: float
    furnishing_cost: float
    cash_invested: float       # total cash out of pocket upfront
    annual_debt_service: float


def _annual_debt_service(
    loan_amount: float, rate: float, term_years: int, interest_only: bool
) -> float:
    """Annual mortgage payment. Amortizing by default; interest-only optional
    (interest-only minimizes payment -> maximizes leverage/cash flow, but you
    pay down no principal -- a real high-leverage trade-off)."""
    if loan_amount <= 0:
        return 0.0
    if interest_only:
        return loan_amount * rate
    r = rate / 12.0
    n = term_years * 12
    if r == 0:
        return (loan_amount / n) * 12
    monthly = loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return monthly * 12


def financing(purchase_price: float, fin: FinancingAssumptions) -> Financing:
    """Compute the financing/leverage breakdown."""
    down_payment = purchase_price * fin.down_payment_pct
    loan_amount = purchase_price - down_payment
    ltv = loan_amount / purchase_price if purchase_price else 0.0
    closing = purchase_price * fin.closing_cost_pct
    cash_invested = down_payment + closing + fin.furnishing_cost
    debt = _annual_debt_service(
        loan_amount, fin.interest_rate, fin.loan_term_years, fin.interest_only
    )
    return Financing(
        down_payment=down_payment,
        loan_amount=loan_amount,
        ltv=ltv,
        closing_costs=closing,
        furnishing_cost=fin.furnishing_cost,
        cash_invested=cash_invested,
        annual_debt_service=debt,
    )


# ---------------------------------------------------------------------------
# Quick self-test when run directly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    price = 450_000.0
    rev = 72_000.0
    oc = operating_costs(price, rev, CostAssumptions())
    print(f"opex: fixed ${oc.fixed_dollar:,.0f} | "
          f"tax ${oc.price_based:,.0f} | rev% ${oc.revenue_based:,.0f} "
          f"| TOTAL ${oc.total:,.0f}")
    for dp in (0.25, 0.10, 1.00):
        f = financing(price, FinancingAssumptions(down_payment_pct=dp))
        print(f"down {dp:>4.0%} -> LTV {f.ltv:>4.0%} | "
              f"cash in ${f.cash_invested:>8,.0f} | "
              f"debt ${f.annual_debt_service:>8,.0f}/yr")
