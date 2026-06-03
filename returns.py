"""
returns.py  —  RETURNS layer + orchestrator of the STR investment model.

Pipeline position:
    data -> revenue -> cost -> [ RETURNS ] -> risk -> rank
                                ^^^^^^^^^^^ this file (also ties the others together)

JOB OF THIS FILE
----------------
Take a revenue estimate (from revenue.py) + cost & financing assumptions (from
costs.py) and produce ALL the headline numbers:

    NOI            = revenue - operating costs        (BEFORE mortgage; the asset)
    cash flow      = NOI - debt service               (AFTER mortgage; your pocket)
    cash-on-cash   = cash flow / cash invested        (PRIMARY metric)
    cap rate       = NOI / price
    gross yield    = revenue / price
    DSCR           = NOI / debt service               (lenders want > ~1.2x)
    break-even occ = occupancy where cash flow = 0    (your margin of safety)

Because revenue.py hands us a RANGE of scenarios, this file evaluates EACH one,
so the output is honest: conservative .. base .. optimistic, not a single
falsely-precise number.

Run it:   python3 returns.py
Edit the property + assumptions in main() at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import costs as costs_mod
import revenue as revenue_mod


# ---------------------------------------------------------------------------
# The result of analyzing ONE revenue scenario.
# ---------------------------------------------------------------------------


@dataclass
class ReturnsResult:
    label: str                 # which revenue scenario this is
    purchase_price: float
    annual_revenue: float
    occupancy: float

    # cost + financing breakdowns (kept whole so the report can show detail)
    operating: costs_mod.OperatingCosts
    financing: costs_mod.Financing

    # the two profit lines
    noi: float
    cash_flow: float

    # headline ratios
    cash_on_cash: float
    cap_rate: float
    gross_yield: float
    dscr: float
    break_even_occupancy: float


def analyze(
    purchase_price: float,
    rev: revenue_mod.RevenueEstimate,
    cost_assumptions: costs_mod.CostAssumptions,
    financing_assumptions: costs_mod.FinancingAssumptions,
) -> ReturnsResult:
    """Combine revenue + costs + financing into every return metric."""

    # 1. operating costs and financing come straight from costs.py
    oc = costs_mod.operating_costs(purchase_price, rev.annual_revenue, cost_assumptions)
    fin = costs_mod.financing(purchase_price, financing_assumptions)

    # 2. the two profit lines
    noi = rev.annual_revenue - oc.total
    cash_flow = noi - fin.annual_debt_service

    # 3. headline ratios
    cash_on_cash = cash_flow / fin.cash_invested if fin.cash_invested else 0.0
    cap_rate = noi / purchase_price if purchase_price else 0.0
    gross_yield = rev.annual_revenue / purchase_price if purchase_price else 0.0
    dscr = noi / fin.annual_debt_service if fin.annual_debt_service else float("inf")

    # 4. break-even occupancy (algebra, because type-(C) costs scale with revenue)
    #    max_revenue = revenue at 100% occ = annual_revenue / occupancy
    #    CF(occ) = R(occ)*(1 - rev_pct) - fixed - price_based - debt = 0
    #    R_be    = (fixed + price_based + debt) / (1 - rev_pct)
    #    occ_be  = R_be / max_revenue
    rev_pct = cost_assumptions.revenue_cost_pct()
    if rev.occupancy > 0 and (1 - rev_pct) > 0:
        max_revenue = rev.annual_revenue / rev.occupancy
        r_be = (oc.fixed_dollar + oc.price_based + fin.annual_debt_service) / (1 - rev_pct)
        break_even_occ = r_be / max_revenue if max_revenue else float("inf")
    else:
        break_even_occ = float("inf")

    return ReturnsResult(
        label=rev.label,
        purchase_price=purchase_price,
        annual_revenue=rev.annual_revenue,
        occupancy=rev.occupancy,
        operating=oc,
        financing=fin,
        noi=noi,
        cash_flow=cash_flow,
        cash_on_cash=cash_on_cash,
        cap_rate=cap_rate,
        gross_yield=gross_yield,
        dscr=dscr,
        break_even_occupancy=break_even_occ,
    )


def analyze_scenarios(
    purchase_price: float,
    estimates: List[revenue_mod.RevenueEstimate],
    cost_assumptions: costs_mod.CostAssumptions,
    financing_assumptions: costs_mod.FinancingAssumptions,
) -> List[ReturnsResult]:
    """Run analyze() across a whole revenue range (conservative..optimistic)."""
    return [analyze(purchase_price, e, cost_assumptions, financing_assumptions)
            for e in estimates]


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _pct(x: float) -> str:
    return f"{x * 100:,.1f}%"


def print_detail(r: ReturnsResult) -> None:
    """Full breakdown for a single scenario."""
    print("=" * 64)
    print(f"SCENARIO: {r.label.upper()}   (placeholders, not financial advice)")
    print("=" * 64)
    print(f"  Purchase price          {_money(r.purchase_price)}")
    print(f"  Revenue/yr              {_money(r.annual_revenue)}  "
          f"(at {_pct(r.occupancy)} occupancy)")
    print(f"  Leverage (LTV)          {_pct(r.financing.ltv)}  "
          f"borrowed | cash in {_money(r.financing.cash_invested)}")
    print("-" * 64)
    print("  OPERATING COSTS (annual)")
    print(f"    Fixed-dollar          {_money(r.operating.fixed_dollar)}")
    print(f"    % of price (tax)      {_money(r.operating.price_based)}")
    print(f"    % of revenue          {_money(r.operating.revenue_based)}")
    print(f"    TOTAL opex            {_money(r.operating.total)}")
    print("-" * 64)
    print(f"  NOI (before mortgage)   {_money(r.noi)}")
    print(f"  Debt service           -{_money(r.financing.annual_debt_service)}")
    print(f"  CASH FLOW (after mort.) {_money(r.cash_flow)}")
    print("-" * 64)
    print(f"  >> Cash-on-cash         {_pct(r.cash_on_cash)}   (PRIMARY)")
    print(f"     Cap rate             {_pct(r.cap_rate)}")
    print(f"     Gross yield          {_pct(r.gross_yield)}")
    dscr = "inf" if r.dscr == float("inf") else f"{r.dscr:.2f}x"
    print(f"     DSCR                 {dscr}")
    beo = ("infeasible" if r.break_even_occupancy == float("inf")
           else _pct(r.break_even_occupancy))
    print(f"     Break-even occupancy {beo}")
    if r.break_even_occupancy != float("inf"):
        cushion = r.occupancy - r.break_even_occupancy
        verdict = "OK" if cushion > 0.10 else "THIN -- risky"
        print(f"     Occupancy cushion    {_pct(cushion)}  ({verdict})")
    print()


def print_range(results: List[ReturnsResult]) -> None:
    """One-line-per-scenario comparison table -- the honest range view."""
    print("=" * 64)
    print("RETURNS RANGE  (conservative .. optimistic)")
    print("=" * 64)
    print(f"  {'scenario':>12} | {'revenue':>9} | {'cash flow':>9} | "
          f"{'CoC':>6} | {'break-even':>10}")
    print("  " + "-" * 58)
    for r in results:
        beo = "n/a" if r.break_even_occupancy == float("inf") else f"{r.break_even_occupancy:.0%}"
        print(f"  {r.label:>12} | {_money(r.annual_revenue):>9} | "
              f"{_money(r.cash_flow):>9} | {r.cash_on_cash:>6.1%} | {beo:>10}")
    print()


# ---------------------------------------------------------------------------
# RUN IT  -- the full pipeline for one candidate property.
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- the candidate property ----
    purchase_price = 450_000.0

    # ---- REVENUE: build a range of scenarios (revenue.py) ----
    # Either flex ADR x occupancy into conservative/base/optimistic...
    rev_scenarios = revenue_mod.scenarios(base_adr=300.0, base_occupancy=0.65)
    # ...or use a single manual figure:
    #   rev_scenarios = [revenue_mod.from_annual_revenue(72_000, occupancy=0.65)]
    # ...or pull a live AirROI estimate (needs AIRROI_API_KEY):
    #   est = revenue_mod.from_airroi(30.2672, -97.7431, 3, 2, 6)
    #   rev_scenarios = [est] if est else rev_scenarios

    # ---- COSTS + FINANCING: tune the fee + leverage knobs here (costs.py) ----
    cost_assumptions = costs_mod.CostAssumptions(
        management_pct=0.20,      # try 0.0 for self-managed
        cleaning_pct=0.0,         # net cleaning you eat; see cleaning_pct_from_per_stay
        maintenance_pct=0.05,
    )
    financing_assumptions = costs_mod.FinancingAssumptions(
        down_payment_pct=0.25,    # <-- leverage knob: 0.10 = high leverage, 1.0 = all-cash
        interest_rate=0.07,
    )

    # ---- RETURNS: evaluate every scenario ----
    results = analyze_scenarios(
        purchase_price, rev_scenarios, cost_assumptions, financing_assumptions
    )

    print_range(results)                 # the honest range, one line each
    for r in results:                    # full detail per scenario
        print_detail(r)


if __name__ == "__main__":
    main()
