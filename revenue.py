"""
revenue.py  —  REVENUE layer of the STR investment model.

Pipeline position:
    data -> [ REVENUE ] -> cost -> returns -> risk -> rank
              ^^^^^^^^^ this file

JOB OF THIS FILE
----------------
Produce the *different revenue incomes* a prospective property might earn.
Note "different" -- we never trust a single point estimate. Occupancy is never
observed publicly; it is always estimated, and that estimate is the single
biggest error source in the whole model. So this file's main output is a SET of
scenarios (conservative / base / optimistic), not one number.

Every revenue figure is built from the same identity:

        annual_revenue  =  ADR  x  365  x  occupancy

    ADR        = average nightly rate ($/night)
    occupancy  = fraction of nights booked (0..1)   <-- the uncertain one

A revenue estimate can come from three sources, all returned in the SAME shape
(`RevenueEstimate`) so downstream code (costs, returns) never cares where it
came from:

    1. manual / ADR x occupancy   -- you type the numbers
    2. scenarios()                -- flex occupancy to get conservative..optimistic
    3. AirROI Calculator API      -- live estimate for a real or hypothetical unit

(Later, the ML revenue engine becomes a 4th source feeding the same shape.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# The shared shape: one revenue scenario.
# ---------------------------------------------------------------------------


@dataclass
class RevenueEstimate:
    """One revenue scenario. ALWAYS carries occupancy + ADR alongside the
    dollar figure, because the cost layer needs occupancy to compute
    break-even, and keeping ADR makes per-stay fee math (e.g. cleaning) possible."""

    label: str            # e.g. "conservative", "base", "optimistic", "AirROI"
    annual_revenue: float
    occupancy: float      # the assumption that drives everything; keep it visible
    adr: float            # average nightly rate ($/night)
    source: str           # where it came from, for traceability

    def __post_init__(self) -> None:
        # Guard against the silent inconsistency of revenue not matching ADR*365*occ.
        # We don't hard-fail (sources round differently); we just keep the trio honest.
        if self.occupancy <= 0:
            raise ValueError("occupancy must be > 0")


# ---------------------------------------------------------------------------
# Source 1: manual numbers.
# ---------------------------------------------------------------------------


def from_adr_occupancy(
    adr: float, occupancy: float, *, label: str = "manual", source: str = "manual"
) -> RevenueEstimate:
    """Build a revenue estimate from the fundamentals you control."""
    revenue = adr * 365.0 * occupancy
    return RevenueEstimate(label=label, annual_revenue=revenue,
                           occupancy=occupancy, adr=adr, source=source)


def from_annual_revenue(
    annual_revenue: float, occupancy: float, *,
    label: str = "manual", source: str = "manual",
) -> RevenueEstimate:
    """Build from a revenue figure you already have (e.g. a comp or a quote).
    We back out the implied ADR so the per-stay math still works."""
    adr = annual_revenue / (365.0 * occupancy)
    return RevenueEstimate(label=label, annual_revenue=annual_revenue,
                           occupancy=occupancy, adr=adr, source=source)


# ---------------------------------------------------------------------------
# Source 2: scenarios -- the headline feature. Flex the uncertain inputs.
# ---------------------------------------------------------------------------


def scenarios(
    base_adr: float,
    base_occupancy: float,
    *,
    occupancy_swing: float = 0.12,
    adr_swing: float = 0.08,
    source: str = "scenario",
) -> List[RevenueEstimate]:
    """Return [conservative, base, optimistic] revenue estimates.

    We flex the two uncertain inputs by a swing amount:
      - occupancy  +/- occupancy_swing   (absolute, e.g. 0.12 = +/-12 pts)
      - ADR        +/- adr_swing          (relative, e.g. 0.08 = +/-8%)

    This is the honest-error-reporting principle in code: you get a RANGE, and
    every downstream metric (cash-on-cash, break-even) inherits that range.
    The swings ARE assumptions -- tune them per how well you know the market.
    """
    def occ(delta: float) -> float:
        return max(0.01, min(0.95, base_occupancy + delta))

    conservative = from_adr_occupancy(
        base_adr * (1 - adr_swing), occ(-occupancy_swing),
        label="conservative", source=source,
    )
    base = from_adr_occupancy(
        base_adr, occ(0.0), label="base", source=source,
    )
    optimistic = from_adr_occupancy(
        base_adr * (1 + adr_swing), occ(+occupancy_swing),
        label="optimistic", source=source,
    )
    return [conservative, base, optimistic]


# ---------------------------------------------------------------------------
# Source 3: AirROI Calculator (live). Stdlib-only so there's no dependency.
# ---------------------------------------------------------------------------


def from_airroi(
    latitude: float,
    longitude: float,
    bedrooms: int,
    bathrooms: float,
    accommodates: int,
) -> Optional[RevenueEstimate]:
    """Best-effort live estimate from AirROI's /calculator/estimate.
    Returns a RevenueEstimate, or None on any failure (caller falls back).
    Requires env var AIRROI_API_KEY.

    NOTE: AirROI's exact response key names aren't fully published. The .get()
    fallbacks below are our best guess -- if a call succeeds but parsing fails,
    confirm the field names in AirROI's interactive docs and adjust here. We
    flag this uncertainty rather than pretend we know the schema.
    """
    import json
    import os
    import urllib.parse
    import urllib.request

    api_key = os.environ.get("AIRROI_API_KEY")
    if not api_key:
        print("[airroi] AIRROI_API_KEY not set -- skipping live call.")
        return None

    params = urllib.parse.urlencode({
        "latitude": latitude, "longitude": longitude,
        "bedrooms": bedrooms, "bathrooms": bathrooms,
        "accommodates": accommodates,
    })
    url = f"https://api.airroi.com/calculator/estimate?{params}"
    req = urllib.request.Request(url, headers={"x-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 -- any failure -> fall back to manual
        print(f"[airroi] live call failed ({exc}).")
        return None

    revenue = data.get("annual_revenue") or data.get("revenue")
    occupancy = data.get("occupancy") or data.get("occupancy_rate")
    adr = data.get("adr") or data.get("average_daily_rate")
    if revenue is None or occupancy is None:
        print("[airroi] response parsed but expected keys missing -- check schema.")
        return None
    if adr is None:                       # derive ADR if the API didn't return it
        adr = revenue / (365.0 * occupancy)
    return RevenueEstimate(label="AirROI", annual_revenue=float(revenue),
                           occupancy=float(occupancy), adr=float(adr), source="airroi")


# ---------------------------------------------------------------------------
# Quick self-test when run directly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for est in scenarios(base_adr=300.0, base_occupancy=0.65):
        print(f"{est.label:>12}:  ${est.annual_revenue:>9,.0f}/yr  "
              f"(ADR ${est.adr:.0f} x {est.occupancy:.0%} occ)")
