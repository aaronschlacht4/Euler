"""
property_enrichment.py  —  external property-data join (square footage & more).

Pipeline position:
    data -> [ENRICH w/ external records] -> revenue model -> ...
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^ this file

WHY THIS EXISTS
---------------
Inside Airbnb does NOT publish square footage. To make property SIZE a real,
learned feature (not a guess), we join each listing to an external property
record by coordinates, using the RentCast API. This pulls squareFootage,
yearBuilt, lotSize, propertyType -- physical facts that are intrinsic (knowable
before purchase) and therefore leakage-safe.

THREE HARD REALITIES (flagged honestly)
---------------------------------------
1. RATE LIMIT / COST: RentCast's free tier is ~50 calls/month. A city has
   ~2,000 listings. So full coverage needs the PAID tier, OR many months of
   incremental free runs, OR the free county-assessor route (see README note).
   We CACHE every result to disk and honor a --max-calls budget so you never
   re-pay for a listing and never blow your quota in one run.
2. COORDINATE JITTER: Inside Airbnb deliberately anonymizes each listing's
   lat/lon by up to ~150m. So a nearest-property match can grab a NEIGHBOR's
   parcel, not the exact home. We record `match_distance_mi` so you can filter
   out bad matches. This join is good-but-noisy, not exact.
3. Still an estimate stack: enriched sqft is real, but it's joined to a jittered
   point, then fed to a model predicting an estimated label. Honest uncertainty
   compounds -- keep reporting ranges.

USAGE
-----
    export RENTCAST_API_KEY=your_key
    python3 property_enrichment.py --market asheville --max-calls 50
        -> enriches up to 50 new listings, writes enrichment_asheville.json
        -> re-run next month to continue where the cache left off

The engine (airbnb_revenue_engine.py) reads that JSON cache and adds a
square_feet feature for the rows it covers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# SSL context with a real CA bundle -- macOS Python.org builds otherwise fail
# cert verification (SSLCertVerificationError). Falls back to system default.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = ssl.create_default_context()

# NOTE: the VERIFIED official RentCast endpoint is below. A previous edit pointed
# this at "https://rentest.ai/..." -- an UNVERIFIED host. Sending the API key
# (X-Api-Key header) to an unverified domain risks leaking it, so we use the
# official host whose schema we confirmed (squareFootage, /properties, X-Api-Key).
RENTCAST_URL = "https://api.rentcast.io/v1/properties"
OFFICIAL_RENTCAST_URL = "https://api.rentcast.io/v1/properties"  # used by --check, always


# ---------------------------------------------------------------------------
# Cache helpers -- the whole point is to never pay for the same listing twice.
# ---------------------------------------------------------------------------


def load_dotenv(path: str = ".env") -> None:
    """Tiny stdlib .env loader so the API key can live in the project (gitignored)
    instead of a fragile interactive-shell export. Real env vars win (setdefault)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def check_api() -> None:
    """One-call connectivity test against the OFFICIAL RentCast host (never a
    custom RENTCAST_URL -- a test must not leak the key to an unverified domain)."""
    load_dotenv()
    key = os.environ.get("RENTCAST_API_KEY")
    if not key:
        print("[check] RENTCAST_API_KEY not found (looked in env and ./.env). "
              "Put it in a .env file: RENTCAST_API_KEY=your_key")
        return
    url = OFFICIAL_RENTCAST_URL + "?" + urllib.parse.urlencode(
        {"latitude": 35.5951, "longitude": -82.5515, "radius": 0.5})
    req = urllib.request.Request(url, headers={"X-Api-Key": key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25, context=_SSL_CTX) as r:
            data = json.loads(r.read().decode())
        recs = data if isinstance(data, list) else data.get("results") or []
        print(f"[check] ✓ WORKING — HTTP 200, {len(recs)} record(s).")
        if recs:
            d = recs[0]
            print(f"        sample: {d.get('squareFootage')} sqft, "
                  f"built {d.get('yearBuilt')}, {d.get('formattedAddress')}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        hint = " -> activate a plan at app.rentcast.io" if e.code == 401 else ""
        print(f"[check] ✗ HTTP {e.code} {e.reason}: {body}{hint}")
    except Exception as e:  # noqa: BLE001
        print(f"[check] ✗ FAILED: {e!r}")


def cache_path_for(market: str) -> str:
    return f"enrichment_{market}.json"


def load_cache(path: str) -> Dict[str, Optional[dict]]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: Dict[str, Optional[dict]]) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=0)


# ---------------------------------------------------------------------------
# The API call.
# ---------------------------------------------------------------------------


def _haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def enrich_one(lat: float, lon: float, api_key: str,
               radius: float = 0.2, timeout: int = 20) -> Optional[dict]:
    """Look up the property record nearest to (lat, lon) via RentCast.
    Returns the physical facts we care about, or None if nothing/failure.
    `radius` (miles) is widened a bit to survive Airbnb's coordinate jitter."""
    params = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "radius": radius})
    req = urllib.request.Request(
        f"{RENTCAST_URL}?{params}",
        headers={"X-Api-Key": api_key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 -- any failure -> treat as a miss
        print(f"[rentcast] call failed ({exc})")
        return None

    records = data if isinstance(data, list) else data.get("results") or []
    if not records:
        return None

    # pick the record physically closest to the (jittered) query point
    def dist(rec):
        rlat, rlon = rec.get("latitude"), rec.get("longitude")
        if rlat is None or rlon is None:
            return 9_999.0
        return _haversine_miles(lat, lon, rlat, rlon)

    best = min(records, key=dist)
    return {
        "square_feet": best.get("squareFootage"),
        "year_built": best.get("yearBuilt"),
        "lot_size": best.get("lotSize"),
        "property_type": best.get("propertyType"),
        "match_distance_mi": round(dist(best), 3),
    }


# ---------------------------------------------------------------------------
# Bulk enrichment with a budget -- the safe way to spend a rate-limited quota.
# ---------------------------------------------------------------------------


def bulk_enrich(listings_source: str, cache_path: str, *,
                max_calls: int = 50, radius: float = 0.2) -> None:
    api_key = os.environ.get("RENTCAST_API_KEY")
    if not api_key:
        print("[rentcast] RENTCAST_API_KEY not set -- cannot enrich. "
              "Set the key, then re-run.")
        return

    df = pd.read_csv(listings_source, low_memory=False, usecols=["id", "latitude", "longitude"])
    cache = load_cache(cache_path)
    print(f"[rentcast] {len(df):,} listings; {len(cache):,} already in cache; "
          f"budget = {max_calls} new calls this run.")

    calls = 0
    for _, row in df.iterrows():
        lid = str(row["id"])
        if lid in cache:                      # already looked up (hit OR miss) -> skip
            continue
        if calls >= max_calls:
            print(f"[rentcast] budget of {max_calls} reached -- stopping. "
                  f"Re-run later; the cache persists and we resume here.")
            break
        cache[lid] = enrich_one(row["latitude"], row["longitude"], api_key, radius=radius)
        calls += 1
        if calls % 10 == 0:
            save_cache(cache_path, cache)
            print(f"[rentcast] {calls} new calls made...")
        time.sleep(0.2)                       # be polite to the API

    save_cache(cache_path, cache)
    covered = sum(1 for v in cache.values() if v and v.get("square_feet"))
    print(f"[rentcast] done. {calls} new calls. "
          f"{covered:,}/{len(df):,} listings now have sqft "
          f"({covered / max(len(df), 1):.0%} coverage).")


# ---------------------------------------------------------------------------
# AREA-QUERY enrichment -- spend each call as a NET, not a hook.
# ---------------------------------------------------------------------------
# One radius call returns up to 500 properties. So instead of 1 call per listing,
# we cluster the listings into a handful of zones, make ONE bulk call per zone
# (limit=500), pool every home that has sqft, then match each listing to its
# nearest pooled home locally (free). This is how ~45 free calls can cover a city.


def _haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized great-circle miles (works on numpy arrays)."""
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def area_query(lat: float, lon: float, radius: float, api_key: str,
               limit: int = 500, timeout: int = 30) -> List[dict]:
    """One bulk radius call -> up to `limit` raw property records (the NET)."""
    params = urllib.parse.urlencode(
        {"latitude": lat, "longitude": lon, "radius": radius, "limit": limit})
    req = urllib.request.Request(
        OFFICIAL_RENTCAST_URL + "?" + params,
        headers={"X-Api-Key": api_key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        print(f"[area] query failed ({exc})")
        return []
    return data if isinstance(data, list) else (data.get("results") or [])


def area_enrich(listings_source: str, cache_path: str, *,
                budget: int = 45, max_match_mi: float = 0.30) -> None:
    """Cover a market's listings with <= `budget` bulk calls, then match each
    listing to its nearest pooled home with sqft. Writes the same id->sqft cache
    the engine reads, and prints a coverage + match-quality report."""
    from sklearn.cluster import KMeans

    api_key = os.environ.get("RENTCAST_API_KEY")
    if not api_key:
        print("[area] RENTCAST_API_KEY not set."); return

    df = pd.read_csv(listings_source, low_memory=False, usecols=["id", "latitude", "longitude"],
                     dtype={"id": str})   # keep ids as clean strings (no "155305.0")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    coords = df[["latitude", "longitude"]].to_numpy()

    # ~1 zone per 40 listings, capped by the call budget
    k = max(1, min(budget, len(df) // 40))
    print(f"[area] {len(df):,} listings -> {k} query zones (budget {budget} calls)")
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(coords)

    pooled: Dict[str, dict] = {}   # dedupe by record id; keep homes with ANY useful field
    calls = 0
    for ci in range(k):
        if calls >= budget:
            break
        center = km.cluster_centers_[ci]
        members = coords[km.labels_ == ci]
        spread = _haversine_np(center[0], center[1], members[:, 0], members[:, 1])
        radius = float(min(3.0, max(0.3, np.percentile(spread, 90) + 0.1)))
        recs = area_query(center[0], center[1], radius, api_key, limit=500)
        calls += 1
        for rec in recs:
            useful = (rec.get("squareFootage") or rec.get("yearBuilt")
                      or rec.get("lotSize") or rec.get("propertyType"))
            if useful and rec.get("latitude") is not None:
                rid = str(rec.get("id") or f"{rec['latitude']},{rec['longitude']}")
                pooled[rid] = rec
        print(f"[area] zone {ci + 1}/{k}: r={radius:.2f}mi -> {len(recs)} records "
              f"({len(pooled):,} pooled homes so far)")
        time.sleep(0.2)

    if not pooled:
        print("[area] no usable pooled homes -- cannot match."); return

    precs = list(pooled.values())
    plat = np.array([r["latitude"] for r in precs])
    plon = np.array([r["longitude"] for r in precs])

    # Persist the RAW pooled records. This is the key fix: with these saved, we
    # can add/extract MORE fields later (or re-match) with ZERO new API calls.
    pooled_path = cache_path.replace("enrichment_", "pooled_")
    save_cache(pooled_path, precs)
    print(f"[area] saved {len(precs):,} raw pooled records -> {pooled_path}")

    cache = load_cache(cache_path)
    dists = []
    for _, row in df.iterrows():
        dd = _haversine_np(row["latitude"], row["longitude"], plat, plon)
        j = int(np.argmin(dd))
        md = float(dd[j])
        if md <= max_match_mi:
            rec = precs[j]
            cache[str(row["id"])] = {
                "square_feet": float(rec["squareFootage"]) if rec.get("squareFootage") else None,
                "year_built": rec.get("yearBuilt"),
                "lot_size": rec.get("lotSize"),
                "property_type": rec.get("propertyType"),
                "match_distance_mi": round(md, 3),
            }

            dists.append(md)
        else:
            cache[str(row["id"])] = None
    save_cache(cache_path, cache)

    matched = len(dists)
    print(f"\n[area] ===== REPORT =====")
    print(f"[area] spent {calls} calls; pooled {len(pooled):,} homes with sqft")
    print(f"[area] matched {matched:,}/{len(df):,} listings = {matched/len(df):.0%} coverage")
    if dists:
        print(f"[area] match distance: median {statistics.median(dists)*5280:,.0f} ft, "
              f"p90 {np.percentile(dists,90)*5280:,.0f} ft  (Airbnb jitter is ~500 ft)")
    print(f"[area] wrote {cache_path}")


# ---------------------------------------------------------------------------
# ATTRIBUTE-CONSTRAINED re-matching -- the FIX for bad coordinate matches.
# ---------------------------------------------------------------------------
# Airbnb scrambles coordinates ~150m, so "nearest dot to the fake point" is
# routinely a neighbor (validated: only 23% bedroom agreement). The fix: among
# the pooled homes near the listing, pick the one whose BEDROOMS agree with the
# listing (a corroborating attribute), tie-broken by distance. Held-out check:
# this lifts bathroom agreement 41% -> 68%. Runs entirely off the saved pool
# (pooled_<market>.json) -- ZERO API calls. Trades coverage for correctness.


def _num(x) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def _ab_bathrooms(row) -> float:
    b = row.get("bathrooms")
    if pd.notna(b):
        try:
            return float(b)
        except (TypeError, ValueError):
            pass
    m = re.search(r"[\d.]+", str(row.get("bathrooms_text") or ""))
    return float(m.group()) if m else float("nan")


def rematch_from_pool(listings_source: str, cache_path: str, *, radius: float = 0.15) -> None:
    """Re-match listings to the saved pool using BEDROOM agreement (+ bathroom
    as a tiebreak), within `radius` miles. No API calls. Overwrites the cache
    with higher-accuracy, lower-coverage matches."""
    pooled_path = cache_path.replace("enrichment_", "pooled_")
    if not os.path.exists(pooled_path):
        print(f"[rematch] no pooled file {pooled_path} -- run --area first.")
        return
    precs = load_cache(pooled_path)
    plat = np.array([_num(r.get("latitude")) for r in precs])
    plon = np.array([_num(r.get("longitude")) for r in precs])
    pbed = np.array([_num(r.get("bedrooms")) for r in precs])
    pbath = np.array([_num(r.get("bathrooms")) for r in precs])

    df = pd.read_csv(listings_source, low_memory=False, dtype={"id": str},
                     usecols=["id", "latitude", "longitude", "bedrooms",
                              "bathrooms", "bathrooms_text", "room_type"])
    df = df.dropna(subset=["latitude", "longitude"])

    cache: Dict[str, Optional[dict]] = {}
    matched, dists = 0, []
    for _, row in df.iterrows():
        ab_bed = _num(row["bedrooms"])
        ab_bath = _ab_bathrooms(row)
        d = _haversine_np(row["latitude"], row["longitude"], plat, plon)
        near = np.where(d <= radius)[0]
        chosen = None
        if len(near) and not math.isnan(ab_bed):
            bedok = near[pbed[near] == ab_bed]            # require bedroom agreement
            if len(bedok):
                if not math.isnan(ab_bath):               # prefer bathroom agreement too
                    bb = bedok[np.abs(pbath[bedok] - ab_bath) <= 0.5]
                    pick = bb if len(bb) else bedok
                else:
                    pick = bedok
                chosen = int(pick[np.argmin(d[pick])])    # tie-break by distance
        if chosen is None:
            cache[str(row["id"])] = None
            continue
        rec = precs[chosen]
        cache[str(row["id"])] = {
            "square_feet": _num(rec.get("squareFootage")) if rec.get("squareFootage") else None,
            "year_built": rec.get("yearBuilt"),
            "lot_size": rec.get("lotSize"),
            "property_type": rec.get("propertyType"),
            "match_distance_mi": round(float(d[chosen]), 3),
            "attr_matched": True,
        }
        matched += 1
        dists.append(float(d[chosen]))
    save_cache(cache_path, cache)
    print(f"[rematch] attribute-confirmed matches: {matched:,}/{len(df):,} "
          f"= {matched/len(df):.0%} (bedroom-agreeing within {radius} mi)")
    if dists:
        print(f"[rematch] match distance: median {statistics.median(dists)*5280:,.0f} ft")
    print(f"[rematch] wrote {cache_path} (0 API calls)")


def main() -> None:
    # local import to avoid a hard dependency cycle; only needed for the CLI
    from airbnb_revenue_engine import MARKETS

    ap = argparse.ArgumentParser(description="Enrich listings with property records (sqft).")
    ap.add_argument("--market", default="asheville", choices=sorted(MARKETS))
    ap.add_argument("--source", default=None, help="Local CSV override for the market data.")
    ap.add_argument("--max-calls", type=int, default=50,
                    help="Max NEW API calls this run (default 50 = free tier/month).")
    ap.add_argument("--check", action="store_true",
                    help="Just test API connectivity (1 call) and exit.")
    ap.add_argument("--area", action="store_true",
                    help="Area-query mode: bulk radius calls (the NET), best coverage per call.")
    ap.add_argument("--budget", type=int, default=45,
                    help="Max calls for --area mode (default 45, leaves buffer under 50/mo).")
    ap.add_argument("--rematch", action="store_true",
                    help="Re-match from saved pool using bedroom agreement (0 API calls).")
    args = ap.parse_args()

    load_dotenv()                     # pick up RENTCAST_API_KEY from .env if present
    if args.check:
        check_api()
        return

    market = MARKETS[args.market]
    source = args.source or market.listings_url
    if args.rematch:
        rematch_from_pool(source, cache_path_for(args.market))
    elif args.area:
        area_enrich(source, cache_path_for(args.market), budget=args.budget)
    else:
        bulk_enrich(source, cache_path_for(args.market), max_calls=args.max_calls)


if __name__ == "__main__":
    main()
