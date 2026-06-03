"""
airbnb_revenue_engine.py  —  REVENUE model (the ML layer).

Pipeline position:
    data -> [ REVENUE MODEL ] -> revenue.py -> costs.py -> returns.py -> risk -> rank
              ^^^^^^^^^^^^^^^ this file

JOB OF THIS FILE
----------------
Learn, from real Inside Airbnb listings, how a property's INTRINSIC features map
to annual revenue -- then predict revenue for a property you're considering
buying (which has no reviews, no ratings, no history yet).

Its output (a predicted annual revenue + occupancy) plugs into revenue.py as a
4th source, feeding the cost/returns stack we already built.

THE TWO IRON RULES (do not break)
---------------------------------
1. LEAKAGE: train ONLY on features knowable BEFORE purchase -- size, type,
   location, amenities. NEVER reviews, ratings, superhost, response rate, or the
   estimated_* outcome columns. Those don't exist for a property you haven't
   bought; using them = a model that validates great and is useless. We enforce
   this STRUCTURALLY: there is an explicit ALLOWED-feature allowlist, and the
   banned columns can never enter the matrix.
2. EVERY occupancy/revenue number is an ESTIMATE. The label itself
   (`estimated_revenue_l365d`) is Inside Airbnb's review-rate estimate, not
   actual bookings. We report honest accuracy (median abs % error), never a
   single confident number.

TWO KINDS OF "NICENESS" FEATURE
-------------------------------
(a) AMENITIES  -> hard booleans extracted from the data (hot tub Y/N, pool Y/N...).
    These ARE real trained features; they exist for every training row.
(b) LUXURIOUSNESS -> a 1..10 subjective quality score (design/finish/"wow").
    We do NOT have this for the 21k training rows, so for the MVP it is NOT a
    trained feature -- it is a calibrated MULTIPLIER applied on top of the base
    prediction. You set it manually, or an LLM scores it from photos/description.
    See apply_luxuriousness() -- the multiplier-per-point is a flagged ASSUMPTION.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


# ===========================================================================
# 1. AMENITY FEATURES  -- the Yes/No knobs. Add/remove freely.
# ===========================================================================
# Each amenity is matched case-insensitively as a SUBSTRING against the listing's
# amenity strings (they're verbose, e.g. "Private hot tub - available all year").
# `include` = any of these substrings present -> True.
# `exclude` = ...unless one of these is present (guards false positives like
#             "pool" matching "pool table").

AMENITY_FEATURES: Dict[str, Dict[str, List[str]]] = {
    "hot_tub":      {"include": ["hot tub", "jacuzzi"], "exclude": []},
    "pool":         {"include": ["pool"], "exclude": ["pool table"]},
    "sauna":        {"include": ["sauna"], "exclude": []},
    "gym":          {"include": ["gym", "exercise equipment", "fitness"], "exclude": []},
    "ac":           {"include": ["air conditioning", "central air"], "exclude": []},
    "ev_charger":   {"include": ["ev charger", "electric vehicle"], "exclude": []},
    "free_parking": {"include": ["free parking", "garage", "carport"], "exclude": ["street"]},
    "waterfront":   {"include": ["waterfront", "lake access", "beachfront", "lakefront"], "exclude": []},
    "fire_pit":     {"include": ["fire pit", "fireplace"], "exclude": []},
    "bbq":          {"include": ["bbq grill", "barbecue", "bbq"], "exclude": []},
    "patio":        {"include": ["patio or balcony", "backyard", "private patio"], "exclude": []},
    "workspace":    {"include": ["dedicated workspace"], "exclude": []},
    "sound_system": {"include": ["sound system", "sonos"], "exclude": []},
    "pets_allowed": {"include": ["pets allowed"], "exclude": []},
}


# ---------------------------------------------------------------------------
# 1a. TEXT-QUALITY FEATURES  -- mine the listing's words for "niceness" signal.
# ---------------------------------------------------------------------------
# Each group -> a 0/1 feature = "does the name/description/neighborhood text
# mention any of these?". These describe PROPERTY qualities (renovated, views,
# modern, charm) that are assessable before purchase, so leakage-safe. Matched
# with word boundaries so "view" doesn't fire inside "review"/"overview".

TEXT_FEATURES: Dict[str, List[str]] = {
    "txt_luxury":    ["luxury", "luxurious", "upscale", "high-end", "elegant",
                      "designer", "gourmet", "stunning", "spectacular", "exquisite"],
    "txt_renovated": ["renovated", "remodeled", "updated", "newly", "brand new",
                      "brand-new", "new construction", "restored"],
    "txt_modern":    ["modern", "contemporary", "sleek", "stylish", "chic"],
    "txt_views":     ["view", "views", "scenic", "overlook", "vista", "panoramic"],
    "txt_charm":     ["charming", "cozy", "quaint", "cottage", "cabin", "chalet",
                      "retreat", "rustic", "historic"],
    "txt_private":   ["private", "secluded", "quiet", "peaceful", "tranquil", "serene"],
    "txt_walkable":  ["walkable", "walking distance", "walk to", "steps from",
                      "short walk", "stroll"],
}


# Photo-quality scores from photo_scorer.py (Claude vision). The actual test of
# whether visual "niceness" drives revenue beyond beds/location/amenities.
PHOTO_FEATURES = ["kitchen", "bathroom", "exterior", "view",
                  "interior_decor", "uniqueness", "overall"]


def has_amenity(amenity_list: List[str], spec: Dict[str, List[str]]) -> bool:
    """True if any include-substring is present and no exclude-substring is."""
    blob = " || ".join(a.lower() for a in amenity_list)
    if any(bad in blob for bad in spec["exclude"]):
        # only veto if the bad term is the ONLY reason it matched
        hits = [inc for inc in spec["include"] if inc in blob]
        bad_only = all(any(bad in a.lower() for bad in spec["exclude"])
                       for a in amenity_list
                       if any(inc in a.lower() for inc in hits))
        if bad_only:
            return False
    return any(inc in blob for inc in spec["include"])


# ===========================================================================
# 1b. MARKETS + LOCATION via DISTANCE-TO-ATTRACTION  (replaces raw lat/lon)
# ===========================================================================
# WHY distances: raw latitude/longitude let a tree memorize coordinates -- it
# learns "this exact spot earns X", which doesn't generalize. What actually
# drives STR demand is PROXIMITY to where visitors go. So we turn (lat, lon) into
# distances (miles) to the market's tourist anchors -- a feature that MEANS
# something an investor recognizes.
#
# Anchors are MARKET-SPECIFIC (the #1 thing to change between cities), so each
# Market carries its own data URL + anchor set. A model remembers the anchors it
# was trained on (see RevenueModel) so predictions use the same ones.


@dataclass
class Market:
    """One STR market: where to get its data + the anchors that drive its demand."""
    name: str
    listings_url: str
    anchors: Dict[str, Tuple[float, float]]   # name -> (lat, lon)


MARKETS: Dict[str, Market] = {
    # Urban market: demand = nightlife, business districts, events.
    "austin": Market(
        name="Austin, TX",
        listings_url="https://data.insideairbnb.com/united-states/tx/austin/2025-09-16/data/listings.csv.gz",
        anchors={
            "downtown":       (30.2685, -97.7425),  # 6th St / Congress core
            "zilker":         (30.2669, -97.7729),  # Zilker Park / Barton Springs / ACL
            "south_congress": (30.2489, -97.7501),  # SoCo shopping & food
            "domain":         (30.4012, -97.7252),  # The Domain (north retail/business)
            "airport":        (30.1945, -97.6699),  # ABIA -- where visitors arrive
        },
    ),
    # Mountain-tourism market (Catskills analog): demand = nature, the Estate,
    # breweries, scenic drives. NOT a "downtown" market -- anchors look different.
    "asheville": Market(
        name="Asheville, NC",
        listings_url="https://data.insideairbnb.com/united-states/nc/asheville/2025-09-22/data/listings.csv.gz",
        anchors={
            "downtown":      (35.5951, -82.5515),  # Pack Square / breweries / restaurants
            "biltmore":      (35.5401, -82.5520),  # Biltmore Estate -- the #1 attraction
            "river_arts":    (35.5860, -82.5680),  # River Arts District
            "blue_ridge":    (35.5938, -82.4818),  # Blue Ridge Parkway / Folk Art Center
            "airport":       (35.4362, -82.5418),  # AVL regional airport
        },
    ),
    # NOTE: Los Angeles is NOT usable on the free data -- the Dec-2025 snapshot
    # ships estimated_occupancy but EMPTY price + estimated_revenue columns, so
    # there's no label to train on. (Kept out of the registry for that reason.)

    # Luxury resort market (multi-ISLAND): demand = beaches + island resort hubs.
    # Highest ADR market in the US. Location spans islands, so the meaningful
    # location signal is dist_nearest_attraction (proximity to the island's hub).
    "hawaii": Market(
        name="Hawaii",
        listings_url="https://data.insideairbnb.com/united-states/hi/hawaii/2025-09-16/data/listings.csv.gz",
        anchors={
            "waikiki":     (21.2793, -157.8294),  # Oahu tourist core
            "kaanapali":   (20.9296, -156.6940),  # West Maui resorts
            "wailea":      (20.6870, -156.4420),  # South Maui luxury
            "kona":        (19.6400, -155.9969),  # Big Island (Kailua-Kona)
            "poipu":       (21.8800, -159.4600),  # Kauai south resort
            "princeville": (22.2230, -159.4830),  # Kauai north luxury
            "hnl_airport": (21.3187, -157.9225),  # main airport (Oahu)
        },
    ),
}


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles. Works on scalars OR numpy/pandas arrays."""
    R = 3958.8  # Earth radius, miles
    lat1, lon1, lat2, lon2 = (np.radians(lat1), np.radians(lon1),
                              np.radians(lat2), np.radians(lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def distance_features(lat, lon, anchors: Dict[str, Tuple[float, float]]) -> Dict[str, object]:
    """Map (lat, lon) -> {dist_<anchor>: miles, ..., dist_nearest_attraction}
    for the given market's anchors. Accepts scalars (one property) or pandas
    Series (the whole dataset)."""
    cols: Dict[str, object] = {}
    for key, (alat, alon) in anchors.items():
        cols[f"dist_{key}"] = haversine_miles(lat, lon, alat, alon)
    dists = list(cols.values())
    if hasattr(dists[0], "values"):                       # pandas Series -> dataset
        cols["dist_nearest_attraction"] = pd.concat(dists, axis=1).min(axis=1)
    else:                                                 # scalars -> one property
        cols["dist_nearest_attraction"] = min(dists)
    return cols


# ===========================================================================
# 2. PROPERTY SCHEMA  -- the shape used both for training rows and predictions.
# ===========================================================================


@dataclass
class PropertyFeatures:
    """Everything the model uses to describe a property. Structural + location +
    amenity booleans + the (optional) luxuriousness lever."""

    # --- structural (intrinsic, knowable before purchase) ---
    bedrooms: float
    bathrooms: float
    accommodates: int
    beds: float
    latitude: float
    longitude: float
    room_type: str = "Entire home/apt"     # Entire home/apt | Private room | Shared room
    property_type: str = "Entire home"
    minimum_nights: int = 2
    instant_bookable: bool = False

    # --- external facts (from property_enrichment.py; you know them for a candidate) ---
    square_feet: Optional[float] = None     # None -> imputed to market median at predict
    year_built: Optional[float] = None
    lot_size: Optional[float] = None
    rc_property_type: Optional[str] = None  # e.g. "Single Family", "Condo"

    # --- amenity booleans (auto-filled from data; set by hand for a candidate) ---
    amenities: Dict[str, bool] = field(default_factory=dict)
    # --- text-quality flags (does it have views? renovated? etc.) ---
    text_flags: Dict[str, bool] = field(default_factory=dict)

    # --- the subjective lever (NOT a trained feature; a predict-time multiplier) ---
    luxuriousness: Optional[float] = None   # 1..10, 5 = market-average. None = neutral.


# ===========================================================================
# 3. LEAKAGE GUARD  -- the allowlist is the enforcement.
# ===========================================================================
# Only these columns may become model features. Anything not here CANNOT leak in.
# Reviews / ratings / superhost / host_* / estimated_* are deliberately absent.

STRUCTURAL_FEATURES = [
    "bedrooms", "bathrooms", "accommodates", "beds",
    "minimum_nights",
    # external join (property_enrichment.py): physical facts + their _known flags
    "square_feet", "square_feet_known",
    "year_built", "year_built_known",
    "lot_size", "lot_size_known",
    # location is now distance-to-attraction (see distance_features), NOT raw lat/lon
]

# Numeric enrichment fields and how to impute a missing value:
#   "by_bedrooms" -> per-bedroom median (size scales with bedrooms)
#   "median"      -> overall median (age/lot don't track bedroom count)
ENRICHMENT_NUMERIC = {
    "square_feet": "by_bedrooms",
    "year_built": "median",
    "lot_size": "median",
}
# RentCast's propertyType becomes a categorical feature (rc_proptype_*).
ENRICHMENT_CATEGORICAL = "property_type"

# Minimum coverage before we trust an enriched field. Below this, a mostly-imputed
# column is noise pretending to be signal, so we skip that field.
MIN_SQFT_COVERAGE = 0.30
CATEGORICAL_FEATURES = ["room_type", "property_type_grouped"]
BOOL_FEATURES = ["instant_bookable"]
# amenity booleans (AMENITY_FEATURES keys) are added on top.

BANNED_SUBSTRINGS = [          # belt-and-suspenders: assert none of these sneak in
    "review", "rating", "superhost", "host_", "score",
    "estimated_", "reviews_per_month", "price",   # price excluded: it's a decision
]

LABEL_COLUMN = "estimated_revenue_l365d"


# ===========================================================================
# 4. LOAD + FEATURIZE  -- turn the raw CSV into (X, y).
# ===========================================================================


def _parse_price(val) -> float:
    if pd.isna(val):
        return np.nan
    return float(str(val).replace("$", "").replace(",", "").strip() or "nan")


def _parse_bathrooms(row) -> float:
    b = row.get("bathrooms")
    if pd.notna(b) and str(b).strip() != "":
        try:
            return float(b)
        except ValueError:
            pass
    txt = str(row.get("bathrooms_text", "")).lower()
    for tok in txt.split():
        try:
            return float(tok)
        except ValueError:
            continue
    return np.nan


def _group_property_type(pt: str) -> str:
    """Collapse Airbnb's ~100 property_type values into a few buckets so one-hot
    encoding stays small and stable between train and predict."""
    pt = (pt or "").lower()
    if "entire" in pt or pt.startswith("home") or "house" in pt or "villa" in pt:
        return "entire_home"
    if "condo" in pt or "apartment" in pt or "loft" in pt or "rental unit" in pt:
        return "apartment_condo"
    if "private room" in pt or "room in" in pt:
        return "private_room"
    return "other"


def _enrichment_lookup_fn(cache: dict, field: str):
    """Build a robust id->value lookup for one enrichment field.
    Airbnb has short ids (155305) AND ~18-digit ids that lose precision if ever
    read as float. A cache may carry clean keys ("654...098") or float-mangled
    ones ("6.54e+17"/"155305.0"), so we try several key forms per listing."""
    field_map = {str(k): (v.get(field) if v else None) for k, v in cache.items()}

    def _float_key(x: object) -> str:
        try:
            return str(float(x))
        except Exception:  # noqa: BLE001
            return str(x)

    def _lookup(idv: object):
        s = str(idv)
        for cand in (s, s[:-2] if s.endswith(".0") else s, _float_key(idv)):
            if cand in field_map:
                return field_map[cand]
        return None

    return _lookup


def _join_enrichment(df: pd.DataFrame, feat: pd.DataFrame,
                     enrichment_cache: Optional[str]) -> Dict[str, float]:
    """Join all external property facts (sqft, year_built, lot_size, property_type)
    from property_enrichment.py's cache onto `feat` in place. Each field is added
    only if its coverage clears MIN_SQFT_COVERAGE; below that it's noise. Missing
    values are imputed and a `<field>_known` flag marks real vs imputed rows.
    Returns medians of the numeric fields that were added (for predict-time fill)."""
    medians: Dict[str, float] = {}
    if not enrichment_cache or not os.path.exists(enrichment_cache):
        print("[load] no enrichment cache -> external features skipped "
              "(run property_enrichment.py --area to populate).")
        return medians

    with open(enrichment_cache) as f:
        cache = json.load(f)
    print(f"[load] enrichment cache: {len(cache):,} entries")

    # --- numeric fields ---
    for field, how in ENRICHMENT_NUMERIC.items():
        series = pd.to_numeric(df["id"].map(_enrichment_lookup_fn(cache, field)),
                               errors="coerce")
        cov = float(series.notna().mean())
        if cov < MIN_SQFT_COVERAGE:
            print(f"[load]   {field}: coverage {cov:.0%} < "
                  f"{MIN_SQFT_COVERAGE:.0%} -> skipped")
            continue
        feat[f"{field}_known"] = series.notna().astype(int)
        med = float(series.median())
        filler = series.groupby(feat["bedrooms"]).transform("median") \
            if how == "by_bedrooms" else med
        feat[field] = series.fillna(filler).fillna(med)
        medians[field] = med
        print(f"[load]   {field}: coverage {cov:.0%} -> added (median {med:,.0f})")

    # --- categorical: RentCast propertyType -> rc_proptype_* one-hot ---
    cat = df["id"].map(_enrichment_lookup_fn(cache, ENRICHMENT_CATEGORICAL))
    cov = float(cat.notna().mean())
    if cov >= MIN_SQFT_COVERAGE:
        dummies = pd.get_dummies(cat.fillna("unknown"), prefix="rc_proptype")
        for c in dummies.columns:
            feat[c] = dummies[c].astype(int)
        print(f"[load]   property_type: coverage {cov:.0%} -> "
              f"{dummies.shape[1]} rc_proptype_* columns")
    else:
        print(f"[load]   property_type: coverage {cov:.0%} < "
              f"{MIN_SQFT_COVERAGE:.0%} -> skipped")
    return medians


def load_and_featurize(
    source: str, anchors: Dict[str, Tuple[float, float]],
    enrichment_cache: Optional[str] = None, *, use_location: bool = True,
    target: str = "revenue",
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, float]]:
    """Read an Inside Airbnb listings CSV (local path or URL, .csv or .csv.gz)
    and return (X, y, feature_columns, enrichment_medians). X contains ONLY
    allowlisted features. `anchors` are the market's tourist anchors used for
    distance features; `enrichment_cache` (optional) adds external property
    facts (sqft, year built, lot size, property type)."""
    print(f"[load] reading {source} ...")
    df = pd.read_csv(source, low_memory=False)
    print(f"[load] {len(df):,} raw listings, {df.shape[1]} columns")

    # --- drop hotel rooms: they run on hotel economics (high turnover, front
    #     desks, no purchase to model) -- not STR investment properties. They
    #     were also the single most distorting feature in the model. ---
    before = len(df)
    df = df[df["room_type"].astype(str).str.strip() != "Hotel room"].copy()
    print(f"[load] dropped {before - len(df):,} hotel-room listings -> {len(df):,} remain")

    # --- label: revenue estimate, OR the advertised nightly PRICE (ADR) ---
    # price is a REAL number (not a review-rate estimate) and is set by what the
    # property IS -- so it's far more predictable than revenue. As the target it's
    # clean (it's banned only as a FEATURE). We clip placeholder/blocker prices.
    if target == "price":
        label_series = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
        label_valid = label_series.between(20, 10000)
    else:
        label_series = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
        label_valid = label_series > 0
    df["_label"] = label_series

    # --- amenity booleans from the JSON list ---
    def amen_flags(raw) -> Dict[str, bool]:
        try:
            items = json.loads(raw) if isinstance(raw, str) else []
        except json.JSONDecodeError:
            items = []
        return {name: has_amenity(items, spec) for name, spec in AMENITY_FEATURES.items()}

    amen_df = df["amenities"].apply(lambda r: pd.Series(amen_flags(r)))

    # --- structural / categorical ---
    feat = pd.DataFrame(index=df.index)
    feat["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
    feat["bathrooms"] = df.apply(_parse_bathrooms, axis=1)
    feat["accommodates"] = pd.to_numeric(df["accommodates"], errors="coerce")
    feat["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    feat["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")
    # location -> distance-to-attraction features (replaces raw lat/lon)
    if use_location:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        for name, series in distance_features(lat, lon, anchors).items():
            feat[name] = series
    else:
        print("[load] location features OFF (markets + distance-to-attraction skipped)")
    feat["instant_bookable"] = (df["instant_bookable"].astype(str).str.lower() == "t").astype(int)
    feat["room_type"] = df["room_type"].fillna("Entire home/apt")
    feat["property_type_grouped"] = df["property_type"].apply(_group_property_type)
    feat = pd.concat([feat, amen_df.astype(int)], axis=1)

    # --- text-quality flags from name + description + neighborhood text ---
    blob = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
            + df["neighborhood_overview"].fillna("")).str.lower()
    for fname, kws in TEXT_FEATURES.items():
        pattern = r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b"
        feat[fname] = blob.str.contains(pattern, regex=True).astype(int)

    # --- external property facts (sqft, year built, lot size, type) via join ---
    enrichment_medians = _join_enrichment(df, feat, enrichment_cache)

    # --- one-hot the categoricals ---
    feat = pd.get_dummies(feat, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)

    # --- assemble (X, y), drop rows with no usable label or core features ---
    y = df["_label"]
    mask = label_valid & feat["bedrooms"].notna() & feat["accommodates"].notna()
    X, y = feat[mask].copy(), y[mask].copy()
    X = X.fillna(0)

    # --- LEAKAGE ASSERTION: no banned column may exist in X ---
    leaked = [c for c in X.columns if any(b in c.lower() for b in BANNED_SUBSTRINGS)]
    assert not leaked, f"LEAKAGE: banned columns in feature matrix: {leaked}"

    print(f"[load] {len(X):,} usable rows, {X.shape[1]} leakage-free features")
    return X, y, list(X.columns), enrichment_medians


# ===========================================================================
# 5. TRAIN  -- fit the model and report HONEST accuracy.
# ===========================================================================


@dataclass
class RevenueModel:
    """Trained model + the metadata predict() needs to rebuild a feature row."""
    model: RandomForestRegressor
    feature_columns: List[str]
    mdape: float                         # median absolute % error (the honest metric)
    anchors: Dict[str, Tuple[float, float]]  # the market's anchors (predict must match train)
    market_name: str = "unknown"
    mdape_std: float = 0.0               # spread of MdAPE across CV folds (stability)
    enrichment_medians: Dict[str, float] = field(default_factory=dict)  # impute candidate fields
    luxury_pct_per_point: float = 0.06   # ASSUMPTION: each luxury point = +/-6% revenue


def median_abs_pct_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MdAPE -- the project's preferred accuracy metric. Robust to the big
    outliers that single-point % errors blow up on."""
    return float(np.median(np.abs((y_pred - y_true) / y_true)))


def _new_forest() -> RandomForestRegressor:
    """The model spec, in one place so CV and the final fit use the SAME config."""
    return RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    )


def cross_validate_mdape(X: pd.DataFrame, y: pd.Series, *, n_splits: int = 5):
    """K-fold cross-validation of MdAPE -- a more honest, more stable accuracy
    estimate than a single train/test split.

    HOW IT WORKS: shuffle the data into `n_splits` equal folds. For each fold,
    train on the OTHER k-1 folds and test on the held-out fold. Every listing
    gets predicted exactly once, by a model that never saw it. We get one MdAPE
    per fold; their MEAN is the accuracy estimate and their SPREAD (std) tells us
    how stable that estimate is (big spread = the number depends on luck).

    NOTE: still a RANDOM split, not spatial -- two listings on the same block can
    land in different folds, so the model can 'peek' at a neighborhood. K-fold
    fixes the *which-20%-did-we-draw* luck, NOT the spatial-leakage optimism.
    Spatial CV (hold out whole neighborhoods) is the stricter next step.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores: List[float] = []
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
        m = _new_forest()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        score = median_abs_pct_error(y.iloc[te_idx].values, m.predict(X.iloc[te_idx]))
        fold_scores.append(score)
        print(f"          fold {fold}/{n_splits}: MdAPE = {score:.1%}")
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores


def train(market: Market, *, source_override: Optional[str] = None,
          enrichment_cache: Optional[str] = None, target: str = "revenue",
          luxury_pct_per_point: float = 0.06) -> RevenueModel:
    """Train on a Market (uses its data URL + anchors). `source_override` lets
    you point at a local cached copy of that market's CSV. `enrichment_cache`
    (optional) adds external square footage from property_enrichment.py.
    `target` is "revenue" (default) or "price" (the more-predictable ADR)."""
    print(f"[train] market: {market.name}  |  target: {target}")
    source = source_override or market.listings_url
    X, y, cols, enrichment_medians = load_and_featurize(
        source, market.anchors, enrichment_cache, target=target)

    # --- EVALUATE accuracy honestly with k-fold cross-validation ---
    print("[train] 5-fold cross-validation:")
    mdape, mdape_std, _ = cross_validate_mdape(X, y, n_splits=5)
    print(f"[train] MdAPE = {mdape:.1%} +/- {mdape_std:.1%}  "
          f"(mean across folds; +/- = fold-to-fold spread)")

    # --- FIT the final model on ALL the data (use every row, now that we've
    #     already gotten an honest accuracy estimate from CV above) ---
    model = _new_forest()
    model.fit(X, y)

    # show what's driving it -- sanity-check there's no surprise leakage
    imp = sorted(zip(cols, model.feature_importances_), key=lambda t: -t[1])[:10]
    print("[train] top features (final model on all data):")
    for name, val in imp:
        print(f"          {val:6.3f}  {name}")

    return RevenueModel(model=model, feature_columns=cols, mdape=mdape,
                        anchors=market.anchors, market_name=market.name,
                        mdape_std=mdape_std, enrichment_medians=enrichment_medians,
                        luxury_pct_per_point=luxury_pct_per_point)


# ===========================================================================
# 6. PREDICT  -- base revenue from features, then the luxuriousness lever.
# ===========================================================================


def _features_to_row(feat: PropertyFeatures, columns: List[str],
                     anchors: Dict[str, Tuple[float, float]],
                     enrichment_medians: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Turn a PropertyFeatures into a single-row matrix aligned to training cols."""
    enrichment_medians = enrichment_medians or {}
    row = {c: 0 for c in columns}
    for k in ["bedrooms", "bathrooms", "accommodates", "beds", "minimum_nights"]:
        if k in row:
            row[k] = getattr(feat, k)
    # location -> distance-to-attraction (computed from the candidate's lat/lon)
    for name, val in distance_features(feat.latitude, feat.longitude, anchors).items():
        if name in row:
            row[name] = val
    # external numeric facts (sqft/year_built/lot_size): candidate value or median
    for fld in ENRICHMENT_NUMERIC:
        if fld in row:
            val = getattr(feat, fld, None)
            row[fld] = val if val is not None else enrichment_medians.get(fld, 0)
            if f"{fld}_known" in row:
                row[f"{fld}_known"] = 1 if val is not None else 0
    # external property type -> the matching rc_proptype_* dummy
    if feat.rc_property_type:
        col = f"rc_proptype_{feat.rc_property_type}"
        if col in row:
            row[col] = 1
    if "instant_bookable" in row:
        row["instant_bookable"] = int(feat.instant_bookable)
    # amenity booleans
    for name in AMENITY_FEATURES:
        if name in row:
            row[name] = int(feat.amenities.get(name, False))
    # text-quality flags
    for name in TEXT_FEATURES:
        if name in row:
            row[name] = int(feat.text_flags.get(name, False))
    # one-hot categoricals (set the matching dummy to 1 if it exists)
    rt = f"room_type_{feat.room_type}"
    if rt in row:
        row[rt] = 1
    pt = f"property_type_grouped_{_group_property_type(feat.property_type)}"
    if pt in row:
        row[pt] = 1
    return pd.DataFrame([row], columns=columns)


def apply_luxuriousness(base_revenue: float, luxuriousness: Optional[float],
                        pct_per_point: float) -> float:
    """Adjust base revenue for subjective quality.

    ASSUMPTION (flag it): luxuriousness is 1..10 with 5 = market-average. Each
    point above/below 5 moves revenue by `pct_per_point` (default 6%). So a 8 ->
    +18%, a 3 -> -12%. This represents quality BEYOND what amenities already
    encode (design, finish, photos). The 6%/point is a placeholder to be
    calibrated against residuals or AirROI -- it is NOT measured yet.
    """
    if luxuriousness is None:
        return base_revenue
    return base_revenue * (1 + (luxuriousness - 5.0) * pct_per_point)


def predict(rm: RevenueModel, feat: PropertyFeatures) -> Dict[str, float]:
    """Predict annual revenue for one candidate property."""
    row = _features_to_row(feat, rm.feature_columns, rm.anchors, rm.enrichment_medians)
    base = float(rm.model.predict(row)[0])
    adjusted = apply_luxuriousness(base, feat.luxuriousness, rm.luxury_pct_per_point)
    return {
        "base_revenue": base,
        "luxuriousness": feat.luxuriousness,
        "adjusted_revenue": adjusted,
        "mdape": rm.mdape,
        # honest band: base prediction +/- the model's typical error
        "low": adjusted * (1 - rm.mdape),
        "high": adjusted * (1 + rm.mdape),
    }


# ===========================================================================
# 7. LUXURIOUSNESS via LLM  -- the MVP "ask Claude/ChatGPT" path (stub).
# ===========================================================================


def score_luxuriousness_via_llm(name: str, description: str,
                                picture_url: Optional[str] = None) -> Optional[float]:
    """MVP: ask an LLM to rate luxury 1..10 from the listing text (and, later,
    photos). Returns None if no API is configured -- caller falls back to manual.

    This is intentionally a thin stub: wire it to the Claude API (anthropic SDK)
    or OpenAI here. The PROMPT is the real artifact -- keep the rubric explicit
    so scores are consistent across listings:

        "Rate this short-term rental's luxury/modernity from 1 (dated, basic) to
         10 (high-end, designer, recently renovated). 5 = average for its market.
         Judge finish quality, design, and 'wow' factor -- NOT size or location
         (the model already handles those). Reply with only the number."
    """
    print("[llm] score_luxuriousness_via_llm is a stub -- wire to Claude/OpenAI, "
          "or pass luxuriousness manually.")
    return None


# ===========================================================================
# 8. RUN IT
# ===========================================================================


def dump_dataset(market: Market, market_key: str = None, *,
                 source_override: Optional[str] = None,
                 enrichment_cache: Optional[str] = None,
                 use_location: bool = True, target: str = "revenue",
                 out_path: str = None) -> None:
    """Write the unified per-listing dataset (id + every feature + the revenue
    label + photo-quality scores) to ONE CSV, and print a human summary. The
    photo_* columns are auto-joined from photo_scores_<market>.json whenever it
    exists -- so the dataset stays unified as scoring proceeds."""
    source = source_override or market.listings_url
    X, y, cols, _ = load_and_featurize(source, market.anchors, enrichment_cache,
                                       use_location=use_location, target=target)

    # recover listing ids aligned to the surviving rows (X keeps the raw row index)
    raw = pd.read_csv(source, low_memory=False, usecols=["id"], dtype={"id": str})
    out = X.copy()
    out.insert(0, "id", raw.loc[X.index, "id"].values)
    out["revenue_label"] = y.values

    # ---- auto-join photo-quality scores (the unification) ----
    market_key = market_key or market.name.split(",")[0].lower().replace(" ", "_")
    spath = f"photo_scores_{market_key}.json"
    if os.path.exists(spath):
        raw_scores = json.load(open(spath))

        def _n(s: object) -> str:
            s = str(s)
            return s[:-2] if s.endswith(".0") else s

        smap = {_n(k): v for k, v in raw_scores.items() if v}
        for f in PHOTO_FEATURES:
            out[f"photo_{f}"] = [(smap.get(_n(i)) or {}).get(f) for i in out["id"]]
        ncov = int(out["photo_overall"].notna().sum())
        print(f"[dump] joined photo scores: {ncov:,}/{len(out):,} listings have "
              f"photo_* columns ({ncov/max(len(out),1):.1%})")
    else:
        print(f"[dump] no {spath} yet -> photo_* columns omitted (run photo_scorer.py)")

    out_path = out_path or f"dataset_{market_key}.csv"
    out.to_csv(out_path, index=False)

    # ---- human-readable summary ----
    print("\n" + "=" * 64)
    print(f"DATASET: {market.name}  ({len(out):,} listings x {X.shape[1]} features)")
    print(f"written to: {out_path}")
    print("=" * 64)
    binary = [c for c in X.columns if set(X[c].dropna().unique()) <= {0, 1}]
    numeric = [c for c in X.columns if c not in binary]
    print(f"\nNUMERIC features ({len(numeric)}): range across listings")
    for c in numeric:
        print(f"  {c:<26} min {X[c].min():>10,.1f}  median {X[c].median():>10,.1f}  max {X[c].max():>12,.1f}")
    print(f"\nLABEL  revenue_label             min {y.min():>10,.0f}  median {y.median():>10,.0f}  max {y.max():>12,.0f}")
    print(f"\nBINARY features ({len(binary)}): % of listings = 1")
    for c in binary:
        print(f"  {c:<26} {X[c].mean():>5.0%}")
    print("\n--- 3 sample listings (transposed) ---")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(out.head(3).T.to_string())


def photo_test(market: Market, market_key: str, *,
               source_override: Optional[str] = None, target: str = "revenue") -> None:
    """THE test: does Claude's photo-quality score move accuracy? Trains ONLY on
    the photo-scored subset, comparing photos-OFF vs photos-ON on identical folds."""
    source = source_override or market.listings_url
    X, y, cols, _ = load_and_featurize(source, market.anchors, None,
                                       use_location=True, target=target)

    # align listing ids to the surviving rows, then keep only photo-scored ones
    raw = pd.read_csv(source, low_memory=False, usecols=["id"], dtype={"id": str})
    ids = raw.loc[X.index, "id"].values

    spath = f"photo_scores_{market_key}.json"
    if not os.path.exists(spath):
        print(f"[photo-test] no {spath} -- run photo_fetch + photo_scorer first."); return
    scores = {str(k): v for k, v in json.load(open(spath)).items() if v}

    def norm(s: object) -> str:
        s = str(s)
        return s[:-2] if s.endswith(".0") else s

    smap = {norm(k): v for k, v in scores.items()}

    # only trust listings with >=3 REAL property photos -- the rest are "can't
    # assess" (all-5 neutral), which would just add noise to the test.
    MIN_PROPERTY_PHOTOS = 3

    def usable(i: object) -> bool:
        v = smap.get(norm(i))
        return bool(v) and (v.get("property_photo_count") or 0) >= MIN_PROPERTY_PHOTOS

    have = np.array([usable(i) for i in ids])
    n = int(have.sum())
    if n < 50:
        print(f"[photo-test] only {n} photo-scored listings in the data -- "
              "fetch + score more first."); return

    Xs = X[have].reset_index(drop=True)
    ys = y[have].reset_index(drop=True)
    sub_ids = ids[have]
    photo_cols = pd.DataFrame(
        {f: [float(smap[norm(i)].get(f, 5)) for i in sub_ids] for f in PHOTO_FEATURES}
    )
    print(f"[photo-test] {market.name}: {n} photo-scored listings\n")

    print("[photo-test] WITHOUT photo features:")
    m0, s0, _ = cross_validate_mdape(Xs, ys)
    Xp = pd.concat([Xs, photo_cols], axis=1)
    print("[photo-test] WITH photo features:")
    m1, s1, _ = cross_validate_mdape(Xp, ys)

    print(f"\n[photo-test] ===== VERDICT ({n} listings) =====")
    print(f"[photo-test] WITHOUT photos: {m0:.1%} +/- {s0:.1%}")
    print(f"[photo-test] WITH photos:    {m1:.1%} +/- {s1:.1%}")
    delta = (m1 - m0) * 100
    verdict = ("photos HELP" if m1 < m0 - 0.01 else
               "photos HURT" if m1 > m0 + 0.01 else "within noise (no effect)")
    print(f"[photo-test] delta: {delta:+.1f} pp  -> {verdict}")

    model = _new_forest()
    model.fit(Xp, ys)
    imp = sorted(zip(Xp.columns, model.feature_importances_), key=lambda t: -t[1])[:12]
    print("[photo-test] top features (with photos):")
    for name, val in imp:
        tag = "  <-- PHOTO" if name in PHOTO_FEATURES else ""
        print(f"          {val:6.3f}  {name}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="STR revenue model (Inside Airbnb).")
    ap.add_argument("--market", default="asheville", choices=sorted(MARKETS),
                    help="Which market to model (see MARKETS registry).")
    ap.add_argument("--source", default=None,
                    help="Optional local CSV path overriding the market's data URL.")
    ap.add_argument("--enrichment", default=None,
                    help="Optional sqft cache JSON (default: auto-detect enrichment_<market>.json).")
    ap.add_argument("--no-location", action="store_true",
                    help="Drop the markets + distance-to-attraction location features.")
    ap.add_argument("--dump", action="store_true",
                    help="Write the featurized per-listing dataset to CSV + print a summary, then exit.")
    ap.add_argument("--photo-test", action="store_true",
                    help="Test whether photo-quality scores improve accuracy (subset, photos on/off).")
    ap.add_argument("--target", default="revenue", choices=["revenue", "price"],
                    help="What to predict: 'revenue' (estimate) or 'price' (advertised ADR, more predictable).")
    args = ap.parse_args()
    market = MARKETS[args.market]
    use_location = not args.no_location

    if args.photo_test:
        photo_test(market, args.market, source_override=args.source, target=args.target)
        return

    # auto-detect the enrichment cache for this market if the user didn't name one
    cache = args.enrichment or f"enrichment_{args.market}.json"
    cache = cache if os.path.exists(cache) else None

    if args.dump:
        dump_dataset(market, args.market, source_override=args.source,
                     enrichment_cache=cache, use_location=use_location, target=args.target)
        return

    rm = train(market, source_override=args.source, enrichment_cache=cache, target=args.target)

    # --- demo: a candidate near downtown of the chosen market, mountain-cabin amenities ---
    # (Asheville defaults; coords sit ~downtown so distance features see a prime spot.)
    near = next(iter(market.anchors.values()))   # the market's first/primary anchor
    candidate = PropertyFeatures(
        bedrooms=3, bathrooms=2, accommodates=6, beds=4,
        latitude=near[0], longitude=near[1],
        room_type="Entire home/apt", property_type="Entire home",
        minimum_nights=2, instant_bookable=True,
        amenities={"hot_tub": True, "fire_pit": True, "patio": True,
                   "free_parking": True, "pets_allowed": True},
    )

    print("\n" + "=" * 60)
    print(f"CANDIDATE PREDICTION  ({market.name}: 3bd/2ba, hot tub, fire pit, near core)")
    print("=" * 60)
    for lux in (3, 5, 8):
        candidate.luxuriousness = lux
        out = predict(rm, candidate)
        print(f"  luxuriousness {lux}: ${out['adjusted_revenue']:,.0f}/yr "
              f"(band ${out['low']:,.0f}–${out['high']:,.0f})")
    print("\n  (base, luxuriousness=5, before lever: "
          f"${predict(rm, PropertyFeatures(**{**asdict(candidate), 'luxuriousness': None}))['base_revenue']:,.0f})")


if __name__ == "__main__":
    main()
