"""
photo_fetch.py  —  get a listing's hero photos from its Airbnb page.

Pipeline position:
    listing id -> [FETCH PHOTOS] -> photo_scorer.py -> features -> model

HOW IT WORKS
------------
Inside Airbnb only gives us the COVER photo. But every listing has a public page
at airbnb.com/rooms/<id>, and that page's HTML contains the hero photos. Crucially,
each photo URL embeds the listing id, base64-encoded:

    .../hosting/Hosting-<base64("StaySupplyListing:<id>")>/original/<uuid>.jpeg

So we can match photos to the listing with certainty. We grab the `hosting/...`
URLs, group them by that encoded blob, and keep the most common one (= this
listing's own gallery, not "similar listings" thumbnails).

HONEST CAVEATS
--------------
1. This is SCRAPING Airbnb (against their ToS — fine for personal research, not
   something to productize). Be polite: slow rate, realistic headers.
2. Airbnb has bot detection. One fetch is fine; thousands of fast fetches get
   CAPTCHA'd. We rate-limit and CACHE every result so a block just means
   "re-run later" -- it resumes where it left off.
3. The static page yields the host's CURATED hero shots (~5-15), not the full
   gallery. That's actually ideal for a quality score -- it's the best rooms,
   and it's what guests judge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import time
import urllib.request
from collections import Counter
from typing import Dict, List

import pandas as pd

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = ssl.create_default_context()

# Hero photos live at this URL shape; group(1) is the base64-id blob.
_PHOTO_RE = re.compile(
    r"https://[a-z0-9.]*muscache\.com/im/pictures/hosting/Hosting-"
    r"([A-Za-z0-9_\-]+)/original/[a-f0-9\-]+\.(?:jpe?g|png)"
)

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_listing_photos(listing_id: str, *, max_photos: int = 12,
                         timeout: int = 20) -> List[str]:
    """Return up to `max_photos` hero photo URLs for one listing, or [] on failure."""
    url = f"https://www.airbnb.com/rooms/{listing_id}"
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
            html = r.read().decode("utf-8", "ignore")
    except Exception as exc:  # noqa: BLE001
        print(f"[photos] {listing_id}: fetch failed ({exc})")
        return []

    # collect (blob, full_url) pairs; keep the most common blob = this listing's set
    matches = [(m.group(1), m.group(0)) for m in _PHOTO_RE.finditer(html)]
    if not matches:
        return []
    top_blob = Counter(b for b, _ in matches).most_common(1)[0][0]
    seen, urls = set(), []
    for blob, full in matches:
        if blob == top_blob and full not in seen:
            seen.add(full)
            urls.append(full)
    return urls[:max_photos]


def bulk_fetch(listings_source: str, cache_path: str, *,
               limit: int = None, sample: int = None, delay: float = 2.5,
               entire_home_only: bool = True) -> None:
    """Fetch hero photos for listings, caching by id (resumable). `delay` seconds
    between requests keeps us polite and under the bot-detection radar.
    `sample` takes a RANDOM subset (seeded) -- better than the first-N, which are
    the oldest listings (mostly delisted)."""
    cols = ["id", "room_type"]
    df = pd.read_csv(listings_source, low_memory=False, usecols=cols, dtype={"id": str})
    if entire_home_only:
        df = df[df["room_type"].astype(str).str.strip() == "Entire home/apt"]
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    ids = df["id"].tolist()
    if limit:
        ids = ids[:limit]

    cache: Dict[str, List[str]] = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))

    fetched, hits = 0, 0
    for lid in ids:
        if lid in cache:                       # resumable: skip already-fetched
            continue
        photos = fetch_listing_photos(lid)
        cache[lid] = photos
        fetched += 1
        if photos:
            hits += 1
        print(f"[photos] {lid}: {len(photos)} hero photos")
        if fetched % 10 == 0:
            json.dump(cache, open(cache_path, "w"))
        time.sleep(delay)                      # be polite

    json.dump(cache, open(cache_path, "w"))
    covered = sum(1 for v in cache.values() if v)
    print(f"[photos] done. {fetched} new fetches ({hits} with photos). "
          f"{covered}/{len(cache)} cached listings have photos -> {cache_path}")


def main() -> None:
    from airbnb_revenue_engine import MARKETS
    ap = argparse.ArgumentParser(description="Fetch listing hero photos from Airbnb.")
    ap.add_argument("--market", default="asheville", choices=sorted(MARKETS))
    ap.add_argument("--source", default=None, help="Local listings CSV override.")
    ap.add_argument("--limit", type=int, default=None, help="Max listings (for a pilot).")
    ap.add_argument("--sample", type=int, default=None, help="Random subset size (seeded).")
    ap.add_argument("--delay", type=float, default=2.5, help="Seconds between fetches.")
    args = ap.parse_args()
    market = MARKETS[args.market]
    source = args.source or market.listings_url
    bulk_fetch(source, f"photos_{args.market}.json", limit=args.limit,
               sample=args.sample, delay=args.delay)


if __name__ == "__main__":
    main()
