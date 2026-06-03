"""
photo_scorer.py  —  cheap Claude-vision quality scores for listing photos.

Pipeline position:
    listing id -> photo_fetch.py -> [SCORE PHOTOS] -> features -> model

WHAT IT DOES
------------
Reads the hero photos collected by photo_fetch.py and asks Claude (Haiku, the
cheapest capable model) to score each listing's QUALITY on a 1-5 rubric:
kitchen, bathroom, exterior, view, interior_decor, uniqueness, overall. Results
are cached by listing id, then joined into the model as real features.

WHY IT'S CHEAP (the levers)
---------------------------
1. Downsized images  -> tokens scale with (w x h)/750, so we request ~480px.
2. One call per listing, whole gallery in it -> not one call per photo.
3. Haiku 4.5         -> $1 / $5 per 1M in/out (cheapest capable).
4. Batch API (--batch) -> flat 50% off for the offline bulk run.
5. Structured output -> clean JSON, zero parsing.
Net: ~$0.002/listing -> a few dollars for a whole market.

SETUP (needed before this runs)
-------------------------------
    pip install anthropic
    echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env     # a real API key, not the Code login

USAGE
-----
    python3 photo_scorer.py --sync  --limit 10      # pilot: immediate, full price (pennies)
    python3 photo_scorer.py --batch                 # full market: 50% off, ~1h async
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

# 1-10 scale. NOTE: structured outputs forbid min/max, so we use an enum.
_SCORE = {"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
SCORE_FIELDS = ["kitchen", "bathroom", "exterior", "view",
                "interior_decor", "uniqueness", "overall"]
SCHEMA = {
    "type": "object",
    "properties": {
        **{f: _SCORE for f in SCORE_FIELDS},
        # transparency fields -- let a human audit whether it grounded correctly
        "property_photo_count": {"type": "integer"},   # how many photos actually show the unit
        "notes": {"type": "string"},                    # 1 sentence: what it saw + what it ignored
    },
    "required": SCORE_FIELDS + ["property_photo_count", "notes"],
    "additionalProperties": False,
}

RUBRIC = (
    "You score short-term-rental QUALITY from listing photos. Be STRICT and "
    "evidence-based -- never guess or inflate.\n\n"
    "CRITICAL -- IGNORE NON-PROPERTY PHOTOS. Galleries often include the "
    "neighborhood, nearby beaches, city skylines, sunsets, pools/amenities of the "
    "building, aerial/drone shots, maps, or stock images. Those are NOT the rental "
    "unit -- ignore them completely. Rate ONLY the actual unit: its own rooms, its "
    "own building/entrance, its own private outdoor space. A scenic beach or "
    "skyline photo must NOT raise any score.\n\n"
    "Scores are 1-10: 1-2=dated/basic/poor, 5=average for the market, "
    "9-10=high-end/designer/recently renovated. Use the FULL range -- most "
    "listings are not a 5.\n"
    "- kitchen: the unit's kitchen (counters, cabinets, appliances, styling)\n"
    "- bathroom: the unit's bathroom (fixtures, vanity, shower/tub)\n"
    "- exterior: the unit's OWN building/entrance/private yard -- NOT a nearby "
    "beach, street, or shared resort grounds\n"
    "- view: score HIGH only if the view is clearly shown FROM the unit (its own "
    "window, balcony, or lanai). A standalone beach/sunset/skyline shot that could "
    "be taken anywhere is NOT the unit's view -- if unsure, score view 5\n"
    "- interior_decor: the unit's furnishings, cohesion, lighting\n"
    "- uniqueness: distinctive design OF THE UNIT (not the location)\n"
    "- overall: holistic quality of the unit itself\n\n"
    "If a dimension is not shown in a PROPERTY photo, score it 5 (unknown) -- not "
    "high. Return property_photo_count (how many of the photos actually show the "
    "unit) and notes (one sentence: what you saw + which photos you ignored)."
)

MODEL = "claude-haiku-4-5"   # cheapest capable model; user-selected for cost


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _downsize(url: str, w: int = 480) -> str:
    """Request a small render -> fewer image tokens. muscache honors ?im_w=."""
    return f"{url}{'&' if '?' in url else '?'}im_w={w}"


def _content(photo_urls: List[str], max_photos: int = 12) -> list:
    blocks = [{"type": "image", "source": {"type": "url", "url": _downsize(u)}}
              for u in photo_urls[:max_photos]]
    blocks.append({"type": "text", "text": "Score this listing's photos."})
    return blocks


# ---------------------------------------------------------------------------
# Sync mode -- immediate, for the pilot (full price, but pennies).
# ---------------------------------------------------------------------------


def score_sync(galleries: Dict[str, List[str]], cache_path: str,
               limit: Optional[int] = None) -> None:
    import anthropic
    client = anthropic.Anthropic()      # reads ANTHROPIC_API_KEY
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    todo = [(lid, ph) for lid, ph in galleries.items() if ph and lid not in cache]
    if limit:
        todo = todo[:limit]
    print(f"[score] sync-scoring {len(todo)} listings with {MODEL}...")
    for lid, photos in todo:
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=500, system=RUBRIC,
                messages=[{"role": "user", "content": _content(photos)}],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            )
            text = next(b.text for b in resp.content if b.type == "text")
            cache[lid] = json.loads(text)
            print(f"[score] {lid}: {cache[lid]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[score] {lid}: failed ({exc})")
        json.dump(cache, open(cache_path, "w"))
    print(f"[score] wrote {len(cache)} scored listings -> {cache_path}")


# ---------------------------------------------------------------------------
# Batch mode -- 50% off, for the full market run.
# ---------------------------------------------------------------------------


def score_batch(galleries: Dict[str, List[str]], cache_path: str) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [(lid, ph) for lid, ph in galleries.items() if ph and lid not in cache]
    if not todo:
        print("[score] nothing new to score."); return

    requests = [
        Request(custom_id=lid, params=MessageCreateParamsNonStreaming(
            model=MODEL, max_tokens=500, system=RUBRIC,
            messages=[{"role": "user", "content": _content(photos)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        ))
        for lid, photos in todo
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"[score] batch {batch.id} submitted ({len(requests)} listings, 50% off)")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"[score] status={b.processing_status}, processing={b.request_counts.processing}")
        time.sleep(60)
    for r in client.messages.batches.results(batch.id):
        if r.result.type == "succeeded":
            text = next(b.text for b in r.result.message.content if b.type == "text")
            cache[r.custom_id] = json.loads(text)
    json.dump(cache, open(cache_path, "w"))
    print(f"[score] wrote {len(cache)} scored listings -> {cache_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score listing photos with Claude vision.")
    ap.add_argument("--market", default="asheville")
    ap.add_argument("--sync", action="store_true", help="Immediate scoring (pilot).")
    ap.add_argument("--batch", action="store_true", help="Batch scoring, 50% off (scale).")
    ap.add_argument("--limit", type=int, default=None, help="Cap listings (pilot).")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[score] ANTHROPIC_API_KEY not set. Add it to .env:\n"
              "        ANTHROPIC_API_KEY=sk-ant-...   (get one at console.anthropic.com)")
        return

    photos_path = f"photos_{args.market}.json"
    if not os.path.exists(photos_path):
        print(f"[score] no {photos_path} -- run photo_fetch.py first."); return
    galleries = json.load(open(photos_path))

    cache_path = f"photo_scores_{args.market}.json"
    if args.batch:
        score_batch(galleries, cache_path)
    else:
        score_sync(galleries, cache_path, limit=args.limit)


if __name__ == "__main__":
    main()
