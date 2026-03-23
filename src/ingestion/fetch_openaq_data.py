"""
fetch_openaq_data.py
--------------------
Pulls air quality measurements from the OpenAQ v3 API and uploads
the raw JSON to the S3 landing zone.

OpenAQ API docs: https://docs.openaq.org/
Free API key:    https://docs.openaq.org/docs/getting-started
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import requests
import yaml

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("openaq_ingestion")

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "aws_config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

OPENAQ_BASE   = cfg["openaq"]["base_url"]
OPENAQ_KEY    = cfg["openaq"]["api_key"]
CITY          = cfg["openaq"]["city"]
LOOKBACK_DAYS = cfg["openaq"]["lookback_days"]
LIMIT         = cfg["openaq"]["limit"]
BUCKET        = cfg["s3"]["bucket_name"]
LANDING       = cfg["s3"]["landing_prefix"]
AWS_REGION    = cfg["aws"]["region"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"X-API-Key": OPENAQ_KEY, "Accept": "application/json"}


def get_locations(city: str) -> list[dict]:
    """Return all monitoring locations for a given city."""
    url = f"{OPENAQ_BASE}/locations"
    params = {"city": city, "limit": 100}
    log.info("Fetching locations for city='%s'", city)
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    locations = data.get("results", [])
    log.info("Found %d location(s)", len(locations))
    return locations


def get_measurements(location_id: int, date_from: str, date_to: str) -> list[dict]:
    """
    Return all measurements for a single location within a date range.
    Handles pagination automatically.
    """
    url = f"{OPENAQ_BASE}/measurements"
    all_results: list[dict] = []
    page = 1

    while True:
        params = {
            "location_id": location_id,
            "date_from": date_from,
            "date_to": date_to,
            "limit": LIMIT,
            "page": page,
            "parameter": "pm25",   # Fine particulate matter — core AQI driver
        }
        log.debug("  GET %s page=%d", url, page)
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)
        # Stop if we received fewer results than the page limit
        if len(results) < LIMIT:
            break
        page += 1

    return all_results


def upload_to_s3(payload: dict, s3_key: str, bucket: str, region: str) -> None:
    """Serialise payload to JSON and upload to S3."""
    s3 = boto3.client("s3", region_name=region)
    body = json.dumps(payload, default=str)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=body, ContentType="application/json")
    log.info("Uploaded → s3://%s/%s (%d bytes)", bucket, s3_key, len(body))


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    now      = datetime.now(timezone.utc)
    date_to  = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_from = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_ts   = now.strftime("%Y%m%dT%H%M%S")

    log.info("Ingestion window: %s → %s", date_from, date_to)

    # 1. Discover locations
    locations = get_locations(CITY)
    if not locations:
        log.error("No locations returned for city='%s'. Check your API key and city name.", CITY)
        sys.exit(1)

    # Upload raw locations manifest
    upload_to_s3(
        payload={"city": CITY, "run_ts": run_ts, "locations": locations},
        s3_key=f"{LANDING}openaq/locations/{run_ts}_locations.json",
        bucket=BUCKET,
        region=AWS_REGION,
    )

    # 2. Fetch measurements per location
    all_measurements: list[dict] = []
    for loc in locations:
        loc_id   = loc["id"]
        loc_name = loc.get("name", str(loc_id))
        log.info("Fetching measurements for location '%s' (id=%s)", loc_name, loc_id)

        measurements = get_measurements(loc_id, date_from, date_to)
        # Enrich each row with location metadata
        for m in measurements:
            m["location_id"]   = loc_id
            m["location_name"] = loc_name
            m["city"]          = CITY

        all_measurements.extend(measurements)
        log.info("  → %d measurements collected", len(measurements))

    log.info("Total measurements: %d", len(all_measurements))

    # 3. Upload combined measurements file to landing zone
    upload_to_s3(
        payload={"run_ts": run_ts, "city": CITY, "measurements": all_measurements},
        s3_key=f"{LANDING}openaq/measurements/{run_ts}_measurements.json",
        bucket=BUCKET,
        region=AWS_REGION,
    )

    log.info("✓ OpenAQ ingestion complete.")


if __name__ == "__main__":
    run()