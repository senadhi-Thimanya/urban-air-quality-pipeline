"""
fetch_traffic_data.py
---------------------
Pulls real-time traffic flow data from the TomTom Traffic Flow API
and uploads the raw JSON to the S3 landing zone.

TomTom free tier: 2,500 requests/day
Sign up at: https://developer.tomtom.com/
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import requests
import yaml

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("traffic_ingestion")

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "aws_config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

TOMTOM_KEY = cfg["tomtom"]["api_key"]
BBOX       = cfg["tomtom"]["bbox"]          # "min_lon,min_lat,max_lon,max_lat"
BUCKET     = cfg["s3"]["bucket_name"]
LANDING    = cfg["s3"]["landing_prefix"]
AWS_REGION = cfg["aws"]["region"]

# TomTom Traffic Flow Segment Data endpoint
TOMTOM_BASE = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# ── Grid helpers ──────────────────────────────────────────────────────────────

def build_sample_points(bbox: str, grid_steps: int = 5) -> list[tuple[float, float]]:
    """
    Create a regular grid of (lat, lon) sample points within the bounding box.
    TomTom returns traffic data for the road segment nearest each point.
    """
    min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
    lon_step = (max_lon - min_lon) / grid_steps
    lat_step = (max_lat - min_lat) / grid_steps

    points = []
    for i in range(grid_steps + 1):
        for j in range(grid_steps + 1):
            lat = min_lat + i * lat_step
            lon = min_lon + j * lon_step
            points.append((round(lat, 6), round(lon, 6)))

    log.info("Generated %d sample points from bounding box", len(points))
    return points


def fetch_flow_segment(lat: float, lon: float, api_key: str) -> dict | None:
    """
    Call the TomTom Flow Segment Data API for a single coordinate.
    Returns the parsed JSON or None on failure.
    """
    params = {
        "point": f"{lat},{lon}",
        "unit": "KMPH",
        "openLr": "false",
        "key": api_key,
    }
    try:
        resp = requests.get(TOMTOM_BASE, params=params, timeout=15)
        if resp.status_code == 404:
            # No road segment near this point — skip silently
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.warning("  TomTom request failed for (%s, %s): %s", lat, lon, exc)
        return None


# ── S3 upload ─────────────────────────────────────────────────────────────────

def upload_to_s3(payload: dict, s3_key: str, bucket: str, region: str) -> None:
    s3   = boto3.client("s3", region_name=region)
    body = json.dumps(payload, default=str)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=body, ContentType="application/json")
    log.info("Uploaded → s3://%s/%s (%d bytes)", bucket, s3_key, len(body))


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    now    = datetime.now(timezone.utc)
    run_ts = now.strftime("%Y%m%dT%H%M%S")
    hour   = now.strftime("%H")

    log.info("Starting traffic ingestion at %s (hour=%s)", run_ts, hour)

    points   = build_sample_points(BBOX, grid_steps=5)
    segments = []

    for idx, (lat, lon) in enumerate(points):
        log.debug("  Querying point %d/%d: (%s, %s)", idx + 1, len(points), lat, lon)
        data = fetch_flow_segment(lat, lon, TOMTOM_KEY)

        if data and "flowSegmentData" in data:
            segment = data["flowSegmentData"]
            # Enrich with our metadata
            segment["sample_lat"]  = lat
            segment["sample_lon"]  = lon
            segment["captured_at"] = now.isoformat()
            segment["hour"]        = int(hour)
            # Derive a station ID by rounding coordinates to ~1 km grid
            segment["station_id"]  = f"{round(lat, 2)}_{round(lon, 2)}"
            segments.append(segment)

        # Be polite to the API (free tier rate limit)
        time.sleep(0.1)

    log.info("Collected %d traffic segments", len(segments))

    if not segments:
        log.error("No traffic data collected. Check your TomTom API key and bounding box.")
        sys.exit(1)

    upload_to_s3(
        payload={"run_ts": run_ts, "hour": int(hour), "segments": segments},
        s3_key=f"{LANDING}traffic/{run_ts}_traffic.json",
        bucket=BUCKET,
        region=AWS_REGION,
    )

    log.info("✓ Traffic ingestion complete. %d segments saved.", len(segments))


if __name__ == "__main__":
    run()