"""
run_pipeline.py
---------------
Orchestrator — runs the full end-to-end pipeline in a single execution:
  1. Ingest OpenAQ air quality data  → S3 landing/
  2. Ingest TomTom traffic data      → S3 landing/
  3. ETL transform + join            → S3 processed/ & gold/
  4. Create Athena table             → queryable via SQL

Run with:
    python src/orchestration/run_pipeline.py

The pipeline stops at the first failure and logs the step that failed.
"""

import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),   # local log file
    ],
)
log = logging.getLogger("orchestrator")

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "aws_config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

BUCKET   = cfg["s3"]["bucket_name"]
GOLD     = cfg["s3"]["gold_prefix"]
REGION   = cfg["aws"]["region"]
ATHENA_DB  = cfg["athena"]["database"]
ATHENA_WG  = cfg["athena"]["workgroup"]
RESULTS_S3 = cfg["athena"]["results_bucket"]


# ── Step runner ───────────────────────────────────────────────────────────────

def run_step(step_name: str, func, *args, **kwargs):
    """Execute a pipeline step, log timing, and propagate exceptions."""
    log.info("━━━ Step: %-40s ━━━", step_name)
    start = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - start
        log.info("  ✓ %s completed in %.1fs", step_name, elapsed)
    except Exception as exc:
        elapsed = time.time() - start
        log.error("  ✗ %s FAILED after %.1fs: %s", step_name, elapsed, exc)
        log.debug(traceback.format_exc())
        raise


# ── Athena table setup ────────────────────────────────────────────────────────

def create_athena_table() -> None:
    """
    Register the gold Parquet dataset as an external Athena table
    so it can be queried with standard SQL.
    """
    athena  = boto3.client("athena", region_name=REGION)
    gold_s3 = f"s3://{BUCKET}/{GOLD}aqi_traffic_joined/"

    create_db_sql = f"CREATE DATABASE IF NOT EXISTS {ATHENA_DB};"
    create_table_sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DB}.aqi_traffic_hourly (
        station_id         STRING,
        date_key           DATE,
        hour               INT,
        avg_aqi            DOUBLE,
        avg_pm25           DOUBLE,
        max_aqi            DOUBLE,
        sensor_readings    BIGINT,
        avg_speed_kmph     DOUBLE,
        avg_congestion_index DOUBLE,
        is_rush_hour       BOOLEAN,
        traffic_samples    BIGINT
    )
    STORED AS PARQUET
    LOCATION '{gold_s3}'
    TBLPROPERTIES ('parquet.compress'='SNAPPY');
    """

    for sql in [create_db_sql, create_table_sql]:
        response = athena.start_query_execution(
            QueryString=sql,
            WorkGroup=ATHENA_WG,
            ResultConfiguration={"OutputLocation": RESULTS_S3},
        )
        qid = response["QueryExecutionId"]
        _wait_for_athena(athena, qid)

    log.info("Athena table '%s.aqi_traffic_hourly' is ready.", ATHENA_DB)


def _wait_for_athena(athena_client, query_execution_id: str, timeout: int = 120) -> None:
    """Poll Athena until the query finishes or times out."""
    start = time.time()
    while True:
        resp   = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state  = resp["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            return
        if state in ("FAILED", "CANCELLED"):
            reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
            raise RuntimeError(f"Athena query {query_execution_id} {state}: {reason}")
        if time.time() - start > timeout:
            raise TimeoutError(f"Athena query {query_execution_id} timed out.")
        time.sleep(3)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    pipeline_start = datetime.now(timezone.utc)
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║   Urban Air Quality Pipeline — Starting          ║")
    log.info("║   %s UTC                     ║", pipeline_start.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("╚══════════════════════════════════════════════════╝")

    # Import here so Spark doesn't initialise until needed
    from ingestion.fetch_openaq_data  import run as ingest_aqi
    from ingestion.fetch_traffic_data import run as ingest_traffic
    from transformation.clean_and_join import run as transform

    steps = [
        ("1. Ingest OpenAQ Air Quality",  ingest_aqi),
        ("2. Ingest TomTom Traffic",      ingest_traffic),
        ("3. ETL Transform & Join",        transform),
        ("4. Register Athena Table",       create_athena_table),
    ]

    for step_name, func in steps:
        run_step(step_name, func)

    total = time.time() - pipeline_start.timestamp()
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║   Pipeline COMPLETE in %.1fs                    ║", total)
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("Query your data in Athena:")
    log.info("  SELECT * FROM %s.aqi_traffic_hourly LIMIT 20;", ATHENA_DB)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("Pipeline aborted. See errors above.")
        sys.exit(1)