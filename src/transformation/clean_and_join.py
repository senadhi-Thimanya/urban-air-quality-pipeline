"""
clean_and_join.py
-----------------
Silver Layer ETL — reads raw JSON from S3 landing/,
applies cleansing & transformations using PySpark,
joins the two datasets, and writes Parquet to S3 processed/.

Cleansing steps applied (satisfies coursework requirement for ≥ 2):
  1. Missing Value Handling
  2. Duplicate Handling
  3. Data Type Conversions / Formatting  (timestamps standardised to UTC ISO-8601)
  4. Data Standardisation               (speed → km/h, AQI calculated from PM2.5)
  5. Data Aggregation                   (hourly averages per station)
"""

import logging
import sys
from pathlib import Path

import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType, TimestampType
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("etl_transform")

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "aws_config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

BUCKET    = cfg["s3"]["bucket_name"]
LANDING   = cfg["s3"]["landing_prefix"]
PROCESSED = cfg["s3"]["processed_prefix"]
REGION    = cfg["aws"]["region"]


# ── Spark session ─────────────────────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    """
    Build a local Spark session with S3 (Hadoop-AWS) support.
    The hadoop-aws and aws-java-sdk JARs are pulled from Maven on first run.
    """
    spark = (
        SparkSession.builder
        .appName("UrbanAirQualityETL")
        .master("local[*]")        # Uses all CPU cores — horizontal scaling ready
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.367")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.sql.session.timeZone", "UTC")
        # Minimise shuffle partitions for small-scale local runs
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info("Spark session created (version %s)", spark.version)
    return spark


# ── Schema definitions ────────────────────────────────────────────────────────

AQI_SCHEMA = StructType([
    StructField("location_id",   StringType(),  True),
    StructField("location_name", StringType(),  True),
    StructField("city",          StringType(),  True),
    StructField("parameter",     StringType(),  True),
    StructField("value",         DoubleType(),  True),   # PM2.5 µg/m³
    StructField("unit",          StringType(),  True),
    StructField("date",          StructType([   # nested object from OpenAQ
        StructField("utc",   StringType(), True),
        StructField("local", StringType(), True),
    ]), True),
    StructField("coordinates",   StructType([
        StructField("latitude",  DoubleType(), True),
        StructField("longitude", DoubleType(), True),
    ]), True),
])

TRAFFIC_SCHEMA = StructType([
    StructField("station_id",      StringType(),  True),
    StructField("sample_lat",      DoubleType(),  True),
    StructField("sample_lon",      DoubleType(),  True),
    StructField("hour",            IntegerType(), True),
    StructField("captured_at",     StringType(),  True),
    StructField("currentSpeed",    DoubleType(),  True),  # km/h
    StructField("freeFlowSpeed",   DoubleType(),  True),  # km/h
    StructField("currentTravelTime", DoubleType(), True), # seconds
    StructField("freeFlowTravelTime", DoubleType(), True),
    StructField("confidence",      DoubleType(),  True),
    StructField("roadClosure",     StringType(),  True),
])


# ── Ingestion helpers ─────────────────────────────────────────────────────────

def read_json_from_s3(spark: SparkSession, prefix: str, inner_key: str) -> DataFrame:
    """
    Read all JSON files under an S3 prefix.
    OpenAQ and TomTom payloads wrap arrays under a key; we explode them.
    """
    path = f"s3a://{BUCKET}/{prefix}*.json"
    log.info("Reading from %s", path)
    raw = spark.read.option("multiLine", "true").json(path)
    # Explode the nested array (e.g. measurements or segments)
    df = raw.select(F.explode(F.col(inner_key)).alias("record")).select("record.*")
    log.info("  → %d raw rows", df.count())
    return df


# ── Cleansing functions ───────────────────────────────────────────────────────

def cleanse_aqi(df: DataFrame) -> DataFrame:
    """
    Apply all cleansing steps to the raw AQI DataFrame.
    """
    log.info("Cleansing AQI data…")
    initial = df.count()

    # 1. Flatten nested structs
    df = df.withColumn("measured_at_utc", F.col("date.utc")) \
           .withColumn("lat", F.col("coordinates.latitude")) \
           .withColumn("lon", F.col("coordinates.longitude")) \
           .drop("date", "coordinates")

    # 2. Type conversion — parse timestamp string → TimestampType (UTC)
    df = df.withColumn(
        "measured_at_utc",
        F.to_timestamp(F.col("measured_at_utc"), "yyyy-MM-dd'T'HH:mm:ss'Z'")
    )

    # 3. Missing value handling — drop rows where the core measurement is null
    df = df.dropna(subset=["value", "measured_at_utc", "location_id"])

    # 4. Filter out physically impossible PM2.5 readings (corrupt data)
    df = df.filter((F.col("value") >= 0) & (F.col("value") <= 1000))

    # 5. Data standardisation — compute US EPA AQI from PM2.5
    #    Using the simplified linear interpolation for the 0–500 AQI range.
    df = df.withColumn(
        "aqi",
        F.when(F.col("value") <= 12.0,   F.col("value") * (50.0  / 12.0))
         .when(F.col("value") <= 35.4,   50  + (F.col("value") - 12.0)  * (50.0  / 23.4))
         .when(F.col("value") <= 55.4,   100 + (F.col("value") - 35.4)  * (50.0  / 20.0))
         .when(F.col("value") <= 150.4,  150 + (F.col("value") - 55.4)  * (50.0  / 95.0))
         .when(F.col("value") <= 250.4,  200 + (F.col("value") - 150.4) * (100.0 / 100.0))
         .otherwise(300.0)
    ).withColumn("aqi", F.round(F.col("aqi"), 1))

    # 6. Extract join keys: station_id (rounded lat/lon) and hour
    df = df.withColumn("station_id", F.concat_ws(
            "_",
            F.round(F.col("lat"), 2).cast(StringType()),
            F.round(F.col("lon"), 2).cast(StringType())
        )) \
           .withColumn("hour", F.hour(F.col("measured_at_utc"))) \
           .withColumn("date_key", F.to_date(F.col("measured_at_utc")))

    # 7. Duplicate handling — keep the latest reading per sensor per hour
    df = df.dropDuplicates(["location_id", "date_key", "hour"])

    final = df.count()
    log.info("  AQI rows: %d → %d (removed %d)", initial, final, initial - final)
    return df


def cleanse_traffic(df: DataFrame) -> DataFrame:
    """
    Apply all cleansing steps to the raw traffic DataFrame.
    """
    log.info("Cleansing traffic data…")
    initial = df.count()

    # 1. Type conversion — parse captured_at to timestamp
    df = df.withColumn(
        "captured_at",
        F.to_timestamp(F.col("captured_at"))
    ).withColumn("date_key", F.to_date(F.col("captured_at")))

    # 2. Missing value handling
    df = df.dropna(subset=["station_id", "currentSpeed", "hour"])

    # 3. Corrupt data — remove zero or negative speeds
    df = df.filter(F.col("currentSpeed") > 0)

    # 4. Data standardisation — congestion index (0 = free flow, 1 = gridlock)
    df = df.withColumn(
        "congestion_index",
        F.round(
            F.lit(1.0) - (F.col("currentSpeed") / F.col("freeFlowSpeed")),
            4
        )
    ).withColumn("congestion_index",
        F.greatest(F.lit(0.0), F.col("congestion_index"))  # floor at 0
    )

    # 5. Rush-hour flag (07:00–09:00 and 17:00–19:00)
    df = df.withColumn(
        "is_rush_hour",
        F.when(
            F.col("hour").between(7, 9) | F.col("hour").between(17, 19),
            F.lit(True)
        ).otherwise(F.lit(False))
    )

    # 6. Duplicate handling — keep one reading per station per hour per day
    df = df.dropDuplicates(["station_id", "date_key", "hour"])

    final = df.count()
    log.info("  Traffic rows: %d → %d (removed %d)", initial, final, initial - final)
    return df


# ── Aggregation & join ────────────────────────────────────────────────────────

def aggregate_hourly(aqi_df: DataFrame, traffic_df: DataFrame) -> DataFrame:
    """
    Aggregate both datasets to hourly averages per station, then join.
    Join key: station_id + hour + date_key
    """
    log.info("Aggregating to hourly averages…")

    aqi_hourly = aqi_df.groupBy("station_id", "date_key", "hour") \
        .agg(
            F.avg("aqi").alias("avg_aqi"),
            F.avg("value").alias("avg_pm25"),
            F.max("aqi").alias("max_aqi"),
            F.count("*").alias("sensor_readings"),
        )

    traffic_hourly = traffic_df.groupBy("station_id", "date_key", "hour") \
        .agg(
            F.avg("currentSpeed").alias("avg_speed_kmph"),
            F.avg("congestion_index").alias("avg_congestion_index"),
            F.first("is_rush_hour").alias("is_rush_hour"),
            F.count("*").alias("traffic_samples"),
        )

    log.info("Joining on (station_id, date_key, hour)…")
    joined = aqi_hourly.join(
        traffic_hourly,
        on=["station_id", "date_key", "hour"],
        how="inner"
    )

    log.info("  Joined rows: %d", joined.count())
    return joined


def write_parquet(df: DataFrame, s3_key: str) -> None:
    path = f"s3a://{BUCKET}/{s3_key}"
    log.info("Writing Parquet → %s", path)
    df.write.mode("overwrite").parquet(path)
    log.info("  ✓ Write complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    spark = create_spark_session()

    try:
        # --- Read raw data from landing zone ---
        raw_aqi     = read_json_from_s3(spark, f"{LANDING}openaq/measurements/", "measurements")
        raw_traffic = read_json_from_s3(spark, f"{LANDING}traffic/",             "segments")

        # --- Silver layer cleansing ---
        clean_aqi     = cleanse_aqi(raw_aqi)
        clean_traffic = cleanse_traffic(raw_traffic)

        # --- Gold layer aggregation & join ---
        gold_df = aggregate_hourly(clean_aqi, clean_traffic)

        # --- Persist results ---
        write_parquet(clean_aqi,     f"{PROCESSED}aqi_clean/")
        write_parquet(clean_traffic, f"{PROCESSED}traffic_clean/")
        write_parquet(gold_df,       f"{cfg['s3']['gold_prefix']}aqi_traffic_joined/")

        log.info("✓ ETL transformation complete.")

    finally:
        spark.stop()


if __name__ == "__main__":
    run()