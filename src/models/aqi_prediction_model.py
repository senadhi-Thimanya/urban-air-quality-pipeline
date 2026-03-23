"""
aqi_prediction_model.py
-----------------------
Gold Layer — trains a Linear Regression model to predict hourly AQI
from traffic congestion features.

Input:  S3 gold/aqi_traffic_joined/ (Parquet)
Output: Trained model metrics printed to stdout + model saved locally.

This script is intentionally self-contained so it can be run from
a local Jupyter Notebook or a SageMaker instance without changes.
"""

import logging
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("aqi_model")

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "aws_config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

BUCKET = cfg["s3"]["bucket_name"]
GOLD   = cfg["s3"]["gold_prefix"]
REGION = cfg["aws"]["region"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gold_data() -> pd.DataFrame:
    """
    Read the joined Parquet dataset from S3 into a Pandas DataFrame.
    awswrangler handles S3 path resolution and credential passing automatically.
    """
    import awswrangler as wr

    s3_path = f"s3://{BUCKET}/{GOLD}aqi_traffic_joined/"
    log.info("Loading gold data from %s", s3_path)
    df = wr.s3.read_parquet(path=s3_path)
    log.info("  Loaded %d rows × %d columns", *df.shape)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-ready features from the joined dataset.

    Features (X):
      - avg_speed_kmph        : average traffic speed
      - avg_congestion_index  : (1 - speed/free_flow_speed), 0–1
      - is_rush_hour          : boolean rush-hour flag
      - hour                  : hour of day (captures diurnal patterns)
      - hour_sin / hour_cos   : cyclical encoding of hour (0–23)

    Target (y):
      - avg_aqi               : average AQI for the station/hour
    """
    df = df.dropna(subset=["avg_aqi", "avg_speed_kmph", "avg_congestion_index"])

    # Cyclical encoding for hour (avoids jump from 23 → 0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_rush_hour"] = df["is_rush_hour"].astype(int)

    feature_cols = [
        "avg_speed_kmph",
        "avg_congestion_index",
        "is_rush_hour",
        "hour_sin",
        "hour_cos",
    ]

    X = df[feature_cols]
    y = df["avg_aqi"]

    log.info("Features: %s", feature_cols)
    log.info("Target: avg_aqi  |  Samples: %d", len(y))
    return X, y, feature_cols


# ── Model training ────────────────────────────────────────────────────────────

def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """
    Split data, scale features, train a Linear Regression model,
    and print evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    # Standardise features (zero mean, unit variance)
    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train
    model = LinearRegression()
    model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    log.info("═" * 45)
    log.info("  Linear Regression — Evaluation Metrics")
    log.info("  MAE  : %.2f AQI points", mae)
    log.info("  RMSE : %.2f AQI points", rmse)
    log.info("  R²   : %.4f", r2)
    log.info("═" * 45)

    # Print feature coefficients
    coef_df = pd.DataFrame({
        "Feature":     X.columns,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", key=abs, ascending=False)
    log.info("Feature coefficients:\n%s", coef_df.to_string(index=False))

    return model, scaler, y_pred, y_test


# ── Persist ───────────────────────────────────────────────────────────────────

def save_model(model, scaler, feature_cols: list, output_dir: str = "models/") -> None:
    """Save model artifacts locally (upload to S3 in production)."""
    Path(output_dir).mkdir(exist_ok=True)
    joblib.dump(model,        f"{output_dir}linear_regression.pkl")
    joblib.dump(scaler,       f"{output_dir}scaler.pkl")
    joblib.dump(feature_cols, f"{output_dir}feature_cols.pkl")
    log.info("Model artifacts saved to '%s'", output_dir)


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_aqi(
    avg_speed_kmph: float,
    avg_congestion_index: float,
    is_rush_hour: bool,
    hour: int,
    model_dir: str = "models/",
) -> float:
    """
    Load saved model and predict AQI for a single observation.
    Example usage:
        aqi = predict_aqi(avg_speed_kmph=25, avg_congestion_index=0.6,
                          is_rush_hour=True, hour=8)
    """
    model        = joblib.load(f"{model_dir}linear_regression.pkl")
    scaler       = joblib.load(f"{model_dir}scaler.pkl")
    feature_cols = joblib.load(f"{model_dir}feature_cols.pkl")

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    row = pd.DataFrame([{
        "avg_speed_kmph":       avg_speed_kmph,
        "avg_congestion_index": avg_congestion_index,
        "is_rush_hour":         int(is_rush_hour),
        "hour_sin":             hour_sin,
        "hour_cos":             hour_cos,
    }])[feature_cols]

    row_scaled = scaler.transform(row)
    return float(model.predict(row_scaled)[0])


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    df               = load_gold_data()
    X, y, feat_cols  = engineer_features(df)
    model, scaler, y_pred, y_test = train_and_evaluate(X, y)
    save_model(model, scaler, feat_cols)

    # Quick sanity check
    sample_aqi = predict_aqi(
        avg_speed_kmph=20,
        avg_congestion_index=0.65,
        is_rush_hour=True,
        hour=8,
    )
    log.info("Sample prediction — Rush hour, high congestion → AQI ≈ %.1f", sample_aqi)


if __name__ == "__main__":
    run()