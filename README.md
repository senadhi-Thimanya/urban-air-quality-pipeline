# Urban Air Quality & Traffic Correlation Pipeline

End-to-end data engineering pipeline that ingests air quality data
(OpenAQ API) and traffic flow data (TomTom API), transforms them
using PySpark, stores results on AWS S3, and trains a Linear
Regression model to predict the Air Quality Index (AQI) from traffic
conditions.

---

## Architecture Overview

```
OpenAQ API ──┐
             ├──► S3 landing/  ──► PySpark ETL ──► S3 processed/  ──► S3 gold/  ──► Athena ──► Linear Regression
TomTom API ──┘                                                                          (SQL)      (Jupyter)
             └── Orchestrator (run_pipeline.py) runs all steps in sequence
```

### S3 Layer Definitions
| Layer     | S3 Prefix    | Format | Contents                          |
|-----------|-------------|--------|-----------------------------------|
| Landing   | `landing/`  | JSON   | Raw API responses                 |
| Processed | `processed/`| Parquet| Cleaned, typed individual datasets|
| Gold      | `gold/`     | Parquet| Joined, aggregated, model-ready   |

---

## File Structure

```
urban-air-quality-pipeline/
├── src/
│   ├── ingestion/
│   │   ├── fetch_openaq_data.py     # Pulls PM2.5 measurements from OpenAQ v3 API
│   │   └── fetch_traffic_data.py   # Pulls flow segment data from TomTom API
│   ├── transformation/
│   │   └── clean_and_join.py       # PySpark ETL — cleanse, transform, join
│   ├── orchestration/
│   │   └── run_pipeline.py         # Single-command pipeline runner
│   └── models/
│       └── aqi_prediction_model.py # scikit-learn Linear Regression
├── config/
│   └── aws_config.yaml             # All config in one place (fill in your keys)
├── notebooks/
│   └── analysis_and_visualization.ipynb
├── README.md
└── requirements.txt
```

---

## Prerequisites

| Tool              | Version  | Purpose                          |
|-------------------|----------|----------------------------------|
| Python            | 3.11+    | Runtime                          |
| Java (JDK)        | 11 or 17 | Required by PySpark              |
| AWS CLI           | 2.x      | Credential management            |
| AWS account       | Free tier | S3, Athena                      |
| OpenAQ API key    | Free     | Air quality data                 |
| TomTom API key    | Free     | Traffic data (2,500 req/day)     |

---

## Step-by-Step Setup Guide

### Step 1 — Get your API keys

**OpenAQ (free):**
1. Go to https://docs.openaq.org/docs/getting-started
2. Click "Request API Key"
3. Copy the key — you will paste it into `aws_config.yaml`

**TomTom (free tier):**
1. Go to https://developer.tomtom.com/
2. Register → Create an App → Copy the API Key
3. The free tier gives you 2,500 requests/day — more than enough

---

### Step 2 — Set up your AWS environment

**a) Install the AWS CLI** (skip if already done):
- Windows: https://aws.amazon.com/cli/
- Mac: `brew install awscli`
- Linux: `sudo apt install awscli`

**b) Configure credentials:**
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), output format (json)
```

**c) Create your S3 bucket** (bucket names must be globally unique):
```bash
aws s3 mb s3://your-unique-bucket-name --region us-east-1
```

**d) Create the Athena results folder:**
```bash
aws s3api put-object --bucket your-unique-bucket-name --key athena-results/
```

---

### Step 3 — Fill in the config file

Open `config/aws_config.yaml` and replace every placeholder:

```yaml
s3:
  bucket_name: "your-unique-bucket-name"   # ← the bucket you just created

openaq:
  api_key: "your-openaq-api-key"           # ← from Step 1
  city: "London"                           # ← change to your chosen city

tomtom:
  api_key: "your-tomtom-api-key"           # ← from Step 1
  bbox: "-0.1276,51.5074,0.0077,51.5200"  # ← bounding box for your city
```

**Finding a bounding box for your city:**
- Go to https://boundingbox.klokantech.com/
- Search for your city → select "CSV" format → copy the four numbers

---

### Step 4 — Install Java (required for PySpark)

**Mac:**
```bash
brew install openjdk@17
export JAVA_HOME=$(brew --prefix openjdk@17)
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install openjdk-17-jdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

**Windows:**
Download from https://adoptium.net/ and set `JAVA_HOME` in Environment Variables.

Verify: `java -version`  (should show 17.x)

---

### Step 5 — Install Python dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# Install all packages
pip install -r requirements.txt
```

---

### Step 6 — Run the full pipeline

```bash
# From the project root directory
python src/orchestration/run_pipeline.py
```

The orchestrator will:
1. Fetch OpenAQ measurements and upload to `s3://your-bucket/landing/openaq/`
2. Fetch TomTom traffic segments and upload to `s3://your-bucket/landing/traffic/`
3. Run PySpark ETL — cleanse, join, aggregate → write to `processed/` and `gold/`
4. Register the gold Parquet as an Athena external table

You will see timestamped log output for each step. A `pipeline.log` file
is also created in the project root.

Expected runtime: 3–8 minutes depending on your internet speed and dataset size.

---

### Step 7 — Query with Athena

1. Open the AWS Console → Athena
2. Select workgroup "primary" and database "air_quality_db"
3. Run:

```sql
-- Preview the gold table
SELECT * FROM air_quality_db.aqi_traffic_hourly LIMIT 20;

-- Rush hour vs non-rush hour AQI comparison
SELECT
    is_rush_hour,
    ROUND(AVG(avg_aqi), 1)            AS mean_aqi,
    ROUND(AVG(avg_congestion_index), 3) AS mean_congestion
FROM air_quality_db.aqi_traffic_hourly
GROUP BY is_rush_hour;

-- Worst pollution hours
SELECT hour, ROUND(AVG(avg_aqi), 1) AS avg_aqi
FROM air_quality_db.aqi_traffic_hourly
GROUP BY hour
ORDER BY avg_aqi DESC;
```

---

### Step 8 — Train the prediction model & view charts

```bash
# Option A: Run the model script directly
python src/models/aqi_prediction_model.py

# Option B: Use the Jupyter notebook (recommended for visuals)
pip install jupyter
jupyter notebook notebooks/analysis_and_visualization.ipynb
```

The notebook produces four charts:
- `aqi_vs_congestion_by_hour.png`
- `rush_hour_aqi_comparison.png`
- `congestion_vs_aqi_scatter.png`
- `actual_vs_predicted_aqi.png`

Trained model artifacts are saved to `models/`:
- `linear_regression.pkl`
- `scaler.pkl`
- `feature_cols.pkl`

---

## ETL Cleansing Steps Applied

| Step                      | Where applied       | Details                                         |
|---------------------------|---------------------|-------------------------------------------------|
| Missing Value Handling    | Both datasets       | Rows with null core fields dropped              |
| Duplicate Handling        | Both datasets       | One reading per sensor per hour kept            |
| Corrupt Data Handling     | AQI                 | PM2.5 values outside 0–1000 µg/m³ removed      |
| Data Type Conversions     | Both datasets       | Timestamps parsed and normalised to UTC         |
| Data Standardisation      | AQI + Traffic       | PM2.5 → US EPA AQI; speed → congestion index   |
| Data Aggregation          | Gold layer join     | Hourly averages per station                     |

---

## Cost Estimate (AWS Free Tier)

| Service | Usage                   | Cost                           |
|---------|-------------------------|--------------------------------|
| S3      | < 5 GB storage          | Free (5 GB free tier)          |
| Athena  | < 1 TB queries/month    | Free (1 TB free tier)          |
| Glue    | Not used                | $0                             |
| EMR     | Not used                | $0                             |
| **Total** |                       | **~$0 for this project**       |

> ⚠️ **Important:** Delete your S3 bucket after marking to avoid any storage charges.
> ```bash
> aws s3 rb s3://your-unique-bucket-name --force
> ```

---

## Troubleshooting

**`JAVA_HOME not set` error:**
Set the environment variable as shown in Step 4 and restart your terminal.

**`NoCredentialsError` from boto3:**
Run `aws configure` again and make sure your access key and region are correct.

**`KeyError: measurements` when reading JSON:**
The OpenAQ API returned an empty response. Check your API key and city name in `aws_config.yaml`.

**PySpark runs slowly or out of memory:**
Reduce `lookback_days` in `aws_config.yaml` to 1–2 days to shrink the dataset.

**TomTom returns 403 Forbidden:**
Your API key is incorrect or you have exceeded the 2,500 req/day free limit.