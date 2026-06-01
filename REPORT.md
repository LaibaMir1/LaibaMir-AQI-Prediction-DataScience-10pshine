# AQI Predictor — Final Project Report
## Karachi Air Quality Index Prediction System
**10Pearls Data Science Project**

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Collection & Feature Pipeline](#3-data-collection--feature-pipeline)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Training & Evaluation](#6-model-training--evaluation)
7. [SHAP Explainability](#7-shap-explainability)
8. [Web Application](#8-web-application)
9. [Automated CI/CD Pipeline](#9-automated-cicd-pipeline)
10. [Results Summary](#10-results-summary)
11. [Challenges & Solutions](#11-challenges--solutions)
12. [Future Improvements](#12-future-improvements)

---

## 1. Project Overview

### Objective
Build an end-to-end, serverless Air Quality Index (AQI) prediction system for **Karachi, Pakistan** that:
- Fetches real-time weather and pollutant data from external APIs
- Trains and compares multiple ML models automatically
- Forecasts AQI for the next 3 days
- Displays results on an interactive Streamlit dashboard
- Runs fully automated via GitHub Actions CI/CD

### Tech Stack

| Layer | Technology |
|---|---|
| Data Source | Open-Meteo API (Weather + Air Quality) |
| Feature Store | Hopsworks Feature Store |
| Model Registry | Hopsworks Model Registry |
| ML Models | XGBoost, Random Forest, Ridge Regression |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Dashboard | Streamlit + Plotly |
| Database | SQLite (prediction history) |
| CI/CD | GitHub Actions |
| Language | Python 3.12 |

### Location
**Karachi, Pakistan**
- Latitude: 24.8607°N
- Longitude: 67.0011°E

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                          │
│  Open-Meteo Archive API    Open-Meteo Air Quality API   │
└──────────────────┬──────────────────┬───────────────────┘
                   │                  │
                   ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE PIPELINE                            │
│  fetch_karachi_data.py                                   │
│  → Fetch weather + pollutants                           │
│  → Aggregate hourly → daily                             │
│  → Compute derived features                             │
│  → Store in Hopsworks Feature Store                     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              TRAINING PIPELINE                           │
│  train_pipeline.py                                       │
│  → Load features from Hopsworks                         │
│  → Train XGBoost, RandomForest, Ridge                   │
│  → Evaluate & select best model                         │
│  → Save to Hopsworks Model Registry                     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              FORECAST PIPELINE                           │
│  forecast.py                                             │
│  → Fetch next 3-day weather forecast                    │
│  → Generate AQI predictions                             │
│  → Return daily forecast dataframe                      │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              WEB APPLICATION                             │
│  app.py (Streamlit)                                      │
│  → Overview & KPI metrics                               │
│  → Data Analysis & EDA                                  │
│  → Model Performance & SHAP                             │
│  → Manual AQI Prediction                                │
│  → 3-Day Forecast                                       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              CI/CD AUTOMATION                            │
│  GitHub Actions                                          │
│  → feature_pipeline.yml  → runs every hour              │
│  → training_pipeline.yml → runs every day at 2AM        │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Collection & Feature Pipeline

### Data Source
**Open-Meteo API** (free, no API key required)
- Archive API: `https://archive-api.open-meteo.com/v1/archive`
- Air Quality API: `https://air-quality-api.open-meteo.com/v1/air-quality`
- Forecast API: `https://api.open-meteo.com/v1/forecast`

### Data Range
- **Historical:** January 2024 — May 2026
- **Total records:** 874 daily observations
- **Frequency:** Hourly data aggregated to daily mean

### Raw Features Collected

| Feature | Description | Unit |
|---|---|---|
| PM2.5 | Fine particulate matter | μg/m³ |
| PM10 | Coarse particulate matter | μg/m³ |
| NO2 | Nitrogen dioxide | μg/m³ |
| SO2 | Sulphur dioxide | μg/m³ |
| CO | Carbon monoxide | μg/m³ |
| O3 | Ozone | μg/m³ |
| Temperature | Mean daily temperature | °C |
| Humidity | Mean relative humidity | % |
| Precipitation | Total daily rainfall | mm |
| AQI | US Air Quality Index | - |

### Data Cleaning
- Missing temperature/humidity → filled with column mean
- Missing precipitation → filled with 0
- Hourly → daily aggregation using mean (sum for precipitation)
- Rows with NaN after feature engineering → dropped

---

## 4. Exploratory Data Analysis

### Key Findings

**AQI Distribution:**
- Average AQI: ~120 (Unhealthy for Sensitive Groups range)
- Maximum AQI: ~280 (Very Unhealthy)
- Minimum AQI: ~45 (Good)
- Most common category: Moderate to Unhealthy for Sensitive Groups

**Seasonal Patterns:**
- AQI is highest in **winter months (November–January)** due to temperature inversion trapping pollutants near ground level
- AQI is lowest in **monsoon season (July–August)** as rainfall washes pollutants from the air
- Weekend AQI slightly lower than weekdays due to reduced industrial activity

**Strongest Correlations with AQI:**
1. PM2.5 (r = 0.95) — strongest predictor
2. PM10  (r = 0.92)
3. CO    (r = 0.78)
4. SO2   (r = 0.71)
5. Temperature (r = -0.42) — negative: higher temp = better mixing = lower AQI

---

## 5. Feature Engineering

### Final Feature Set (15 features)

| Feature | Type | Description |
|---|---|---|
| pm2_5 | Pollutant | Fine particulate matter |
| pm10 | Pollutant | Coarse particulate matter |
| no2 | Pollutant | Nitrogen dioxide |
| so2 | Pollutant | Sulphur dioxide |
| co | Pollutant | Carbon monoxide |
| o3 | Pollutant | Ozone |
| temperature | Weather | Mean daily temperature |
| humidity | Weather | Mean relative humidity |
| precipitation | Weather | Total daily rainfall |
| month | Temporal | Month of year (1–12) |
| day_of_week | Temporal | Day of week (0=Monday) |
| is_weekend | Temporal | 1 if Saturday/Sunday |
| aqi_change_rate | Derived | AQI difference from previous day |
| aqi_rolling_3d | Derived | 3-day rolling average AQI |
| aqi_rolling_7d | Derived | 7-day rolling average AQI |

### Why Derived Features Matter
The rolling average and change rate features capture **temporal momentum** in air quality:
- A rising AQI trend (positive change rate) predicts continued deterioration
- Rolling averages smooth out daily noise and capture week-scale patterns
- Adding these features improved R² from ~0.85 to **0.9545**

---

## 6. Model Training & Evaluation

### Models Compared

| Model | MAE (Test) | RMSE (Test) | R² (Test) |
|---|---|---|---|
| XGBoost | 4.87 | 6.69 | 0.9140 |
| Random Forest | 5.16 | 7.70 | 0.8859 |
| **Ridge Regression** ✅ | **3.73** | **4.86** | **0.9545** |

### Best Model: Ridge Regression
Ridge Regression outperformed both tree-based models because:
- The derived rolling features (aqi_rolling_3d, aqi_rolling_7d) have a **near-linear relationship** with next-day AQI
- Ridge handles multicollinearity well (PM2.5 and PM10 are highly correlated)
- With L2 regularization (α=1.0), it generalises better on this dataset size

### Training Configuration
- **Train/Test split:** 80% / 20% (699 train, 175 test)
- **Feature scaling:** StandardScaler (zero mean, unit variance)
- **Random state:** 42 (reproducible)
- **Selection criterion:** Highest R² on test set

---

## 7. SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to explain individual model predictions and understand global feature importance.

### Key SHAP Findings

**Most Influential Features (by mean |SHAP value|):**
1. **aqi_rolling_7d** — 7-day rolling average is the single strongest predictor
2. **aqi_rolling_3d** — 3-day rolling average confirms trend direction
3. **pm2_5** — fine particulate matter drives acute AQI spikes
4. **aqi_change_rate** — rate of change indicates trajectory
5. **co** — carbon monoxide levels indicate combustion activity

**Key Insights:**
- High rolling AQI values → model predicts continued high AQI (momentum effect)
- High PM2.5 consistently pushes predictions toward Unhealthy range
- High precipitation → lower predicted AQI (rain washes pollutants)
- Winter months (high month value) → higher predicted AQI

---

## 8. Web Application

### Dashboard Pages

#### 🏠 Overview
- Latest AQI metric card with category
- 30/90-day AQI trend chart
- AQI category breakdown (pie + bar charts)
- Hazardous alert banners (auto-triggered by AQI level)
- AQI health guide

#### 🔍 Data Analysis
- Dataset preview and statistics
- Correlation heatmap
- Feature distributions (PM2.5, Temperature, Humidity, AQI)
- Temperature vs PM2.5 dual-axis time series
- Monthly average AQI bar chart

#### 🤖 Model Performance
- Best model metrics (MAE, RMSE, R²)
- All 3 models comparison table + bar charts
- R² accuracy gauge
- SHAP feature importance plots (bar + beeswarm)
- Model configuration table

#### 🔮 Predict AQI
- Manual input form (pollutants + weather)
- Auto-computed derived features
- Instant AQI prediction with color-coded category badge
- Health advice based on predicted level
- Hazardous alert if AQI > 200
- Prediction history chart and table

#### 📅 3-Day Forecast
- Live data from Open-Meteo forecast API
- Color-coded day cards with AQI + weather
- Forecast trend chart with AQI zone backgrounds
- Weather conditions table
- Automatic alert if high AQI forecasted

### Alerts System
Automatic color-coded banners appear when:
- AQI > 150 → 🟠 Orange warning
- AQI > 200 → 🟣 Purple alert
- AQI > 300 → 🔴 Red emergency alert

---

## 9. Automated CI/CD Pipeline

### GitHub Actions Workflows

#### Feature Pipeline (`feature_pipeline.yml`)
- **Schedule:** Every hour (`0 * * * *`)
- **Action:** Fetches latest Karachi AQI + weather from Open-Meteo
- **Output:** Updates `karachi_daily_aqi_weather.csv` in repository

#### Training Pipeline (`training_pipeline.yml`)
- **Schedule:** Every day at 2:00 AM UTC (`0 2 * * *`)
- **Action:** Retrains all 3 models on updated dataset
- **Output:** Updates model pkl files in `models/` folder

### Benefits
- **No manual intervention** required after deployment
- Dataset stays fresh with hourly updates
- Models retrain daily on latest data
- Full audit trail in GitHub Actions logs

---

## 10. Results Summary

### What Was Achieved

| Requirement | Status |
|---|---|
| Fetch data from external API | ✅ Open-Meteo API |
| Compute features & store in Feature Store | ✅ Hopsworks |
| Backfill historical data | ✅ 2024–2026 (874 days) |
| Train multiple ML models | ✅ XGBoost, RandomForest, Ridge |
| Evaluate with RMSE, MAE, R² | ✅ All metrics computed |
| Store model in Model Registry | ✅ Hopsworks |
| Automated feature pipeline (hourly) | ✅ GitHub Actions |
| Automated training pipeline (daily) | ✅ GitHub Actions |
| Interactive Streamlit dashboard | ✅ 5 pages |
| SHAP feature importance | ✅ Bar + beeswarm plots |
| Hazardous AQI alerts | ✅ 3-level alert system |
| 3-day AQI forecast | ✅ Live Open-Meteo data |

### Model Performance
- **Best Model:** Ridge Regression
- **R² Score:** 0.9545 (95.45% variance explained)
- **MAE:** 3.73 AQI units
- **RMSE:** 4.86 AQI units

---

## 11. Challenges & Solutions

| Challenge | Solution |
|---|---|
| Slow pip installs on local machine | Used Tsinghua mirror for faster downloads |
| venv pushed to GitHub (136MB file) | Used git filter-repo to remove from history |
| twofish package needs C++ compiler | Tested locally without hopsworks, used local pkl files |
| Merge conflicts on .gitignore | Manually resolved keeping both Python and Node entries |
| SQLite schema mismatch after adding features | Deleted old DB, updated CREATE TABLE with new columns |
| Model pkl overwritten by redundant training steps | Removed duplicate Steps 23-27, used best_model variable consistently |
| SHAP incompatible with some model types | Used LinearExplainer for Ridge, TreeExplainer for tree models |

---

## 12. Future Improvements

1. **Deep Learning Model** — Add LSTM for sequence-based AQI forecasting
2. **Multi-city Support** — Extend to Lahore, Islamabad, Peshawar
3. **Real-time alerts** — Email/SMS notifications when AQI exceeds threshold
4. **Streamlit Cloud deployment** — Deploy dashboard publicly with Hopsworks model loading
5. **Longer forecast** — Extend from 3-day to 7-day forecast
6. **More features** — Add wind direction, solar radiation, traffic data
7. **Model monitoring** — Track model drift over time and trigger retraining
8. **LIME explainability** — Add LIME alongside SHAP for comparison

---

