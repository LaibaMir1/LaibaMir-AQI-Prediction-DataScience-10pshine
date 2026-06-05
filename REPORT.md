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
7. [Model Selection Justification](#7-model-selection-justification)
8. [SHAP Explainability](#8-shap-explainability)
9. [Web Application](#9-web-application)
10. [Automated CI/CD Pipeline](#10-automated-cicd-pipeline)
11. [Challenges & Solutions](#11-challenges--solutions)
12. [Results Summary](#12-results-summary)
13. [Future Improvements](#13-future-improvements)

---

## 1. Project Overview

### Objective
Build an end-to-end, serverless Air Quality Index (AQI) prediction system for **Karachi, Pakistan** that:
- Fetches real-time weather and pollutant data from external APIs automatically
- Engineers meaningful features and stores them in a managed Feature Store
- Trains and compares multiple ML models to select the best performer
- Forecasts AQI for the next 3 days using live weather forecast data
- Displays results on an interactive Streamlit dashboard accessible publicly
- Runs fully automated via GitHub Actions CI/CD - no manual intervention required

### Why This Project Matters
Karachi is one of the most densely populated cities in the world with chronic air quality issues. Accurate AQI forecasting helps:
- Citizens plan outdoor activities safely
- Sensitive groups (elderly, children, asthma patients) take precautions
- Policymakers identify pollution trends and take action
- Health authorities issue timely warnings

### Tech Stack

| Layer | Technology | Why Chosen |
|---|---|---|
| Data Source | Open-Meteo API | Free, no API key, reliable, global coverage |
| Feature Store | Hopsworks | Industry-standard MLOps platform, free tier available |
| Model Registry | Hopsworks | Centralized model versioning and deployment |
| ML Models | XGBoost, Random Forest, Ridge | Range from linear to ensemble — best selected automatically |
| Explainability | SHAP | Gold standard for ML interpretability |
| Dashboard | Streamlit + Plotly | Fast to build, Python-native, free deployment |
| Database | SQLite | Lightweight, zero-config, perfect for prediction history |
| CI/CD | GitHub Actions | Free, integrated with GitHub, reliable scheduling |
| Language | Python 3.11 | Industry standard for ML/data science |

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
│  → Fetch weather + pollutants (hourly)                  │
│  → Aggregate hourly → daily                             │
│  → Compute derived AQI trend features                   │
│  → Store in Hopsworks Feature Store                     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              TRAINING PIPELINE                           │
│  train_pipeline.py / colab_pipeline.py                  │
│  → Load features from Hopsworks Feature Store           │
│  → Train XGBoost, RandomForest, Ridge                   │
│  → Evaluate all 3 models (MAE, RMSE, R²)               │
│  → Select best model automatically by R²               │
│  → Save to Hopsworks Model Registry                     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              FORECAST PIPELINE                           │
│  forecast.py                                             │
│  → Fetch next 3-day weather forecast from Open-Meteo   │
│  → Engineer features (rolling avg, change rate)         │
│  → Generate AQI predictions using best model            │
│  → Return daily forecast with health categories         │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              WEB APPLICATION                             │
│  app.py (Streamlit)                                      │
│  → Load model from Hopsworks Registry                   │
│  → Load features from Hopsworks Feature Store           │
│  → Overview & KPI metrics                               │
│  → Data Analysis & EDA (13 interactive charts)          │
│  → Model Performance & SHAP explainability              │
│  → Manual AQI Prediction with history                   │
│  → 3-Day Forecast with live data                        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              CI/CD AUTOMATION                            │
│  GitHub Actions                                          │
│  → feature_pipeline.yml  → runs every hour              │
│  → training_pipeline.yml → runs every day at 2AM UTC    │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture?
We chose a **serverless, modular architecture** because:
- Each component is independently deployable and testable
- Hopsworks acts as the central hub connecting all stages
- GitHub Actions eliminates the need for dedicated servers
- Streamlit Cloud provides free, public dashboard hosting
- The fallback system (local files if Hopsworks unavailable) ensures robustness

---

## 3. Data Collection & Feature Pipeline

### Why Open-Meteo API?
We evaluated several weather and AQI APIs:

| API | Cost | Key Required | Historical Data | AQI Data |
|---|---|---|---|---|
| OpenWeather | Paid after limit | Yes | Limited | Yes |
| AQICN | Free with limits | Yes | No | Yes |
| **Open-Meteo** | **Free** | **No** | **Yes (2+ years)** | **Yes** |
| IQAir | Paid | Yes | Yes | Yes |

**Open-Meteo was chosen** because it provides both historical archive data and future forecast data for free, with no API key required, making CI/CD automation seamless.

### Data Source
**Open-Meteo API** (free, no API key required)

### Data Range
- **Historical:** January 2024 — May 2026
- **Total records:** 874 daily observations
- **Frequency:** Hourly data aggregated to daily mean

### Why Hopsworks as Feature Store?
We considered several options for storing features:

- **MongoDB** (used by reference project): Good for raw data but lacks feature versioning, statistics tracking, and ML-specific capabilities
- **CSV files**: No versioning, no statistics, not scalable
- **Hopsworks**: Purpose-built for ML features, tracks statistics, enables feature reuse across models, has a free tier

Hopsworks gives us **feature lineage** (knowing exactly which features trained which model version) and **point-in-time correctness** (retrieving the feature values as they were at training time).

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
| AQI | US Air Quality Index | — |

### Data Cleaning Strategy
- **Missing temperature/humidity** → filled with column mean (stable variable, mean imputation appropriate)
- **Missing precipitation** → filled with 0 (no data = no rain assumption)
- **Hourly → daily aggregation** → mean for all variables except precipitation (sum)
- **Rows with NaN after feature engineering** → dropped (7 rows lost from rolling window initialization)

---

## 4. Exploratory Data Analysis

### Key Findings

**AQI Distribution:**
- Average AQI: ~120 (Unhealthy for Sensitive Groups range)
- Maximum AQI recorded: ~280 (Very Unhealthy)
- Minimum AQI recorded: ~45 (Good — rare)
- Most common category: Moderate to Unhealthy for Sensitive Groups
- Karachi never truly reaches "Good" AQI — even best days are Moderate

**Seasonal Patterns:**
- AQI is highest in **Winter (December–February)**: Temperature inversion traps pollutants near ground level. Cold air prevents vertical mixing, causing pollutant accumulation
- AQI drops significantly in **Monsoon (July–August)**: Rainfall physically washes particulate matter from the atmosphere
- **Spring and Autumn** show moderate AQI levels
- This seasonal pattern directly motivated the `month` feature in our model

**Weekly Patterns:**
- Weekend AQI is ~8% lower than weekday AQI on average
- This reflects reduced industrial activity, fewer commercial vehicles, and less construction on weekends
- This pattern motivated the `is_weekend` feature

**Strongest Correlations with AQI:**
1. PM2.5 (r = 0.95) — fine particles are the primary AQI driver
2. PM10  (r = 0.92) — coarse particles closely related to PM2.5
3. CO    (r = 0.78) — combustion indicator
4. SO2   (r = 0.71) — industrial emission indicator
5. Temperature (r = -0.42) — negative correlation: higher temperature means better atmospheric mixing, which dilutes pollutants

**Pollutant Trends:**
- PM2.5 and CO peaked in January 2025, coinciding with coldest temperatures
- NO2 showed weekly cyclicity consistent with weekday traffic patterns
- O3 (ozone) showed an inverse relationship with other pollutants — higher in summer due to photochemical reactions

---

## 5. Feature Engineering

### Why Feature Engineering Matters
Raw pollutant readings alone give us a snapshot of current conditions. But AQI tomorrow is not just about today's PM2.5 - it's about **where AQI has been trending** over the past week. Feature engineering captures this temporal momentum.

### Final Feature Set (15 features)

| Feature | Type | Why Included |
|---|---|---|
| pm2_5 | Pollutant | Primary AQI driver (r=0.95 with AQI) |
| pm10 | Pollutant | Secondary particle driver |
| no2 | Pollutant | Traffic and industrial indicator |
| so2 | Pollutant | Industrial emission indicator |
| co | Pollutant | Combustion activity indicator |
| o3 | Pollutant | Photochemical smog indicator |
| temperature | Weather | Atmospheric mixing capacity |
| humidity | Weather | Affects particle hygroscopic growth |
| precipitation | Weather | Natural pollutant washout mechanism |
| month | Temporal | Captures seasonal patterns |
| day_of_week | Temporal | Captures weekly industrial cycles |
| is_weekend | Temporal | Binary weekend effect indicator |
| aqi_change_rate | Derived | Day-over-day momentum signal |
| aqi_rolling_3d | Derived | Short-term trend (3-day average) |
| aqi_rolling_7d | Derived | Medium-term trend (7-day average) |

### Why Derived Features Made the Biggest Difference
Adding `aqi_change_rate`, `aqi_rolling_3d`, and `aqi_rolling_7d` improved R² from ~0.85 to **0.9545**. This is because:

1. **AQI has strong temporal autocorrelation** — today's AQI is the best predictor of tomorrow's AQI
2. **Rolling averages smooth noise** — a single unusually high day doesn't distort the 7-day picture
3. **Change rate signals direction** — a rising rate predicts continued deterioration; a falling rate predicts improvement
4. These features gave the model "memory" of recent air quality without requiring a full time-series architecture like LSTM

### Why We Didn't Add More Features
We deliberately kept 15 features rather than adding more:
- More features with 874 rows increases overfitting risk
- The 15 features already capture pollutants, weather, temporal cycles, and trend momentum
- Adding redundant features (e.g., both PM2.5 and a PM2.5 lag) would introduce multicollinearity

---

## 6. Model Training & Evaluation

### Why These Three Models?

We selected one model from each major ML paradigm:

**XGBoost (Gradient Boosting):**
- Handles non-linear interactions between pollutants
- Built-in L1/L2 regularization prevents overfitting
- Industry standard for tabular data competitions
- We chose XGBoost over sklearn's GradientBoosting because it's faster, has early stopping, and handles missing values natively

**Random Forest:**
- Ensemble of decision trees — reduces variance through bagging
- Robust to outliers (AQI spikes during pollution events)
- Provides natural feature importance ranking
- Requires less hyperparameter tuning than XGBoost

**Ridge Regression:**
- Linear model with L2 regularization
- Specifically designed to handle multicollinear features (PM2.5 and PM10 correlation = 0.95)
- Interpretable coefficients — each feature has a direct weight
- Very low variance — ideal for small datasets (874 rows)

### Model Results

| Model | MAE (Train) | MAE (Test) | RMSE (Train) | RMSE (Test) | R² (Train) | R² (Test) |
|---|---|---|---|---|---|---|
| XGBoost | 0.36 | 4.87 | 0.51 | 6.69 | 0.9995 | 0.9140 |
| Random Forest | 2.58 | 5.16 | 3.59 | 7.70 | 0.9756 | 0.8859 |
| **Ridge** | **3.06** | **3.73** | **4.24** | **4.86** | **0.9659** | **0.9545** |

### Training Configuration
- **Train/Test split:** 80% / 20% (699 train, 175 test)
- **Temporal ordering:** Data sorted by date before splitting — no future data leaking into training
- **Feature scaling:** StandardScaler — zero mean, unit variance for all features
- **Selection criterion:** Highest R² score on test set

---

## 7. Model Selection Justification

### Why Ridge Regression Won

**1. The Overfitting Problem with Tree Models**

XGBoost achieved a near-perfect R²=0.9995 on training data but only 0.9140 on test data, a gap of 0.0855. This is severe overfitting. The model memorized training patterns instead of learning generalizable relationships. With only 874 rows, XGBoost had too many degrees of freedom.

Random Forest showed the same pattern: R²=0.9756 train vs 0.8859 test with gap of 0.0897.

Ridge Regression showed: R²=0.9659 train vs 0.9545 test with gap of only **0.0114**. This tiny gap indicates genuine learning, not memorization.

**2. The Linearity of Rolling Features**

The three derived features (`aqi_rolling_7d`, `aqi_rolling_3d`, `aqi_change_rate`) have a **near-linear relationship** with next-day AQI. When tomorrow's air quality is largely a continuation of this week's trend, a linear model captures this directly. Tree-based models build complex decision boundaries that are unnecessary for smooth continuous trends.

**3. L2 Regularization Handles Multicollinearity**

PM2.5 and PM10 are correlated at r=0.95. The three rolling features are also highly correlated with each other. Ridge's L2 penalty shrinks the coefficients of correlated features together, producing stable estimates. Tree models handle this by randomly selecting splits, introducing variance.

**4. Small Dataset Favors Simpler Models**

With 874 observations and 15 features, the model complexity should be proportional to data size. Ridge's parameter space (15 coefficients + 1 alpha) is far smaller than XGBoost's (300 trees × multiple split parameters).

### What Else Could Have Been Used?

**LSTM (Long Short-Term Memory Neural Network):**
- Would capture multi-week temporal patterns better than rolling averages
- Not used because: requires 5,000+ samples for reliable training; our 874-row dataset would cause severe overfitting; our mentor confirmed deep learning is not required
- Ridge's rolling average features approximate LSTM's memory at a fraction of the cost

**Support Vector Regression (SVR):**
- Works well on small datasets; robust to outliers
- Not used because: SVR training scales as O(n²) to O(n³) - problematic as hourly CI/CD adds daily rows; requires careful kernel selection; Ridge achieved better R² with zero tuning

**Gradient Boosting (sklearn):**
- Similar to XGBoost conceptually
- Not used because: XGBoost already represents this model family with a superior implementation; adding sklearn's GradientBoosting would be redundant with identical weaknesses (overfitting on small data)

---

## 8. SHAP Explainability

### Why SHAP?
SHAP (SHapley Additive exPlanations) was chosen over alternatives because:
- **LIME**: Provides local explanations only (one prediction at a time); less stable
- **Feature importance (built-in)**: Only available for tree models; Ridge has coefficients instead; not model-agnostic
- **SHAP**: Works for any model type (LinearExplainer for Ridge, TreeExplainer for tree models); provides both global and local explanations; widely accepted in industry and academia

### Key SHAP Findings

**Most Influential Features (by mean |SHAP value|):**
1. **aqi_rolling_7d** — The 7-day rolling average is the single strongest predictor. High recent AQI → model predicts continued high AQI
2. **aqi_rolling_3d** — Confirms the short-term trend direction
3. **pm2_5** — Fine particulate matter drives acute AQI spikes beyond the baseline trend
4. **aqi_change_rate** — Rising rate signals deteriorating conditions; falling rate signals recovery
5. **co** — Carbon monoxide elevations indicate combustion events (traffic, burning)

**Key Insight:** The top 4 features are all derived temporal features, confirming that our feature engineering decision was the most impactful contribution of this project. Without these features, the model would have R²≈0.85 instead of 0.9545.

---

## 9. Web Application

### Why Streamlit?
We evaluated three frontend options:

- **Flask/FastAPI + HTML**: Full control but requires significant frontend development time
- **Gradio**: Simple but limited customization; primarily for model demos
- **Streamlit**: Pure Python, rich component library, free cloud deployment, native Plotly integration, ideal for data science dashboards

Streamlit allowed us to build a production-quality dashboard with custom CSS, interactive Plotly charts, and real-time Hopsworks integration in a fraction of the time of a traditional web stack.

### Dashboard Pages

#### 🏠 Overview
- Latest AQI metric card with category and health status
- 90-day AQI trend chart
- AQI category breakdown (donut + bar charts)
- Automatic hazardous alert banners (3 severity levels)
- AQI health guide (expandable)

#### 🔍 Data Analysis (13 charts)
- Dataset preview and statistics
- Correlation heatmap
- Feature distributions (PM2.5, Temperature, Humidity, AQI)
- Temperature vs PM2.5 dual-axis time series
- Monthly average AQI bar chart
- AQI trend with health threshold lines
- 7-day and 30-day rolling average trend
- Seasonal analysis (Winter/Spring/Monsoon/Autumn) — bar + box
- Weekend vs Weekday comparison — bar + box
- Pollutant trends over time (PM2.5, PM10, CO, NO2)

#### 🤖 Model Performance
- Best model metrics (MAE, RMSE, R²)
- All 3 models comparison table and bar charts
- R² accuracy gauge
- Feature importance / coefficient chart
- SHAP bar and beeswarm plots
- Model configuration table

#### 🔮 Predict AQI
- Manual input form (pollutants + weather + AQI trend inputs)
- Auto-computed derived features from inputs
- Instant AQI prediction with color-coded category badge
- Health advice based on predicted level
- Hazardous alert banners for dangerous predictions
- Prediction history trend chart and table (SQLite)

#### 📅 3-Day Forecast
- Live data from Open-Meteo forecast API
- Color-coded day cards with predicted AQI + weather
- Forecast trend chart with AQI zone backgrounds
- Weather conditions table
- Automatic alert if high AQI forecasted

### Alerts System
Automatic color-coded alert banners appear when AQI exceeds thresholds:
- AQI > 150 → 🟠 Orange **Unhealthy** notice
- AQI > 200 → 🟣 Purple **Very Unhealthy** warning
- AQI > 300 → 🔴 Red **Hazardous** emergency alert

---

## 10. Automated CI/CD Pipeline

### Why GitHub Actions?
We evaluated:
- **Apache Airflow**: Powerful but requires server; not serverless
- **Prefect**: Good MLOps tool but requires cloud account and complex setup
- **GitHub Actions**: Free for public repos, triggers on schedule, integrates directly with our repo, zero infrastructure management

### Workflows

#### Feature Pipeline (`feature_pipeline.yml`)
- **Schedule:** Every hour (`0 * * * *` cron)
- **Action:** Runs `fetch_karachi_data.py` → fetches latest Karachi AQI + weather from Open-Meteo → updates `karachi_daily_aqi_weather.csv` → commits back to repo
- **Why hourly?** Air quality can change significantly within hours; hourly updates keep the dashboard fresh

#### Training Pipeline (`training_pipeline.yml`)
- **Schedule:** Every day at 2:00 AM UTC (`0 2 * * *` cron)
- **Action:** Runs `train_pipeline.py` → reads updated dataset → retrains all 3 models → saves best model → commits updated pkl files
- **Why 2 AM?** Runs after midnight when GitHub Actions has less competition for runners; avoids peak usage hours

### Hopsworks Integration in CI/CD
The training pipeline uploads models to Hopsworks Model Registry, enabling:
- **Version tracking**: Every daily retrain is a tracked version
- **Rollback capability**: Can revert to previous model if new one performs worse
- **Deployment**: Streamlit app always pulls the latest registered model

---

## 11. Challenges & Solutions

### Challenge 1: Hopsworks Windows Compatibility
**Problem:** Hopsworks client library uses hardcoded `/tmp` directory paths (Linux convention). On Windows, this path doesn't exist, causing `[WinError 3] The system cannot find the path specified: '/tmp\eu-west.cloud.hopsworks.ai'` every time the app started locally.

**Solution:** Added environment variable overrides and automatic directory creation at the very top of `app.py` before any imports:
```python
os.makedirs("C:/tmp", exist_ok=True)
os.environ["TMPDIR"] = "C:/tmp"
os.environ["TEMP"]   = "C:/tmp"
os.environ["TMP"]    = "C:/tmp"
```
This runs before Hopsworks initializes and provides a valid temp directory on Windows.

---

### Challenge 2: Python 3.14 Breaking Streamlit Cloud Deployment
**Problem:** Streamlit Cloud automatically selected Python 3.14 (released weeks before our deployment). The `protobuf` package (v4.x, required by hopsworks) uses C extensions with metaclasses that Python 3.14 no longer supports, causing `TypeError: Metaclasses with custom tp_new are not supported` and crashing the entire app before it could start.

**Solution:** Added `.python-version` file and `runtime.txt` to force Python 3.11, and set Python version explicitly in Streamlit Cloud's Advanced Settings dropdown. Python 3.11 is fully compatible with protobuf 4.x and all our other dependencies.

---

### Challenge 3: Large Files Blocking GitHub Push
**Problem:** The `venv/` folder (containing `xgboost.dll` at 136MB) was accidentally committed to git history. GitHub rejected all pushes with `GH001: Large files detected. File venv/Lib/site-packages/xgboost/lib/xgboost.dll is 136.94 MB; this exceeds GitHub's file size limit of 100.00 MB`.

**Solution:** Used `git filter-repo` to surgically remove the `venv/` directory from all git history:
```bash
git filter-repo --path venv/ --invert-paths --force
```
Then added `venv/` to `.gitignore` to prevent future accidental commits.

---

### Challenge 4: Duplicate Training Steps Overwriting Best Model
**Problem:** The original Colab pipeline had both a single XGBoost training block (Steps 16–18) AND the multi-model comparison block (Steps 17–22). The loop `for name, model in models.items()` overwrote the `model` variable with the last iteration result (Ridge). Then Steps 23–27 re-saved this `model` variable, silently overwriting the best model pkl with Ridge regardless of which model actually won.

**Solution:** Removed the redundant single-model training block entirely. Changed the loop variable to `for name, clf in all_models.items()` to avoid variable name collision. Used `best_model = trained_models[best_name]` to explicitly capture the winning model before saving.

---

### Challenge 5: SQLite Schema Mismatch After Adding Features
**Problem:** When we added 3 new derived features (`aqi_change_rate`, `aqi_rolling_3d`, `aqi_rolling_7d`), the `save_prediction()` function now passed 12 inputs instead of the 9 it was designed for. The existing SQLite database had a 12-column schema but received 15 values, causing `sqlite3.OperationalError: table predictions has 12 columns but 15 values were supplied`.

**Solution:** Updated the `CREATE TABLE` statement to include all new columns and updated the INSERT placeholder count from 11 `?` marks to 15. Also deleted the existing database file to force schema recreation with the new structure.

---

### Challenge 6: Streamlit Cloud SQLite Write Permission
**Problem:** On Streamlit Cloud, the app directory (`/mount/src/...`) is read-only. Attempting to create `aqi_predictions.db` in the same directory as `app.py` caused `sqlite3.OperationalError` when users clicked the Predict button.

**Solution:** Changed the database path to use the user's home directory, which is always writable on both local and cloud environments:
```python
DB_PATH = os.path.join(os.path.expanduser("~"), "aqi_predictions.db")
```
On local Windows this resolves to `C:\Users\Laiba Mir\`, on Streamlit Cloud to `/home/adminuser/`.

---

## 12. Results Summary

### What Was Achieved

| Requirement | Implementation | Status |
|---|---|---|
| Fetch data from external API | Open-Meteo API (weather + AQI) | ✅ |
| Compute features from raw data | 15 features (pollutants + weather + temporal + derived) | ✅ |
| Store features in Feature Store | Hopsworks Feature Store (aqi_features v1) | ✅ |
| Backfill historical data | 874 days (Jan 2024 – May 2026) | ✅ |
| Train multiple ML models | XGBoost, Random Forest, Ridge Regression | ✅ |
| Evaluate with RMSE, MAE, R² | All 3 metrics for all 3 models | ✅ |
| Store model in Model Registry | Hopsworks Model Registry (xgboost_karachi_aqi v1) | ✅ |
| Automate feature pipeline hourly | GitHub Actions feature_pipeline.yml | ✅ |
| Automate training pipeline daily | GitHub Actions training_pipeline.yml | ✅ |
| Interactive Streamlit dashboard | 5 pages, 13 EDA charts, live forecast | ✅ |
| SHAP feature importance | LinearExplainer for Ridge, bar + beeswarm plots | ✅ |
| Hazardous AQI alerts | 3-level system (Unhealthy/Very Unhealthy/Hazardous) | ✅ |
| EDA to identify trends | 13 interactive charts across seasonal, weekly, temporal | ✅ |
| Detailed report | This document | ✅ |
| Deployed publicly | Streamlit Cloud | ✅ |

### Model Performance (Best Model)
- **Algorithm:** Ridge Regression
- **R² Score:** 0.9545 (95.45% of AQI variance explained)
- **MAE:** 3.73 AQI units (average prediction error)
- **RMSE:** 4.86 AQI units

### Key Contributions
1. **Feature engineering was the highest-impact decision** — adding 3 derived AQI trend features improved R² by ~0.10
2. **Model selection was data-driven** — all 3 models trained and compared automatically; Ridge selected by R² score
3. **Full MLOps stack** — Hopsworks for feature store and model registry, GitHub Actions for automation, Streamlit Cloud for deployment
4. **Explainability** — SHAP analysis confirmed that temporal momentum features (rolling averages) are the strongest predictors

---

## 13. Future Improvements

### Short-Term
1. **Retrain on scikit-learn 1.8.0** — Current model was trained on 1.6.1; retraining eliminates `InconsistentVersionWarning` and ensures optimal performance with latest sklearn
2. **Add more cities** — Extend the pipeline to Lahore, Islamabad, Peshawar with minimal code changes (parameterize coordinates)
3. **Email/SMS alerts** — Notify users via email when forecasted AQI exceeds dangerous thresholds

### Medium-Term
4. **LSTM model** — With more data (2+ years of hourly readings), an LSTM would capture multi-week seasonal patterns more accurately than rolling averages
5. **7-day forecast** — Extend from 3-day to 7-day forecast using Open-Meteo's extended forecast endpoint
6. **Model drift monitoring** — Track prediction accuracy over time; trigger retraining when error exceeds threshold
7. **Persistent prediction history** — Store SQLite predictions in Hopsworks instead of local file so history persists across Streamlit Cloud restarts

### Long-Term
8. **Traffic and industrial data integration** — Add vehicle count and industrial activity data as features
9. **Hyperparameter optimization** — Use Optuna for automated hyperparameter tuning during daily retraining
10. **Multi-pollutant forecasting** — Predict PM2.5, PM10, CO individually in addition to composite AQI

---

*Report prepared for 10Pearls Shine Internship Data Science Project*
*Karachi AQI Predictor*
*Author: Laiba Mir | June 2026*
