# 🍃 AQI Predictor Karachi Air Quality Dashboard

An end-to-end Air Quality Index (AQI) prediction system for **Karachi, Pakistan** built with a fully serverless  stack.

---

## 🌐 Live Dashboard
👉 **[View Live App on Streamlit Cloud](https://laibamir-aqi-prediction-datascience-10pshine-ecdguvxp6xmnqe2uv.streamlit.app/)**

---

## 📌 What It Does
- Fetches real-time weather and pollutant data from Open-Meteo API
- Stores features in Hopsworks Feature Store
- Trains and compares 3 ML models — best one selected automatically
- Forecasts AQI for the next 3 days using live weather data
- Displays everything on an interactive 5-page Streamlit dashboard
- Fully automated via GitHub Actions (hourly data fetch + daily retraining)

---

## 🗂️ Project Structure

```
AQI-prediction-datascience-10pshine/
├── app.py                        # Streamlit dashboard (5 pages)
├── colab_pipeline.py             # Full training pipeline (run in Colab)
├── fetch_karachi_data.py         # Fetch data from Open-Meteo API
├── forecast.py                   # 3-day AQI forecast generator
├── train_pipeline.py             # Automated training for CI/CD
├── karachi_daily_aqi_weather.csv # Raw dataset
├── karachi_aqi_cleaned.csv       # Cleaned + engineered dataset
├── models/
│   ├── xgboost_karachi.pkl       # Best trained model (Ridge)
│   ├── scaler_karachi.pkl        # StandardScaler
│   ├── model_comparison.csv      # All 3 models metrics
│   ├── shap_summary_bar.png      # SHAP feature importance
│   └── shap_beeswarm.png         # SHAP beeswarm plot
├── .github/workflows/
│   ├── feature_pipeline.yml      # Runs hourly — fetches data
│   └── training_pipeline.yml     # Runs daily — retrains models
├── requirements.txt
└── REPORT.md                     # Full project report
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Data Source | Open-Meteo API |
| Feature Store | Hopsworks |
| Model Registry | Hopsworks |
| ML Models | XGBoost, Random Forest, Ridge Regression |
| Best Model | Ridge Regression (R² = 0.9545) |
| Explainability | SHAP |
| Dashboard | Streamlit + Plotly |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/LaibaMir1/LaibaMir-AQI-Prediction-DataScience-10pshine.git
cd LaibaMir-AQI-Prediction-DataScience-10pshine
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add secrets**

Create `.streamlit/secrets.toml`:
```toml
HOPSWORKS_API_KEY = "your_api_key"
HOPSWORKS_HOST    = "eu-west.cloud.hopsworks.ai"
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📊 Model Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | 4.87 | 6.69 | 0.914 |
| Random Forest | 5.16 | 7.70 | 0.886 |
| **Ridge Regression** ✅ | **3.73** | **4.86** | **0.954** |

---

## 📁 Data
- **Source:** Open-Meteo API (free, no key required)
- **Location:** Karachi, Pakistan (24.8607°N, 67.0011°E)
- **Range:** January 2024 — May 2026
- **Records:** 874 daily observations
- **Features:** 15 (pollutants + weather + temporal + derived AQI trends)

---

## 🔄 Automation
- **Every hour** → `feature_pipeline.yml` fetches latest data from Open-Meteo
- **Every day at 2AM UTC** → `training_pipeline.yml` retrains all 3 models and saves the best one

---

*10Pearls Data Science Project | Karachi AQI Predictor | 2026*
