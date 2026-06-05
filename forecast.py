"""
forecast.py
Fetches next 3 days of weather and air quality forecast
from Open-Meteo and generates AQI predictions using
the trained model.
"""

import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# ── Karachi coordinates ───────────────────────────────────────────
LAT = 24.8607
LON = 67.0011

def fetch_forecast():
    """Fetch 3-day forecast from Open-Meteo APIs"""

    today     = datetime.now().date()
    end_date  = today + timedelta(days=3)

    # ── Weather forecast ──────────────────────────────────────────
    wx_url = "https://api.open-meteo.com/v1/forecast"
    wx_params = {
        "latitude":   LAT,
        "longitude":  LON,
        "daily":      "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
        "start_date": str(today),
        "end_date":   str(end_date),
        "timezone":   "Asia/Karachi"
    }

    wx_resp = requests.get(wx_url, params=wx_params, timeout=60)
    wx_resp.raise_for_status()
    wx_data = wx_resp.json()

    wx_df = pd.DataFrame({
        "date":          wx_data["daily"]["time"],
        "temperature":   wx_data["daily"]["temperature_2m_mean"],
        "humidity":      wx_data["daily"]["relative_humidity_2m_mean"],
        "precipitation": wx_data["daily"]["precipitation_sum"],
    })

    # ── Air quality forecast ──────────────────────────────────────
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude":  LAT,
        "longitude": LON,
        "hourly":    "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "start_date": str(today),
        "end_date":   str(end_date),
        "timezone":  "Asia/Karachi"
    }

    aqi_resp = requests.get(aqi_url, params=aqi_params, timeout=60)
    aqi_resp.raise_for_status()
    aqi_data = aqi_resp.json()

    aqi_hourly = pd.DataFrame({
        "datetime": aqi_data["hourly"]["time"],
        "pm2_5":    aqi_data["hourly"]["pm2_5"],
        "pm10":     aqi_data["hourly"]["pm10"],
        "no2":      aqi_data["hourly"]["nitrogen_dioxide"],
        "so2":      aqi_data["hourly"]["sulphur_dioxide"],
        "co":       aqi_data["hourly"]["carbon_monoxide"],
        "o3":       aqi_data["hourly"]["ozone"],
    })

    aqi_hourly["datetime"] = pd.to_datetime(aqi_hourly["datetime"])
    aqi_hourly["date"]     = aqi_hourly["datetime"].dt.date.astype(str)

    aqi_daily = aqi_hourly.groupby("date").mean(numeric_only=True).reset_index()

    # ── Merge weather + air quality ───────────────────────────────
    df = pd.merge(wx_df, aqi_daily, on="date", how="inner")
    df["date"] = pd.to_datetime(df["date"])

    # Fill any nulls
    df = df.fillna(df.median(numeric_only=True))

    return df


def generate_forecast():
    """Load model and generate 3-day AQI predictions"""

    print("🌍 Fetching 3-day forecast data from Open-Meteo...")
    df = fetch_forecast()

    print(f"✅ Fetched {len(df)} days of forecast data")

    # ── Load model + scaler ───────────────────────────────────────
    model  = joblib.load("models/xgboost_karachi.pkl")
    scaler = joblib.load("models/scaler_karachi.pkl")

    # ── Load recent historical data for rolling features ──────────
    hist = pd.read_csv("karachi_daily_aqi_weather.csv")
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").tail(7)

    recent_aqi     = hist["AQI"].values if "AQI" in hist.columns else hist["aqi"].values
    last_aqi       = float(recent_aqi[-1])
    prev_aqi       = float(recent_aqi[-2])
    rolling_7d_avg = float(np.mean(recent_aqi[-7:]))
    rolling_3d_avg = float(np.mean(recent_aqi[-3:]))

    # ── Build features for each forecast day ─────────────────────
    predictions = []

    for i, row in df.iterrows():
        month      = row["date"].month
        dow        = row["date"].dayofweek
        is_weekend = 1 if dow >= 5 else 0

        # Rolling features — update as we predict forward
        aqi_change_rate = last_aqi - prev_aqi
        aqi_roll_3d     = rolling_3d_avg
        aqi_roll_7d     = rolling_7d_avg

        features = np.array([[
            row["pm2_5"],
            row["pm10"],
            row["no2"],
            row["so2"],
            row["co"],
            row["o3"],
            row["temperature"],
            row["humidity"],
            row["precipitation"],
            month,
            dow,
            is_weekend,
            aqi_change_rate,
            aqi_roll_3d,
            aqi_roll_7d
        ]])

        scaled     = scaler.transform(features)
        predicted  = float(model.predict(scaled)[0])
        predicted  = max(0, predicted)   # no negative AQI

        # Get category
        if predicted <= 50:   cat = "Good"
        elif predicted <= 100: cat = "Moderate"
        elif predicted <= 150: cat = "Unhealthy for Sensitive Groups"
        elif predicted <= 200: cat = "Unhealthy"
        elif predicted <= 300: cat = "Very Unhealthy"
        else:                  cat = "Hazardous"

        predictions.append({
            "date":          row["date"].strftime("%Y-%m-%d"),
            "day":           row["date"].strftime("%A"),
            "predicted_aqi": round(predicted, 1),
            "category":      cat,
            "temperature":   round(row["temperature"], 1),
            "humidity":      round(row["humidity"],    1),
            "precipitation": round(row["precipitation"], 1),
            "pm2_5":         round(row["pm2_5"], 2),
            "pm10":          round(row["pm10"],  2),
        })

        # Update rolling values for next day
        prev_aqi       = last_aqi
        last_aqi       = predicted
        rolling_3d_avg = (rolling_3d_avg * 2 + predicted) / 3
        rolling_7d_avg = (rolling_7d_avg * 6 + predicted) / 7

    forecast_df = pd.DataFrame(predictions)
    print("\n✅ 3-Day Forecast Generated:")
    print(forecast_df[["date", "day", "predicted_aqi", "category"]].to_string(index=False))

    return forecast_df


if __name__ == "__main__":
    df = generate_forecast()
