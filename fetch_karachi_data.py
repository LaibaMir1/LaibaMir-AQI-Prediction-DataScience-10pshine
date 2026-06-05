"""
Fetch Karachi AQI Weather data from Open-Meteo API
Covers: 2024-01-01 to 2026-05-31
Saves to: karachi_daily_aqi_weather.csv
"""

import requests
import pandas as pd
from datetime import datetime

# ── Karachi coordinates ───────────────────────────────────────────
LAT       = 24.8607
LON       = 67.0011
START     = "2024-01-01"
END       = "2026-05-31"

print("🌍 Fetching Karachi AQI + Weather from Open-Meteo...")

# ═══════════════════════════════════════════════════════
# STEP 1 — Fetch Air Quality (hourly → aggregate to daily)
# ═══════════════════════════════════════════════════════

aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

aqi_params = {
    "latitude":    LAT,
    "longitude":   LON,
    "hourly":      "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,us_aqi",
    "start_date":  START,
    "end_date":    END,
    "timezone":    "Asia/Karachi"
}

print("  → Fetching air quality data...")
aqi_resp = requests.get(aqi_url, params=aqi_params, timeout=60)
aqi_resp.raise_for_status()
aqi_json = aqi_resp.json()

aqi_df = pd.DataFrame({
    "datetime":            aqi_json["hourly"]["time"],
    "PM2.5":               aqi_json["hourly"]["pm2_5"],
    "PM10":                aqi_json["hourly"]["pm10"],
    "NO2":                 aqi_json["hourly"]["nitrogen_dioxide"],
    "SO2":                 aqi_json["hourly"]["sulphur_dioxide"],
    "CO":                  aqi_json["hourly"]["carbon_monoxide"],
    "O3":                  aqi_json["hourly"]["ozone"],
    "AQI":                 aqi_json["hourly"]["us_aqi"],
})

aqi_df["datetime"] = pd.to_datetime(aqi_df["datetime"])
aqi_df["date"]     = aqi_df["datetime"].dt.date

# Aggregate hourly → daily mean
aqi_daily = aqi_df.groupby("date").agg({
    "PM2.5": "mean",
    "PM10":  "mean",
    "NO2":   "mean",
    "SO2":   "mean",
    "CO":    "mean",
    "O3":    "mean",
    "AQI":   "mean",
}).reset_index()

print(f"  ✅ Air quality: {len(aqi_daily)} days fetched")

# ═══════════════════════════════════════════════════════
# STEP 2 — Fetch Weather (hourly → aggregate to daily)
# ═══════════════════════════════════════════════════════

weather_url = "https://archive-api.open-meteo.com/v1/archive"

weather_params = {
    "latitude":   LAT,
    "longitude":  LON,
    "hourly":     "temperature_2m,relative_humidity_2m,precipitation",
    "start_date": START,
    "end_date":   END,
    "timezone":   "Asia/Karachi"
}

print("  → Fetching weather data...")
wx_resp = requests.get(weather_url, params=weather_params, timeout=60)
wx_resp.raise_for_status()
wx_json = wx_resp.json()

wx_df = pd.DataFrame({
    "datetime":    wx_json["hourly"]["time"],
    "Temperature": wx_json["hourly"]["temperature_2m"],
    "Humidity":    wx_json["hourly"]["relative_humidity_2m"],
    "Precipitation": wx_json["hourly"]["precipitation"],
})

wx_df["datetime"] = pd.to_datetime(wx_df["datetime"])
wx_df["date"]     = wx_df["datetime"].dt.date

wx_daily = wx_df.groupby("date").agg({
    "Temperature":   "mean",
    "Humidity":      "mean",
    "Precipitation": "sum",   # total daily rainfall
}).reset_index()

print(f"  ✅ Weather: {len(wx_daily)} days fetched")

# ═══════════════════════════════════════════════════════
# STEP 3 — Merge AQI + Weather
# ═══════════════════════════════════════════════════════

df = pd.merge(aqi_daily, wx_daily, on="date", how="inner")
df["date"] = pd.to_datetime(df["date"])

# ═══════════════════════════════════════════════════════
# STEP 4 — Clean up
# ═══════════════════════════════════════════════════════

# Fill missing values
df["Temperature"]  = df["Temperature"].fillna(df["Temperature"].median())
df["Humidity"]     = df["Humidity"].fillna(df["Humidity"].median())
df["Precipitation"]= df["Precipitation"].fillna(0)
df["AQI"]          = df["AQI"].fillna(df["AQI"].median())
df["PM2.5"]        = df["PM2.5"].fillna(df["PM2.5"].median())
df["PM10"]         = df["PM10"].fillna(df["PM10"].median())
df["NO2"]          = df["NO2"].fillna(df["NO2"].median())
df["SO2"]          = df["SO2"].fillna(df["SO2"].median())
df["CO"]           = df["CO"].fillna(df["CO"].median())
df["O3"]           = df["O3"].fillna(df["O3"].median())

# Add Next_Day_AQI target column (for model training)
df = df.sort_values("date").reset_index(drop=True)
df["Next_Day_AQI"] = df["AQI"].shift(-1)
df = df.dropna(subset=["Next_Day_AQI"])

# ═══════════════════════════════════════════════════════
# STEP 5 — Save
# ═══════════════════════════════════════════════════════

df.to_csv("karachi_daily_aqi_weather.csv", index=False)

print(f"\n✅ Dataset saved → karachi_daily_aqi_weather.csv")
print(f"   Rows  : {len(df)}")
print(f"   From  : {df['date'].min().date()}")
print(f"   To    : {df['date'].max().date()}")
print(f"\nColumns:")
print(df.columns.tolist())
print(f"\nSample:")
print(df.head(3))
