import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

print("=" * 50)
print("AQI PREDICTOR — AUTOMATED TRAINING PIPELINE")
print("=" * 50)

# ── Load dataset ──────────────────────────────────────────────────
print("\n✅ Loading dataset...")
df = pd.read_csv("karachi_daily_aqi_weather.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"   Rows: {len(df)}")
print(f"   From: {df['date'].min().date()} → {df['date'].max().date()}")

# ── Clean ─────────────────────────────────────────────────────────
print("\n✅ Cleaning data...")
df["Temperature"]   = df["Temperature"].fillna(df["Temperature"].mean())
df["Humidity"]      = df["Humidity"].fillna(df["Humidity"].mean())
df["Precipitation"] = df["Precipitation"].fillna(0)

# ── Feature engineering ───────────────────────────────────────────
print("\n✅ Engineering features...")
df["month"]           = df["date"].dt.month
df["day_of_week"]     = df["date"].dt.dayofweek
df["is_weekend"]      = df["day_of_week"].isin([5, 6]).astype(int)
df["aqi_change_rate"] = df["AQI"].diff()
df["aqi_rolling_3d"]  = df["AQI"].rolling(window=3).mean()
df["aqi_rolling_7d"]  = df["AQI"].rolling(window=7).mean()
df["Next_Day_AQI"]    = df["AQI"].shift(-1)
df = df.dropna().reset_index(drop=True)

# Rename columns
df.columns = [c.lower().replace(".", "_") for c in df.columns]

print(f"   Rows after feature engineering: {len(df)}")

# ── Features + target ─────────────────────────────────────────────
features = [
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "precipitation",
    "month", "day_of_week", "is_weekend",
    "aqi_change_rate", "aqi_rolling_3d", "aqi_rolling_7d"
]

X = df[features]
y = df["aqi"]

# ── Split + scale ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train all models ──────────────────────────────────────────────
print("\n✅ Training models...")

all_models = {
    "XGBoost": XGBRegressor(
        objective="reg:squarederror", n_estimators=300,
        learning_rate=0.05, max_depth=6,
        random_state=42, eval_metric="rmse"
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=8,
        min_samples_split=10, random_state=42, n_jobs=-1
    ),
    "Ridge": Ridge(alpha=1.0)
}

results        = []
trained_models = {}

for name, clf in all_models.items():
    clf.fit(X_train_scaled, y_train)
    test_preds = clf.predict(X_test_scaled)

    result = {
        "Model":      name,
        "MAE_train":  round(mean_absolute_error(y_train, clf.predict(X_train_scaled)), 4),
        "MAE_test":   round(mean_absolute_error(y_test,  test_preds), 4),
        "RMSE_train": round(np.sqrt(mean_squared_error(y_train, clf.predict(X_train_scaled))), 4),
        "RMSE_test":  round(np.sqrt(mean_squared_error(y_test,  test_preds)), 4),
        "R2_train":   round(r2_score(y_train, clf.predict(X_train_scaled)), 4),
        "R2_test":    round(r2_score(y_test,  test_preds), 4),
    }
    results.append(result)
    trained_models[name] = clf
    print(f"   {name}: R²={result['R2_test']} MAE={result['MAE_test']}")

# ── Pick best model ───────────────────────────────────────────────
results_df = pd.DataFrame(results)
best_row   = results_df.loc[results_df["R2_test"].idxmax()]
best_name  = best_row["Model"]
best_model = trained_models[best_name]

print(f"\n🏆 Best Model: {best_name} (R²={best_row['R2_test']})")

# ── Save ──────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/xgboost_karachi.pkl")
joblib.dump(scaler,     "models/scaler_karachi.pkl")
joblib.dump(features,   "models/feature_names.pkl")
results_df.to_csv("models/model_comparison.csv", index=False)

print("\n✅ Models saved successfully")
print(f"   Best model : {best_name}")
print(f"   R²         : {best_row['R2_test']}")
print(f"   MAE        : {best_row['MAE_test']}")
print(f"   RMSE       : {best_row['RMSE_test']}")
print("\n✅ TRAINING PIPELINE COMPLETE")
