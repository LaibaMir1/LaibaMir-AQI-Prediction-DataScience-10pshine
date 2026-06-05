# =========================================================
# STEP 1 — INSTALL LIBRARIES (RUN ONLY IN GOOGLE COLAB)
# =========================================================

# !pip install hopsworks==4.7.*
# !pip install xgboost
# !pip install shap
# !pip install confluent-kafka

# =========================================================
# STEP 2 — IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import os
import joblib
import hopsworks
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# =========================================================
# STEP 3 — LOAD DATASET
# =========================================================

df = pd.read_csv("karachi_daily_aqi_weather.csv")

print("✅ Dataset Loaded Successfully")
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 Rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# =========================================================
# STEP 4 — DATA CLEANING
# =========================================================

print("\n✅ Starting Data Cleaning...")

df['Temperature']   = df['Temperature'].fillna(df['Temperature'].mean())
df['Humidity']      = df['Humidity'].fillna(df['Humidity'].mean())
df['Precipitation'] = df['Precipitation'].fillna(0)

print("✅ Missing Values Handled")

# =========================================================
# STEP 5 — FEATURE ENGINEERING
# =========================================================

print("\n✅ Creating Features...")

df['date']        = pd.to_datetime(df['date'])
df               = df.sort_values('date').reset_index(drop=True)

# Time-based features
df['month']       = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)

# Derived AQI features
df['aqi_change_rate'] = df['AQI'].diff()
df['aqi_rolling_3d']  = df['AQI'].rolling(window=3).mean()
df['aqi_rolling_7d']  = df['AQI'].rolling(window=7).mean()

# Next-day AQI as target
df['Next_Day_AQI'] = df['AQI'].shift(-1)

# Drop rows with NaN (from diff/rolling/shift)
df = df.dropna().reset_index(drop=True)

print(f"✅ Feature Engineering Completed")
print(f"   Rows after dropna: {len(df)}")
print(f"   New features: aqi_change_rate, aqi_rolling_3d, aqi_rolling_7d")

# =========================================================
# STEP 6 — RENAME COLUMNS FOR HOPSWORKS COMPATIBILITY
# =========================================================

print("\n✅ Renaming Columns...")

df.columns = [
    col.lower().replace('.', '_') if col != 'date' else col
    for col in df.columns
]

print("Updated Columns:", df.columns.tolist())

# =========================================================
# STEP 7 — SAVE CLEANED DATASET
# =========================================================

df.to_csv("karachi_aqi_cleaned.csv", index=False)
print("\n✅ Cleaned Dataset Saved")

# =========================================================
# STEP 8 — CONNECT TO HOPSWORKS
# =========================================================

print("\n✅ Connecting to Hopsworks...")

project = hopsworks.login()
fs      = project.get_feature_store()

print("✅ Connected to Hopsworks")

# =========================================================
# STEP 9 — DELETE OLD FEATURE GROUP (OPTIONAL)
# =========================================================

print("\n✅ Checking Existing Feature Group...")

try:
    old_fg = fs.get_feature_group(name="aqi_features", version=1)
    print("⚠ Existing Feature Group Found — Deleting...")
    old_fg.delete()
    print("✅ Old Feature Group Deleted")
except:
    print("No Existing Feature Group Found")

# =========================================================
# STEP 10 — CREATE NEW FEATURE GROUP
# =========================================================

print("\n✅ Creating Feature Group...")

feature_group = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    description="Karachi AQI and weather features with derived AQI trend features",
    primary_key=["date"],
    event_time="date"
)

print("✅ Feature Group Created")

# =========================================================
# STEP 11 — INSERT DATA INTO FEATURE STORE
# =========================================================

print("\n✅ Uploading Data to Feature Store...")

feature_group.insert(df, write_options={"wait_for_job": True})

print("✅ Data Uploaded Successfully")

# =========================================================
# STEP 12 — READ DATA FROM FEATURE STORE
# =========================================================

print("\n✅ Reading Data from Feature Store...")

feature_df = feature_group.select_all().read()

print("✅ Data Loaded — Shape:", feature_df.shape)

# =========================================================
# STEP 13 — SELECT FEATURES + TARGET
# =========================================================

features = [
    'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3',
    'temperature', 'humidity', 'precipitation',
    'month', 'day_of_week', 'is_weekend',
    'aqi_change_rate', 'aqi_rolling_3d', 'aqi_rolling_7d'
]

target = 'aqi'

X = feature_df[features]
y = feature_df[target]

print("\n✅ Features and Target Selected")
print(f"   Total features : {len(features)}")
print(f"   Features       : {features}")

# =========================================================
# STEP 14 — TRAIN TEST SPLIT
# =========================================================

print("\n✅ Splitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# =========================================================
# STEP 15 — FEATURE SCALING
# =========================================================

print("\n✅ Scaling Features...")

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("✅ Scaling Completed")

# =========================================================
# STEP 16 — TRAIN MULTIPLE MODELS
# =========================================================

print("\n✅ Training Multiple Models...")

all_models = {
    "XGBoost": XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='rmse'
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    ),
    "Ridge": Ridge(alpha=1.0)
}

# =========================================================
# STEP 17 — EVALUATE ALL MODELS
# =========================================================

results        = []
trained_models = {}

for name, clf in all_models.items():
    print(f"\n  Training {name}...")
    clf.fit(X_train_scaled, y_train)

    train_preds = clf.predict(X_train_scaled)
    test_preds  = clf.predict(X_test_scaled)

    result = {
        "Model":      name,
        "MAE_train":  round(mean_absolute_error(y_train, train_preds), 4),
        "MAE_test":   round(mean_absolute_error(y_test,  test_preds),  4),
        "RMSE_train": round(np.sqrt(mean_squared_error(y_train, train_preds)), 4),
        "RMSE_test":  round(np.sqrt(mean_squared_error(y_test,  test_preds)),  4),
        "R2_train":   round(r2_score(y_train, train_preds), 4),
        "R2_test":    round(r2_score(y_test,  test_preds),  4),
    }

    results.append(result)
    trained_models[name] = clf

    print(f"    MAE  : {result['MAE_test']}")
    print(f"    RMSE : {result['RMSE_test']}")
    print(f"    R²   : {result['R2_test']}")

# =========================================================
# STEP 18 — COMPARE + PICK BEST MODEL
# =========================================================

results_df = pd.DataFrame(results)

print("\n=============================")
print("MODEL COMPARISON")
print("=============================")
print(results_df.to_string(index=False))

best_row   = results_df.loc[results_df["R2_test"].idxmax()]
best_name  = best_row["Model"]
best_model = trained_models[best_name]

print(f"\n🏆 Best Model  : {best_name}")
print(f"   R²          : {best_row['R2_test']}")
print(f"   MAE         : {best_row['MAE_test']}")
print(f"   RMSE        : {best_row['RMSE_test']}")

# =========================================================
# STEP 19 — SAVE ALL FILES
# =========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/xgboost_karachi.pkl")
joblib.dump(scaler,     "models/scaler_karachi.pkl")
joblib.dump(features,   "models/feature_names.pkl")

results_df.to_csv("models/model_comparison.csv", index=False)
joblib.dump(results_df, "models/model_comparison.pkl")

print("\n✅ Saved Files:")
print(f"  - models/xgboost_karachi.pkl  ← best model: {best_name}")
print("  - models/scaler_karachi.pkl")
print("  - models/feature_names.pkl")
print("  - models/model_comparison.csv")
print("  - models/model_comparison.pkl")

# =========================================================
# STEP 20 — TEST SAVED MODEL
# =========================================================

print("\n✅ Testing Saved Model...")

loaded_model  = joblib.load("models/xgboost_karachi.pkl")
loaded_scaler = joblib.load("models/scaler_karachi.pkl")

sample        = X_test.iloc[:1]
sample_scaled = loaded_scaler.transform(sample)
prediction    = loaded_model.predict(sample_scaled)

print(f"Sample Prediction: {prediction[0]:.2f}")
print("\n✅ PIPELINE COMPLETE")

# =========================================================
# STEP 21 — UPLOAD TO HOPSWORKS MODEL REGISTRY
# =========================================================

print("\n✅ Uploading to Hopsworks Model Registry...")

mr = project.get_model_registry()

# Delete old version if exists
try:
    old = mr.get_model("xgboost_karachi_aqi", version=1)
    old.delete()
    print("⚠ Old model version deleted")
except:
    pass

karachi_model = mr.python.create_model(
    name="xgboost_karachi_aqi",
    version=1,
    metrics={
        "mae":  float(best_row["MAE_test"]),
        "rmse": float(best_row["RMSE_test"]),
        "r2":   float(best_row["R2_test"])
    },
    description=f"Best model: {best_name} — Karachi AQI prediction"
)

karachi_model.save("models/")

print(f"\n✅ Uploaded to Hopsworks Registry")
print(f"   Best Model : {best_name}")
print(f"   Files uploaded:")
for f in os.listdir("models/"):
    print(f"   - {f}")

# =========================================================
# STEP 22 — SHAP EXPLAINABILITY
# =========================================================

print("\n✅ Computing SHAP Values...")

if best_name == "Ridge":
    explainer   = shap.LinearExplainer(best_model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
elif best_name in ["RandomForest", "XGBoost"]:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled)

print("✅ SHAP Values Computed")

# Plot 1: Summary Bar Plot
shap.summary_plot(
    shap_values, X_test_scaled,
    feature_names=features,
    plot_type="bar", show=False
)
plt.title("SHAP Feature Importance — Mean |SHAP Value|")
plt.tight_layout()
plt.savefig("models/shap_summary_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: shap_summary_bar.png")

# Plot 2: Beeswarm Plot
shap.summary_plot(
    shap_values, X_test_scaled,
    feature_names=features, show=False
)
plt.title("SHAP Beeswarm — Feature Impact on AQI Prediction")
plt.tight_layout()
plt.savefig("models/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: shap_beeswarm.png")

# Save SHAP values
np.save("models/shap_values.npy", shap_values)

print("\n✅ SHAP Analysis Complete")
print("   - models/shap_summary_bar.png")
print("   - models/shap_beeswarm.png")
print("   - models/shap_values.npy")

print("\n" + "="*50)
print("✅ COMPLETE PIPELINE EXECUTED SUCCESSFULLY")
print("="*50)
print(f"\n🏆 Best Model : {best_name}")
print(f"   R²         : {best_row['R2_test']}")
print(f"   MAE        : {best_row['MAE_test']}")
print(f"   RMSE       : {best_row['RMSE_test']}")
print("\n📦 Download from models/ folder:")
print("   xgboost_karachi.pkl")
print("   scaler_karachi.pkl")
print("   feature_names.pkl")
print("   model_comparison.csv")
print("   shap_summary_bar.png")
print("   shap_beeswarm.png")
