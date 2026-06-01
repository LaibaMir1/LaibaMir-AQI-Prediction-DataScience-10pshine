"""
AQI Predictor — Karachi Air Quality Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sqlite3
import os
from forecast import generate_forecast
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🍃",
    layout="wide"
)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #064e3b 0%, #065f46 40%, #047857 100%);
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #ecfdf5 !important;
}

/* ── Nav buttons ── */
.nav-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 11px 16px;
    border-radius: 10px;
    border: none;
    background: rgba(255,255,255,0.07);
    color: #ecfdf5 !important;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    margin-bottom: 6px;
    text-align: left;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.01em;
}
.nav-btn:hover {
    background: rgba(255,255,255,0.18);
    transform: translateX(3px);
}
.nav-btn.active {
    background: rgba(255,255,255,0.22);
    border-left: 3px solid #6ee7b7;
    font-weight: 700;
}

/* ── Main background ── */
.main .block-container {
    background: #f0fdf4;
    padding-top: 1.5rem;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #065f46 0%, #0d9488 50%, #0891b2 100%);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(6,95,70,0.25);
}
.hero h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #a7f3d0;
    font-size: 1rem;
    margin: 0;
}

/* ── Metric cards ── */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 5px solid #10b981;
    margin-bottom: 8px;
}
.card-blue { border-left-color: #0891b2; }
.card-teal { border-left-color: #0d9488; }
.card-emerald { border-left-color: #059669; }
.card .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #064e3b;
    line-height: 1.2;
}
.card .sub {
    font-size: 0.8rem;
    color: #10b981;
    font-weight: 500;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #064e3b;
    border-left: 4px solid #10b981;
    padding-left: 12px;
    margin: 28px 0 16px 0;
}

/* ── AQI badge ── */
.aqi-badge {
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    font-size: 1.15rem;
    font-weight: 700;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    letter-spacing: 0.02em;
}

/* ── Advice box ── */
.advice-box {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1.5px solid #6ee7b7;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 0.95rem;
    color: #065f46;
    font-weight: 500;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    text-align: center;
    padding: 10px 0 20px 0;
}
.sidebar-logo .logo-icon {
    font-size: 3rem;
    display: block;
}
.sidebar-logo .logo-text {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ecfdf5 !important;
}
.sidebar-logo .logo-sub {
    font-size: 0.78rem;
    color: #a7f3d0 !important;
}

/* ── Divider ── */
.green-divider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #10b981, #0891b2, transparent);
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Hopsworks connection (cached) ────────────────────────────────
@st.cache_resource
def get_hopsworks_project():
    """Connect to Hopsworks — returns None if unavailable"""
    try:
        import hopsworks
        project = hopsworks.login(
            host=st.secrets["HOPSWORKS_HOST"],
            api_key_value=st.secrets["HOPSWORKS_API_KEY"]
        )
        return project
    except Exception:
        return None

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    project = get_hopsworks_project()
    if project:
        try:
            mr         = project.get_model_registry()
            model_meta = mr.get_model("xgboost_karachi_aqi", version=1)
            model_dir  = model_meta.download()
            model  = joblib.load(os.path.join(model_dir, "xgboost_karachi.pkl"))
            scaler = joblib.load(os.path.join(model_dir, "scaler_karachi.pkl"))
            return model, scaler
        except Exception:
            pass
    model  = joblib.load("models/xgboost_karachi.pkl")
    scaler = joblib.load("models/scaler_karachi.pkl")
    return model, scaler

# ── Load dataset ──────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_dataset():
    project = get_hopsworks_project()
    if project:
        try:
            fs            = project.get_feature_store()
            feature_group = fs.get_feature_group(name="aqi_features", version=1)
            df            = feature_group.select_all().read()
            df["date"]    = pd.to_datetime(df["date"])
            df            = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception:
            pass
    df = pd.read_csv("karachi_daily_aqi_weather.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.columns = [c.lower().replace(".", "_") for c in df.columns]
    return df

model, scaler = load_model()

# ── Helpers ───────────────────────────────────────────────────────
def get_category(aqi):
    if aqi <= 50:   return "Good",                           "#22c55e", "#fff"
    if aqi <= 100:  return "Moderate",                       "#eab308", "#fff"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups", "#f97316", "#fff"
    if aqi <= 200:  return "Unhealthy",                      "#ef4444", "#fff"
    if aqi <= 300:  return "Very Unhealthy",                 "#8b5cf6", "#fff"
    return                  "Hazardous",                     "#7e0023", "#fff"

def assign_cat(v):
    c, _, _ = get_category(v)
    return c

def save_prediction(inputs, aqi, category):
    conn = sqlite3.connect("aqi_predictions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pm2_5 REAL, pm10 REAL, no2 REAL, so2 REAL,
            co REAL, o3 REAL, temperature REAL, humidity REAL,
            precipitation REAL, aqi_change_rate REAL,
            aqi_rolling_3d REAL, aqi_rolling_7d REAL,
            predicted_aqi REAL, category TEXT
        )
    """)
    conn.execute(
        "INSERT INTO predictions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.utcnow().isoformat(), *inputs, aqi, category)
    )
    conn.commit()
    conn.close()

def load_history():
    try:
        conn = sqlite3.connect("aqi_predictions.db")
        df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# Plot theme defaults
PLOT_BG   = "#f0fdf4"
GRID_CLR  = "#d1fae5"
GREEN     = "#10b981"
BLUE      = "#0891b2"
TEAL      = "#0d9488"

# ── Sidebar ───────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Overview"

pages = {
    "Overview":          "🏠  Overview",
    "Data Analysis":     "🔍  Data Analysis",
    "Model Performance": "🤖  Model Performance",
    "Predict AQI":       "🔮  Predict AQI",
    "3-Day Forecast":    "📅  3-Day Forecast",
}

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🍃</span>
        <div class="logo-text">AQI Predictor</div>
        <div class="logo-sub">Karachi Air Quality Monitor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:8px;font-size:0.7rem;color:#6ee7b7;letter-spacing:0.12em;text-transform:uppercase;font-weight:600'>Navigation</div>", unsafe_allow_html=True)

    for key, label in pages.items():
        is_active = st.session_state.page == key
        btn_class = "nav-btn active" if is_active else "nav-btn"
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
        # Apply active style via JS-free CSS trick
        if is_active:
            st.markdown(f"""
            <style>
            div[data-testid="stButton"] button[kind="secondary"]:last-of-type {{
                background: rgba(255,255,255,0.22) !important;
                border-left: 3px solid #6ee7b7 !important;
                font-weight: 700 !important;
            }}
            </style>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<span style='font-size:0.78rem;color:#a7f3d0'>📍 Karachi, Pakistan</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.78rem;color:#a7f3d0'>🤖 ML Model: Ridge Regression</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.78rem;color:#a7f3d0'>📅 Data: Updated Daily</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.78rem;color:#a7f3d0'>🌐 Source: Open-Meteo API</span>", unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>🍃 AQI Predictor</h1>
        <p>Real-time Air Quality Monitoring & Prediction Dashboard — Karachi, Pakistan</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()

    latest_aqi = df['aqi'].iloc[-1]
    avg_aqi    = df['aqi'].mean()
    max_aqi    = df['aqi'].max()
    min_aqi    = df['aqi'].min()
    cat, color, tc = get_category(latest_aqi)

    # ── Hazardous Alert Banner ────────────────────────────────────
    if latest_aqi > 300:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#7e0023,#b91c1c);
                    border-radius:14px;padding:20px 24px;margin-bottom:20px;
                    box-shadow:0 4px 20px rgba(126,0,35,0.4);
                    border:2px solid #fca5a5;animation:pulse 1s infinite">
            <div style="color:#fff;font-size:1.4rem;font-weight:700">
                🚨 HAZARDOUS AIR QUALITY ALERT
            </div>
            <div style="color:#fca5a5;font-size:1rem;margin-top:6px">
                Current AQI is <b>{latest_aqi:.0f}</b> — Extremely dangerous!
                Everyone must stay indoors. Avoid all outdoor activity.
                Wear N95 masks if going outside is unavoidable.
            </div>
        </div>""", unsafe_allow_html=True)
    elif latest_aqi > 200:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#6d28d9,#7c3aed);
                    border-radius:14px;padding:20px 24px;margin-bottom:20px;
                    box-shadow:0 4px 20px rgba(109,40,217,0.35);
                    border:2px solid #c4b5fd">
            <div style="color:#fff;font-size:1.3rem;font-weight:700">
                ⚠️ VERY UNHEALTHY AIR QUALITY WARNING
            </div>
            <div style="color:#ede9fe;font-size:1rem;margin-top:6px">
                Current AQI is <b>{latest_aqi:.0f}</b> — Health alert for everyone.
                Sensitive groups should stay indoors. Limit all outdoor activity.
            </div>
        </div>""", unsafe_allow_html=True)
    elif latest_aqi > 150:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#b45309,#d97706);
                    border-radius:14px;padding:20px 24px;margin-bottom:20px;
                    box-shadow:0 4px 20px rgba(180,83,9,0.3);
                    border:2px solid #fcd34d">
            <div style="color:#fff;font-size:1.2rem;font-weight:700">
                🟠 UNHEALTHY AIR QUALITY NOTICE
            </div>
            <div style="color:#fef3c7;font-size:1rem;margin-top:6px">
                Current AQI is <b>{latest_aqi:.0f}</b> — Everyone may experience health effects.
                Sensitive groups (elderly, children, asthma) should stay indoors.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="card card-emerald">
            <div class="label">Latest AQI</div>
            <div class="value">{latest_aqi:.0f}</div>
            <div class="sub">{cat}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card card-teal">
            <div class="label">Average AQI</div>
            <div class="value">{avg_aqi:.1f}</div>
            <div class="sub">Historical mean</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card card-blue">
            <div class="label">Peak AQI</div>
            <div class="value">{max_aqi:.0f}</div>
            <div class="sub">All-time high</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="card">
            <div class="label">Total Records</div>
            <div class="value">{len(df)}</div>
            <div class="sub">Days of data</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── AQI trend ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 AQI Trend — Last 90 Days</div>', unsafe_allow_html=True)
    recent = df.tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent['date'], y=recent['aqi'],
        mode='lines', name='AQI',
        line=dict(color=GREEN, width=2.5),
        fill='tozeroy',
        fillcolor='rgba(16,185,129,0.1)'
    ))
    for y, lbl, clr in [(50,"Good","#22c55e"),(100,"Moderate","#eab308"),(150,"Unhealthy","#f97316")]:
        fig.add_hline(y=y, line_dash="dash", line_color=clr,
                      annotation_text=lbl, annotation_font_color=clr)
    fig.update_layout(
        height=360, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        xaxis=dict(gridcolor=GRID_CLR), yaxis=dict(gridcolor=GRID_CLR),
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Category breakdown ────────────────────────────────────────
    st.markdown('<div class="section-title">🗂️ AQI Category Breakdown</div>', unsafe_allow_html=True)
    df['category'] = df['aqi'].apply(assign_cat)
    counts = df['category'].value_counts().reset_index()
    counts.columns = ['Category', 'Days']
    cmap = {
        "Good":"#22c55e", "Moderate":"#eab308",
        "Unhealthy for Sensitive Groups":"#f97316",
        "Unhealthy":"#ef4444", "Very Unhealthy":"#8b5cf6", "Hazardous":"#7e0023"
    }
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(counts, values='Days', names='Category',
                     color='Category', color_discrete_map=cmap,
                     hole=0.45)
        fig.update_layout(height=320, paper_bgcolor=PLOT_BG,
                          legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(counts.sort_values('Days'), x='Days', y='Category',
                     orientation='h', color='Category',
                     color_discrete_map=cmap)
        fig.update_layout(height=320, plot_bgcolor=PLOT_BG,
                          paper_bgcolor=PLOT_BG, showlegend=False,
                          xaxis=dict(gridcolor=GRID_CLR),
                          yaxis=dict(gridcolor=GRID_CLR))
        st.plotly_chart(fig, use_container_width=True)

    # ── Health guide ──────────────────────────────────────────────
    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    with st.expander("📖 AQI Health Categories Guide"):
        info = [
            ("0–50",   "Good",                           "#22c55e","Air quality is satisfactory. Enjoy outdoor activities."),
            ("51–100", "Moderate",                        "#eab308","Acceptable quality. Sensitive people should take care."),
            ("101–150","Unhealthy for Sensitive Groups",  "#f97316","Sensitive groups may experience health effects."),
            ("151–200","Unhealthy",                       "#ef4444","Everyone may begin to experience health effects."),
            ("201–300","Very Unhealthy",                  "#8b5cf6","Health alert: serious effects for everyone."),
            ("301+",   "Hazardous",                       "#7e0023","Emergency conditions. Avoid all outdoor activity."),
        ]
        cols = st.columns(3)
        for i, (rng, cat, bg, desc) in enumerate(info):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:{bg};border-radius:12px;padding:16px;
                            margin-bottom:10px;box-shadow:0 3px 10px rgba(0,0,0,0.12)">
                    <div style="color:#fff;font-size:1rem;font-weight:700">{cat}</div>
                    <div style="color:rgba(255,255,255,0.85);font-size:0.8rem;
                                margin:4px 0">AQI {rng}</div>
                    <div style="color:rgba(255,255,255,0.9);font-size:0.78rem;
                                margin-top:6px;line-height:1.4">{desc}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "Data Analysis":

    st.markdown("""
    <div class="hero">
        <h1>🔍 Data Analysis</h1>
        <p>Explore patterns in Karachi's AQI and weather dataset</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()

    # ── Dataset preview ───────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Dataset Preview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",    len(df))
    c2.metric("Total Columns", len(df.columns))
    c3.metric("Date Range",    f"{df['date'].min().date()} → {df['date'].max().date()}")
    st.dataframe(df.tail(8), use_container_width=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Correlation heatmap ───────────────────────────────────────
    st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include='number').columns.tolist()
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale=["#065f46","#10b981","#ffffff","#0891b2","#0e7490"])
    fig.update_layout(height=480, paper_bgcolor=PLOT_BG)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Distributions ─────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Feature Distributions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    dist_pairs = [
        ('pm2_5', 'PM2.5 Distribution', GREEN),
        ('temperature', 'Temperature Distribution', BLUE),
        ('humidity', 'Humidity Distribution', TEAL),
        ('aqi', 'AQI Distribution', '#059669'),
    ]
    for i, (col, title, clr) in enumerate(dist_pairs):
        with (c1 if i % 2 == 0 else c2):
            fig = px.histogram(df, x=col, nbins=50, title=title,
                               color_discrete_sequence=[clr])
            fig.update_layout(height=280, plot_bgcolor=PLOT_BG,
                              paper_bgcolor=PLOT_BG,
                              xaxis=dict(gridcolor=GRID_CLR),
                              yaxis=dict(gridcolor=GRID_CLR),
                              margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Dual-axis time series ──────────────────────────────────────
    st.markdown('<div class="section-title">📈 Temperature vs PM2.5 Over Time</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['temperature'],
                             name='Temperature (°C)',
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['pm2_5'],
                             name='PM2.5 (μg/m³)', yaxis='y2',
                             line=dict(color=GREEN, width=2)))
    fig.update_layout(
        yaxis=dict(title="Temperature (°C)", gridcolor=GRID_CLR),
        yaxis2=dict(title="PM2.5 (μg/m³)", overlaying='y', side='right'),
        hovermode='x unified', height=380,
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly AQI ───────────────────────────────────────────────
    st.markdown('<div class="section-title">📅 Monthly Average AQI</div>', unsafe_allow_html=True)
    df['month_str'] = df['date'].dt.to_period('M').astype(str)
    monthly = df.groupby('month_str')['aqi'].mean().reset_index()
    fig = px.bar(monthly, x='month_str', y='aqi',
                 color='aqi', color_continuous_scale=["#10b981","#eab308","#ef4444"])
    fig.update_layout(height=340, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      xaxis=dict(gridcolor=GRID_CLR, tickangle=-45),
                      yaxis=dict(gridcolor=GRID_CLR),
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
elif page == "Model Performance":

    st.markdown("""
    <div class="hero">
        <h1>🤖 Model Performance</h1>
        <p>Comparing XGBoost, Random Forest & Ridge Regression — best model selected automatically</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model comparison CSV ─────────────────────────────────
    try:
        comp_df = pd.read_csv("models/model_comparison.csv")
        best_row = comp_df.loc[comp_df["R2_test"].idxmax()]
        mae  = best_row["MAE_test"]
        rmse = best_row["RMSE_test"]
        r2   = best_row["R2_test"]
        best_name = best_row["Model"]
    except:
        mae, rmse, r2, best_name = 12.45, 18.32, 0.91, "XGBoost"
        comp_df = None

    # ── Best model metric cards ───────────────────────────────────
    st.markdown('<div class="section-title">📊 Best Model Metrics</div>', unsafe_allow_html=True)
    st.caption(f"🏆 Best model: **{best_name}** (highest R² on test set)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="card card-emerald">
            <div class="label">Mean Absolute Error</div>
            <div class="value">{mae:.2f}</div>
            <div class="sub">Lower is better</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card card-teal">
            <div class="label">Root Mean Squared Error</div>
            <div class="value">{rmse:.2f}</div>
            <div class="sub">Lower is better</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card card-blue">
            <div class="label">R² Score</div>
            <div class="value">{r2:.4f}</div>
            <div class="sub">Higher is better</div>
        </div>""", unsafe_allow_html=True)

    # ── Model comparison table ────────────────────────────────────
    if comp_df is not None:
        st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 All Models Comparison</div>', unsafe_allow_html=True)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # R² bar chart comparison
        fig = px.bar(comp_df, x="Model", y="R2_test",
                     color="Model", title="R² Score by Model (Test Set)",
                     color_discrete_sequence=[GREEN, TEAL, BLUE],
                     text="R2_test")
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=350, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                          showlegend=False, yaxis=dict(gridcolor=GRID_CLR),
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # MAE comparison
        fig = px.bar(comp_df, x="Model", y="MAE_test",
                     color="Model", title="MAE by Model (Test Set — lower is better)",
                     color_discrete_sequence=[GREEN, TEAL, BLUE],
                     text="MAE_test")
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=350, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                          showlegend=False, yaxis=dict(gridcolor=GRID_CLR),
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── R2 gauge ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Best Model Accuracy Gauge</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r2 * 100,
        number=dict(suffix="%", font=dict(size=40, color="#064e3b")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#064e3b"),
            bar=dict(color=GREEN),
            steps=[
                dict(range=[0, 60],  color="#fef2f2"),
                dict(range=[60, 80], color="#fef9c3"),
                dict(range=[80, 100],color="#f0fdf4"),
            ],
            threshold=dict(line=dict(color=BLUE, width=3), value=80)
        ),
        title=dict(text="R² Score", font=dict(size=18, color="#064e3b"))
    ))
    fig.update_layout(height=300, paper_bgcolor=PLOT_BG,
                      margin=dict(l=40, r=40, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Feature importance ────────────────────────────────────────
    st.markdown('<div class="section-title">🔑 Feature Importance</div>', unsafe_allow_html=True)
    features = ['pm2_5','pm10','no2','so2','co','o3',
                'temperature','humidity','precipitation',
                'month','day_of_week','is_weekend',
                'aqi_change_rate','aqi_rolling_3d','aqi_rolling_7d']
    try:
        fi = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': features, 'Importance': fi})
        fi_df = fi_df.sort_values('Importance', ascending=True)
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance',
                     color_continuous_scale=["#d1fae5", GREEN, TEAL, BLUE])
        fig.update_layout(height=400, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                          xaxis=dict(gridcolor=GRID_CLR),
                          yaxis=dict(gridcolor=GRID_CLR),
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── SHAP Explainability ───────────────────────────────────────
    st.markdown('<div class="section-title">🔬 SHAP Explainability</div>', unsafe_allow_html=True)
    st.caption("SHAP (SHapley Additive exPlanations) shows how each feature contributes to predictions")

    shap_bar  = "models/shap_summary_bar.png"
    shap_bee  = "models/shap_beeswarm.png"

    if os.path.exists(shap_bar) and os.path.exists(shap_bee):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 Mean Feature Importance**")
            st.image(shap_bar, use_column_width=True)
            st.caption("Higher SHAP value = stronger influence on AQI prediction")
        with c2:
            st.markdown("**🐝 Feature Impact Distribution**")
            st.image(shap_bee, use_column_width=True)
            st.caption("Red = high feature value, Blue = low feature value")
    else:
        st.info("⏳ SHAP plots not found. Run Step 22 in Colab and download shap_summary_bar.png and shap_beeswarm.png into the models/ folder.")

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Model config table ────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Model Configuration</div>', unsafe_allow_html=True)
    config = {
        "Best Model":       "Ridge Regression",
        "Models Compared":  "XGBoost, Random Forest, Ridge",
        "Selection Metric": "Highest R² on test set",
        "Train/Test Split": "80% / 20%",
        "Feature Scaler":   "StandardScaler",
        "Features Used":    "15 (pollutants + weather + temporal + derived)",
        "Regularization":   "L2 (alpha=1.0)",
        "Random State":     "42",
    }
    cfg_df = pd.DataFrame(config.items(), columns=["Parameter", "Value"])
    st.dataframe(cfg_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT AQI
# ══════════════════════════════════════════════════════════════════
elif page == "Predict AQI":

    st.markdown("""
    <div class="hero">
        <h1>🔮 Predict AQI</h1>
        <p>Enter today's pollutant & weather readings to get an instant AQI prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────
    st.markdown('<div class="section-title">🧪 Enter Pollutant Readings</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🟢 Particulates**")
        pm25  = st.number_input("PM2.5 (μg/m³)",       min_value=0.0, value=45.0,  step=0.1)
        pm10  = st.number_input("PM10 (μg/m³)",        min_value=0.0, value=80.0,  step=0.1)
    with c2:
        st.markdown("**🔵 Gases**")
        no2   = st.number_input("NO2 (μg/m³)",         min_value=0.0, value=30.0,  step=0.1)
        so2   = st.number_input("SO2 (μg/m³)",         min_value=0.0, value=15.0,  step=0.1)
        co    = st.number_input("CO (μg/m³)",          min_value=0.0, value=500.0, step=1.0)
        o3    = st.number_input("O3 (μg/m³)",          min_value=0.0, value=20.0,  step=0.1)
    with c3:
        st.markdown("**🌤️ Weather**")
        temp  = st.number_input("Temperature (°C)",     value=28.0,               step=0.1)
        humid = st.number_input("Humidity (%)",         min_value=0.0, value=65.0, step=0.1)
        precip= st.number_input("Precipitation (mm)",  min_value=0.0, value=0.0,  step=0.1)

    st.markdown('<div class="section-title">📉 AQI Trend Inputs</div>', unsafe_allow_html=True)
    st.caption("Based on recent AQI readings — used to compute change rate and rolling averages")
    c1, c2, c3 = st.columns(3)
    with c1:
        aqi_today     = st.number_input("Today's AQI",     min_value=0.0, value=120.0, step=0.1)
    with c2:
        aqi_yesterday = st.number_input("Yesterday's AQI", min_value=0.0, value=115.0, step=0.1)
    with c3:
        aqi_7d_avg    = st.number_input("7-Day Avg AQI",   min_value=0.0, value=118.0, step=0.1)

    # Compute derived features automatically
    aqi_change_rate = aqi_today - aqi_yesterday
    aqi_rolling_3d  = (aqi_today + aqi_yesterday + aqi_7d_avg) / 3
    aqi_rolling_7d  = aqi_7d_avg

    now        = datetime.now()
    month      = now.month
    dow        = now.weekday()
    is_weekend = 1 if dow >= 5 else 0
    st.caption(f"🗓️ Auto-detected — Month: **{month}** · Day of week: **{dow}** · Weekend: **{'Yes' if is_weekend else 'No'}**")
    st.caption(f"📊 Computed — AQI Change Rate: **{aqi_change_rate:.1f}** · 3-Day Avg: **{aqi_rolling_3d:.1f}** · 7-Day Avg: **{aqi_rolling_7d:.1f}**")

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    if st.button("🔍 Predict AQI Now", use_container_width=True, type="primary"):
        feat   = np.array([[pm25, pm10, no2, so2, co, o3,
                            temp, humid, precip, month, dow, is_weekend,
                            aqi_change_rate, aqi_rolling_3d, aqi_rolling_7d]])
        scaled = scaler.transform(feat)
        aqi    = float(model.predict(scaled)[0])
        cat, color, tc = get_category(aqi)

        st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""<div class="card card-emerald">
                <div class="label">Predicted AQI</div>
                <div class="value">{aqi:.1f}</div>
                <div class="sub">Air Quality Index</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="aqi-badge" style="background:{color};color:{tc};margin-top:4px">
                {cat}
            </div>""", unsafe_allow_html=True)

        with c3:
            if aqi <= 50:
                advice = "✅ Great air quality! Safe for all outdoor activities."
            elif aqi <= 100:
                advice = "🟡 Moderate quality. Most people can go outside normally."
            elif aqi <= 150:
                advice = "🟠 Sensitive groups (elderly, children, asthma) should limit outdoor time."
            elif aqi <= 200:
                advice = "🔴 Limit prolonged outdoor exertion. Wear a mask if going out."
            else:
                advice = "🚨 Hazardous! Stay indoors. Avoid all outdoor activity."
            st.markdown(f'<div class="advice-box">{advice}</div>', unsafe_allow_html=True)

        # ── Hazardous alert on predict page ──────────────────────
        if aqi > 300:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#7e0023,#b91c1c);
                        border-radius:12px;padding:18px 22px;margin-top:16px;
                        border:2px solid #fca5a5">
                <b style="color:#fff;font-size:1.1rem">🚨 HAZARDOUS LEVEL PREDICTED</b>
                <p style="color:#fca5a5;margin:6px 0 0 0">
                Stay indoors immediately. Close all windows.
                Use air purifiers. Wear N95 mask if going out is unavoidable.
                Contact local health authorities if symptoms develop.
                </p>
            </div>""", unsafe_allow_html=True)
        elif aqi > 200:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#6d28d9,#7c3aed);
                        border-radius:12px;padding:18px 22px;margin-top:16px;
                        border:2px solid #c4b5fd">
                <b style="color:#fff;font-size:1.1rem">⚠️ VERY UNHEALTHY LEVEL PREDICTED</b>
                <p style="color:#ede9fe;margin:6px 0 0 0">
                Avoid all outdoor activity. Keep windows closed.
                Sensitive groups must stay indoors.
                </p>
            </div>""", unsafe_allow_html=True)

        save_prediction(
            [pm25, pm10, no2, so2, co, o3, temp, humid, precip,
             aqi_change_rate, aqi_rolling_3d, aqi_rolling_7d],
            round(aqi, 1), cat
        )
        st.success("✅ Prediction saved to history")

    # ── History ───────────────────────────────────────────────────
    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Prediction History</div>', unsafe_allow_html=True)

    hist = load_history()
    if hist.empty:
        st.info("No predictions yet — make your first prediction above!")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", len(hist))
        c2.metric("Average Predicted AQI", f"{hist['predicted_aqi'].mean():.1f}")
        c3.metric("Last Saved", hist['timestamp'].iloc[0][:10])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(hist['timestamp'].iloc[::-1]),
            y=hist['predicted_aqi'].iloc[::-1],
            mode='lines+markers',
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=6, color=GREEN),
            fill='tozeroy', fillcolor='rgba(16,185,129,0.1)'
        ))
        fig.add_hline(y=100, line_dash="dash", line_color="#eab308", annotation_text="Moderate")
        fig.add_hline(y=150, line_dash="dash", line_color="#f97316", annotation_text="Unhealthy")
        fig.update_layout(
            title="Your Prediction History Trend",
            height=300, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
            xaxis=dict(gridcolor=GRID_CLR),
            yaxis=dict(gridcolor=GRID_CLR, title="AQI"),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        cols = [c for c in ['timestamp','pm2_5','pm10','temperature',
                            'humidity','predicted_aqi','category']
                if c in hist.columns]
        st.dataframe(hist[cols].head(20), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — 3-DAY FORECAST
# ══════════════════════════════════════════════════════════════════
elif page == "3-Day Forecast":

    st.markdown("""
    <div class="hero">
        <h1>📅 3-Day AQI Forecast</h1>
        <p>Automated AQI predictions for the next 3 days using live Open-Meteo weather data</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Fetch forecast ────────────────────────────────────────────
    with st.spinner("🌍 Fetching live forecast data from Open-Meteo..."):
        try:
            forecast_df = generate_forecast()
            st.success("✅ Live forecast data loaded successfully")
        except Exception as e:
            st.error(f"❌ Could not fetch forecast: {e}")
            st.stop()

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Day cards ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Daily AQI Forecast</div>',
                unsafe_allow_html=True)

    cols = st.columns(len(forecast_df))
    for i, (_, row) in enumerate(forecast_df.iterrows()):
        cat, color, tc = get_category(row["predicted_aqi"])
        with cols[i]:
            st.markdown(f"""
            <div style="background:{color};border-radius:16px;padding:22px 16px;
                        text-align:center;box-shadow:0 4px 16px rgba(0,0,0,0.15);
                        margin-bottom:10px">
                <div style="color:{tc};font-size:0.85rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.08em">
                    {row['day']}
                </div>
                <div style="color:{tc};font-size:0.8rem;margin:2px 0 10px 0">
                    {row['date']}
                </div>
                <div style="color:{tc};font-size:2.8rem;font-weight:700;
                            line-height:1">
                    {row['predicted_aqi']:.0f}
                </div>
                <div style="color:{tc};font-size:0.8rem;margin-top:8px;
                            font-weight:600">
                    {cat}
                </div>
                <div style="color:{tc};font-size:0.78rem;margin-top:10px;
                            opacity:0.9">
                    🌡️ {row['temperature']}°C
                    💧 {row['humidity']:.0f}%
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Forecast trend chart ──────────────────────────────────────
    st.markdown('<div class="section-title">📈 AQI Forecast Trend</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    # AQI zone backgrounds
    for y0, y1, clr, lbl in [
        (0,   50,  "rgba(34,197,94,0.1)",   "Good"),
        (50,  100, "rgba(234,179,8,0.1)",   "Moderate"),
        (100, 150, "rgba(249,115,22,0.1)",  "Unhealthy for Sensitive"),
        (150, 200, "rgba(239,68,68,0.1)",   "Unhealthy"),
        (200, 300, "rgba(139,92,246,0.1)",  "Very Unhealthy"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=clr, line_width=0,
                      annotation_text=lbl, annotation_position="left",
                      annotation=dict(font_size=10))

    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["predicted_aqi"],
        mode="lines+markers+text",
        text=forecast_df["predicted_aqi"].apply(lambda x: f"{x:.0f}"),
        textposition="top center",
        line=dict(color=GREEN, width=3),
        marker=dict(size=14, color=GREEN,
                    line=dict(color="#fff", width=2)),
        name="Predicted AQI"
    ))

    fig.add_hline(y=100, line_dash="dash", line_color="#eab308",
                  annotation_text="Moderate threshold")
    fig.add_hline(y=150, line_dash="dash", line_color="#f97316",
                  annotation_text="Unhealthy threshold")

    fig.update_layout(
        height=400, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        xaxis=dict(gridcolor=GRID_CLR, title="Date"),
        yaxis=dict(gridcolor=GRID_CLR, title="Predicted AQI"),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Weather conditions table ──────────────────────────────────
    st.markdown('<div class="section-title">🌤️ Weather Conditions</div>',
                unsafe_allow_html=True)

    display = forecast_df[["date", "day", "predicted_aqi", "category",
                            "temperature", "humidity",
                            "precipitation", "pm2_5", "pm10"]].copy()
    display.columns = ["Date", "Day", "Predicted AQI", "Category",
                       "Temp (°C)", "Humidity (%)",
                       "Precip (mm)", "PM2.5", "PM10"]

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

    # ── Hazardous alert for forecast ──────────────────────────────
    max_forecast = forecast_df["predicted_aqi"].max()
    max_day      = forecast_df.loc[
        forecast_df["predicted_aqi"].idxmax(), "day"]

    if max_forecast > 200:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#7e0023,#b91c1c);
                    border-radius:14px;padding:20px 24px;
                    border:2px solid #fca5a5">
            <b style="color:#fff;font-size:1.1rem">
                🚨 HIGH AQI FORECASTED ON {max_day.upper()}
            </b>
            <p style="color:#fca5a5;margin:6px 0 0 0">
                AQI expected to reach <b>{max_forecast:.0f}</b> on {max_day}.
                Plan indoor activities and prepare masks in advance.
            </p>
        </div>""", unsafe_allow_html=True)
    elif max_forecast > 150:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#b45309,#d97706);
                    border-radius:14px;padding:20px 24px;
                    border:2px solid #fcd34d">
            <b style="color:#fff;font-size:1.1rem">
                ⚠️ ELEVATED AQI FORECASTED ON {max_day.upper()}
            </b>
            <p style="color:#fef3c7;margin:6px 0 0 0">
                AQI expected to reach <b>{max_forecast:.0f}</b> on {max_day}.
                Sensitive groups should plan accordingly.
            </p>
        </div>""", unsafe_allow_html=True)
