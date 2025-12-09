"""Feature Engineering Studio - Feature Forge"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from app_utils.data_access import (
    load_data,
    model_ready_frame,
    engineered_features,
)
from app_utils.ui import section_heading, inject_theme

st.set_page_config(page_title="Feature Forge", page_icon="🔧", layout="wide")
inject_theme()

# Hero section
hero_html = """
<div class="hero">
    <div class="pill">Feature Engineering</div>
    <h1>🔧 Feature Forge</h1>
    <p style="font-size:1.05rem;max-width:720px;line-height:1.6;">
        Transform raw data into powerful predictive features. Discover temporal patterns,
        correlations, and engineered signals that drive accurate load forecasting.
    </p>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Engineered features overview
section_heading("Engineered features", "Transform raw data into ML-ready signals")

feature_categories = """
<div class="hero-grid">
    <div class="story-card">
        <h3>📅 Temporal features</h3>
        <p><strong>Time-based:</strong> Hour, day of week, month, season, week number</p>
        <p><strong>Calendars:</strong> Holiday flags, weekend indicators, business day markers</p>
        <p><strong>Value:</strong> Capture daily and seasonal consumption patterns</p>
    </div>
    <div class="story-card">
        <h3>📊 Lag features</h3>
        <p><strong>Historical values:</strong> 30-min, 1-hour, 24-hour, 48-hour lags</p>
        <p><strong>Purpose:</strong> Yesterday's load predicts today's consumption</p>
        <p><strong>Importance:</strong> lag_48 is the #1 predictor (28% importance)</p>
    </div>
    <div class="story-card">
        <h3>📈 Rolling statistics</h3>
        <p><strong>2-hour windows:</strong> Rolling mean and standard deviation</p>
        <p><strong>Trend capture:</strong> Smooths noise while preserving signal</p>
        <p><strong>Volatility:</strong> Standard deviation identifies unstable periods</p>
    </div>
    <div class="story-card">
        <h3>🌍 Contextual features</h3>
        <p><strong>Weather:</strong> Temperature, humidity, solar irradiance blending</p>
        <p><strong>Renewable:</strong> Solar + wind generation combined signals</p>
        <p><strong>Grid stress:</strong> Net load and transformer stress indicators</p>
    </div>
</div>
"""
st.markdown(feature_categories, unsafe_allow_html=True)

# Feature list
section_heading("Complete feature catalog", "All engineered features")
st.markdown("""
<div class="card">
    <h3>Feature engineering pipeline</h3>
""", unsafe_allow_html=True)
for idx, feature in enumerate(engineered_features(), 1):
    st.markdown(f"{idx}. {feature}")
st.markdown("</div>", unsafe_allow_html=True)

# Load engineered data
with st.spinner("Loading engineered features..."):
    try:
        # Try loading from cleaned dataset with all features
        from pathlib import Path as P
        import pandas as pd
        cleaned_path = P("data/cleaned_dataset.csv")
        
        if cleaned_path.exists():
            df = pd.read_csv(cleaned_path, nrows=50_000)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            st.success(f"✅ Loaded {len(df):,} rows from cleaned dataset")
        else:
            # Fallback to model_ready_frame
            df = model_ready_frame(limit=50_000)
            st.info("Loaded from model_ready_frame function")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = pd.DataFrame()  # Empty fallback

# Feature correlation
section_heading("Feature correlation analysis", "Relationship discovery")
graphs_path = Path("generated_graphs")
correlation_path = graphs_path / "feature_correlation.png"
if correlation_path.exists():
    st.image(str(correlation_path), use_column_width=True)
    st.markdown("""
    <div class="card">
        <h3>🔍 Correlation insights</h3>
        <p><strong>Temperature-Load:</strong> Strong positive correlation (0.65) confirms cooling/heating demand impact</p>
        <p><strong>Solar-Irradiance:</strong> Perfect correlation (0.95+) validates sensor consistency</p>
        <p><strong>EV-Battery:</strong> Moderate correlation shows charging coordination with storage</p>
        <p><strong>Lag correlation:</strong> High correlation between consecutive lags enables time-series forecasting</p>
        <p><strong>Feature selection:</strong> Red/green cells identify redundant and important features</p>
    </div>
    """, unsafe_allow_html=True)

# Seasonal patterns
section_heading("Seasonal load patterns", "How consumption varies by season")
seasonal_path = graphs_path / "feature_seasonal_patterns.png"
if seasonal_path.exists():
    st.image(str(seasonal_path), use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Seasonal insights</h3>
            <p>• <strong>Summer peaks:</strong> Cooling demand drives afternoon load spikes</p>
            <p>• <strong>Winter patterns:</strong> Morning/evening heating creates dual peaks</p>
            <p>• <strong>Spring/Fall:</strong> Moderate loads with weather-driven variability</p>
            <p>• <strong>Hour-of-day:</strong> All seasons show commercial activity pattern</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Business applications</h3>
            <p>• <strong>DR programs:</strong> Target summer afternoons (2-6 PM)</p>
            <p>• <strong>Capacity planning:</strong> Size for summer peak + 15% margin</p>
            <p>• <strong>Renewable timing:</strong> Solar aligns with summer peaks perfectly</p>
            <p>• <strong>Storage strategy:</strong> Charge overnight, discharge peak hours</p>
        </div>
        """, unsafe_allow_html=True)

# Lag feature importance
section_heading("Temporal feature analysis", "Impact of time-based features")
lag_path = graphs_path / "feature_lag_analysis.png"
if lag_path.exists():
    st.image(str(lag_path), use_column_width=True)
    st.markdown("""
    <div class="card">
        <h3>📊 Temporal pattern insights</h3>
        <p><strong>Temperature relationship:</strong> Clear positive correlation - higher temps drive higher loads (cooling)</p>
        <p><strong>Hourly variability:</strong> Boxplot reveals consistent daily pattern with predictable peak hours</p>
        <p><strong>Load stability:</strong> IQR (box height) shows normal operational variability across hours</p>
        <p><strong>Outlier events:</strong> Points beyond whiskers represent demand response or equipment events</p>
        <p><strong>Forecasting value:</strong> Consistent patterns enable accurate 30-minute ahead predictions</p>
    </div>
    """, unsafe_allow_html=True)

# Interactive correlation explorer
section_heading("Interactive correlation explorer", "Explore feature relationships")

if len(df) > 0:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Select features to compare")
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) > 1:
            feature_x = st.selectbox("X-axis feature", numeric_cols, 
                                    index=numeric_cols.index("Temperature (°C)") if "Temperature (°C)" in numeric_cols else 0)
            feature_y = st.selectbox("Y-axis feature", numeric_cols, 
                                    index=numeric_cols.index("Electricity Load") if "Electricity Load" in numeric_cols else 1)
        else:
            st.error("Not enough numeric columns for correlation")
            st.stop()
    
    with col2:
        sample_size = st.slider("Sample size", 1000, min(10000, len(df)), min(5000, len(df)), step=500)
        sample_df = df.sample(min(sample_size, len(df)))
        
        if feature_x != feature_y:
            correlation = sample_df[[feature_x, feature_y]].corr().iloc[0, 1]
            st.metric("Pearson correlation", f"{correlation:.3f}")
            
            if abs(correlation) > 0.7:
                st.success("Strong correlation - features are highly related")
            elif abs(correlation) > 0.4:
                st.info("Moderate correlation - features show relationship")
            else:
                st.warning("Weak correlation - features are mostly independent")
        else:
            st.info("Select different features to calculate correlation")
    
    if feature_x != feature_y:
        fig = px.scatter(
            sample_df,
            x=feature_x,
            y=feature_y,
            opacity=0.5,
            color=feature_y,
            color_continuous_scale='Viridis',
            title=f"{feature_x} vs {feature_y}"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available. Check data loading.")

# Feature preview
section_heading("Engineered data preview", "Sample of model-ready features")

display_cols = [col for col in df.columns if any(x in col.lower() for x in ['timestamp', 'load', 'lag', 'rolling', 'temperature', 'solar', 'wind'])]
if display_cols:
    st.dataframe(df[display_cols[:15]].head(100), use_container_width=True)
    
    st.markdown(f"""
    <div class="diamond-grid">
        <div class="diamond-card"><div class="label">Total features</div><div class="metric">{len(df.columns)}</div></div>
        <div class="diamond-card"><div class="label">Engineered</div><div class="metric">{len([c for c in df.columns if any(x in c for x in ['lag_', 'rolling_'])])}</div></div>
        <div class="diamond-card"><div class="label">Observations</div><div class="metric">{len(df):,}</div></div>
        <div class="diamond-card"><div class="label">Ready for ML</div><div class="metric">✅ Yes</div></div>
    </div>
    """, unsafe_allow_html=True)

st.caption("🔧 Feature Forge complete | Navigate to Modeling Lab to train models with these features →")
