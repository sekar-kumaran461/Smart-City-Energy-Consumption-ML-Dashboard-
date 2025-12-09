"""Dataset summary page - Executive Dashboard"""
from __future__ import annotations

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from app_utils.data_access import (
    cleaning_steps,
    dataset_overview,
    engineered_features,
    load_data,
    missing_profile,
    compute_kpis,
    format_number,
)
from app_utils.ui import section_heading, inject_theme

st.set_page_config(page_title="Data Pulse", page_icon="📊", layout="wide")
inject_theme()

overview = dataset_overview()
kpis = compute_kpis()

# Hero section
hero_html = f"""
<div class="hero">
    <div class="pill">Dataset Intelligence</div>
    <h1>📊 Data Pulse</h1>
    <p style="font-size:1.05rem;max-width:720px;line-height:1.6;">
        Complete dataset summary with 72,960 observations spanning smart city energy consumption,
        renewable generation, weather patterns, and mobility indicators.
    </p>
    <div class="hero-grid">
        <div class="story-card">
            <h3>Dataset Scale</h3>
            <p>{overview['rows']:,} rows × {overview['columns']} features covering {(overview['time_end'] - overview['time_start']).days} days of continuous smart city operations.</p>
        </div>
        <div class="story-card">
            <h3>Data Quality</h3>
            <p>Pre-cleaning missingness: {overview['missing_pct']:.1f}%. Post-processing completeness: 99.8% with IQR outlier capping.</p>
        </div>
        <div class="story-card">
            <h3>Time Coverage</h3>
            <p>From {overview['time_start'].date()} to {overview['time_end'].date()} at 30-minute intervals.</p>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Executive metrics
section_heading("Executive metrics", "Key performance indicators at a glance")
insight_html = f"""
<div class="insight-grid">
    <div class="insight-card"><h4>Average Load</h4><div class="value">{format_number(kpis['avg_load'])} kW</div><p>Baseline consumption</p></div>
    <div class="insight-card"><h4>Peak Load</h4><div class="value">{format_number(kpis['peak_load'])} kW</div><p>Maximum observed</p></div>
    <div class="insight-card"><h4>Renewable Mix</h4><div class="value">{format_number(kpis['avg_renewables'])} kW</div><p>Solar + wind average</p></div>
    <div class="insight-card"><h4>Temperature</h4><div class="value">{kpis['avg_temperature']:.1f} °C</div><p>Average climate</p></div>
    <div class="insight-card"><h4>Data Quality</h4><div class="value">{100 - overview['missing_pct']:.1f}%</div><p>Completeness score</p></div>
</div>
"""
st.markdown(insight_html, unsafe_allow_html=True)

# Load distribution visualization
section_heading("Load distribution analysis", "Understanding consumption patterns")
graphs_path = Path("generated_graphs")
load_overview_path = graphs_path / "main_load_overview.png"
if load_overview_path.exists():
    st.image(str(load_overview_path), use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Distribution insights</h3>
            <p>• Load follows normal distribution with slight right skew</p>
            <p>• Mean and median are closely aligned, indicating balanced data</p>
            <p>• IQR capping successfully removed extreme outliers</p>
            <p>• Distribution shape enables reliable predictive modeling</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Business impact</h3>
            <p>• Predictable load patterns enable proactive grid management</p>
            <p>• Outlier removal improves forecast accuracy</p>
            <p>• Understanding variability informs capacity planning</p>
            <p>• Distribution metrics guide demand response strategies</p>
        </div>
        """, unsafe_allow_html=True)

# Hourly pattern
section_heading("24-hour load profile", "Daily consumption rhythm")
hourly_pattern_path = graphs_path / "main_hourly_pattern.png"
if hourly_pattern_path.exists():
    st.image(str(hourly_pattern_path), use_column_width=True)
    st.markdown("""
    <div class="card">
        <h3>📊 Pattern analysis</h3>
        <p><strong>Morning ramp:</strong> Load increases sharply between 6-9 AM as commercial and industrial facilities activate</p>
        <p><strong>Peak hours:</strong> Maximum consumption occurs during business hours (9 AM - 6 PM)</p>
        <p><strong>Evening decline:</strong> Gradual decrease after 7 PM as operations wind down</p>
        <p><strong>Overnight minimum:</strong> Base load maintained during 2-5 AM period</p>
        <p><strong>Actionable insight:</strong> Peak shaving programs should target 10 AM - 4 PM window for maximum impact</p>
    </div>
    """, unsafe_allow_html=True)

# Dataset snapshot
section_heading("Dataset snapshot", "Sample observations with key features")

preview = load_data(limit=600)
col1, col2 = st.columns((1.5, 1))
with col1:
    st.dataframe(preview.head(50), use_container_width=True)

with col2:
    st.markdown(
        f"""
        <div class="diamond-grid">
            <div class="diamond-card"><div class="label">Total rows</div><div class="metric">{overview['rows']:,}</div></div>
            <div class="diamond-card"><div class="label">Features</div><div class="metric">{overview['columns']}</div></div>
            <div class="diamond-card"><div class="label">Time span</div><div class="metric">{(overview['time_end'] - overview['time_start']).days} days</div></div>
            <div class="diamond-card"><div class="label">Completeness</div><div class="metric">{100 - overview['missing_pct']:.1f}%</div></div>
        </div>
        <br/>
        <div class="card">
            <h3>Feature categories</h3>
            <p><strong>Generation:</strong> Solar PV, Wind Power, Battery Storage</p>
            <p><strong>Demand:</strong> Electricity Load, Net Load, EV Charging</p>
            <p><strong>Weather:</strong> Temperature, Humidity, Solar Irradiance, Wind Speed</p>
            <p><strong>Operations:</strong> Demand Response, Grid Stability, Curtailment</p>
            <p><strong>Temporal:</strong> Hour, Day, Week, Season, Holiday indicators</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Energy mix
section_heading("Energy source contribution", "Average power by source")
energy_mix_path = graphs_path / "main_energy_mix.png"
if energy_mix_path.exists():
    st.image(str(energy_mix_path), use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Source breakdown</h3>
            <p>• Grid load represents total consumption demand</p>
            <p>• Solar PV provides daytime renewable contribution</p>
            <p>• Wind power offers complementary generation profile</p>
            <p>• Net load shows demand after renewable offset</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Strategic implications</h3>
            <p>• Renewable penetration reduces grid dependency</p>
            <p>• Net load variability drives storage requirements</p>
            <p>• Source diversity improves grid resilience</p>
            <p>• Generation mix informs infrastructure planning</p>
        </div>
        """, unsafe_allow_html=True)

# Feature grouping
section_heading("Feature organization", "Logical grouping for analysis")
st.markdown(
    """
    <div class="diamond-grid">
        <div class="diamond-card"><div class="label">Generation</div><div class="metric">Solar · Wind · Battery</div></div>
        <div class="diamond-card"><div class="label">Demand</div><div class="metric">Load · EV · Transit</div></div>
        <div class="diamond-card"><div class="label">Weather</div><div class="metric">Temp · Humidity · Wind</div></div>
        <div class="diamond-card"><div class="label">Operations</div><div class="metric">DR · Stability · Curtailment</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Detailed views
section_heading("Data quality views", "Technical and business perspectives")
simple, technical = st.tabs(["📋 Business view", "🔬 Technical view"])
with simple:
    st.markdown("""
    <div class="card">
        <h3>Why this dataset matters</h3>
        <p>• Comprehensive coverage of smart city energy ecosystem</p>
        <p>• Real-world data enables accurate operational planning</p>
        <p>• Multi-dimensional signals support advanced analytics</p>
        <p>• Clean data structure accelerates model development</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Data preparation highlights")
    for item in cleaning_steps()[:5]:
        st.markdown(f"✓ {item}")
    
    st.markdown("### Engineered features")
    for item in engineered_features()[:6]:
        st.markdown(f"🔧 {item}")

with technical:
    st.markdown("### Statistical summary")
    st.dataframe(preview.describe(include="all"), use_container_width=True)
    
    missing = missing_profile()
    if len(missing) > 0:
        top_missing = missing.head(12)
        heat = alt.Chart(top_missing).mark_rect().encode(
            x=alt.X("feature:N", sort=None, title="Feature"),
            y=alt.Y("missing_percent:Q", title="Missing %"),
            color=alt.Color("missing_percent:Q", scale=alt.Scale(scheme="tealblues")),
            tooltip=["feature", alt.Tooltip("missing_percent", format=".2f")],
        ).properties(height=300)
        st.altair_chart(heat, use_container_width=True)

# Interactive distribution explorer
section_heading("Interactive distribution explorer", "Analyze any numeric column")
numeric_cols = [col for col in preview.columns if pd.api.types.is_numeric_dtype(preview[col])]
if numeric_cols:
    col_choice = st.selectbox("Select column to analyze", numeric_cols, index=0)
    
    col1, col2 = st.columns((2, 1))
    with col1:
        chart = px.histogram(preview, x=col_choice, nbins=50, opacity=0.85, 
                           color_discrete_sequence=["#14c8c0"])
        chart.update_layout(title=f"Distribution: {col_choice}", height=350)
        st.plotly_chart(chart, use_container_width=True)
    
    with col2:
        col_data = preview[col_choice].dropna()
        st.markdown(
            f"""
            <div class="diamond-grid">
                <div class="diamond-card"><div class="label">Mean</div><div class="metric">{col_data.mean():.2f}</div></div>
                <div class="diamond-card"><div class="label">Median</div><div class="metric">{col_data.median():.2f}</div></div>
                <div class="diamond-card"><div class="label">Std Dev</div><div class="metric">{col_data.std():.2f}</div></div>
                <div class="diamond-card"><div class="label">Range</div><div class="metric">{col_data.max() - col_data.min():.1f}</div></div>
            </div>
            <br/>
            <div class="card">
                <h3>Distribution insights</h3>
                <p>• Analyze shape, spread, and central tendency</p>
                <p>• Identify outliers and data quality issues</p>
                <p>• Understand feature characteristics for modeling</p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("Numeric columns unavailable in current preview")

st.caption("Navigate to other pages for deeper analysis: Data Quality → Feature Forge → Modeling Lab → Actionable Insights")
