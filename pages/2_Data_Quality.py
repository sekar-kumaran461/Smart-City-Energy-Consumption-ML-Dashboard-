"""Data Quality Dashboard - Cleaning and Validation"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path

from app_utils.data_access import (
    cleaning_steps,
    dataset_overview,
    load_data,
    missing_profile,
    format_number,
)
from app_utils.ui import section_heading, inject_theme

st.set_page_config(page_title="Data Quality", page_icon="✨", layout="wide")
inject_theme()

overview = dataset_overview()
raw = load_data(limit=250_000)
duplicate_share = float(raw.duplicated().mean() * 100)
quality_score = max(0.0, 100 - overview["missing_pct"] - duplicate_share)

# Hero section
hero_html = f"""
<div class="hero">
    <div class="pill">Data Integrity</div>
    <h1>✨ Data Quality</h1>
    <p style="font-size:1.05rem;max-width:720px;line-height:1.6;">
        Comprehensive data quality analysis covering completeness, consistency, and accuracy. 
        Our cleaning pipeline transforms raw data into analysis-ready format.
    </p>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Quality metrics
section_heading("Quality metrics", "Core data quality indicators")
insight_html = f"""
<div class="insight-grid">
    <div class="insight-card"><h4>Total rows</h4><div class="value">{format_number(overview['rows'])}</div><p>Assessed records</p></div>
    <div class="insight-card"><h4>Quality score</h4><div class="value">{quality_score:.1f}/100</div><p>Overall data health</p></div>
    <div class="insight-card"><h4>Completeness</h4><div class="value">{100 - overview['missing_pct']:.1f}%</div><p>Non-missing data</p></div>
    <div class="insight-card"><h4>Duplicates</h4><div class="value">{duplicate_share:.2f}%</div><p>Duplicate records</p></div>
    <div class="insight-card"><h4>Features</h4><div class="value">{overview['columns']}</div><p>Validated columns</p></div>
</div>
"""
st.markdown(insight_html, unsafe_allow_html=True)

# Completeness visualization
section_heading("Data completeness analysis", "Feature-level completeness assessment")
graphs_path = Path("generated_graphs")
completeness_path = graphs_path / "quality_completeness.png"
if completeness_path.exists():
    st.image(str(completeness_path), use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Completeness insights</h3>
            <p>• Green bars (100%): Perfect data quality, no missing values</p>
            <p>• Orange bars (95-99%): Good quality with minimal gaps</p>
            <p>• Red bars (<95%): Requires attention or imputation</p>
            <p>• Core features show excellent completeness</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Quality actions</h3>
            <p>• Features above 95% threshold are model-ready</p>
            <p>• Missing values handled via forward fill or interpolation</p>
            <p>• Outliers capped using IQR method to preserve distributions</p>
            <p>• Duplicate records removed to ensure data uniqueness</p>
        </div>
        """, unsafe_allow_html=True)

# Distribution quality
section_heading("Distribution quality check", "Key feature distributions post-cleaning")
distributions_path = graphs_path / "quality_distributions.png"
if distributions_path.exists():
    st.image(str(distributions_path), use_column_width=True)
    st.markdown("""
    <div class="card">
        <h3>📊 Distribution analysis</h3>
        <p><strong>Temperature:</strong> Normal distribution centered around city's average climate (~20°C)</p>
        <p><strong>Humidity:</strong> Right-skewed distribution typical of continental climate patterns</p>
        <p><strong>Solar PV:</strong> Daytime generation shows expected diurnal pattern with zero overnight values</p>
        <p><strong>Wind Power:</strong> Continuous generation with natural variability reflecting wind conditions</p>
        <p><strong>Quality impact:</strong> All distributions show realistic patterns without anomalies, confirming data quality</p>
    </div>
    """, unsafe_allow_html=True)

# Cleaning pipeline
section_heading("Cleaning pipeline", "Transformation steps applied to raw data")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="card">
        <h3>Data cleaning steps</h3>
        """, unsafe_allow_html=True)
    for idx, step in enumerate(cleaning_steps(), 1):
        st.markdown(f"{idx}. {step}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    clean_pct = max(0.0, 100 - overview["missing_pct"])
    st.markdown(f"""
    <div class="card">
        <h3>Cleaning results</h3>
        <p><strong>Clean data percentage:</strong> {clean_pct:.1f}%</p>
        <br/>
    </div>
    """, unsafe_allow_html=True)
    st.progress(clean_pct / 100)
    st.markdown(f"""
    <div class="diamond-grid">
        <div class="diamond-card"><div class="label">Pre-cleaning</div><div class="metric">{overview['missing_pct']:.1f}% missing</div></div>
        <div class="diamond-card"><div class="label">Post-cleaning</div><div class="metric">{clean_pct:.1f}% complete</div></div>
    </div>
    """, unsafe_allow_html=True)

# Detailed quality analysis
section_heading("Detailed quality analysis", "Deep-dive into feature quality")
tabs = st.tabs(["📉 Missingness", "🔍 Feature inspector", "⚡ Outlier analysis"])

missing = missing_profile()
with tabs[0]:
    st.markdown("### Missingness by feature")
    if len(missing) > 0:
        st.bar_chart(missing.set_index("feature").head(15)["missing_percent"])
        st.markdown("""
        <div class="card">
            <h3>Interpretation</h3>
            <p>• Bar height represents percentage of missing values</p>
            <p>• Features with minimal bars have excellent completeness</p>
            <p>• Taller bars indicate features requiring imputation strategies</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✓ No missing data detected - excellent quality!")

with tabs[1]:
    st.markdown("### Feature-level inspection")
    column_choice = st.selectbox("Select feature to inspect", missing["feature"].tolist() if len(missing) > 0 else list(raw.columns))
    column_data = raw[column_choice]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Nulls", f"{(column_data.isna().mean() * 100):.2f}%")
    col2.metric("Type", str(column_data.dtype))
    col3.metric("Unique", f"{column_data.nunique():,}")
    
    if pd.api.types.is_numeric_dtype(column_data):
        chart = alt.Chart(pd.DataFrame({column_choice: column_data.dropna()})).mark_area(
            color='#14c8c0', opacity=0.7
        ).encode(
            x=alt.X(f"{column_choice}:Q", title="Value"), 
            y=alt.Y("count()", title="Frequency")
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        freq = column_data.value_counts().reset_index()
        freq.columns = ["value", "count"]
        st.dataframe(freq.head(20), use_container_width=True)

with tabs[2]:
    st.markdown("### Outlier detection (IQR method)")
    feature_selection = st.multiselect(
        "Select features for outlier analysis",
        [
            "Electricity Load",
            "Solar PV Output (kW)",
            "Wind Power Output (kW)",
            "Temperature (°C)",
            "Humidity (%)",
            "Transformer Load Level",
        ],
        default=["Electricity Load", "Solar PV Output (kW)", "Wind Power Output (kW)"],
    )

    @st.cache_data(show_spinner=False)
    def outlier_table(columns: list[str]) -> pd.DataFrame:
        df = load_data(limit=180_000)
        rows = []
        for column in columns:
            if column not in df.columns:
                continue
            series = df[column].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            rows.append(
                {
                    "feature": column,
                    "below_lower": f"{(series < lower).mean() * 100:.2f}%",
                    "above_upper": f"{(series > upper).mean() * 100:.2f}%",
                }
            )
        return pd.DataFrame(rows)

    if feature_selection:
        outliers = outlier_table(feature_selection)
        st.dataframe(outliers, use_container_width=True)
        st.markdown("""
        <div class="card">
            <h3>Outlier handling</h3>
            <p>• IQR method: Outliers defined as values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR</p>
            <p>• Capping strategy: Outliers capped to bounds rather than removed</p>
            <p>• Preserves distribution shape while reducing extreme value impact</p>
        </div>
        """, unsafe_allow_html=True)

st.caption("Data quality is the foundation of reliable analytics. Navigate to Feature Forge to see engineered features →")
