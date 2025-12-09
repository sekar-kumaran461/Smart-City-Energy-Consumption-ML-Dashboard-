"""Presentation helpers for the Streamlit showcase."""
from __future__ import annotations

import streamlit as st

PRIMARY = "#041a2f"
ACCENT = "#14c8c0"
HIGHLIGHT = "#f6a623"
LIGHT = "#f7fbff"


def inject_theme() -> None:
    """Inject custom CSS aligned with the reference mock-up."""

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Space+Mono:wght@600&display=swap');

        :root {{
            --primary: {PRIMARY};
            --accent: {ACCENT};
            --highlight: {HIGHLIGHT};
            --page-bg: {LIGHT};
            --ink: #0f253a;
        }}

        /* SIDEBAR: Dark Blue BG -> White Text */
        [data-testid="stSidebar"] {{
            background-color: #041a2f !important;
        }}
        
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, 
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {{
            color: #ffffff !important;
        }}
        
        /* Fix sidebar nav links if they are not caught by above */
        [data-testid="stSidebarNav"] span {{
            color: #ffffff !important;
        }}

        /* GLOBAL: Force Light Theme (White BG -> Black Text) */
        html, body, .main, .stApp {{
            background-color: #ffffff !important;
            color: #000000 !important;
        }}

        /* Force all standard text elements to black (except in sidebar/hero) */
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, 
        .main p, .main li, .main span, .main div, .main label, 
        .main .stMarkdown, .main .stText {{
            color: #000000 !important;
        }}

        /* INPUTS: Specific styling for input widgets to be Dark BG + White Text */
        .main input, .main textarea, .main select {{
            color: #ffffff !important;
            background-color: #262730 !important; /* Dark grey background */
        }}
        
        /* Fix for Selectbox/Multiselect which use different structure */
        .main [data-baseweb="select"] > div {{
            background-color: #262730 !important;
            color: #ffffff !important;
        }}
        .main [data-baseweb="select"] span {{
            color: #ffffff !important;
        }}
        
        /* Fix for NumberInput +/- buttons */
        .main [data-testid="stNumberInput"] button {{
            color: #ffffff !important;
            background-color: #262730 !important;
        }}

        /* WIDGETS: Force labels to black in main area */
        .main .stSlider label, .main .stNumberInput label, .main .stSelectbox label, 
        .main .stDateInput label, .main .stTimeInput label, 
        .main [data-testid="stWidgetLabel"], .main [data-testid="stMetricLabel"] {{
            color: #000000 !important;
        }}
        
        /* Metric Values */
        [data-testid="stMetricValue"] {{
            color: #041a2f !important; /* Dark blue for emphasis */
        }}

        /* COMPONENT: Hero Section (Dark BG -> White Text) */
        .hero {{
            background: linear-gradient(135deg, #041a2f, #0f3856) !important;
            padding: 2.8rem;
            border-radius: 34px;
            color: #ffffff !important;
            box-shadow: 0 25px 60px rgba(0,0,0,0.35);
            position: relative;
            overflow: hidden;
        }}
        
        /* Force text INSIDE hero to be white */
        .hero h1, .hero h2, .hero h3, .hero p, .hero span, .hero div {{
            color: #ffffff !important;
        }}

        .hero:after {{
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 80% 10%, rgba(255,255,255,0.25), transparent 50%);
            opacity: 0.6;
            pointer-events: none;
        }}

        .hero-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 1.2rem; margin-top: 2rem; }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.2rem 1rem;
            border-radius: 999px;
            background-color: rgba(255,255,255,0.18);
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.1em;
            color: #ffffff !important;
        }}

        .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 1rem; margin: 1.4rem 0; }}
        
        .insight-card {{ 
            background: #f8f9fa !important; /* Light grey card */
            border-radius: 20px; 
            padding: 1.2rem; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            border: 1px solid rgba(0,0,0,0.05); 
        }}
        .insight-card h4 {{ margin: 0; font-size: 1rem; color: #000000 !important; }}
        .insight-card .value {{ font-size: 1.9rem; font-weight: 600; margin-top: 0.4rem; color: #041a2f !important; }}
        .insight-card p {{ color: #333333 !important; }}

        .diamond-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: 1.2rem; }}
        .diamond-card {{
            background: #f8f9fa !important;
            border-radius: 22px;
            padding: 1rem 1.4rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .diamond-card .label {{ font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; color: #555555 !important; }}
        .diamond-card .metric {{ font-size: 1.4rem; font-weight: 600; color: #041a2f !important; }}

        .chip {{
            background: linear-gradient(120deg, #14c8c0, #6cf2da);
            color: #042339 !important;
            font-weight: 600;
            border-radius: 999px;
            padding: 0.2rem 0.9rem;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
        }}

        .card {{
            background: #f8f9fa !important;
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            color: #000000 !important;
        }}
        
        .story-card {{
            background: #eaf6ff !important; /* Light blue card for visibility */
            border: 1px solid #d0e1f0;
            border-radius: 16px;
            padding: 1.5rem;
            color: #000000 !important;
        }}
        
        .story-card h3 {{
            color: #041a2f !important;
            margin-top: 0;
        }}
        
        .hero h1, .hero p, .hero div {{
             color: #eaf6ff !important;
        }}

        .hero:after {{
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 80% 10%, rgba(255,255,255,0.25), transparent 50%);
            opacity: 0.6;
            pointer-events: none;
        }}

        .hero-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 1.2rem; margin-top: 2rem; }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.2rem 1rem;
            border-radius: 999px;
            background-color: rgba(255,255,255,0.18);
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.1em;
            color: #ffffff !important;
        }}

        .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 1rem; margin: 1.4rem 0; }}
        
        .insight-card {{ 
            background: var(--secondary-background-color); 
            border-radius: 20px; 
            padding: 1.2rem; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            border: 1px solid rgba(128,128,128,0.1); 
        }}
        .insight-card h4 {{ margin: 0; font-size: 1rem; color: var(--text-color); }}
        .insight-card .value {{ font-size: 1.9rem; font-weight: 600; margin-top: 0.4rem; color: var(--primary-color); }}
        .insight-card p {{ color: var(--text-color); opacity: 0.8; }}

        .diamond-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: 1.2rem; }}
        .diamond-card {{
            background: var(--secondary-background-color);
            border-radius: 22px;
            padding: 1rem 1.4rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .diamond-card .label {{ font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-color); opacity: 0.7; }}
        .diamond-card .metric {{ font-size: 1.4rem; font-weight: 600; color: var(--primary-color); }}

        .chip {{
            background: linear-gradient(120deg, var(--primary-color), var(--secondary-background-color));
            color: white;
            font-weight: 600;
            border-radius: 999px;
            padding: 0.2rem 0.9rem;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
        }}

        .card {{
            background: var(--secondary-background-color);
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            color: var(--text-color);
        }}
        
        .story-card {{
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 1.5rem;
            color: inherit;
        }}
            border: 1px solid rgba(4,26,47,0.04);
            box-shadow: 0 15px 45px rgba(4,26,47,0.08);
        }}

        .card h3 {{ margin-bottom: 0.5rem; font-size: 1.05rem; }}
        .stat-value {{ font-size: 2.1rem; font-weight: 600; color: var(--primary); }}

        .viz-frame {{
            background: white;
            border-radius: 24px;
            padding: 1rem;
            box-shadow: 0 12px 30px rgba(4,26,47,0.08);
            border: 1px solid rgba(4,26,47,0.05);
        }}

        .story-card {{
            background: linear-gradient(180deg, rgba(20,200,192,0.12), rgba(255,255,255,0));
            border-radius: 18px;
            padding: 1rem 1.2rem;
            border: 1px solid rgba(20,200,192,0.3);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: str, subtitle: str | None = None) -> None:
    subtitle = subtitle or ""
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <div class="stat-value">{value}</div>
            <div style="color:#5b6c86">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<div style='color:#566882;margin-top:0.25rem'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div style="margin-top:2.2rem;margin-bottom:1rem;">
            <span class="chip">INSIGHT</span>
            <h2 style="margin-bottom:0.2rem;font-size:1.8rem;font-weight:600;color:var(--primary);">{title}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
