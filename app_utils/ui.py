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

        html, body, .main {{
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            background: radial-gradient(circle at 20% 20%, rgba(20,200,192,0.12), transparent 45%),
                        radial-gradient(circle at 80% 0%, rgba(246,166,35,0.12), transparent 35%),
                        var(--page-bg);
            color: var(--ink);
        }}

        .hero {{
            background: linear-gradient(135deg, rgba(4,26,47,0.95), rgba(15,56,86,0.92));
            padding: 2.8rem;
            border-radius: 34px;
            color: #eaf6ff;
            box-shadow: 0 25px 60px rgba(4,26,47,0.35);
            position: relative;
            overflow: hidden;
        }}

        .hero:after {{
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 80% 10%, rgba(255,255,255,0.25), transparent 50%);
            opacity: 0.6;
            pointer-events: none;
        }}

        .hero h1 {{ font-size: 2.6rem; margin-bottom: 0.4rem; letter-spacing: -0.01em; }}
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
        }}

        .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 1rem; margin: 1.4rem 0; }}
        .insight-card {{ background: white; border-radius: 20px; padding: 1.2rem; box-shadow: 0 18px 40px rgba(4,26,47,0.12); border: 1px solid rgba(4,26,47,0.05); }}
        .insight-card h4 {{ margin: 0; font-size: 1rem; }}
        .insight-card .value {{ font-size: 1.9rem; font-weight: 600; margin-top: 0.4rem; color: var(--primary); }}

        .diamond-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: 1.2rem; }}
        .diamond-card {{
            background: white;
            border-radius: 22px;
            padding: 1rem 1.4rem;
            text-align: center;
            box-shadow: inset 0 0 0 1px rgba(4,26,47,0.08), 0 12px 25px rgba(4,26,47,0.08);
        }}

        .diamond-card .label {{ font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; color: #5f7088; }}
        .diamond-card .metric {{ font-size: 1.4rem; font-weight: 600; color: var(--primary); }}

        .chip {{
            background: linear-gradient(120deg, var(--accent), #6cf2da);
            color: #042339;
            font-weight: 600;
            border-radius: 999px;
            padding: 0.2rem 0.9rem;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
        }}

        .card {{
            background: rgba(255,255,255,0.95);
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
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
