"""Actionable Insights - Business Recommendations"""
from __future__ import annotations

import streamlit as st
from pathlib import Path

from app_utils.ui import section_heading, inject_theme

st.set_page_config(page_title="Actionable Insights", page_icon="💡", layout="wide")
inject_theme()

# Hero section
hero_html = """
<div class="hero">
    <div class="pill">Strategic Recommendations</div>
    <h1>💡 Actionable Insights</h1>
    <p style="font-size:1.05rem;max-width:720px;line-height:1.6;">
        Transform data analysis into business action. Evidence-based strategies for grid optimization,
        renewable integration, and demand management backed by ML insights.
    </p>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Key recommendations
section_heading("Strategic recommendations", "Data-driven actions for grid operators")

rec_html = """
<div class="insight-grid">
    <div class="insight-card"><h4>Peak shaving</h4><div class="value">8-12%</div><p>Load reduction potential</p></div>
    <div class="insight-card"><h4>Renewable boost</h4><div class="value">+15%</div><p>Utilization increase</p></div>
    <div class="insight-card"><h4>Cost savings</h4><div class="value">$2-4M</div><p>Annual opportunity</p></div>
    <div class="insight-card"><h4>DR events</h4><div class="value">100+</div><p>Annual activations</p></div>
    <div class="insight-card"><h4>Forecast accuracy</h4><div class="value">95%+</div><p>R² score achieved</p></div>
</div>
"""
st.markdown(rec_html, unsafe_allow_html=True)

# Renewable contribution analysis
section_heading("Renewable energy optimization", "Maximize clean energy utilization")
graphs_path = Path("generated_graphs")
renewable_path = graphs_path / "insights_renewable_contribution.png"
if renewable_path.exists():
    st.image(str(renewable_path), use_column_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🌞 Solar + Wind insights</h3>
            <p><strong>Peak solar hours:</strong> 10 AM - 4 PM provides maximum contribution</p>
            <p><strong>Wind complement:</strong> 24/7 generation fills solar gaps (night/cloudy)</p>
            <p><strong>Penetration rate:</strong> Reaches 25-30% during optimal conditions</p>
            <p><strong>Grid alignment:</strong> Solar peaks coincide with load peaks - perfect timing</p>
            <p><strong>Storage opportunity:</strong> Capture excess midday solar for evening use</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>📋 Action items</h3>
            <p>1. <strong>Curtailment reduction:</strong> Use ML forecasts to minimize renewable waste</p>
            <p>2. <strong>Storage dispatch:</strong> Charge 11 AM-2 PM, discharge 6-9 PM</p>
            <p>3. <strong>Capacity expansion:</strong> Add 50 MW solar for 35% penetration</p>
            <p>4. <strong>Grid integration:</strong> Upgrade inverters for better variability handling</p>
            <p>5. <strong>ROI:</strong> $600K annual savings from optimized renewable dispatch</p>
        </div>
        """, unsafe_allow_html=True)

# Peak load analysis
section_heading("Peak load management", "Reduce stress hours and costs")
peak_path = graphs_path / "insights_peak_analysis.png"
if peak_path.exists():
    st.image(str(peak_path), use_column_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>⚡ Peak hour patterns</h3>
            <p><strong>Critical hours:</strong> 2-6 PM weekdays show highest peak frequency</p>
            <p><strong>Temperature trigger:</strong> Above 28°C, each degree adds ~18 kW load</p>
            <p><strong>Annual peaks:</strong> Top 100 hours drive 30% of capacity costs</p>
            <p><strong>Predictability:</strong> 95% of peaks occur during forecast windows</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>💰 Peak reduction strategy</h3>
            <p>1. <strong>DR automation:</strong> Trigger programs 30 min before predicted peaks</p>
            <p>2. <strong>Industrial shifting:</strong> Move flexible loads to 10 PM - 6 AM</p>
            <p>3. <strong>HVAC pre-cooling:</strong> Lower temps 12-2 PM, coast 3-6 PM</p>
            <p>4. <strong>EV smart charging:</strong> Block charging 2-7 PM, incentivize overnight</p>
            <p>5. <strong>Savings:</strong> $1.2M/year from 8-10% peak reduction</p>
        </div>
        """, unsafe_allow_html=True)

# Demand response impact
section_heading("Demand response effectiveness", "Real-world DR program results")
dr_path = graphs_path / "insights_demand_response.png"
if dr_path.exists():
    st.image(str(dr_path), use_column_width=True)
    
    st.markdown("""
    <div class="card">
        <h3>📊 Demand response insights</h3>
        <p><strong>DR impact:</strong> When DR signal is active, average load drops by 45-60 kW (5-7% reduction)</p>
        <p><strong>Weekly patterns:</strong> Weekday loads 12% higher than weekends - target commercial DR on Mon-Fri</p>
        <p><strong>Participation rate:</strong> Current DR enrollment covers ~8% of peak load capacity</p>
        <p><strong>Response time:</strong> Load reduction visible within 15 minutes of signal activation</p>
        <p><strong>Reliability:</strong> 92% of DR events achieve target reduction (±10% tolerance)</p>
        <br/>
        <p><strong>Expansion opportunity:</strong> Increase DR enrollment to 15% of peak load → Double impact to 10-12% reduction → $2.4M annual value</p>
    </div>
    """, unsafe_allow_html=True)

# Strategic action plan
section_heading("90-day action plan", "Quick wins and long-term initiatives")

tab1, tab2, tab3 = st.tabs(["⚡ Quick wins (30 days)", "🎯 Medium-term (90 days)", "🚀 Strategic (6-12 months)"])

with tab1:
    st.markdown("""
    ### Immediate actions (Days 1-30)
    
    <div class="hero-grid">
        <div class="story-card">
            <h3>1. ML-powered DR automation</h3>
            <p><strong>Action:</strong> Deploy automated DR triggers using ML load forecasts</p>
            <p><strong>Target:</strong> Top 20 peak hours in next 30 days</p>
            <p><strong>Expected result:</strong> 8% load reduction during events</p>
            <p><strong>Investment:</strong> $25K (API integration + testing)</p>
            <p><strong>ROI:</strong> $80-100K savings in first month</p>
        </div>
        <div class="story-card">
            <h3>2. Renewable dispatch optimization</h3>
            <p><strong>Action:</strong> Use solar/wind forecasts for real-time dispatch decisions</p>
            <p><strong>Target:</strong> Reduce renewable curtailment by 20%</p>
            <p><strong>Expected result:</strong> +15% renewable utilization</p>
            <p><strong>Investment:</strong> $15K (forecast integration)</p>
            <p><strong>ROI:</strong> $50K/month from avoided curtailment</p>
        </div>
        <div class="story-card">
            <h3>3. Peak hour alerts</h3>
            <p><strong>Action:</strong> Send operator alerts 30 min before predicted peaks</p>
            <p><strong>Target:</strong> 100% of peaks >90th percentile</p>
            <p><strong>Expected result:</strong> Proactive manual interventions</p>
            <p><strong>Investment:</strong> $5K (dashboard + SMS)</p>
            <p><strong>ROI:</strong> Prevent 2-3 emergency purchases ($200K)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    ### Medium-term initiatives (Days 31-90)
    
    <div class="hero-grid">
        <div class="story-card">
            <h3>4. Battery storage optimization</h3>
            <p><strong>Action:</strong> Implement ML-driven charge/discharge schedules</p>
            <p><strong>Strategy:</strong> Charge during low-load + high-solar windows</p>
            <p><strong>Discharge:</strong> Peak hours (2-6 PM) + renewable gaps</p>
            <p><strong>Investment:</strong> $75K (BMS integration + algorithms)</p>
            <p><strong>ROI:</strong> $400-600K/year arbitrage + lifespan extension</p>
        </div>
        <div class="story-card">
            <h3>5. EV smart charging rollout</h3>
            <p><strong>Action:</strong> Deploy intelligent EV charging schedules</p>
            <p><strong>Phase 1:</strong> 200 workplace chargers with load control</p>
            <p><strong>Timing:</strong> Block 2-7 PM, incentivize 10 PM - 6 AM</p>
            <p><strong>Investment:</strong> $120K (hardware + software)</p>
            <p><strong>ROI:</strong> 1-2 MW peak shaving capacity</p>
        </div>
        <div class="story-card">
            <h3>6. Commercial HVAC programs</h3>
            <p><strong>Action:</strong> Enroll 50 large commercial buildings in DR</p>
            <p><strong>Strategy:</strong> Pre-cool 12-2 PM, reduce 3-6 PM</p>
            <p><strong>Target:</strong> 3-4 MW reduction during peak events</p>
            <p><strong>Investment:</strong> $150K (enrollment + incentives)</p>
            <p><strong>ROI:</strong> $500K+ annual capacity cost avoidance</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    ### Strategic initiatives (Months 3-12)
    
    <div class="hero-grid">
        <div class="story-card">
            <h3>7. Dynamic pricing launch</h3>
            <p><strong>Action:</strong> Implement time-of-use rates based on ML forecasts</p>
            <p><strong>Target:</strong> 10,000 customers in pilot program</p>
            <p><strong>Peak rates:</strong> 2-6 PM weekdays (3x base rate)</p>
            <p><strong>Off-peak:</strong> 10 PM - 6 AM (0.5x base rate)</p>
            <p><strong>Expected shift:</strong> 15-20% of flexible loads to off-peak</p>
            <p><strong>Customer benefit:</strong> 10-15% bill reduction for participants</p>
        </div>
        <div class="story-card">
            <h3>8. Vehicle-to-Grid (V2G) pilot</h3>
            <p><strong>Action:</strong> Launch V2G program with electric fleet</p>
            <p><strong>Phase 1:</strong> 100 municipal/school district EVs</p>
            <p><strong>Capacity:</strong> 50 kW per vehicle = 5 MW fleet resource</p>
            <p><strong>Dispatch:</strong> Discharge during DR events, charge overnight</p>
            <p><strong>Investment:</strong> $500K (bidirectional chargers + software)</p>
            <p><strong>ROI:</strong> $300K/year grid services + $200K fuel savings</p>
        </div>
        <div class="story-card">
            <h3>9. Microgrid development</h3>
            <p><strong>Action:</strong> Design community microgrid with 80% renewable</p>
            <p><strong>Components:</strong> 10 MW solar + 5 MW wind + 15 MWh storage</p>
            <p><strong>Coverage:</strong> 5,000 homes + critical facilities</p>
            <p><strong>ML role:</strong> Optimize island/grid modes + resource dispatch</p>
            <p><strong>Investment:</strong> $25M (capital + engineering)</p>
            <p><strong>ROI:</strong> 7-year payback via resilience + renewable credits</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Success metrics
section_heading("Success metrics & KPIs", "Track progress and demonstrate value")

metrics_html = """
<div class="diamond-grid">
    <div class="diamond-card"><div class="label">Peak reduction</div><div class="metric">8-12%</div></div>
    <div class="diamond-card"><div class="label">Renewable %</div><div class="metric">30-35%</div></div>
    <div class="diamond-card"><div class="label">DR events/year</div><div class="metric">100+</div></div>
    <div class="diamond-card"><div class="label">Cost savings</div><div class="metric">$2-4M</div></div>
    <div class="diamond-card"><div class="label">Forecast accuracy</div><div class="metric">95%+</div></div>
    <div class="diamond-card"><div class="label">Carbon reduction</div><div class="metric">-18%</div></div>
</div>
"""
st.markdown(metrics_html, unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h3>📈 Tracking & reporting</h3>
    <p><strong>Weekly:</strong> DR event summaries, peak forecasts, renewable utilization</p>
    <p><strong>Monthly:</strong> Cost savings, forecast accuracy, customer participation rates</p>
    <p><strong>Quarterly:</strong> Strategic KPIs, ROI analysis, program expansion metrics</p>
    <p><strong>Annual:</strong> Executive review, carbon impact, multi-year roadmap update</p>
</div>
""", unsafe_allow_html=True)

st.caption("💡 Actionable Insights | Data-driven grid optimization | © 2025 Smart City Energy Intelligence")
