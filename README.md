# ⚡ Smart City Energy Consumption ML Dashboard

An end-to-end **machine learning + analytics dashboard** for forecasting urban electricity demand, optimizing renewable integration, and supporting data-driven grid operations.

This project combines:
- 📊 Exploratory Data Analysis (EDA)
- 🤖 Predictive modeling (Linear Regression + Random Forest)
- 🧭 Interactive Streamlit dashboard pages for business and technical users
- 🌱 Renewable energy and peak-demand strategy insights

---

## 🚀 Project Highlights

- **Dataset scale:** ~72,960 rows (30-minute interval smart-city energy records)
- **Target variable:** `Electricity Load`
- **Best model:** Random Forest Regressor
- **Model performance:** ~**96.8% R²** (reported)
- **Use case:** Peak-load forecasting, demand response planning, renewable utilization optimization

---

## 🧠 Key Features

### 1) Data & EDA
- Distribution analysis for load, temperature, humidity, solar, and wind
- Outlier handling and data quality checks
- Temporal patterns (hourly, daily, seasonal)
- Correlation and multivariate diagnostics

### 2) Feature Engineering
- Time features (`hour`, `dayofweek`, `month`, etc.)
- Cyclical encodings (`sin_hour`, `cos_hour`, seasonal cycles)
- Lag features (short-term and daily memory)
- Rolling statistics (mean/std windows)
- Interaction features (e.g., temperature × humidity)

### 3) Modeling
- Baseline: Linear Regression
- Champion: Random Forest Regressor
- Chronological train/test split for leakage-safe evaluation
- Performance metrics: R², MAE, RMSE

### 4) Streamlit Dashboard
Multi-page dashboard with business + technical narratives:
- **Home / Strategy page** – executive summary, ROI, recommendations
- **Data Pulse** – dataset profile and load behavior overview
- **Data Quality** – completeness, outliers, consistency checks
- **Feature Forge** – feature analysis and correlation exploration
- **Modeling Lab** – model loading, prediction, and evaluation views
- **Actionable Insights** – operational recommendations
- **Data Visualizations** – comprehensive graph gallery

---

## 🗂️ Repository Structure

```text
.
├── app.py
├── app_utils/
│   ├── data_access.py
│   └── ui.py
├── data/
├── generated_graphs/
├── models/
├── notebooks/
├── pages/
├── requirements.txt
├── EDA_Comprehensive_Report.txt
└── UPGRADE_SUMMARY.md
```

---

## ⚙️ Installation

### 1) Clone the repository
```bash
git clone https://github.com/sekar-kumaran461/Smart-City-Energy-Consumption-ML-Dashboard-.git
cd Smart-City-Energy-Consumption-ML-Dashboard-
```

### 2) Create and activate virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows (PowerShell)
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Dashboard

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 📦 Dependencies

Core packages (from `requirements.txt`):
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `scikit-learn`
- `altair`

---

## 📈 Business Impact (project-reported)

- Improved forecast reliability for proactive operations
- Support for demand response during critical peak windows
- Better renewable dispatch and reduced curtailment potential
- Strong potential for operational cost savings in smart-city grids

---

## 🔮 Future Enhancements

- Real-time ingestion (streaming/SCADA integration)
- Automated retraining pipeline
- Model monitoring & drift alerts
- Deployment with containerized cloud architecture
- Scenario simulation for storage + EV charging strategies

---

## 🤝 Contributing

Contributions are welcome.

If you’d like to improve this project:
1. Fork the repo
2. Create a feature branch
3. Commit changes
4. Open a pull request

---

## 📄 License

Add your preferred license (e.g., MIT) to a `LICENSE` file.

---

## 👤 Author

**Sekar Kumaran**

If you use this project in research or demos, consider giving this repository a ⭐.