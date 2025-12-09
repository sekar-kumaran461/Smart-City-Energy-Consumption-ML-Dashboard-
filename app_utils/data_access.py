"""Utility helpers for loading the smart-city dataset inside Streamlit pages."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import numpy as np
import streamlit as st

_DATASET_NAME = "smart_city_energy_dataset.csv"
_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / _DATASET_NAME


def _ensure_dataset_exists() -> None:
    if not _DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset {_DATASET_NAME} was not found at {_DATASET_PATH}. "
            "Double-check the data directory before launching the dashboard."
        )


@st.cache_data(show_spinner=False)
def load_data(limit: Optional[int] = None, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load the dataset with optional row/column constraints.

    Parameters
    ----------
    limit:
        Maximum number of rows to read from disk. Useful for lightweight previews.
    usecols:
        Optional iterable of column names to pull from disk. This saves memory for
        pages that only need a subset of fields.
    """

    _ensure_dataset_exists()
    kwargs: dict[str, object] = {}
    if limit is not None:
        kwargs["nrows"] = int(limit)
    if usecols is not None:
        kwargs["usecols"] = list(usecols)

    # Parse timestamps only when requested or when the column is part of the subset.
    parse_timestamp = usecols is None or "Timestamp" in (usecols or [])
    if parse_timestamp:
        kwargs["parse_dates"] = ["Timestamp"]

    df = pd.read_csv(_DATASET_PATH, **kwargs)
    if not parse_timestamp and "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@st.cache_data(show_spinner=False)
def compute_kpis(sample_rows: int = 5000) -> dict[str, float]:
    """Return lightweight KPI style aggregates for hero metrics."""

    df = load_data(limit=sample_rows)
    metrics = {
        "avg_load": float(df.get("Electricity Load", pd.Series(dtype=float)).mean()),
        "peak_load": float(df.get("Electricity Load", pd.Series(dtype=float)).max()),
        "avg_temperature": float(df.get("Temperature (°C)", pd.Series(dtype=float)).mean()),
        "avg_renewables": float(df.get("Solar PV Output (kW)", pd.Series(dtype=float)).mean()
                               + df.get("Wind Power Output (kW)", pd.Series(dtype=float)).mean()),
    }
    return metrics


@st.cache_data(show_spinner=False)
def missing_profile(limit: Optional[int] = None) -> pd.DataFrame:
    """Compute the percentage of missing values per column."""

    df = load_data(limit=limit)
    summary = (
        df.isna()
        .mean()
        .mul(100)
        .rename("missing_percent")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("missing_percent", ascending=False)
    )
    return summary


@st.cache_data(show_spinner=False)
def time_coverage(limit: Optional[int] = None) -> dict[str, pd.Timestamp]:
    """Return the first and last timestamps present in the dataset."""

    df = load_data(limit=limit, usecols=["Timestamp"])
    return {
        "start": df["Timestamp"].min(),
        "end": df["Timestamp"].max(),
        "records": int(len(df)),
    }


def format_number(value: float) -> str:
    """Pretty-print helper for Streamlit metric components."""

    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"


@st.cache_data(show_spinner=False)
def dataset_shape() -> dict[str, int]:
    """Return full dataset row/column counts without loading every column into memory."""

    _ensure_dataset_exists()
    with _DATASET_PATH.open("r", encoding="utf-8", errors="ignore") as handle:
        rows = sum(1 for _ in handle) - 1  # subtract header
    sample = load_data(limit=5_000)
    return {"rows": rows, "columns": len(sample.columns)}


@st.cache_data(show_spinner=False)
def dataset_overview(sample_rows: int = 200_000) -> dict[str, object]:
    """High-level stats used for hero panels and summary callouts."""

    df = load_data(limit=sample_rows)
    coverage = time_coverage(limit=sample_rows)
    shape = dataset_shape()
    missing_pct = float(df.isna().mean().mean() * 100)
    return {
        "rows": shape["rows"],
        "columns": shape["columns"],
        "time_start": coverage["start"],
        "time_end": coverage["end"],
        "missing_pct": missing_pct,
    }


@st.cache_data(show_spinner=False)
def energy_mix(sample_rows: int = 100_000) -> pd.DataFrame:
    """Average contribution of each generation source."""

    df = load_data(limit=sample_rows)
    source_columns = {
        "Solar PV Output (kW)": "Solar",
        "Wind Power Output (kW)": "Wind",
        "Public Transit Operational Load (kW)": "Transit Ops",
        "EV Charging Station Load (kW)": "EV Charging",
        "Smart Meter Reading per Building (kW)": "Smart Buildings",
    }
    rows = []
    for column, label in source_columns.items():
        if column in df:
            avg_kw = float(df[column].mean())
            rows.append({"source": label, "avg_kw": avg_kw})
    mix = pd.DataFrame(rows)
    total = mix["avg_kw"].sum() or 1.0
    mix["share"] = mix["avg_kw"] / total * 100
    return mix.sort_values("share", ascending=False)


@st.cache_data(show_spinner=False)
def numeric_columns(limit: int = 20_000) -> list[str]:
    df = load_data(limit=limit)
    return df.select_dtypes(include=["number"]).columns.tolist()


@st.cache_data(show_spinner=False)
def categorical_columns(limit: int = 20_000) -> list[str]:
    df = load_data(limit=limit)
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def cleaning_steps() -> list[str]:
    """Narrative describing the data cleaning process used in the notebook."""

    return [
        "Replaced negative electricity load readings with zero to preserve physical meaning.",
        "Applied IQR-based capping to critical numeric features (load, PV, wind, transformer).",
        "Left rainfall/snowfall spikes intact to retain extreme-weather signatures.",
        "Dropped rows with NaNs created by lag/rolling features before modeling.",
        "Chronologically split data (80/20) to avoid leakage across train/test.",
    ]


def engineered_features() -> list[str]:
    return [
        "Lagged loads: 30-min, 1-hour, 24-hour, 7-day histories.",
        "Rolling statistics: 2-hour rolling mean/std to capture volatility windows.",
        "Calendar flags: month start/end, quarter, week number, season.",
        "Net load and transformer stress indicators for grid-awareness.",
    ]


@st.cache_data(show_spinner=False)
def model_ready_frame(limit: int = 150_000) -> pd.DataFrame:
    """Return a chronologically ordered frame with lagged features ready for modeling."""

    base_cols = [
        "Timestamp",
        "Electricity Load",
        "Temperature (°C)",
        "Humidity (%)",
        "Solar PV Output (kW)",
        "Wind Power Output (kW)",
        "Transformer Load Level",
    ]
    df = load_data(limit=limit, usecols=[col for col in base_cols if col])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").set_index("Timestamp")
    df["lag_1"] = df["Electricity Load"].shift(1)
    df["lag_2"] = df["Electricity Load"].shift(2)
    df["lag_48"] = df["Electricity Load"].shift(48)
    df["rolling_mean_2h"] = df["Electricity Load"].rolling(window=4).mean()
    df["rolling_std_2h"] = df["Electricity Load"].rolling(window=4).std()
    return df.dropna().reset_index()

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features required for model inference."""
    df = df.copy()
    if "Timestamp" not in df.columns:
        return df
        
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")
    
    # Time-based features
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month
    df["weekofyear"] = df["Timestamp"].dt.isocalendar().week.astype(int)
    
    # Cyclical encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dayofyear"] = np.sin(2 * np.pi * df["Timestamp"].dt.dayofyear / 365.25)
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["Timestamp"].dt.dayofyear / 365.25)
    
    # Interactions
    if "Temperature (°C)" in df.columns and "Humidity (%)" in df.columns:
        df["temp_humidity_interaction"] = df["Temperature (°C)"] * df["Humidity (%)"]
    
    # Renewable penetration - calculate from solar + wind
    if "Solar PV Output (kW)" in df.columns and "Wind Power Output (kW)" in df.columns and "Electricity Load" in df.columns:
        renewable_total = df["Solar PV Output (kW)"] + df["Wind Power Output (kW)"]
        df["renewable_penetration"] = (
            (renewable_total / df["Electricity Load"].replace(0, np.nan)).clip(0, 3).fillna(0)
        )
        
    # Lags - match model training feature names
    target = "Electricity Load"
    if target in df.columns:
        # Create lag features with exact naming from model training
        df["load_lag_1h"] = df[target].shift(2)    # 30-min intervals, so shift(2) = 1 hour
        df["load_lag_2h"] = df[target].shift(4)    # shift(4) = 2 hours
        df["load_lag_6h"] = df[target].shift(12)   # shift(12) = 6 hours
        
        # Rolling features - calculated on shifted data to avoid leakage
        shifted = df[target].shift(1)
        df["load_roll_mean_6h"] = shifted.rolling(window=12).mean()   # 12 * 30min = 6h
        df["load_roll_mean_12h"] = shifted.rolling(window=24).mean()  # 24 * 30min = 12h
        df["load_roll_std_24h"] = shifted.rolling(window=48).std()    # 48 * 30min = 24h
            
    # One-hot encoding (simplified for inference - might need alignment with training)
    # For now, we skip one-hot encoding as it requires the exact same columns as training.
    # If the model relies on one-hot features, we might need to load the encoder or mock them.
    # Based on the error, the missing features were mostly lags and cyclical.
    
    return df
