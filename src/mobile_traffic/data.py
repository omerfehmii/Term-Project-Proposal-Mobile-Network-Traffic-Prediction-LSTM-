from __future__ import annotations

import pandas as pd
from typing import Optional, Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_series(
    path: str,
    time_col: str,
    value_col: str,
    cell_col: Optional[str] = None,
    cell_id: Optional[int | str] = None,
    freq: Optional[str] = None,
    tz: Optional[str] = None,
    resample_agg: str = "sum",
) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[time_col])
    if tz:
        df[time_col] = df[time_col].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT", errors="coerce")
    if cell_col and cell_id is not None:
        if pd.api.types.is_numeric_dtype(df[cell_col]):
            cell_id_num = pd.to_numeric(cell_id, errors="coerce")
            if pd.notna(cell_id_num):
                df = df[df[cell_col] == cell_id_num]
            else:
                df = df[df[cell_col].astype(str) == str(cell_id)]
        else:
            df = df[df[cell_col].astype(str) == str(cell_id)]
    df = df[[time_col, value_col]].dropna()
    df = df.sort_values(time_col).drop_duplicates(subset=time_col, keep="last")
    series = df.set_index(time_col)[value_col]
    if freq:
        if resample_agg == "sum":
            series = series.resample(freq).sum()
        elif resample_agg == "mean":
            series = series.resample(freq).mean()
        else:
            raise ValueError(f"Unsupported resample_agg: {resample_agg}")
    series = series.sort_index()
    series = series.asfreq(freq) if freq else series
    return series


def clip_outliers(series: pd.Series, iqr_factor: float = 3.0) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    return series.clip(lower=lower, upper=upper)


def impute_missing(
    series: pd.Series, method: str = "linear", limit: int = 6, seasonal_periods: int | None = None
) -> pd.Series:
    if not series.isna().any():
        return series
    if method == "linear":
        filled = series.interpolate(method="time", limit=limit)
        return filled.ffill().bfill()
    if method == "seasonal":
        seasonal_means = series.groupby(series.index.time).transform("mean")
        filled = series.fillna(seasonal_means)
        filled = filled.interpolate(method="time", limit=limit)
        return filled.ffill().bfill()
    # fallback
    return series.ffill().bfill()


def make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "none" or kind is None:
        return None
    raise ValueError(f"Unknown scaler kind: {kind}")


def scale_series(
    series: pd.Series, scaler_kind: str = "standard", scaler=None
) -> Tuple[pd.Series, Optional[StandardScaler | MinMaxScaler]]:
    scaler = scaler or make_scaler(scaler_kind)
    if scaler is None:
        return series.copy(), None
    values = series.values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)
    scaled_series = pd.Series(scaled.flatten(), index=series.index, name=series.name)
    return scaled_series, scaler


def apply_scaler(series: pd.Series, scaler=None) -> pd.Series:
    if scaler is None:
        return series.copy()
    values = series.values.reshape(-1, 1)
    scaled = scaler.transform(values)
    return pd.Series(scaled.flatten(), index=series.index, name=series.name)
