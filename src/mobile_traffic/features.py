from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def time_encodings(index: pd.DatetimeIndex, add_weekend: bool = True) -> pd.DataFrame:
    hours = index.hour + index.minute / 60.0
    hour_rad = 2 * np.pi * hours / 24.0
    dow = index.dayofweek
    dow_rad = 2 * np.pi * dow / 7.0
    data = {
        "hour_sin": np.sin(hour_rad),
        "hour_cos": np.cos(hour_rad),
        "dow_sin": np.sin(dow_rad),
        "dow_cos": np.cos(dow_rad),
    }
    if add_weekend:
        data["is_weekend"] = (dow >= 5).astype(float)
    return pd.DataFrame(data, index=index)


def build_windows(
    series: pd.Series,
    window: int,
    horizon: int,
    stride: int = 1,
    use_time_features: bool = True,
    add_weekend: bool = True,
    extra_covariates: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    values = series.values
    idx = series.index
    n = len(series)
    if use_time_features:
        tf = time_encodings(idx, add_weekend=add_weekend).values
    else:
        tf = None
    extra = None
    if extra_covariates is not None:
        # Align on index
        extra_covariates = extra_covariates.reindex(idx)
        extra = extra_covariates.values

    X_list, y_list, target_times = [], [], []
    max_start = n - window - horizon + 1
    for start in range(0, max_start, stride):
        end = start + window
        tgt_end = end + horizon
        base = values[start:end].reshape(window, 1)
        parts = [base]
        if tf is not None:
            parts.append(tf[start:end])
        if extra is not None:
            parts.append(extra[start:end])
        X_list.append(np.concatenate(parts, axis=1))
        y_list.append(values[end:tgt_end])
        target_times.append(idx[tgt_end - 1])
    X = np.stack(X_list) if X_list else np.empty((0, window, 0))
    y = np.stack(y_list) if y_list else np.empty((0, horizon))
    return X, y, pd.DatetimeIndex(target_times)
