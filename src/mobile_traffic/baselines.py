from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression

from .utils import Metrics


def _make_last_value_preds(X: np.ndarray, horizon: int) -> np.ndarray:
    last_vals = X[:, -1, 0]
    return np.repeat(last_vals[:, None], horizon, axis=1)


def _make_seasonal_preds(X: np.ndarray, horizon: int, lag: int) -> np.ndarray:
    preds = []
    for sample in X:
        if lag <= sample.shape[0]:
            val = sample[-lag, 0]
        else:
            val = sample[-1, 0]
        preds.append(np.repeat(val, horizon))
    return np.stack(preds) if preds else np.empty((0, horizon))


def _lag_matrix(X: np.ndarray, lags: int) -> np.ndarray:
    feats = []
    for sample in X:
        values = sample[:, 0]
        if len(values) >= lags:
            lvec = values[-lags:]
        else:
            # pad the front with the first value to keep length consistent
            pad = np.full(lags - len(values), values[0])
            lvec = np.concatenate([pad, values])
        feats.append(lvec)
    return np.stack(feats) if feats else np.empty((0, lags))


def run_baselines(
    baseline_cfg: Dict,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    horizon: int,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    if baseline_cfg.get("last", True):
        preds = _make_last_value_preds(X_test, horizon)
        results["last"] = {"metrics": Metrics.from_arrays(y_test, preds), "preds": preds}

    if baseline_cfg.get("seasonal_naive", True):
        lag = int(baseline_cfg.get("seasonal_lag", horizon))
        preds = _make_seasonal_preds(X_test, horizon, lag)
        results["seasonal_naive"] = {"metrics": Metrics.from_arrays(y_test, preds), "preds": preds}

    linear_lags = int(baseline_cfg.get("linear_lags", 0))
    if linear_lags > 0 and len(X_train) > 0:
        Xtr = _lag_matrix(X_train, linear_lags)
        Xte = _lag_matrix(X_test, linear_lags)
        model = LinearRegression()
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        results["linear"] = {"metrics": Metrics.from_arrays(y_test, preds), "preds": preds}

    return results
