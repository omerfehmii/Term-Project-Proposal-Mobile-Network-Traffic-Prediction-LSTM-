from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from .baselines import run_baselines
from .data import apply_scaler, clip_outliers, impute_missing, load_series, make_scaler
from .features import build_windows
from .models.lstm import GRURegressor, LSTMRegressor, SequenceDataset, evaluate, make_loaders, train_model
from .models.hybrid import ResidualLSTMRegressor
from .utils import Metrics, ensure_dir, make_run_dir, plot_error_hist, plot_forecast, save_config_copy, select_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM for mobile traffic forecasting.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path in config.")
    parser.add_argument("--cell_id", type=str, default=None, help="Override cell id selection.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_boundaries(index: pd.DatetimeIndex, split_cfg: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    n = len(index)
    train_ratio = split_cfg.get("train", 0.7)
    val_ratio = split_cfg.get("val", 0.15)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)
    train_boundary = index[train_end - 1]
    val_boundary = index[val_end - 1]
    return train_boundary, val_boundary


def split_by_time(X: np.ndarray, y: np.ndarray, times: pd.DatetimeIndex, train_boundary, val_boundary):
    train_mask = times <= train_boundary
    val_mask = (times > train_boundary) & (times <= val_boundary)
    test_mask = times > val_boundary
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    times_train, times_val, times_test = times[train_mask], times[val_mask], times[test_mask]
    return (X_train, y_train, times_train), (X_val, y_val, times_val), (X_test, y_test, times_test)


def inverse_transform(preds: np.ndarray, scaler, log_transform: bool = False) -> np.ndarray:
    out = preds
    if scaler is not None:
        flat = preds.reshape(-1, 1)
        inv = scaler.inverse_transform(flat)
        out = inv.reshape(preds.shape)
    if log_transform:
        out = np.expm1(out)
    return out


def maybe_rebalance_val(X_train, y_train, X_val, y_val):
    if len(X_val) == 0 and len(X_train) > 2:
        split_at = max(1, int(len(X_train) * 0.8))
        X_val = X_train[split_at:]
        y_val = y_train[split_at:]
        X_train = X_train[:split_at]
    return X_train, y_train, X_val, y_val


def prepare_series(config: Dict, args) -> pd.Series:
    data_cfg = config["data"]
    series = load_series(
        path=args.data_path or data_cfg["path"],
        time_col=data_cfg.get("time_col", "timestamp"),
        value_col=data_cfg.get("value_col", "traffic"),
        cell_col=data_cfg.get("cell_col"),
        cell_id=args.cell_id or data_cfg.get("cell_id"),
        freq=data_cfg.get("freq"),
        tz=data_cfg.get("tz"),
        resample_agg=data_cfg.get("resample_agg", "sum"),
    )
    if data_cfg.get("outlier", {}).get("clip", False):
        factor = float(data_cfg["outlier"].get("iqr_factor", 3.0))
        series = clip_outliers(series, iqr_factor=factor)
    imp_cfg = data_cfg.get("imputation", {})
    series = impute_missing(
        series,
        method=imp_cfg.get("method", "linear"),
        limit=int(imp_cfg.get("limit", 6)),
        seasonal_periods=imp_cfg.get("seasonal_periods"),
    )
    return series


def run_single_split(series: pd.Series, config: Dict, run_dir: str, args) -> Dict:
    split_cfg = config.get("split", {})
    window_cfg = config.get("window", {})
    model_cfg = config.get("model", {})
    baseline_cfg = config.get("baselines", {})
    logging_cfg = config.get("logging", {})
    data_cfg = config.get("data", {})
    log_transform = bool(data_cfg.get("log_transform", False))

    train_boundary, val_boundary = compute_boundaries(series.index, split_cfg)

    # Fit scaler on train partition only
    scaler_kind = data_cfg.get("scaler", "standard")
    scaler = make_scaler(scaler_kind)
    scaled_series = series
    if log_transform:
        scaled_series = np.log1p(scaled_series)
    if scaler is not None:
        train_end_idx = (scaled_series.index <= train_boundary).sum()
        scaler.fit(scaled_series.iloc[:train_end_idx].values.reshape(-1, 1))
        scaled_values = scaler.transform(scaled_series.values.reshape(-1, 1)).flatten()
        scaled_series = pd.Series(scaled_values, index=scaled_series.index, name=series.name)

    window = int(window_cfg.get("size", 48))
    horizon = int(window_cfg.get("horizon", 6))
    stride = int(window_cfg.get("stride", 1))
    use_time_features = bool(window_cfg.get("time_features", True))
    add_weekend = bool(window_cfg.get("add_weekend", True))


    
    # For Residual LSTM: Inject lagged series as covariate
    model_type = model_cfg.get("type", "lstm").lower()
    extra_covariates = None
    if model_type == "residual_lstm":
        # Create a shifted series aligned with the index
        seasonal_lag = int(model_cfg.get("seasonal_lag", 24))
        # shift(lag) moves data forward, so index T has value from T-lag
        lagged_series = scaled_series.shift(seasonal_lag).bfill() # bfill to handle start
        # Lagged series must be a DataFrame
        extra_covariates = lagged_series.to_frame(name="lagged_value")

    # Build windows for scaled and raw series
    X_scaled, y_scaled, target_times = build_windows(
        scaled_series,
        window=window,
        horizon=horizon,
        stride=stride,
        use_time_features=use_time_features,
        add_weekend=add_weekend,
        extra_covariates=extra_covariates,
    )
    X_raw, y_raw, _ = build_windows(
        series,
        window=window,
        horizon=horizon,
        stride=stride,
        use_time_features=use_time_features,
        add_weekend=add_weekend,
        # We don't strictly need covariates for raw X unless measuring raw residuals, 
        # but let's keep it consistent if needed later. For now, X_raw is used for splitting.
    )
    if len(X_scaled) == 0:
        raise ValueError("Not enough data to build any training windows. Adjust window/horizon or provide more data.")

    (Xtr, ytr, ttr), (Xv, yv, tv), (Xte, yte, tte) = split_by_time(
        X_scaled, y_scaled, target_times, train_boundary, val_boundary
    )
    (Xtr_raw, ytr_raw, _), (Xv_raw, yv_raw, _), (Xte_raw, yte_raw, _) = split_by_time(
        X_raw, y_raw, target_times, train_boundary, val_boundary
    )

    Xtr, ytr, Xv, yv = maybe_rebalance_val(Xtr, ytr, Xv, yv)
    if len(Xtr) == 0 or len(Xv) == 0 or len(Xte) == 0:
        raise ValueError("Train/val/test splits are empty. Check split ratios and data length.")

    batch_size = int(model_cfg.get("batch_size", 64))
    train_loader, val_loader = make_loaders(Xtr, ytr, Xv, yv, batch_size=batch_size)
    device = select_device(model_cfg.get("device", "auto"))
    model_type = model_cfg.get("type", "lstm").lower()
    input_size = Xtr.shape[2]
    hidden_size = int(model_cfg.get("hidden_size", 64))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.2))
    if model_type == "gru":
        model = GRURegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
        )
    elif model_type == "residual_lstm":
        seasonal_lag = int(model_cfg.get("seasonal_lag", 24))
        model = ResidualLSTMRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            seasonal_lag=seasonal_lag,
        )
    else:
        model = LSTMRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
        )
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(model_cfg.get("epochs", 50)),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
        grad_clip=float(model_cfg.get("grad_clip", 1.0)),
        patience=int(model_cfg.get("patience", 5)),
    )

    # Evaluate
    test_loader = torch.utils.data.DataLoader(
        SequenceDataset(Xte, yte), batch_size=batch_size, shuffle=False, drop_last=False
    )
    preds_scaled, truths_scaled, test_loss = evaluate(model, test_loader, device)
    preds = inverse_transform(preds_scaled, scaler, log_transform=log_transform)
    truths = yte_raw  # already in original scale

    metrics = Metrics.from_arrays(truths, preds)

    # Baselines
    baseline_results = run_baselines(
        baseline_cfg,
        splits={"train": (Xtr_raw, ytr_raw), "test": (Xte_raw, yte_raw)},
        horizon=horizon,
    )

    # Save artifacts
    ensure_dir(run_dir)
    hist_df = pd.DataFrame({"train_loss": history["train_loss"], "val_loss": history["val_loss"]})
    hist_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)

    pred_cols = ["target_time"] + [f"y_true_t+{i+1}" for i in range(horizon)] + [
        f"y_pred_t+{i+1}" for i in range(horizon)
    ]
    pred_data = np.concatenate([truths, preds], axis=1)
    pred_df = pd.DataFrame(pred_data, columns=[c for c in pred_cols if c != "target_time"])
    pred_df.insert(0, "target_time", tte)
    pred_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    model_key = model_type
    metrics_payload = {model_key: metrics.to_dict()}
    for name, res in baseline_results.items():
        metrics_payload[name] = res["metrics"].to_dict()
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    if logging_cfg.get("save_plots", True) and len(tte) > 0:
        # Plot first forecast window in test set for a quick visual
        first_truth = truths[0]
        first_pred = preds[0]
        horizon_times = pd.date_range(end=tte[0], periods=horizon, freq=config["data"].get("freq", "H"))
        plot_forecast(
            horizon_times,
            truth=first_truth,
            preds=first_pred,
            horizon=horizon,
            title="First test forecast",
            save_path=os.path.join(run_dir, "forecast.png"),
        )
        flat_err = (preds - truths).flatten()
        plot_error_hist(flat_err, title="Forecast errors", save_path=os.path.join(run_dir, "error_hist.png"))

    print(f"Test metrics ({model_key}):", metrics_payload[model_key])
    for name, res in baseline_results.items():
        print(f"Test metrics ({name}): {res['metrics'].to_dict()}")

    return {"metrics": metrics_payload, "history": history}


def run():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    series = prepare_series(config, args)
    if series.empty:
        raise ValueError("Loaded series is empty. Check data path and filters.")

    logging_cfg = config.get("logging", {})
    run_dir = make_run_dir(logging_cfg.get("output_dir", "outputs"), logging_cfg.get("run_name"))
    save_config_copy(run_dir, args.config, config)

    rolling_cfg = config.get("rolling", {})
    if rolling_cfg.get("enabled", False):
        holdout_frac = float(rolling_cfg.get("holdout", 0.15))
        step_frac = float(rolling_cfg.get("step", 0.05))
        folds = int(rolling_cfg.get("folds", 3))
        n = len(series)
        step_n = max(1, int(n * step_frac))
        metrics_list = []
        for fold in range(folds):
            end = n - fold * step_n
            if end <= 0:
                break
            subset = series.iloc[:end]
            fold_dir = os.path.join(run_dir, f"fold_{fold + 1}")
            res = run_single_split(subset, config, fold_dir, args)
            metrics_list.append(res["metrics"])
        if metrics_list:
            def _aggregate(metric_payloads):
                agg = {}
                model_names = set().union(*[m.keys() for m in metric_payloads])
                for model_name in model_names:
                    keys = set().union(*[m.get(model_name, {}).keys() for m in metric_payloads])
                    agg[model_name] = {
                        k: float(
                            np.nanmean(
                                [m.get(model_name, {}).get(k, np.nan) for m in metric_payloads if k in m.get(model_name, {})]
                            )
                        )
                        for k in keys
                    }
                return agg

            aggregated = _aggregate(metrics_list)
            with open(os.path.join(run_dir, "rolling_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"folds": metrics_list, "mean": aggregated}, f, indent=2)
        return {"rolling_metrics": metrics_list}
    else:
        result = run_single_split(series, config, run_dir, args)
        return result


if __name__ == "__main__":
    run()
