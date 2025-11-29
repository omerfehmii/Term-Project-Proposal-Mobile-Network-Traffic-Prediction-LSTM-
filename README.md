Mobile Network Traffic Forecasting (LSTM)
========================================

This repo provides a reproducible scaffold to forecast mobile network traffic for one or more Milan grid cells using time-series baselines and an LSTM regressor. It follows the proposal plan (preprocessing, supervised windowing, baselines, neural model, and rolling evaluation).

What’s here
-----------
- Config-driven pipeline: `configs/default.yaml` captures data paths, preprocessing, window sizes, and model hyperparameters.
- Data prep: imputation, outlier clipping, scaling, time encodings, and sliding-window dataset creation.
- Baselines: Last value, seasonal naive, and a light linear regression on lags.
- Neural model: PyTorch LSTM/GRU with dropout, gradient clipping, early stopping, and validation-based checkpointing.
- Evaluation: Chronological train/val/test split with MAE, RMSE, and sMAPE; optional rolling-origin evaluation.

Getting started
---------------
1) Create an environment (Python ≥3.10 recommended):
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Prepare data: Place a CSV at `data/milan.csv` (or set `data_path` in the config) with columns:
- `timestamp`: UTC or local time in ISO format
- `cell_id`: grid/cell identifier
- `traffic`: numeric traffic volume
If multiple cells are present, set `cell_id` in the config or pass `--cell_id` on the CLI. Resampling to a consistent frequency (e.g., 10/15/30/60 min) is handled by `freq` in the config.

If you already downloaded the Telecom Italia Milan daily CSVs into `archive/`, you can build `data/milan.csv` with:
```
export PYTHONPATH=src
PYTHONPATH=src python -m mobile_traffic.prepare_archive --archive_dir archive --output data/milan.csv --metric internet
```
`--metric internet` uses the Internet traffic column; `--metric all` sums SMS/call/internet for a total load signal.

3) Run an experiment (uses `configs/default.yaml`):
```
export PYTHONPATH=src
PYTHONPATH=src python -m mobile_traffic.train --config configs/default.yaml --data_path data/milan.csv --cell_id 123
```
Outputs (metrics, plots, and predictions) are saved under `outputs/<run_id>/`.

Outputs
-------
- `metrics.json`: MAE/RMSE/sMAPE for LSTM and baselines; `rolling_metrics.json` when rolling is enabled.
- `history.csv`: train/val loss per epoch.
- `predictions.csv`: timestamps with true vs. predicted horizons.
- `forecast.png` and `error_hist.png`: quick-look plots for the first test window.

Key configuration knobs
-----------------------
- Data: `freq`, `imputation` (linear/seasonal), `outlier_clip` (IQR factor), and `scaler` (standard/minmax).
- Windowing: `window` (W), `horizon` (H), `stride`, and `time_features` (sin/cos for hour/day, weekend flag).
- Baselines: enable/disable seasonal naive and linear regression on lags.
- Model: hidden size, layers, dropout, batch size, learning rate, weight decay, gradient clip, patience, and device selection.

Defaults are set for the Telecom Italia Milan data (`freq: 10min`, `seasonal_lag: 144` for 24h seasonality).
Use GRU instead of LSTM by setting `model.type: gru` in the config.

Variants
--------
- `configs/default.yaml`: LSTM, 24h window, 1h horizon (6 steps @10min), linear scaling.
- `configs/gru.yaml`: GRU with same window/horizon.
- `configs/h3_log.yaml`: GRU, 48h window, 30min horizon (3 steps), log1p-transform + scaling for variance stabilization.

Rolling-origin evaluation (optional)
------------------------------------
Enable `rolling.enabled: true` to run repeated train/val/test folds on progressively earlier cut-offs. `rolling.step` controls how far back each fold moves (as a fraction of the full series); metrics are stored per fold and averaged in `rolling_metrics.json`.

Next steps / customizations
---------------------------
- Add auxiliary covariates (neighboring cells, holidays) by extending `mobile_traffic.features.build_feature_matrix`.
- Swap the LSTM for GRU or Temporal Convolution by adding a new model class in `mobile_traffic/models/`.
- Include SARIMA/Prophet by adding to `mobile_traffic.baselines` if you want classical references.

Reproducibility
---------------
Seeds are fixed across NumPy, PyTorch, and Python for deterministic runs where possible. Configs are saved with each run under `outputs/<run_id>/config.yaml`.

Project Documentation
---------------------
- [Technical Report](docs/report.md)
- [Presentation Slides](docs/slides.md)
