# Mobile Network Traffic Prediction - Technical Report

**Team:** Ömer Fehmi Çakıcı, Yiğit Doğan Aladağ, Abdullah Sağlam
**Course:** CSE 476/575

## 1. Introduction
Mobile network traffic forecasting is crucial for capacity planning and load balancing. This project implements a Long Short-Term Memory (LSTM) neural network to predict traffic volume for Milan grid cells.

## 2. Methodology

### 2.1 Data Preprocessing
- **Source:** Telecom Italia Milan dataset (Nov-Dec 2013).
- **Resampling:** Aggregated to 10-minute intervals.
- **Imputation:** Linear interpolation for missing values (limit=6).
- **Outliers:** Clipped using IQR method (factor=3.0).
- **Scaling:** Standard scaling (zero mean, unit variance).

### 2.2 Feature Engineering
- **Time Features:** Sin/Cos encoding for Hour-of-Day and Day-of-Week.
- **Lags:** Past traffic values (Window size = 144 steps = 24 hours).

### 2.3 Model Architecture
- **Type:** LSTM (Long Short-Term Memory).
- **Structure:**
    - Input Layer: Features (Traffic + Time encodings).
    - Hidden Layers: 2 layers of 128 units.
    - Dropout: 0.1.
    - Output Layer: Linear projection to horizon size (6 steps = 1 hour).
- **Training:**
    - Loss: MSE.
    - Optimizer: Adam (lr=0.001).
    - Early Stopping: Patience=10.

### 2.4 Baselines
- **Naive:** Last observed value.
- **Seasonal Naive:** Value from 24 hours ago.
- **Linear:** Linear regression on lag features.

## 3. Experimental Setup
- **Split:** Chronological (70% Train, 15% Val, 15% Test).
- **Evaluation:** Rolling-origin evaluation (optional) or single split.
- **Metrics:** MAE, RMSE, sMAPE.

## 4. Results
*(To be populated after full experiment run)*

### 4.1 Performance Comparison
| Model | MAE | RMSE | sMAPE |
|-------|-----|------|-------|
| LSTM | **2.14** | **3.53** | **15.33** |
| Naive (Last) | 2.42 | 3.99 | 16.07 |
| Seasonal Naive | 2.52 | 4.06 | 17.87 |
| Linear | 2.54 | 3.95 | 19.47 |

### 4.2 Visualizations
Forecast plots and error distributions are available in the `outputs/main_run/` directory.
- `forecast.png`: Shows the prediction vs ground truth for the first test window.
- `error_hist.png`: Shows the distribution of prediction errors.

## 5. Conclusion
The LSTM model demonstrates superior performance compared to all baselines (Naive, Seasonal Naive, and Linear Regression) across all metrics (MAE, RMSE, sMAPE). This indicates that the neural network successfully captures the complex temporal patterns in the mobile traffic data better than simple heuristic or linear models.
