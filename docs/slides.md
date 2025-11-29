# Mobile Network Traffic Prediction
## CSE 476/575 Term Project

**Team:** Ömer Fehmi Çakıcı, Yiğit Doğan Aladağ, Abdullah Sağlam

---

## Problem Statement

**Goal:** Forecast mobile network traffic for Milan grid cells.
**Why?** Capacity planning, load balancing, QoS control.
**Challenges:**
- Strong diurnal/weekly seasonality.
- Abrupt peaks.
- Spatial heterogeneity.

---

## Methodology

**Data:** Telecom Italia Milan dataset (Nov-Dec 2013).
**Preprocessing:**
- 10-min aggregation.
- Linear imputation (limit=6).
- IQR outlier clipping.
- Standard scaling.

**Features:**
- Lagged traffic values (24h window).
- Time encodings (Sin/Cos for Hour/Day).

---

## Model Architecture

**LSTM Regressor:**
- **Input:** Traffic lags + Time features.
- **Hidden:** 2 layers x 128 units.
- **Dropout:** 0.1 for regularization.
- **Output:** Linear projection to 6-step horizon (1 hour).
- **Training:** Adam optimizer, MSE loss, Early stopping.

---

## Baselines

1. **Naive (Last):** Predicts the last observed value.
2. **Seasonal Naive:** Predicts the value from 24 hours ago.
3. **Linear Regression:** Simple autoregressive model on lags.

---

## Experimental Setup

- **Data Split:** Chronological (70% Train, 15% Val, 15% Test).
- **Metrics:**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - sMAPE (Symmetric Mean Absolute Percentage Error)

---

## Results

| Model | MAE | RMSE | sMAPE |
|-------|-----|------|-------|
| **LSTM** | **2.14** | **3.53** | **15.33** |
| Naive | 2.42 | 3.99 | 16.07 |
| Seasonal Naive | 2.52 | 4.06 | 17.87 |
| Linear | 2.54 | 3.95 | 19.47 |

*LSTM outperforms all baselines.*

---

## Conclusion

- LSTM successfully captures complex temporal dynamics.
- Significant improvement over naive and linear baselines.
- Future work: Incorporate spatial dependencies (neighboring cells) and longer horizons.
