# Mobile Network Traffic Prediction
## Advanced Time-Series Forecasting with LSTM

**Team:** Ömer Fehmi Çakıcı, Yiğit Doğan Aladağ, Abdullah Sağlam
**Date:** December 2025

---

# 1. Project Overview

### The Challenge
*   **Capacity Planning:** Over-provisioning wastes energy; under-provisioning causes congestion.
*   **Data Complexity:** Mobile traffic is non-stationary, bursty, and spatially uneven.
*   **Objective:** Develop a robust deep learning model to forecast internet load for the next **60 minutes** at a cell level.

### The Solution
A **Long Short-Term Memory (LSTM)** neural network that learns historical patterns to predict future demand with high accuracy.

---

# 2. Dataset & Preprocessing

**Source:** Telecom Italia Milan (Big Data Challenge)
**Volume:** 2 Months of Call Detail Records (CDRs)

### Data Engineering Pipeline
1.  **Aggregation:** Raw logs aggregated to **10-minute bins** for high-resolution forecasting.
2.  **Imputation:** Filling short signal gaps (max 1 hour) using Linear Interpolation.
3.  **Outlier Removal:** Clipping extreme spikes (above 3x IQR) to stabilize training.
4.  **Scaling:** Standard Scaler ($z = \frac{x - \mu}{\sigma}$) to normalize traffic volume.

---

# 3. Model Architecture

We designed a custom sequence-to-sequence regressor.

### Architecture Details
*   **Input Layer:**
    *   **Traffic Lags:** Past 144 steps (24 hours history).
    *   **Time Embeddings:** Sin/Cos transformations of "Hour" and "Day" to enforce cyclical seasonality.
*   **Hidden Layers:**
    *   **Layer 1:** 128 LSTM units (captures long-term dependencies).
    *   **Layer 2:** 128 LSTM units (refines feature abstraction).
    *   **Dropout (0.1):** Applied between layers to prevent overfitting.
*   **Output Layer:** Dense linear projection to a 6-step vector (1 hour horizon).

---

# 4. Training Strategy

### Hyperparameters
*   **Optimizer:** Adam (Adaptive Moment Estimation) for fast convergence.
*   **Loss Function:** Mean Squared Error (MSE).
*   **Learning Rate:** 0.001 with dynamic reduction on plateau.
*   **Early Stopping:** Monitoring validation loss with a patience of 10 epochs.

### Comparison Models (Baselines)
To prove the value of Deep Learning, we compared against:
1.  **Naive Persistence:** "Next value = Last value".
2.  **Seasonal Naive:** "Next value = Value from 24h ago".
3.  **Linear Regression:** "Next value = Weighted sum of past 24h".

---

# 5. Experimental Results

### Quantitative Metrics
Evaluated on the final 15% of the timeline (Chronological Split).

| Model Strategy | MAE | RMSE | sMAPE (Error %) |
| :--- | :---: | :---: | :---: |
| **LSTM (Deep Learning)** | **2.14** | **3.53** | **15.33%** |
| Naive Approach | 2.42 | 3.99 | 16.07% |
| Seasonal Naive | 2.52 | 4.06 | 17.87% |
| Linear Regression | 2.54 | 3.95 | 19.47% |

### Key Takeaway
The **LSTM reduced error by ~4.7%** compared to the naive baseline and **~21%** compared to Linear Regression.

---

# 6. Performance Insights

### Why LSTM Won?
*   **Memory:** Successfully remembered the daily cycle (recurring morning peaks).
*   **Filtering:** Ignored random noise better than the Linear model.
*   **Stability:** Maintained accuracy even during weekends when traffic patterns shift.

### Visual Confirmation
*(Placeholder for Forecast Graph)*
*   **Blue Line (Actual):** Shows sharp, erratic spikes.
*   **Orange Line (Predicted):** Smooths the noise while hitting the major peaks accurately.

---

# 7. Business Impact & Future Work

### Real-World Application
*   **Dynamic Slicing:** Allocate 5G slices based on predicted load rather than static rules.
*   **Energy Saving:** Put base stations into "sleep mode" during predicted low-traffic nights.

### Roadmap
1.  **Spatial Correlation:** incorporating traffic from neighboring cells (Graph Neural Networks).
2.  **Weather Data:** Integrating external factors (rain/events) to improve anomaly prediction.
3.  **Transformer Models:** Testing Attention mechanisms for longer forecast horizons (e.g., 24 hours ahead).

---

# Thank You
## Q&A
