# Mobile Network Traffic Prediction Using LSTM

**Authors:** Ömer Fehmi Çakıcı, Yiğit Doğan Aladağ, Abdullah Sağlam  
**Date:** December 2025

---

## Abstract

This project addresses the challenge of optimizing mobile network resource allocation by accurately forecasting traffic loads. Mobile traffic data is characterized by high volatility and strong temporal seasonality, making it difficult for traditional linear models to predict effectively. Using the **Telecom Italia Milan dataset**, we developed a **Long Short-Term Memory (LSTM)** neural network model designed to capture complex temporal dependencies. The proposed model achieves a **Symmetric Mean Absolute Percentage Error (sMAPE) of 15.33%** on the test set, significantly outperforming traditional statistical baselines such as Seasonal Naive and Linear Regression. These results demonstrate that deep learning approaches are superior for capturing the non-linear "bursty" patterns of cellular traffic, offering a viable solution for predictive load balancing and energy-efficient network management.

## 1. Introduction

Efficient management of mobile network resources is critical for maintaining quality of service (QoS) and minimizing operational costs. As mobile data consumption grows, networks face the challenge of allocating bandwidth dynamically to meet fluctuating demand. Mobile traffic exhibits "bursty" behavior and strong **temporal seasonality** (e.g., daily and weekly cycles), which complicates accurate forecasting.

Traditional methods often struggle to memorize long-term dependencies while adapting to short-term fluctuations. This project proposes a deep learning approach using **Long Short-Term Memory (LSTM)** networks. LSTMs are well-suited for this task due to their ability to mitigate the vanishing gradient problem in time-series data. The primary contribution of this work is the development and evaluation of an LSTM-based forecasting model that leverages historical traffic data and temporal embeddings to predict future network load with high precision.

## 2. Methodology

### A. Data Description
We utilized the **Telecom Italia Milan dataset**, which provides telecommunications records for the city of Milan. The study focuses on high-frequency Call Detail Records (CDRs).

### B. Preprocessing
To prepare the raw data for analysis and modeling, the following steps were taken:
1.  **Aggregation:** Raw CDRs were resampled to **10-minute intervals** to allow for granular traffic analysis.
2.  **Imputation:** Missing data points were filled using Linear Interpolation.
3.  **Anomaly Suppression:** Outliers were handled using Interquartile Range (IQR) Clipping to prevent model instability.

### C. Feature Engineering
We engineered features to capture temporal dynamics:
*   **Lags:** A 24-hour historical window (144 steps) was used as input to capture daily seasonality.
*   **Time Embeddings:** Cyclical Sine/Cosine encodings were generated for "Hour" and "Day" to preserve the cyclic nature of time.

### D. Model Architecture
The proposed LSTM architecture consists of:
*   **Input Layer:** A sequence of 144 past steps, each containing traffic volume and time features.
*   **Hidden Layers:** Two stacked LSTM layers with **128 units** each.
*   **Regularization:** A **Dropout rate of 0.1** was applied after LSTM layers to prevent overfitting.
*   **Output Layer:** Predicting a 6-step horizon (the next 60 minutes) of traffic volume.
*   **Residual Hybrid Architecture (V2):** We also implemented a hybrid model that combines a **Seasonal Naive** baseline with an LSTM. The LSTM receives the **residuals** (current value - value from 24h ago) as input and predicts the future residual error. This allows the model to focus on learning complex non-seasonal deviations while relying on the solid baseline for the main pattern.

## 3. Experimental Results

### A. Experimental Setup
The model was evaluated on a chronologically split test set comprising the last 15% of the dataset. We compared the LSTM model against three baseline methods:
1.  **Naive Baseline:** Predicts the last observed value.
2.  **Seasonal Naive:** Predicts the value from the same time in the previous cycle.
3.  **Linear Regression:** A standard linear approach.

### B. Performance Evaluation
Performance was measured using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Symmetric Mean Absolute Percentage Error (sMAPE). Lower values indicate better performance.

| Model | MAE | RMSE | sMAPE (%) |
| :--- | :---: | :---: | :---: |
| **LSTM (Proposed)** | **2.14** | **3.53** | **15.33%** |
| Naive Baseline | 2.42 | 3.99 | 16.07% |
| Seasonal Naive | 2.52 | 4.06 | 17.87% |
| Linear Regression | 2.54 | 3.95 | 19.47% |
| **Residual Hybrid (V2)** | **2.31** | **3.88** | **15.71%** |

### C. Discussion
The LSTM model demonstrated superior performance across all metrics. Key observations include:
1.  **Accuracy:** The LSTM model reduced the sMAPE by approximately **4.7%** relative to the Naive baseline.
2.  **Pattern Recognition:** The model successfully learned critical temporal patterns, such as the "morning peak" and "night trough," which are essential for capacity scaling.
3.  **Generalization:** Dropout regularization effectively prevented overfitting, ensuring stable performance on unseen data.
4.  **Hybrid Model Performance:** The Residual Hybrid model (MAE 2.31) significantly outperformed traditional statistical baselines (MAE 2.52) by learning to correct their errors. However, the pure LSTM (MAE 2.14) still achieved the best overall performance, likely due to its flexibility in learning end-to-end representations without being constrained by a fixed seasonal assumption.

## 4. Conclusion

This project demonstrates that LSTM networks are a highly effective solution for cellular traffic forecasting. The model outperforms traditional heuristics by capturing complex non-linear dependencies in the data. The achieved accuracy supports practical applications in **predictive load balancing** and **energy-efficient network management**. Future work could explore attention mechanisms or transformer-based architectures to further improve long-term prediction capabilities.

## 5. References

[1] Telecom Italia, "Big Data Challenge 2013 Dataset," Milan, Italy.
[2] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
