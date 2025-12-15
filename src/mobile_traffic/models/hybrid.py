from __future__ import annotations

import torch
from torch import nn


class ResidualLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        seasonal_lag: int = 24,
    ):
        """
        A hybrid model that sums a Seasonal Naive prediction with an LSTM-predicted residual.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            horizon: Forecasting horizon
            seasonal_lag: The lag to use for the naive component (e.g., 24 for daily pattern if freq=1H)
        """
        super().__init__()
        self.seasonal_lag = seasonal_lag
        self.horizon = horizon
        
        # Adjusted input size: we perform (Feature 0 - Feature -1) and drop Feature -1
        # So we lose 1 dimension overall (Traffic and Lagged merge into Residual)
        lstm_input_size = input_size - 1
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, window_size, input_size)
        Input is expected to have: [Traffic, ..., LaggedValue]
        """
        # 0. Pre-processing: Convert Raw Input to Residual Input
        # Traffic is at index 0, Lagged is at index -1
        val = x[..., 0]
        lag = x[..., -1]
        residual = val - lag
        
        # Other features are in between (Time features etc)
        # Note: If input_size was 2 (Traffic, Lagged), 1:-1 is empty. Correct.
        others = x[..., 1:-1]
        
        # Concatenate residual with others
        # residual shape is (B, W), need (B, W, 1)
        lstm_input = torch.cat([residual.unsqueeze(-1), others], dim=-1)

        # 1. Compute Naive Prediction (Component A)
        # We assume the last dimension index 0 is the target variable 'traffic'
        # We want to pick values from x that correspond to 'lag' steps ago relative to the target horizon.
        
        batch_size, window_size, _ = x.shape
        
        # Validation: check if window is large enough
        if window_size < self.seasonal_lag:
            # Fallback: if window is too short, just use the last available value (persistent naive)
            # or maybe zeros? Let's use last value to be safe, though this defeats the "seasonal" purpose.
            # Ideally the user sets window_size > seasonal_lag.
            naive_base = x[:, -1, 0].unsqueeze(-1).repeat(1, self.horizon)
        else:
            # We want prediction for T+1 ... T+H
            # Naive predictor is T+1-S ... T+H-S
            # Index of T is (window_size - 1).
            # Index of T+1-S is (window_size - S).
            
            start_idx = window_size - self.seasonal_lag
            end_idx = start_idx + self.horizon
            
            if start_idx < 0:
                 naive_base = x[:, -1, 0].unsqueeze(-1).repeat(1, self.horizon)
            elif end_idx > window_size:
                 indices = torch.arange(start_idx, end_idx, device=x.device)
                 indices = torch.clamp(indices, max=window_size-1)
                 naive_base = x[:, indices, 0] 
            else:
                 naive_base = x[:, start_idx:end_idx, 0]

        # 2. Compute Residual Prediction (Component B)
        # Feed the RESIDUAL input to the LSTM
        _, (hn, _) = self.lstm(lstm_input)
        last_hidden = hn[-1]
        residual_pred = self.head(last_hidden) 
        
        # 3. Combine
        return naive_base + residual_pred
