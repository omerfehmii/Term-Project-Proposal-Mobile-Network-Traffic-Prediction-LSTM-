import pandas as pd
import numpy as np
import os

def generate_dummy_data(path="data/milan.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Generate 1 month of 10-min data
    dates = pd.date_range(start="2023-01-01", periods=24*6*30, freq="10min")
    
    # Create a synthetic signal: Trend + Seasonality + Noise
    t = np.arange(len(dates))
    traffic = 100 + 0.01 * t + 50 * np.sin(2 * np.pi * t / (24*6)) + np.random.normal(0, 5, len(dates))
    
    df = pd.DataFrame({
        "timestamp": dates,
        "cell_id": 1,
        "traffic": traffic
    })
    
    df.to_csv(path, index=False)
    print(f"Generated dummy data at {path}")

if __name__ == "__main__":
    generate_dummy_data()
