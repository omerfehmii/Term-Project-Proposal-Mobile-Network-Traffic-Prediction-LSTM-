import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    # Paths
    hybrid_path = "outputs/hybrid_test_run/predictions.csv"
    lstm_path = "outputs/lstm_comparison/predictions.csv"
    
    # Load data
    df_hybrid = pd.read_csv(hybrid_path)
    df_lstm = pd.read_csv(lstm_path)
    
    # Ensure time columns are datetime
    df_hybrid['target_time'] = pd.to_datetime(df_hybrid['target_time'])
    df_lstm['target_time'] = pd.to_datetime(df_lstm['target_time'])
    
    # Merge on target_time
    # Both files have 'target_time', 'y_true_t+1', 'y_pred_t+1', etc.
    # We will focus on 1-step ahead forecast for clarity (t+1)
    
    merged = pd.merge(
        df_hybrid[['target_time', 'y_true_t+1', 'y_pred_t+1']], 
        df_lstm[['target_time', 'y_pred_t+1']], 
        on='target_time', 
        suffixes=('_hybrid', '_lstm')
    )
    
    # Sort
    merged = merged.sort_values('target_time')
    
    # Select a window to plot (e.g., first 3 days -> 144 * 3 = 432 steps)
    subset = merged.iloc[:432]
    
    plt.figure(figsize=(14, 7))
    plt.plot(subset['target_time'], subset['y_true_t+1'], label='Ground Truth', color='black', alpha=0.6, linewidth=2)
    plt.plot(subset['target_time'], subset['y_pred_t+1_hybrid'], label='Hybrid V2 (Naive+LSTM)', color='blue', alpha=0.8)
    plt.plot(subset['target_time'], subset['y_pred_t+1_lstm'], label='Standard LSTM', color='red', linestyle='--', alpha=0.8)
    
    plt.title('Model Comparison: Traffic Prediction (Next 10min)', fontsize=16)
    plt.ylabel('Traffic Volume', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    out_path = "plots/model_comparison.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path)
    print(f"Comparison plot saved to {out_path}")
    
    # Metric Comparison Bar Chart
    # Values from our experiment
    models = ['Standard LSTM', 'Hybrid V2', 'Seasonal Naive', 'Last Value']
    mae_values = [1.99, 2.31, 2.52, 2.42]
    colors = ['green', 'blue', 'orange', 'gray']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, mae_values, color=colors)
    plt.title('Model MAE Comparison (Lower is Better)', fontsize=16)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=11)
        
    out_path_bar = "plots/mae_comparison.png"
    plt.savefig(out_path_bar)
    print(f"Metric bar chart saved to {out_path_bar}")

if __name__ == "__main__":
    plot_comparison()
