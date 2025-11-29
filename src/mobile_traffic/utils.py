import os
import random
import shutil
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto mode
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_run_dir(base_dir: str, run_name: str | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = run_name or f"run-{stamp}"
    path = os.path.join(base_dir, name)
    ensure_dir(path)
    return path


def save_config_copy(run_dir: str, config_path: str | None, resolved_config: Dict) -> None:
    ensure_dir(run_dir)
    dest = os.path.join(run_dir, "config.yaml")
    try:
        import yaml
    except Exception:
        return
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False)
    if config_path and os.path.exists(config_path):
        # Keep original for traceability
        shutil.copy(config_path, os.path.join(run_dir, "config_source.yaml"))


# Metrics
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


@dataclass
class Metrics:
    mae: float
    rmse: float
    smape: float

    @classmethod
    def from_arrays(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "Metrics":
        return cls(mae=mae(y_true, y_pred), rmse=rmse(y_true, y_pred), smape=smape(y_true, y_pred))

    def to_dict(self) -> Dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "smape": self.smape}


def plot_forecast(
    timestamps,
    truth: np.ndarray,
    preds: np.ndarray,
    horizon: int,
    title: str,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, truth, label="Truth", color="black", linewidth=1.5)
    plt.plot(timestamps, preds, label="Forecast", color="tab:blue", linewidth=1.2)
    if horizon > 1:
        plt.axvline(timestamps[-horizon], color="gray", linestyle="--", linewidth=1, label="Forecast start")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Traffic")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_error_hist(errors: np.ndarray, title: str, save_path: str | None = None) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=40, color="tab:orange", alpha=0.8)
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()


def to_numpy(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr
