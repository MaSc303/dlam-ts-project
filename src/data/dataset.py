import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


HF_PATH = "hf://datasets/AIML-TUDA/dlam-ts-project-data-2026/"
CACHE = Path(".cache")

def load_data(name):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        pd.read_csv(HF_PATH + name).to_csv(path, index=False)
    return pd.read_csv(path)

class ForecastDataset(Dataset):
    def __init__(self, frame, history_length=168, horizon=24):
        self.history_length = history_length
        self.horizon = horizon
        self.series = {}
        self.windows = []
        for series_id, group in frame.groupby("series_id"):
            values = torch.tensor(group.sort_values("timestamp")["target"].to_numpy(), dtype=torch.float32)
            self.series[series_id] = values
            for start in range(len(values) - history_length - horizon + 1):
                self.windows.append((series_id, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        series_id, start = self.windows[index]
        values = self.series[series_id]
        end = start + self.history_length
        return values[start:end], values[end:end + self.horizon]

def load_train_validation(history_length=168, validation_hours=24):
    df = load_data("train.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    train, val = [], []
    for _, group in df.groupby("series_id"):
        group = group.sort_values("timestamp")
        split = len(group) - validation_hours
        train.append(group.iloc[:split])
        val.append(group.iloc[split - history_length:])
    return pd.concat(train, ignore_index=True), pd.concat(val, ignore_index=True)