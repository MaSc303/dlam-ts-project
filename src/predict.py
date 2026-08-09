import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch

HF_PATH = "hf://datasets/AIML-TUDA/dlam-ts-project-data-2026/"

def load_data(input_dir):
    if input_dir is None:
        history = pd.read_csv(HF_PATH + "train.csv")
        future = pd.read_csv(HF_PATH + "validation_input.csv")
        index = pd.read_csv(HF_PATH + "forecast_index_validation.csv")
        return history, future, index

    future_candidates = [input_dir / "test_input.csv", input_dir / "validation_input.csv"]
    index_candidates = [input_dir / "forecast_index_test.csv", input_dir / "forecast_index_validation.csv"]
    future_path = next((p for p in future_candidates if p.exists()), None)
    index_path = next((p for p in index_candidates if p.exists()), None)
    history_path = input_dir / "train.csv"

    if not history_path.exists():
        raise FileNotFoundError("No train.csv found for target history")
    if future_path is None:
        raise FileNotFoundError("No test_input.csv or validation_input.csv found")
    if index_path is None:
        raise FileNotFoundError("No forecast index found")

    return pd.read_csv(history_path), pd.read_csv(future_path), pd.read_csv(index_path)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--output_file", required=True, type=Path)
parser.add_argument("--input_dir", type=Path)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

device = torch.device(args.device)
checkpoint = torch.load(args.checkpoint, map_location=device)
module = importlib.import_module(f"src.models.{checkpoint['model']}")
model_args = SimpleNamespace(**checkpoint["args"])
model = module.create_model(model_args, checkpoint["history_length"], checkpoint["horizon"]).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

history_data, future_data, forecast_index = load_data(args.input_dir)
history_data["timestamp"] = pd.to_datetime(history_data["timestamp"])
forecast_index["timestamp"] = pd.to_datetime(forecast_index["timestamp"])

history_length = checkpoint["history_length"]
output = forecast_index[["series_id", "timestamp"]].copy()
output["prediction"] = float("nan")

with torch.no_grad():
    for series_id, index_part in forecast_index.groupby("series_id", sort=False):
        values = history_data[history_data["series_id"].eq(series_id)].sort_values("timestamp")["target"].tail(history_length).to_numpy()

        if len(values) < history_length:
            raise ValueError(f"Not enough history for {series_id}")

        history = torch.tensor(values, dtype=torch.float32, device=device)
        predictions = []

        while len(predictions) < len(index_part):
            pred = model(history[-history_length:].unsqueeze(0)).squeeze(0)
            predictions.extend(pred.cpu().tolist())
            history = torch.cat([history, pred])

        output.loc[index_part.index, "prediction"] = predictions[:len(index_part)]

if output["prediction"].isna().any():
    raise ValueError("Missing predictions")

args.output_file.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(args.output_file, index=False)
print(f"Wrote {len(output)} predictions to {args.output_file}")