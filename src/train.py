import argparse
import importlib
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.dataset import ForecastDataset, load_train_validation
from src.evaluation.baselines import make_all_baselines
from src.evaluation.metrics import wape
from src.evaluation.rollout import rollout

HISTORY_LENGTH = 168
HORIZON = 24
VALIDATION_HORIZON = 336

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args, _ = parser.parse_known_args()
module = importlib.import_module(f"src.models.{args.model}")

parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

if hasattr(module, "add_model_args"):
    module.add_model_args(parser)

args = parser.parse_args()
device = torch.device(args.device)
Path("checkpoints").mkdir(exist_ok=True)

start = time.perf_counter()
train_frame, val_frame = load_train_validation(HISTORY_LENGTH, VALIDATION_HORIZON)
print(f"Data loading: {time.perf_counter() - start:.2f}s")

val_targets = val_frame.groupby("series_id", sort=False).tail(VALIDATION_HORIZON)
forecast_index = val_targets[["series_id", "timestamp"]]

start = time.perf_counter()
baseline_scores = {}
for name, predictions in make_all_baselines(train_frame, forecast_index).items():
    result = val_targets.merge(predictions, on=["series_id", "timestamp"])
    baseline_scores[name] = wape(result["target"], result["prediction"])
print(f"Baselines: {time.perf_counter() - start:.2f}s")

start = time.perf_counter()
train_dataset = ForecastDataset(train_frame, HISTORY_LENGTH, HORIZON)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=device.type == "cuda",
    persistent_workers=args.workers > 0
)
print(f"Dataset setup: {time.perf_counter() - start:.2f}s")
print(f"Samples: {len(train_dataset)}, batches/epoch: {len(train_loader)}")

model = module.create_model(args, HISTORY_LENGTH, HORIZON).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.MSELoss()
best_val = float("inf")

print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(device)}")

for epoch in range(args.epochs):
    epoch_start = time.perf_counter()
    train_start = time.perf_counter()

    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    train_time = time.perf_counter() - train_start
    rollout_start = time.perf_counter()

    train_loss /= len(train_loader)
    predictions = rollout(model, val_frame, VALIDATION_HORIZON, HISTORY_LENGTH, HORIZON, device)
    result = val_targets.merge(predictions, on=["series_id", "timestamp"])
    val = wape(result["target"], result["prediction"])

    if device.type == "cuda":
        torch.cuda.synchronize()

    rollout_time = time.perf_counter() - rollout_start

    if val < best_val:
        best_val = val
        torch.save({
            "model": args.model,
            "args": vars(args),
            "history_length": HISTORY_LENGTH,
            "horizon": HORIZON,
            "state_dict": model.state_dict()
        }, f"checkpoints/{args.model}.pt")

    epoch_time = time.perf_counter() - epoch_start
    print(f"{epoch + 1}: train={train_loss:.4f}, val_wape={val:.4f} | train={train_time:.2f}s rollout={rollout_time:.2f}s total={epoch_time:.2f}s")

print(f"\n{args.model}: {best_val:.4f}")
for name, score in baseline_scores.items():
    print(f"{name}: {score:.4f}")

naive = baseline_scores["naive_last_value"]
best_baseline_name = min(baseline_scores, key=baseline_scores.get)
best_baseline = baseline_scores[best_baseline_name]

print(f"\nMinimum requirement: {'PASS' if best_val < naive else 'FAIL'} ({best_val:.4f} vs {naive:.4f})")
print(f"Best provided baseline: {'BEATEN' if best_val < best_baseline else 'NOT BEATEN'} ({best_val:.4f} vs {best_baseline:.4f}, {best_baseline_name})")