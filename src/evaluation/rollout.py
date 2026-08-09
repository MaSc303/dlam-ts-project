import pandas as pd
import torch

def rollout(model, frame, horizon=336, history_length=168, block=24, device="cpu"):
    groups = [(series_id, group.sort_values("timestamp")) for series_id, group in frame.groupby("series_id", sort=False)]
    histories = [torch.tensor(group.iloc[:-horizon]["target"].tail(history_length).to_numpy(), dtype=torch.float32) for _, group in groups]

    if any(len(history) < history_length for history in histories):
        raise ValueError("Not enough history for rollout")

    history = torch.stack(histories).to(device)
    predictions = []
    model.eval()

    with torch.no_grad():
        while sum(pred.shape[1] for pred in predictions) < horizon:
            pred = model(history[:, -history_length:])
            predictions.append(pred)
            history = torch.cat([history, pred], dim=1)

    predictions = torch.cat(predictions, dim=1)[:, :horizon].cpu()
    rows = []

    for i, (series_id, group) in enumerate(groups):
        result = group.tail(horizon)[["series_id", "timestamp"]].copy()
        result["prediction"] = predictions[i].numpy()
        rows.append(result)

    return pd.concat(rows, ignore_index=True)