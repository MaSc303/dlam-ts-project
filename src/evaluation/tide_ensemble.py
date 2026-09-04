import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.tide_dataset import ForecastDataset, STATIC_COLS
from src.models.tide import TiDE


@torch.no_grad()
def predict(ckpt, L, norm_mode, frame, pipe, horizon, device, starts="last"):
    model = TiDE(L, horizon, len(pipe.feature_cols), len(STATIC_COLS),
                 norm_mode=norm_mode).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    ds = ForecastDataset(frame, L, horizon, pipe.feature_cols,
                         pipe.series_stats, starts=starts)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    P, Y = [], []
    for batch in dl:
        *inp, y = [t.to(device) for t in batch]
        P.append(model(*inp).float().cpu().numpy()); Y.append(y.cpu().numpy())
    meta = [(ds.series[si][4], ds.series[si][5][t+L:t+L+horizon]) for si, t in ds.index]
    return np.concatenate(P), np.concatenate(Y), meta


def predict_ensemble(ckpts, frame, pipe, horizon, device, starts="last"):
    parts, truth, meta = [], None, None
    for ckpt, L, mode in ckpts:
        p, y, m = predict(ckpt, L, mode, frame, pipe, horizon, device, starts=starts)
        parts.append(p)
        truth, meta = y, m
    return np.mean(parts, axis=0), parts, truth, meta
