import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

STATIC_COLS = ["nominal_capacity", "zone_sin", "zone_cos"]


class FeaturePipeline:

    def __init__(self, fit_frame, dynamic_cols):
        self.dynamic_cols = list(dynamic_cols)

        p = fit_frame.assign(
            _dow=fit_frame["timestamp"].dt.dayofweek,
            _hour=fit_frame["timestamp"].dt.hour,
        )
        self.profile = p.groupby(["series_id", "_dow", "_hour"])["target"].mean()
        self.series_mean = fit_frame.groupby("series_id")["target"].mean()
        self.global_mean = float(fit_frame["target"].mean())

        self.series_stats = pd.DataFrame(
            {
                "mean": fit_frame.groupby("series_id")["target"].mean(),
                "std": fit_frame.groupby("series_id")["target"].std(),
            }
        )

        tmp = self._add_profile(fit_frame.copy())
        self.scale_cols = self.dynamic_cols + ["seasonal_profile"] + STATIC_COLS
        self.mu = tmp[self.scale_cols].mean()
        self.sd = tmp[self.scale_cols].std().replace(0, 1.0)
        self.feature_cols = self.dynamic_cols + ["seasonal_profile"]

    def _add_profile(self, frame):
        idx = pd.MultiIndex.from_arrays(
            [
                frame["series_id"],
                frame["timestamp"].dt.dayofweek,
                frame["timestamp"].dt.hour,
            ]
        )
        v = self.profile.reindex(idx).to_numpy()
        v = np.where(
            np.isnan(v), frame["series_id"].map(self.series_mean).to_numpy(), v
        )
        frame["seasonal_profile"] = np.nan_to_num(v, nan=self.global_mean)
        return frame

    def transform(self, frame):
        f = frame.copy()
        cov = self.dynamic_cols + STATIC_COLS
        f[cov] = f.groupby("series_id")[cov].ffill().bfill()
        f = self._add_profile(f)
        f[self.scale_cols] = (f[self.scale_cols] - self.mu) / self.sd
        assert f[self.scale_cols].isna().sum().sum() == 0
        return f


class ForecastDataset(Dataset):

    def __init__(self, frame, L, H, feature_cols, series_stats, starts="all"):
        self.L, self.H = L, H
        self.series, self.index = [], []
        for sid, g in frame.groupby("series_id", sort=False):
            si = len(self.series)
            self.series.append(
                (
                    g["target"].to_numpy(np.float32),
                    g[feature_cols].to_numpy(np.float32),
                    g[STATIC_COLS].to_numpy(np.float32)[0],
                    series_stats.loc[sid].to_numpy(np.float32),
                    sid,
                    g["timestamp"].to_numpy(),
                )
            )
            n_max = len(g) - L - H
            pos = (
                range(n_max + 1)
                if starts == "all"
                else (
                    [n_max]
                    if starts == "last"
                    else [p for p in starts if 0 <= p <= n_max]
                )
            )
            self.index.extend((si, t) for t in pos)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, t = self.index[i]
        y, cov, static, stats, _, _ = self.series[si]
        L, H = self.L, self.H
        return (
            torch.from_numpy(y[t : t + L]),
            torch.from_numpy(cov[t : t + L]),
            torch.from_numpy(cov[t + L : t + L + H]),
            torch.from_numpy(static),
            torch.from_numpy(stats),
            torch.from_numpy(np.nan_to_num(y[t + L : t + L + H])),
        )
