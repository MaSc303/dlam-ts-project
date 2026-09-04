"""Blend analysis on the local holdout. No training, no GPU.

    python blend.py --l1 out/C10/holdout_predictions.csv out/C/holdout_predictions.csv \
                    --mse out/D4/holdout_predictions.csv out/D/holdout_predictions.csv

Averages the members within each group, then reports the six leaderboard
metrics for blends  w * L1 + (1 - w) * MSE,  w = 1.0 ... 0.0.  The composite
proxy is the mean over the six metrics of (value / top-20 median from the
leaderboard export); lower is better, and only the ordering across w matters.
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd

TOP20 = {"MAE": 1.376, "MSE": 6.344, "RMSE": 2.519, "MAPE": 14.284,
         "sMAPE": 13.681, "WAPE": 12.499}
KEY = ["series_id", "timestamp"]


def six(y, p, v):
    m = v > 0
    y, p = y[m].astype(np.float64), p[m].astype(np.float64)
    e = p - y
    ae, ay = np.abs(e), np.abs(y)
    out = {"MAE": ae.mean(), "MSE": (e ** 2).mean()}
    out["RMSE"] = np.sqrt(out["MSE"])
    pos = ay > 0
    out["MAPE"] = 100 * (ae[pos] / ay[pos]).mean()
    den = ay + np.abs(p)
    s = np.zeros_like(ae)
    ok = den > 0
    s[ok] = 2 * ae[ok] / den[ok]
    out["sMAPE"] = 100 * s.mean()
    out["WAPE"] = 100 * ae.sum() / ay.sum()
    out["bias%"] = 100 * e.sum() / ay.sum()
    out["proxy"] = float(np.mean([out[k] / TOP20[k] for k in TOP20]))
    return out


def load_group(paths):
    base = None
    preds = []
    for p in paths:
        d = pd.read_csv(p).sort_values(KEY).reset_index(drop=True)
        if base is None:
            base = d[KEY + ["y", "valid"]].copy()
        else:
            assert (d[KEY].values == base[KEY].values).all(), f"{p}: rows differ"
        preds.append(d["pred"].to_numpy(np.float64))
    return base, np.stack(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", nargs="+", required=True, help="holdout csvs of L1 members")
    ap.add_argument("--mse", nargs="+", required=True, help="holdout csvs of MSE members")
    a = ap.parse_args()

    base, P1 = load_group(a.l1)
    base2, P2 = load_group(a.mse)
    assert (base[KEY].values == base2[KEY].values).all()
    y, v = base["y"].to_numpy(np.float64), base["valid"].to_numpy(np.float64)

    rows = {}
    for i, p in enumerate(a.l1):
        rows[f"L1 member {i}: {p}"] = six(y, P1[i], v)
    for i, p in enumerate(a.mse):
        rows[f"MSE member {i}: {p}"] = six(y, P2[i], v)
    m1, m2 = P1.mean(0), P2.mean(0)
    rows["L1 group mean"] = six(y, m1, v)
    rows["MSE group mean"] = six(y, m2, v)
    for w in np.arange(1.0, -0.01, -0.1):
        rows[f"blend w_L1={w:.1f}"] = six(y, w * m1 + (1 - w) * m2, v)
    pd.set_option("display.width", 200)
    tb = pd.DataFrame(rows).T
    print(tb.round(3).to_string())
    best = tb.loc[[k for k in tb.index if k.startswith("blend")], "proxy"].idxmin()
    print(f"\nlowest composite proxy among blends: {best}")
    print("(proxy = mean over the six metrics of value / top-20 median; compare rows, "
          "not absolute values)")


if __name__ == "__main__":
    main()
