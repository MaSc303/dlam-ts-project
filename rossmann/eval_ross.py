"""Rossmann evaluation: baselines and checkpoints on the last `val_windows`
28-day folds, six metrics over open days plus RMSPE (the Kaggle Rossmann metric).

    python eval_ross.py --data_dir data --val_windows 2 \
        --checkpoints ck_full.pt ck_nohead.pt ck_nofilm.pt ck_nodom.pt ck_noglobal.pt ck_mse.pt

Statistics are re-fitted on the data before the first fold (as train.py does
with --val_windows 2), so nothing from the folds leaks into the profile.
Baselines: naive last observed value (held constant), store x day-of-week
median profile.  Each checkpoint's feature switches are read from its stats.
"""
from __future__ import annotations

import argparse, os, sys, warnings
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from core import (H, SERIES, TIME, ResidualEncoder, build_block, fit_stats,
                  forecast, target_matrix)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


def six(y, p, v):
    m = v > 0
    y, p = y[m].astype(np.float64), p[m].astype(np.float64)
    e = p - y
    ae, ay = np.abs(e), np.abs(y)
    out = {"MAE": ae.mean(), "RMSE": np.sqrt((e ** 2).mean())}
    pos = ay > 0
    out["MAPE"] = 100 * (ae[pos] / ay[pos]).mean()
    den = ay + np.abs(p)
    s = np.zeros_like(ae)
    ok = den > 0
    s[ok] = 2 * ae[ok] / den[ok]
    out["sMAPE"] = 100 * s.mean()
    out["WAPE"] = 100 * ae.sum() / ay.sum()
    out["RMSPE"] = 100 * np.sqrt(((e[pos] / ay[pos]) ** 2).mean())   # Kaggle's Rossmann metric
    out["bias%"] = 100 * e.sum() / ay.sum()
    return out


def load_models(ck, dev):
    cfg = ck["cfg"]
    states = ck["state_dict"]
    states = states if isinstance(states, list) else [states]
    ms = []
    for sd in states:
        m = ResidualEncoder(n_feat=cfg["n_feat"], n_units=cfg["n_units"],
                            d_model=cfg["d_model"], layers=cfg["layers"],
                            nhead=cfg.get("nhead", 8), ff=cfg.get("ff", 4 * cfg["d_model"]),
                            dropout=cfg["dropout"], film=cfg.get("film", True),
                            level_head=cfg.get("level_head", False)).to(dev)
        m.load_state_dict(sd); m.eval(); ms.append(m)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--val_windows", type=int, default=2)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    train[TIME] = pd.to_datetime(train[TIME])
    T_total = int(train[SERIES].value_counts().iloc[0])
    stamps = sorted(train[TIME].unique())
    hold_start = T_total - a.val_windows * H
    cut = stamps[hold_start]
    fit_frame = train[train[TIME] < cut]
    st = fit_stats(fit_frame)
    Y, grid = target_matrix(train, st)
    valid = (~np.isnan(Y)).astype(np.float32)
    Yc = np.nan_to_num(Y)
    U = Y.shape[0]
    starts = [hold_start + w * H for w in range(a.val_windows)]
    folds = [f"{grid[s].strftime('%b %d')}-{grid[s + H - 1].strftime('%b %d')}" for s in starts]
    print(f"{U} stores | grid {grid[0].date()} -> {grid[-1].date()} ({len(grid)} days) | "
          f"stats fitted on < {pd.Timestamp(cut).date()} | folds {folds} | device {dev}")

    def score(pred_fn, name):
        row, num, den = {}, 0.0, 0.0
        ys, ps, vs = [], [], []
        for k, s in enumerate(starts):
            sl = slice(s, s + H)
            p = pred_fn(sl)
            y, v = Yc[:, sl], valid[:, sl]
            row[f"WAPE {folds[k]}"] = six(y, p, v)["WAPE"]
            ys.append(y); ps.append(p); vs.append(v)
        pooled = six(np.concatenate(ys, 1), np.concatenate(ps, 1), np.concatenate(vs, 1))
        row.update({f"pooled {k}": val for k, val in pooled.items()})
        rows[name] = row

    rows = {}
    # naive: last observed value before the cut, held constant
    last = np.array([Yc[i, :hold_start][valid[i, :hold_start] > 0][-1] for i in range(U)])
    score(lambda sl: np.repeat(last[:, None], H, axis=1), "naive last value")
    how_all = ((grid.dayofweek).to_numpy())
    score(lambda sl: st["profile"][:, how_all[sl]], "store x day-of-week profile")

    days = np.arange(Y.shape[1])
    for path in a.checkpoints:
        ck = torch.load(path, map_location=dev, weights_only=False)
        models = load_models(ck, dev)
        st2 = dict(st)
        for k in ("features", "use_dom", "global_of"):
            if k in ck["stats"]:
                st2[k] = ck["stats"][k]
        X, how, prof = build_block(train, st2, Y, days, grid[0], ())
        w = ck.get("weights")
        name = (f"{path}  [{ck['cfg'].get('loss', 'l1')}, head={ck['cfg'].get('level_head')}, "
                f"film={ck['cfg'].get('film')}, dom={st2.get('use_dom')}, "
                f"global={bool(st2.get('global_of'))}, ep={ck.get('best_epoch')}]")
        score(lambda sl: forecast(models, X[:, sl], how[:, sl], prof[:, sl], st2, dev, weights=w),
              name)

    tb = pd.DataFrame(rows).T
    print(tb.round(3).to_string())


if __name__ == "__main__":
    main()
