"""Step-1 diagnostics for the DLAM residual encoder. No training.

    python diag.py --data_dir data --checkpoint ck_calib.pt --out_dir out/diag

Evaluates on the last 336 h of train.csv (the local holdout) with statistics
re-fitted on the data before it -- the same protocol as `train.py --val_windows 1`,
NOT the checkpoint's stored stats (those were fitted on the full train and would
leak the holdout into the profile).

  1. the six leaderboard metrics + bias of the checkpoint on the holdout
  2. where the error lives, per unit
  3. where the error lives, per regime (maintenance / spike / dip / ordinary)
  4. profile-only forecasts: recent-window profiles and profile x recent level,
     at gap 0 (validation regime) and gap 336 (private-test regime)
  5. the `trend` covariate, and the per-unit level over time (drift)

Writes per_unit.csv and holdout_predictions.csv to --out_dir for later use.
"""
from __future__ import annotations

import argparse, os, sys, warnings
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from core import (H, SERIES, TIME, ResidualEncoder, build_block, fit_stats,
                  forecast, min_hist, target_matrix)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
warnings.filterwarnings("ignore", category=RuntimeWarning)
WEEK = 168


# ------------------------------------------------------------------ metrics
def six(y, p, v):
    """Leaderboard-style metrics over observed rows (v > 0), plus bias.

    MAPE is over rows with y > 0; sMAPE counts 0/0 rows as 0. If the local
    MAPE-WAPE and sMAPE-WAPE gaps come out far from the leaderboard's (2.0 and
    1.8 for ck_calib) the zero-handling differs from the graders' and we adjust.
    """
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
    return out


def show(res, digits=3):
    print(pd.DataFrame(res).T.round(digits).to_string())


# ------------------------------------------------------------------ helpers
def load_models(ck, dev):
    cfg = ck["cfg"]
    states = ck["state_dict"]
    states = states if isinstance(states, list) else [states]
    ms = []
    for sd in states:
        m = ResidualEncoder(n_feat=cfg["n_feat"], n_units=cfg["n_units"],
                            d_model=cfg["d_model"], layers=cfg["layers"],
                            nhead=cfg.get("nhead", 8),
                            ff=cfg.get("ff", 4 * cfg["d_model"]),
                            dropout=cfg["dropout"],
                            film=cfg.get("film", True),
                            level_head=cfg.get("level_head", False)).to(dev)
        m.load_state_dict(sd)
        m.eval()
        ms.append(m)
    return ms, tuple(cfg.get("lags", []))


def raw_cov(df, st, stamps, col):
    """(U, L) raw values of one column on the given timestamps, NaN if absent."""
    out = np.full((len(st["units"]), len(stamps)), np.nan, np.float32)
    d = df.drop_duplicates([SERIES, TIME])
    for u in st["units"]:
        out[st["uid"][u]] = (d[d[SERIES] == u].set_index(TIME)
                             .reindex(stamps)[col].to_numpy(np.float32))
    return out


def level_of(Yc, valid, prof_all, i, lo, hi):
    """Level of unit i over hours [lo, hi) relative to the global profile.

    Returns (median of y/profile, sum(y)/sum(profile)). Falls back to 1.0 when
    fewer than 10 usable hours.
    """
    y, v, p = Yc[i, lo:hi], valid[i, lo:hi] > 0, prof_all[i, lo:hi]
    ok = v & (p > 1e-6)
    med = float(np.median(y[ok] / p[ok])) if ok.sum() > 10 else 1.0
    sr = float(y[v].sum() / max(p[v].sum(), 1e-9)) if v.sum() > 10 else 1.0
    return med, sr


def bucket_stats(mask, pred, y, tot_ae, tot_y, n_obs):
    e = pred[mask] - y[mask]
    ay = np.abs(y[mask])
    den = float(ay.sum())
    return dict(hours_share=100 * mask.sum() / n_obs,
                y_share=100 * den / tot_y,
                err_share=100 * float(np.abs(e).sum()) / tot_ae,
                WAPE=100 * float(np.abs(e).sum()) / den if den > 1e-6 else np.nan,
                bias=100 * float(e.sum()) / den if den > 1e-6 else np.nan,
                mean_y=float(y[mask].mean()), mean_pred=float(pred[mask].mean()))


def rng(s):
    return f"[{np.nanmin(s):.4f}, {np.nanmax(s):.4f}]"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--checkpoint", default="ck_calib.pt")
    ap.add_argument("--out_dir", default="out/diag")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    train[TIME] = pd.to_datetime(train[TIME])
    grid = pd.date_range(train[TIME].min(), train[TIME].max(), freq="h")
    hold = len(grid) - H
    cut = grid[hold]
    fit_frame = train[train[TIME] < cut]
    st = fit_stats(fit_frame)                       # never sees the holdout
    Y, grid = target_matrix(train, st)
    valid = (~np.isnan(Y)).astype(np.float32)
    Yc = np.nan_to_num(Y)
    U = Y.shape[0]
    print(f"train grid {grid[0]} -> {grid[-1]}  ({len(grid)} h, {U} series)")
    print(f"holdout {grid[hold]} -> {grid[-1]}  |  stats fitted on < {cut}  |  device {dev}")

    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    models, lags = load_models(ck, dev)
    print(f"checkpoint {a.checkpoint}: models={len(models)}  cfg={ck['cfg']}")
    print(f"  stored holdout_wape={ck.get('holdout_wape')}  best_epoch={ck.get('best_epoch')}  "
          f"val_windows={ck.get('val_windows')}  seed={ck.get('seed')}")

    hours = np.arange(Y.shape[1])
    if lags:
        hours = hours.copy()
        hours[:min_hist(lags)] = min_hist(lags)
    X, how, prof = build_block(train, st, Y, hours, grid[0], lags)
    sl = slice(hold, hold + H)
    pred = forecast(models, X[:, sl], how[:, sl], prof[:, sl], st, dev,
                    weights=ck.get("weights"))
    y, v, pr, hw = Yc[:, sl], valid[:, sl], prof[:, sl], how[0, sl]
    obs = v > 0
    tot_ae = float((np.abs(pred - y) * v).sum())
    tot_y = float((np.abs(y) * v).sum())

    # ------------------------------------------------------------ check 1
    print("\n" + "=" * 90)
    print("CHECK 1 - six metrics of the checkpoint on the holdout (leaderboard definitions)")
    show({"checkpoint": six(y, pred, v)})

    # ------------------------------------------------------------ check 2
    print("\n" + "=" * 90)
    print("CHECK 2 - where the error lives, per unit (sorted by share of total |error|)")
    rows = []
    for i, u in enumerate(st["units"]):
        m = obs[i]
        e = pred[i][m] - y[i][m]
        ay = np.abs(y[i][m])
        rows.append(dict(unit=u, y_share=100 * float(ay.sum()) / tot_y,
                         err_share=100 * float(np.abs(e).sum()) / tot_ae,
                         WAPE=100 * float(np.abs(e).sum()) / max(float(ay.sum()), 1e-9),
                         bias=100 * float(e.sum()) / max(float(ay.sum()), 1e-9),
                         median_y=float(np.median(y[i][m])) if m.any() else np.nan,
                         res_scale=float(st["res_scale"][i]),
                         n_missing=int((~m).sum())))
    per = (pd.DataFrame(rows).sort_values("err_share", ascending=False)
           .reset_index(drop=True))
    per.to_csv(os.path.join(a.out_dir, "per_unit.csv"), index=False)
    print(per.head(20).round(2).to_string())
    print(f"... {U} units; full table -> {a.out_dir}/per_unit.csv")
    print(f"units with negative bias: {int((per.bias < 0).sum())}/{U}   "
          f"|bias| > 5%: {int((per.bias.abs() > 5).sum())}   "
          f"|bias| > 10%: {int((per.bias.abs() > 10).sum())}")
    for k in (10, 20):
        top = per.head(k)
        print(f"top {k} units by error: {top.err_share.sum():.1f}% of |error|, "
              f"{top.y_share.sum():.1f}% of |y|")
    print(f"unit WAPE: min {per.WAPE.min():.1f}  median {per.WAPE.median():.1f}  "
          f"max {per.WAPE.max():.1f}   |   corr(unit bias, unit WAPE) = "
          f"{np.corrcoef(per.bias, per.WAPE)[0, 1]:.2f}")

    # ------------------------------------------------------------ check 3
    print("\n" + "=" * 90)
    print("CHECK 3 - where the error lives, per regime")
    mk_col = train["maintenance_known"]
    uniq = np.unique(mk_col.dropna().to_numpy(np.float32))
    print(f"maintenance_known: {len(uniq)} distinct values, first 10 = {uniq[:10]}, "
          f"NaN share {mk_col.isna().mean():.3f}, nonzero share {(mk_col.fillna(0) != 0).mean():.3f}")
    mk = raw_cov(train, st, grid[sl], "maintenance_known")
    rs = st["res_scale"][:, None]
    resid = y - pr
    maint = obs & (np.nan_to_num(mk) != 0)
    spike = obs & ~maint & (resid > 2 * rs)
    dip = obs & ~maint & (resid < -2 * rs)
    ordinary = obs & ~(maint | spike | dip)
    res = {}
    for name, m in (("maintenance", maint), ("spike (> +2 IQR)", spike),
                    ("dip (< -2 IQR)", dip), ("ordinary", ordinary)):
        res[name] = (bucket_stats(m, pred, y, tot_ae, tot_y, obs.sum())
                     if m.sum() > 0 else dict(hours_share=0.0))
    show(res, 2)
    print("(WAPE/bias are NaN where the bucket's |y| sums to ~0, e.g. maintenance hours that "
          "are all zero -- read mean_pred there instead. Spike bias is expected negative and "
          "dip bias positive under L1; the ORDINARY row is the one that matters.)")

    # ------------------------------------------------------------ check 4
    print("\n" + "=" * 90)
    print("CHECK 4 - profile-only forecasts of the holdout, no model")
    res = {"global profile (all data < cut)": six(y, pr, v)}
    for N in (2, 4, 8, 13):
        start = cut - pd.Timedelta(weeks=N)
        if start < grid[0]:
            continue
        stN = fit_stats(fit_frame[fit_frame[TIME] >= start])
        res[f"profile from last {N} wk only"] = six(y, stN["profile"][:, hw], v)
    prof_all = st["profile"][:, how[0]]            # (U, T) global profile on the grid
    windows = {"gap 0 (2 wk before cut)": (hold - 2 * WEEK, hold),
               "gap 336 (2-4 wk before cut)": (hold - 4 * WEEK, hold - 2 * WEEK)}
    for wname, (lo, hi) in windows.items():
        lev = np.array([level_of(Yc, valid, prof_all, i, lo, hi) for i in range(U)])
        med, sr = lev[:, 0], lev[:, 1]
        print(f"level, {wname}:  ratio-median  min {med.min():.3f}  median {np.median(med):.3f}  "
              f"max {med.max():.3f}   |   sum-ratio  min {sr.min():.3f}  "
              f"median {np.median(sr):.3f}  max {sr.max():.3f}")
        for lname, lv in (("ratio-median", med), ("sum-ratio", sr)):
            for lam in (0.5, 1.0):
                scale = 1.0 + lam * (lv - 1.0)
                res[f"profile x level[{lname}] lam={lam} {wname}"] = six(y, pr * scale[:, None], v)
    show(res)
    print("(row 1 is your 0.3206 reference; 'lam' shrinks the level toward 1.0)")

    # ------------------------------------------------------------ check 5
    print("\n" + "=" * 90)
    print("CHECK 5 - trend covariate, and per-unit level over time (drift)")
    print(f"trend range:  fit (< cut) {rng(fit_frame['trend'])}   "
          f"holdout {rng(train.loc[train[TIME] >= cut, 'trend'])}", end="")
    vp = os.path.join(a.data_dir, "validation_input.csv")
    if os.path.exists(vp):
        vi = pd.read_csv(vp)
        print(f"   validation_input {rng(vi['trend'])}")
    else:
        print("   (validation_input.csv not found)")
    tr = train[[SERIES, TIME, "trend"]]
    cross = tr.groupby(TIME)["trend"].std()
    mx = float(np.nanmax(cross)) if cross.notna().any() else 0.0
    print(f"max std of trend across units at one timestamp: {mx:.6f}  "
          f"-> {'GLOBAL (same for every unit)' if mx < 1e-6 else 'PER-UNIT'}")
    mt = tr.groupby(TIME)["trend"].mean()
    ok = mt.notna().to_numpy()
    t_hours = ((mt.index - mt.index[0]) / pd.Timedelta(hours=1)).to_numpy()
    print(f"corr(trend, elapsed hours) = {np.corrcoef(t_hours[ok], mt.to_numpy()[ok])[0, 1]:.4f}   "
          f"(1.0 = linear ramp)")
    q = np.quantile(t_hours[ok], [0, .25, .5, .75, 1.0])
    vals = np.interp(q, t_hours[ok], mt.to_numpy()[ok])
    print(f"trend at 0/25/50/75/100% of train: {np.round(vals, 4).tolist()}")

    nb = hold // (4 * WEEK)
    if nb >= 2:
        edges = [hold - (nb - k) * 4 * WEEK for k in range(nb + 1)]
        L = np.zeros((U, nb))
        for i in range(U):
            for k in range(nb):
                L[i, k] = level_of(Yc, valid, prof_all, i, edges[k], edges[k + 1])[0]
        x = np.arange(nb)
        slopes = np.array([np.polyfit(x, L[i], 1)[0] for i in range(U)]) * 100
        print(f"{nb} four-week blocks ending at the cut; MEAN level across units per block "
              f"(1.0 = global profile): {np.round(L.mean(0), 3).tolist()}")
        print(f"per-unit slope of level: positive in {int((slopes > 0).sum())}/{U} units, "
              f"median {np.median(slopes):+.2f} %/4wk, "
              f"IQR [{np.percentile(slopes, 25):+.2f}, {np.percentile(slopes, 75):+.2f}]")
        print(f"last block / first block level: median {np.median(L[:, -1] / L[:, 0]):.3f}")
    hl = np.array([level_of(Yc, valid, prof_all, i, hold, hold + H)[0] for i in range(U)])
    print(f"HOLDOUT level vs global profile: median {np.median(hl):.3f}, "
          f"IQR [{np.percentile(hl, 25):.3f}, {np.percentile(hl, 75):.3f}], "
          f"units above 1.0: {int((hl > 1).sum())}/{U}")
    print("(this last line is the level the model had to reach; compare with the gap-0 "
          "and gap-336 levels printed in check 4)")

    # ------------------------------------------------------------ dump
    out = pd.DataFrame({SERIES: np.repeat(st["units"], H), TIME: np.tile(grid[sl], U),
                        "y": y.reshape(-1), "pred": pred.reshape(-1),
                        "valid": v.reshape(-1), "profile": pr.reshape(-1)})
    out.to_csv(os.path.join(a.out_dir, "holdout_predictions.csv"), index=False)
    print(f"\nwrote {a.out_dir}/holdout_predictions.csv and {a.out_dir}/per_unit.csv")


if __name__ == "__main__":
    main()
