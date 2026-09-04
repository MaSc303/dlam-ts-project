"""Step-2 diagnostics: the level process, what drives it, and the spike regime.
No model, no training. Reads train.csv and the holdout_predictions.csv written by
diag.py.

    python diag2.py --data_dir data --pred out/diag/holdout_predictions.csv

  A. per-unit level (median of y/profile) per week and per fortnight, aligned to
     the holdout so phase relationships are exact; autocorrelation at 1-8 weeks,
     pooled same-phase vs half-phase correlation; cross-unit synchrony
  B. which covariates track the level: global-vs-per-unit share of each dynamic
     covariate, correlation of weekly covariate means with the global level,
     leave-one-out ridge R^2 (global level) and week-blocked CV R^2 (unit level)
  C. spikes: squared-error share by regime on the holdout; which covariates
     separate spike hours from the rest (Cohen's d); spike rate by decile of the
     strongest ones
  D. holdout base variants using same-phase levels (lag 672 / 1344) and a
     phase-specific profile
"""
from __future__ import annotations

import argparse, os, sys, warnings
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from core import H, SERIES, TIME, TARGET, FEATURES, fit_stats, target_matrix, _how_of

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
warnings.filterwarnings("ignore", category=RuntimeWarning)
WEEK = 168
STATIC = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "trend",
          "nominal_capacity", "zone_sin", "zone_cos"}
DYN = [f for f in FEATURES if f not in STATIC]


# ------------------------------------------------------------------ helpers
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
    return out


def show(res, digits=3):
    print(pd.DataFrame(res).T.round(digits).to_string())


def level_block(Yc, valid, prof_all, lo, hi):
    """(U,) median of y/profile over hours [lo, hi); 1.0 where < 10 usable hours."""
    y, v, p = Yc[:, lo:hi], valid[:, lo:hi] > 0, prof_all[:, lo:hi]
    ok = v & (p > 1e-6)
    r = np.where(ok, y / np.where(p > 1e-6, p, 1.0), np.nan)
    out = np.nanmedian(r, axis=1)
    out[ok.sum(1) < 10] = 1.0
    return out


def acf(x, lags):
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    d = (x ** 2).sum()
    return [float((x[:-k] * x[k:]).sum() / d) if 0 < k < len(x) else np.nan for k in lags]


def pooled_corr(A, k):
    """Correlation between block j and block j-k, pooled over units and blocks."""
    a, b = A[:, k:].ravel(), A[:, :-k].ravel()
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else np.nan


_COV_CACHE = {}


def cov_matrix(df, st, grid, col):
    """(U, T) raw values of one covariate on the dense grid (cached per column)."""
    if col in _COV_CACHE:
        return _COV_CACHE[col]
    out = np.full((len(st["units"]), len(grid)), np.nan, np.float32)
    d = df.drop_duplicates([SERIES, TIME])
    for u in st["units"]:
        out[st["uid"][u]] = (d[d[SERIES] == u].set_index(TIME)
                             .reindex(grid)[col].to_numpy(np.float32))
    _COV_CACHE[col] = out
    return out


def loo_ridge_r2(Xm, yv, alpha=1.0):
    """Leave-one-out R^2 of ridge regression on standardised features."""
    n = len(yv)
    preds = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-9
        Xt = (Xm[tr] - mu) / sd
        ym = yv[tr].mean()
        w = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(Xt.shape[1]), Xt.T @ (yv[tr] - ym))
        preds[i] = ym + ((Xm[i] - mu) / sd) @ w
    return 1 - ((yv - preds) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()


def blocked_cv_r2(X, yv, groups, nfold=5):
    """R^2 of least squares with folds over contiguous groups (weeks)."""
    ug = np.unique(groups)
    folds = np.array_split(ug, nfold)
    preds = np.zeros_like(yv)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        Xt = np.c_[np.ones(tr.sum()), X[tr]]
        w, *_ = np.linalg.lstsq(Xt, yv[tr], rcond=None)
        preds[te] = np.c_[np.ones(te.sum()), X[te]] @ w
    return 1 - ((yv - preds) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--pred", default="out/diag/holdout_predictions.csv")
    a = ap.parse_args()

    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    train[TIME] = pd.to_datetime(train[TIME])
    grid = pd.date_range(train[TIME].min(), train[TIME].max(), freq="h")
    hold = len(grid) - H
    cut = grid[hold]
    fit_frame = train[train[TIME] < cut]
    st = fit_stats(fit_frame)
    Y, grid = target_matrix(train, st)
    valid = (~np.isnan(Y)).astype(np.float32)
    Yc = np.nan_to_num(Y)
    U, T = Y.shape
    how_all = _how_of(grid)
    prof_all = st["profile"][:, how_all]
    sl = slice(hold, hold + H)
    y, v, pr = Yc[:, sl], valid[:, sl], prof_all[:, sl]
    print(f"grid {grid[0]} -> {grid[-1]} ({T} h, {U} series) | holdout starts {cut}")

    # ================================================================ A
    print("\n" + "=" * 90)
    print("A. THE LEVEL PROCESS (level = median of y / global profile; 1.0 = profile)")
    for L, name in ((WEEK, "week"), (2 * WEEK, "fortnight")):
        K = hold // L                               # full blocks before the cut
        edges = [hold - (K - k) * L for k in range(K + 1)] + [hold + L]  # + holdout block
        nb = len(edges) - 1
        Lv = np.stack([level_block(Yc, valid, prof_all, edges[k], edges[k + 1])
                       for k in range(nb)], axis=1)                       # (U, nb)
        G = np.median(Lv, axis=0)
        starts = [grid[e].strftime("%m-%d") for e in edges[:-1]]
        print(f"\nglobal level per {name} (block start -> median across units; last block = holdout):")
        print("  " + "  ".join(f"{s}:{g:.3f}" for s, g in zip(starts, G)))
        lags = list(range(1, 9 if L == WEEK else 5))
        pre = Lv[:, :-1]                            # exclude holdout from the stats
        Gp = G[:-1]
        print(f"  autocorrelation of the GLOBAL level, lags 1..{lags[-1]} {name}s: "
              + "  ".join(f"{k}:{r:+.2f}" for k, r in zip(lags, acf(Gp, lags))))
        print(f"  pooled per-unit autocorrelation,       lags 1..{lags[-1]} {name}s: "
              + "  ".join(f"{k}:{pooled_corr(pre, k):+.2f}" for k in lags))
        sd_units = np.nanstd(pre, axis=1).mean()
        print(f"  spread: global level sd {Gp.std():.3f} | mean per-unit sd {sd_units:.3f} | "
              f"per-unit sd around the global level {np.nanstd(pre - Gp[None, :], axis=1).mean():.3f}")
        if L == WEEK:
            Z = pre - pre.mean(1, keepdims=True)
            C = np.corrcoef(Z)
            off = C[~np.eye(U, dtype=bool)].mean()
            s = np.linalg.svd(Z, compute_uv=False)
            print(f"  synchrony across units: mean pairwise corr of weekly levels {off:+.2f}, "
                  f"PC1 explains {100 * s[0] ** 2 / (s ** 2).sum():.0f}% of level variance")
            Lw, Gw, edges_w, K_w = Lv, G, edges, K
    # holdout phase check
    print(f"\n  holdout level {Gw[-1]:.3f} | 1 fortnight before {np.median(level_block(Yc, valid, prof_all, hold - 336, hold)):.3f} "
          f"| 2 before (same phase of a 4-wk cycle) {np.median(level_block(Yc, valid, prof_all, hold - 672, hold - 336)):.3f} "
          f"| 4 before {np.median(level_block(Yc, valid, prof_all, hold - 1344, hold - 1008)):.3f}")

    # ================================================================ B
    print("\n" + "=" * 90)
    print("B. WHAT TRACKS THE LEVEL (weekly blocks before the holdout)")
    nbw = Lw.shape[1] - 1
    Cw = {}
    rows = []
    for c in DYN:
        M = cov_matrix(train, st, grid, c)
        tot_sd = float(np.nanstd(M))
        cross_sd = float(np.nanmean(np.nanstd(M, axis=0)))
        Cw[c] = np.array([np.nanmean(M[:, edges_w[k]:edges_w[k + 1]]) for k in range(nbw)])
        r = np.corrcoef(Cw[c], Gw[:-1])[0, 1]
        rows.append(dict(covariate=c, global_share=1 - cross_sd / max(tot_sd, 1e-9),
                         nan_share=float(np.isnan(M).mean()),
                         corr_with_global_level=r, mean=float(np.nanmean(M)), sd=tot_sd))
    tb = pd.DataFrame(rows).sort_values("corr_with_global_level", key=np.abs, ascending=False)
    print(tb.round(3).to_string(index=False))
    print("(global_share ~1 = same value for every unit at each hour; ~0 = fully per-unit)")
    Xw = np.stack([Cw[c] for c in DYN], axis=1)
    print(f"leave-one-out ridge R^2, global weekly level ~ 13 weekly covariate means: "
          f"{loo_ridge_r2(Xw, Gw[:-1]):.2f}   (n={nbw} weeks; <0 means worse than the mean)")
    # unit-week level vs the unit's own weekly covariate means, unit fixed effects removed
    Xu, yu, gu = [], [], []
    Lpre = Lw[:, :-1]
    per_unit_cov = {}
    for c in DYN:
        M = cov_matrix(train, st, grid, c)
        per_unit_cov[c] = np.stack([np.nanmean(M[:, edges_w[k]:edges_w[k + 1]], axis=1)
                                    for k in range(nbw)], axis=1)          # (U, nbw)
    for i in range(U):
        Xi = np.stack([per_unit_cov[c][i] for c in DYN], axis=1)
        Xi = np.nan_to_num(Xi - np.nanmean(Xi, axis=0))
        Xu.append(Xi); yu.append(Lpre[i] - Lpre[i].mean()); gu.append(np.arange(nbw))
    Xu, yu, gu = np.concatenate(Xu), np.concatenate(yu), np.concatenate(gu)
    r2u = blocked_cv_r2(Xu, yu, gu)
    print(f"week-blocked CV R^2, unit-week level ~ unit's own weekly covariate means "
          f"(unit fixed effects removed): {r2u:.2f}   (n={len(yu)})")
    # add the global level as a feature: how much of the unit level is just the global factor
    G_rep = np.tile(Gw[:-1] - Gw[:-1].mean(), U)
    r2g = blocked_cv_r2(G_rep[:, None], yu, gu)
    print(f"same, using only the global level as the predictor: {r2g:.2f}")

    # ================================================================ C
    print("\n" + "=" * 90)
    print("C. SPIKES")
    rs = st["res_scale"][:, None]
    resid = y - pr
    obs = v > 0
    mk = cov_matrix(train, st, grid, "maintenance_known")[:, sl]
    maint = obs & (np.nan_to_num(mk) != 0)
    spike = obs & ~maint & (resid > 2 * rs)
    dip = obs & ~maint & (resid < -2 * rs)
    ordinary = obs & ~(maint | spike | dip)
    if os.path.exists(a.pred):
        P = pd.read_csv(a.pred)
        P[TIME] = pd.to_datetime(P[TIME])
        pred = np.zeros_like(y)
        for u in st["units"]:
            pred[st["uid"][u]] = (P[P[SERIES] == u].set_index(TIME)
                                  .reindex(grid[sl])["pred"].to_numpy(np.float32))
        se = (pred - y) ** 2 * v
        ae = np.abs(pred - y) * v
        print("share of the holdout error by regime (checkpoint predictions):")
        for name, m in (("maintenance", maint), ("spike", spike), ("dip", dip), ("ordinary", ordinary)):
            print(f"  {name:12s} hours {100 * m.sum() / obs.sum():5.1f}%  |y| {100 * (np.abs(y) * m).sum() / (np.abs(y) * obs).sum():5.1f}%  "
                  f"|err| {100 * ae[m].sum() / ae.sum():5.1f}%  squared err {100 * se[m].sum() / se.sum():5.1f}%")
        # what would perfect spike prediction be worth
        p2 = pred.copy(); p2[spike] = y[spike]
        print("six metrics if spike hours were predicted perfectly (everything else unchanged):")
        show({"checkpoint": six(y, pred, v), "checkpoint + perfect spikes": six(y, p2, v)})
        p3 = pred.copy(); p3[spike] = 0.5 * (pred[spike] + y[spike])
        print("... and if spikes were predicted halfway (what a mean-seeking member could do at best):")
        show({"checkpoint + half spikes": six(y, p3, v)})
    else:
        print(f"({a.pred} not found; skipping the checkpoint-based part)")

    # spike predictability over the whole train (residual against the global profile)
    R = (Yc - prof_all) / st["res_scale"][:, None]
    ok = valid > 0
    mk_all = np.nan_to_num(cov_matrix(train, st, grid, "maintenance_known")) != 0
    sp_all = ok & ~mk_all & (R > 2)
    base = ok & ~mk_all & ~sp_all
    print(f"\nspike hours over all of train: {100 * sp_all.sum() / ok.sum():.2f}% of observed hours "
          f"({sp_all.sum()} hours); mean y in spikes {Yc[sp_all].mean():.2f} vs {Yc[base].mean():.2f} otherwise")
    rows = []
    covs = {}
    for c in DYN:
        M = cov_matrix(train, st, grid, c)
        covs[c] = M
        a1, a0 = M[sp_all], M[base]
        a1, a0 = a1[~np.isnan(a1)], a0[~np.isnan(a0)]
        pooled = np.sqrt(0.5 * (a1.var() + a0.var())) + 1e-9
        rows.append(dict(covariate=c, cohens_d=(a1.mean() - a0.mean()) / pooled,
                         mean_in_spikes=a1.mean(), mean_otherwise=a0.mean()))
    tb = pd.DataFrame(rows).sort_values("cohens_d", key=np.abs, ascending=False)
    print("covariates that separate spike hours from the rest (Cohen's d, pooled over units):")
    print(tb.round(3).to_string(index=False))
    for c in list(tb.covariate[:3]) + (["shock_risk"] if "shock_risk" not in list(tb.covariate[:3]) else []):
        M = covs[c]
        vals = M[ok & ~mk_all]
        flags = sp_all[ok & ~mk_all]
        keep = ~np.isnan(vals)
        vals, flags = vals[keep], flags[keep]
        qs = np.quantile(vals, np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(qs, vals, side="right") - 1, 0, 9)
        rate = [100 * flags[dec == d].mean() for d in range(10)]
        print(f"  spike rate (%) by decile of {c}: " + " ".join(f"{r:4.1f}" for r in rate)
              + f"   (top-decile lift x{rate[-1] / max(np.mean(rate), 1e-9):.1f})")

    # ================================================================ D
    print("\n" + "=" * 90)
    print("D. HOLDOUT BASE VARIANTS USING SAME-PHASE LEVELS (no model)")
    res = {"global profile": six(y, pr, v)}
    lv672 = level_block(Yc, valid, prof_all, hold - 672, hold - 336)
    lv1344 = level_block(Yc, valid, prof_all, hold - 1344, hold - 1008)
    lv336 = level_block(Yc, valid, prof_all, hold - 336, hold)
    for name, lv in (("lag-672 level", lv672), ("mean of lag-672 & lag-1344", 0.5 * (lv672 + lv1344)),
                     ("lag-336 level (half phase)", lv336)):
        for lam in (0.25, 0.5, 0.75, 1.0):
            res[f"profile x {name}, lam={lam}"] = six(y, pr * (1 + lam * (lv[:, None] - 1)), v)
    # phase-specific profile: fit only on fortnights with the holdout's phase
    K2 = hold // 336
    fn_idx = np.full(T, -1)
    for k in range(K2):
        lo, hi = hold - (K2 - k) * 336, hold - (K2 - k - 1) * 336
        fn_idx[lo:hi] = k
    same_phase = {k for k in range(K2) if (K2 - k) % 2 == 0}
    hrs = np.where(np.isin(fn_idx, list(same_phase)))[0]
    mask_rows = train[TIME].isin(grid[hrs])
    st_ph = fit_stats(train[mask_rows & (train[TIME] < cut)])
    res["profile from same-phase fortnights only"] = six(y, st_ph["profile"][:, how_all[sl]], v)
    other = {k for k in range(K2) if (K2 - k) % 2 == 1}
    hrs = np.where(np.isin(fn_idx, list(other)))[0]
    st_op = fit_stats(train[train[TIME].isin(grid[hrs]) & (train[TIME] < cut)])
    res["profile from opposite-phase fortnights only"] = six(y, st_op["profile"][:, how_all[sl]], v)
    show(res)


if __name__ == "__main__":
    main()
