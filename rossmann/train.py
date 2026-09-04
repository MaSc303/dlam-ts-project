"""Train the residual encoder.

Selection phase (multi-window holdout, trustworthy comparisons):
    python train.py --data_dir data --val_windows 3 --stride 6 \
        --d_model 256 --layers 4 --batch 64 --epochs 30 --out ck_sel.pt

Confirmation (single holdout, maximum data, same config):
    python train.py --data_dir data --val_windows 1 --stride 6 ... --out ck_conf.pt

Ship (no holdout, all 4320 h, epoch count carried over unchanged):
    python train.py --data_dir data --no_holdout --stride 6 ... --out ck_final.pt

Diagnostics only:
    python train.py --data_dir data --diagnose
"""
from __future__ import annotations

import argparse, copy, os, sys, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from core import (H, SERIES, TIME, ResidualEncoder, WindowDS, compose, fit_stats,
                  forecast, build_block, min_hist, n_feat, parse_lags,
                  target_matrix, wape)


# ------------------------------------------------------------------ diagnostics
def diagnose(train, st):
    Y, grid = target_matrix(train, st)
    U, T = Y.shape
    hold = T - H
    true = Y[:, hold:]
    print(f"\ngrid {grid[0]} -> {grid[-1]}  ({T} h, {U} series)")
    print(f"holdout: {grid[hold]} -> {grid[-1]}\n")
    how = ((grid.dayofweek * 24 + grid.hour).to_numpy())[hold:]
    rows = [("naive last value", np.repeat(Y[:, hold - 1:hold], H, axis=1)),
            ("seasonal profile", st["profile"][:, how])]
    for k in (24, 168, 336, 504, 672, 840, 1008, 1176, 1344):
        rows.append((f"lag-{k} direct", Y[:, hold - k:hold - k + H]))
    print(f"{'baseline':22s} {'WAPE':>8s}   {'usable at test?':>15s}")
    print("-" * 50)
    for name, pred in rows:
        k = int(name.split("-")[1].split()[0]) if name.startswith("lag") else None
        ok = "-" if k is None else ("yes" if k >= 672 else "NO (hidden)")
        print(f"{name:22s} {wape(true, np.nan_to_num(pred)):8.4f}   {ok:>15s}")
    print("\nautocorrelation within each series (mean over series):")
    for k in (24, 168, 336, 672, 1008, 1344):
        rs = []
        for u in range(U):
            a, b = Y[u, 1512:], Y[u, 1512 - k:T - k]
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() > 10 and a[m].std() > 0 and b[m].std() > 0:
                rs.append(np.corrcoef(a[m], b[m])[0, 1])
        print(f"  lag {k:5d}:  r = {np.mean(rs):.4f}")


# ------------------------------------------------------------------ evaluation
def eval_windows(model, X, how, prof, Yc, valid, st, dev, starts):
    """Per-window WAPE plus the pooled figure.

    Pooled accumulates |error| and |y| across every window before dividing,
    which is how the leaderboard computes WAPE.  Averaging per-window ratios
    would weight a quiet fortnight the same as a busy one.
    """
    per, num, den, sq, cnt = [], 0.0, 0.0, 0.0, 0.0
    for s in starts:
        sl = slice(s, s + H)
        p = forecast(model, X[:, sl], how[:, sl], prof[:, sl], st, dev)
        y, v = Yc[:, sl], valid[:, sl]
        n = float((np.abs(y - p) * v).sum())
        d = float((np.abs(y) * v).sum())
        per.append(n / max(d, 1e-9))
        num += n; den += d
        sq += float((((y - p) ** 2) * v).sum()); cnt += float(v.sum())
    return per, num / max(den, 1e-9), sq / max(cnt, 1e-9)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="checkpoint.pt")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--val_windows", type=int, default=1,
                    help="consecutive 336h blocks held out at the end of train")
    ap.add_argument("--no_holdout", action="store_true",
                    help="train on all data, no evaluation, keep final weights")
    ap.add_argument("--lags", default="none", help="'none' or e.g. '672,840' (>=672)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--ff", type=int, default=0, help="0 -> 4 * d_model")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no_film", action="store_true")
    ap.add_argument("--level_head", type=int, default=1,
                    help="1: multiplicative per-window level on the profile (v3)")
    ap.add_argument("--loss", default="l1", choices=["l1", "mse", "huber"],
                    help="l1 = WAPE (median-seeking); mse = mean-seeking member; "
                         "huber = in between, delta in raw units")
    ap.add_argument("--huber_delta", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lags = parse_lags(a.lags)
    ff = a.ff or 4 * a.d_model

    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    train[TIME] = pd.to_datetime(train[TIME])
    print(f"train {train.shape} | {train[SERIES].nunique()} series | device {dev}")

    T_total = int(train[SERIES].value_counts().iloc[0])
    stamps = sorted(train[TIME].unique())
    nwin = 0 if a.no_holdout else max(1, a.val_windows)
    hold_start = T_total - nwin * H            # == T_total when no_holdout
    fit_frame = train if a.no_holdout else train[train[TIME] < stamps[hold_start]]
    st = fit_stats(fit_frame)                  # statistics never see the holdout

    if a.diagnose:
        diagnose(train, st); return

    Y, grid = target_matrix(train, st)
    hours = np.arange(Y.shape[1])
    if lags:
        hours = hours.copy(); hours[:min_hist(lags)] = min_hist(lags)
    X, how, prof = build_block(train, st, Y, hours, grid[0], lags)
    valid = (~np.isnan(Y)).astype(np.float32)
    Yc = np.nan_to_num(Y)

    max_start = None if a.no_holdout else hold_start - H
    ds = WindowDS(X, how, prof, Yc, valid, stride=a.stride,
                  min_start=min_hist(lags), max_start=max_start)
    starts = [hold_start + w * H for w in range(nwin)]
    if a.no_holdout:
        print(f"NO HOLDOUT — training on all {T_total} h, keeping final weights")
    else:
        print(f"holdout {nwin} x {H} h: {grid[hold_start]} -> {grid[-1]}")
    print(f"X {X.shape} | lags {lags or 'none'} | n_feat {n_feat(lags)}")
    print(f"windows {len(ds)} | starts {min_hist(lags)}.."
          f"{max(s for _, s in ds.index)}")

    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, pin_memory=(dev == "cuda"),
                    persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers > 0 else None)

    amp_dtype, scaler = None, None
    if a.amp != "off" and dev == "cuda":
        use = a.amp if a.amp != "auto" else (
            "bf16" if torch.cuda.is_bf16_supported() else "fp16")
        amp_dtype = torch.bfloat16 if use == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda") if use == "fp16" else None
        print(f"amp: {use}")

    model = ResidualEncoder(n_feat=n_feat(lags), n_units=len(st["units"]),
                            d_model=a.d_model, nhead=a.nhead, layers=a.layers,
                            ff=ff, dropout=a.dropout, film=not a.no_film,
                            level_head=bool(a.level_head)).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()):,} | "
          f"d_model {a.d_model} layers {a.layers} ff {ff} film {not a.no_film} "
          f"level_head {bool(a.level_head)} loss {a.loss}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.15)
    rs = torch.tensor(st["res_scale"], device=dev)

    best, best_state, best_ep = 1e9, None, -1
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); N = D = 0.0
        te = time.time()
        for x, u, hw, pr, y, v in dl:
            x, u, hw, pr, y, v = (t.to(dev, non_blocking=True)
                                  for t in (x, u, hw, pr, y, v))
            with torch.autocast(dev, dtype=amp_dtype, enabled=amp_dtype is not None):
                p = compose(model(x, u, hw), pr, u, rs)
                if a.loss == "l1":                      # batch WAPE
                    num = ((p - y).abs() * v).sum()
                    den = (y.abs() * v).sum()
                elif a.loss == "mse":                   # batch relative MSE
                    num = (((p - y) ** 2) * v).sum()
                    den = ((y ** 2) * v).sum()
                else:                                   # huber, raw units
                    e = (p - y).abs(); d = a.huber_delta
                    hub = torch.where(e <= d, 0.5 * e ** 2, d * (e - 0.5 * d))
                    num = (hub * v).sum()
                    den = (y.abs() * v).sum() * d
                loss = num / den.clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler: scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler: scaler.step(opt); scaler.update()
            else: opt.step()
            sched.step()
            N += num.detach().item(); D += den.detach().item()
        tr = N / max(D, 1e-6)
        dt = time.time() - te

        if a.no_holdout:
            print(f"ep {ep:3d}/{a.epochs}  train {tr:.4f}  ({dt:.0f}s)")
            continue

        if ep % a.eval_every == 0 or ep == a.epochs:
            per, pooled, mse = eval_windows(model, X, how, prof, Yc, valid, st, dev, starts)
            crit = mse if a.loss == "mse" else pooled
            if crit < best:
                best, best_ep = crit, ep
                best_state = copy.deepcopy(model.state_dict())
            detail = "  ".join(f"w{i}{p:.4f}" for i, p in enumerate(per))
            print(f"ep {ep:3d}/{a.epochs}  train {tr:.4f}  pooled {pooled:.4f}  "
                  f"mse {mse:.3f}  [{detail}]  best {best:.4f}@{best_ep}  ({dt:.0f}s)")
        else:
            print(f"ep {ep:3d}/{a.epochs}  train {tr:.4f}  ({dt:.0f}s)")

    if a.no_holdout:
        best, best_state, best_ep = float("nan"), model.state_dict(), a.epochs
        print(f"\nfinished {a.epochs} epochs | {time.time()-t0:.0f}s")
    else:
        per, _, _ = eval_windows(model, X, how, prof, Yc, valid, st, dev, starts)
        print(f"\nbest {'MSE' if a.loss == 'mse' else 'pooled WAPE'} {best:.4f} at epoch {best_ep} "
              f"| final per-window {[round(p, 4) for p in per]} "
              f"| {time.time()-t0:.0f}s")
        if len(per) > 1:
            print(f"window spread: {max(per) - min(per):.4f} — differences "
                  f"smaller than this between configs are period noise")

    torch.save({"state_dict": best_state, "stats": fit_stats(train),
                "cfg": {"d_model": a.d_model, "layers": a.layers, "nhead": a.nhead,
                        "ff": ff, "dropout": a.dropout, "n_feat": n_feat(lags),
                        "n_units": len(st["units"]), "film": not a.no_film,
                        "lags": list(lags), "level_head": bool(a.level_head),
                        "loss": a.loss},
                "holdout_wape": best, "best_epoch": best_ep,
                "val_windows": nwin, "seed": a.seed,
                "args": vars(a)}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()