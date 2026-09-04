"""Convert Rossmann Store Sales into the benchmark schema.

    python prep_rossmann.py --train /path/train.csv --store /path/store.csv \
        --out_dir data --n_stores 100 --seed 0

Writes data/train.csv with columns
    series_id, timestamp, target, <FEATURES ...>
where target = Sales, missing (NaN) on closed days so that closed days are
excluded from the loss and the metrics (Kaggle convention: sales on closed
days are trivially zero).  All features are known ahead of time: calendar
harmonics, Promo, an active-Promo2 flag, SchoolHoliday, Open, StateHoliday as
three binaries, competition distance / competition-open flag, and the static
store type and assortment one-hots.  `Customers` is deliberately dropped: it is
only known after the fact.

Stores are subsampled at random among those with a complete 942-day history
(180 stores have a six-month refurbishment gap).
"""
from __future__ import annotations

import argparse, os
import numpy as np, pandas as pd

MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
          8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_stores", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tr = pd.read_csv(a.train, low_memory=False)
    st = pd.read_csv(a.store)
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["StateHoliday"] = tr["StateHoliday"].astype(str)

    counts = tr.groupby("Store").size()
    complete = counts[counts == counts.max()].index.to_numpy()
    rng = np.random.default_rng(a.seed)
    keep = np.sort(rng.choice(complete, size=min(a.n_stores, len(complete)), replace=False))
    print(f"{len(counts)} stores, {len(complete)} with complete history, keeping {len(keep)}")

    df = tr[tr["Store"].isin(keep)].merge(st, on="Store", how="left")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    d = df["Date"]
    out = pd.DataFrame({
        "series_id": "store_" + df["Store"].astype(str).str.zfill(4),
        "timestamp": d.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "target": df["Sales"].astype(float).where(df["Open"] == 1),
        "dow_sin": np.sin(2 * np.pi * d.dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * d.dt.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * (d.dt.month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (d.dt.month - 1) / 12),
        "promo": df["Promo"].astype(float),
        "school_holiday": df["SchoolHoliday"].astype(float),
        "open": df["Open"].astype(float),
        "state_holiday_a": (df["StateHoliday"] == "a").astype(float),
        "state_holiday_b": (df["StateHoliday"] == "b").astype(float),
        "state_holiday_c": (df["StateHoliday"] == "c").astype(float),
        "comp_dist_log": np.log1p(df["CompetitionDistance"]),
    })
    # competition open flag: 1 after the competitor opened, 0 before, NaN if unknown
    known = df["CompetitionOpenSinceYear"].notna() & df["CompetitionOpenSinceMonth"].notna()
    comp_start = pd.to_datetime(dict(year=df["CompetitionOpenSinceYear"].fillna(2000).astype(int),
                                     month=df["CompetitionOpenSinceMonth"].fillna(1).astype(int),
                                     day=1), errors="coerce")
    out["comp_open"] = np.where(known, (d >= comp_start).astype(float), np.nan)
    # Promo2 active: enrolled, after the start week, and in one of the interval months
    p2 = df["Promo2"] == 1
    start = pd.to_datetime((df["Promo2SinceYear"].fillna(2100).astype(int)).astype(str) + "-W"
                           + df["Promo2SinceWeek"].fillna(1).astype(int).astype(str).str.zfill(2)
                           + "-1", format="%G-W%V-%u", errors="coerce")
    month_in = [(MONTHS[m] in str(iv)) if isinstance(iv, str) else False
                for m, iv in zip(d.dt.month, df["PromoInterval"])]
    out["promo2_active"] = (p2 & (d >= start) & np.array(month_in)).astype(float)
    for t in "abcd":
        out[f"store_type_{t}"] = (df["StoreType"] == t).astype(float)
    for t in "abc":
        out[f"assortment_{t}"] = (df["Assortment"] == t).astype(float)

    out.to_csv(os.path.join(a.out_dir, "train.csv"), index=False)
    print(f"wrote {a.out_dir}/train.csv  rows={len(out)}  stores={out.series_id.nunique()}  "
          f"{out.timestamp.min()} -> {out.timestamp.max()}")
    print(f"target observed (open) share {out.target.notna().mean():.3f}  "
          f"promo2_active share {out.promo2_active.mean():.3f}  comp_open NaN share {out.comp_open.isna().mean():.3f}")


if __name__ == "__main__":
    main()
