"""Figure: the calendar-month cycle in the benchmark and in Rossmann.

    python fig_cycles.py --dlam_train DLAM_v3_notrend/data/train.csv \
        --ross_train /path/to/rossmann/train.csv --out fig_cycles

Left panel : benchmark global level over time (median across units of each
             unit's daily median of target / hour-of-week profile), with the
             second half of every month shaded.
Right panel: day-of-month profile of that level, next to Rossmann open-day
             sales by day of month (per-store normalised, all 1,115 stores).
Writes <out>.pdf and <out>.png.
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def benchmark_level(path):
    df = pd.read_csv(path, usecols=["series_id", "timestamp", "target"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["how"] = df["timestamp"].dt.dayofweek * 24 + df["timestamp"].dt.hour
    prof = df.groupby(["series_id", "how"])["target"].transform("median")
    df["ratio"] = np.where(prof > 1e-6, df["target"] / prof.where(prof > 1e-6, 1.0), np.nan)
    df["day"] = df["timestamp"].dt.normalize()
    per_unit_day = df.groupby(["series_id", "day"])["ratio"].median()
    daily = per_unit_day.groupby("day").median()                 # global daily level
    return daily


def rossmann_dom(path):
    df = pd.read_csv(path, usecols=["Store", "Date", "Sales", "Open"], low_memory=False)
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["norm"] = df["Sales"] / df.groupby("Store")["Sales"].transform("mean")
    return df.groupby(df["Date"].dt.day)["norm"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlam_train", required=True)
    ap.add_argument("--ross_train", required=True)
    ap.add_argument("--out", default="fig_cycles")
    a = ap.parse_args()

    daily = benchmark_level(a.dlam_train)
    dom_b = daily.groupby(daily.index.day).mean()
    dom_r = rossmann_dom(a.ross_train)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    # ---- left: level over time with second-half-of-month shading
    ax1.plot(daily.index, daily.values, color="0.55", lw=0.9, label="daily level")
    ax1.plot(daily.index, daily.rolling(7, center=True).mean().values,
             color="C0", lw=2.0, label="7-day mean")
    ax1.axhline(1.0, color="k", lw=0.8, ls="--")
    months = pd.date_range(daily.index.min().replace(day=1), daily.index.max(), freq="MS")
    for m in months:
        start = m + pd.Timedelta(days=15)
        end = (m + pd.offsets.MonthEnd(1)) + pd.Timedelta(days=1)
        ax1.axvspan(start, end, color="C1", alpha=0.12, lw=0)
    ax1.set_ylabel("level relative to hour-of-week profile")
    ax1.set_title("Benchmark: global level over time (shaded = 16th to month end)")
    ax1.legend(loc="lower left", frameon=False, fontsize=8)
    ax1.set_xlim(daily.index.min(), daily.index.max())
    fig.autofmt_xdate()

    # ---- right: day-of-month profiles
    ax2.plot(dom_b.index, dom_b.values, marker="o", ms=3, color="C0",
             label="benchmark level")
    ax2.plot(dom_r.index, dom_r.values, marker="s", ms=3, color="C3",
             label="Rossmann sales (per-store normalised)")
    ax2.axhline(1.0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("day of month")
    ax2.set_ylabel("relative to mean")
    ax2.set_title("Day-of-month profile")
    ax2.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax2.legend(loc="lower left", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=200)
    print(f"benchmark level by day of month: {dom_b.round(3).to_dict()}")
    print(f"rossmann sales by day of month:  {dom_r.round(3).to_dict()}")
    print(f"wrote {a.out}.pdf and {a.out}.png")


if __name__ == "__main__":
    main()
