"""Inference entrypoint for the graded submission.

    python predict.py --input_dir /data/input \
        --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt

The input directory supplies train.csv (observed history), <kind>_input.csv
(future covariates) and forecast_index_<kind>.csv (the authoritative row list).
"""
from __future__ import annotations

import argparse, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from core import (SERIES, TIME, ResidualEncoder, build_block, forecast,
                  min_hist, target_matrix)


def detect_kind(input_dir):
    for kind in ("test", "validation"):
        if os.path.exists(os.path.join(input_dir, f"forecast_index_{kind}.csv")):
            return kind
    raise FileNotFoundError(f"no forecast_index_*.csv in {input_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--checkpoint", required=True)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    st, cfg = ck["stats"], ck["cfg"]
    lags = tuple(cfg.get("lags", []))

    kind = detect_kind(a.input_dir)
    p = lambda f: os.path.join(a.input_dir, f)
    train = pd.read_csv(p("train.csv")); train[TIME] = pd.to_datetime(train[TIME])
    cov = pd.read_csv(p(f"{kind}_input.csv")); cov[TIME] = pd.to_datetime(cov[TIME])
    idx = pd.read_csv(p(f"forecast_index_{kind}.csv"))
    idx[TIME] = pd.to_datetime(idx[TIME])

    Y, grid = target_matrix(train, st)
    stamps = pd.DatetimeIndex(np.sort(cov[TIME].unique()))
    abs_hours = ((stamps - grid[0]) // pd.Timedelta(hours=1)).to_numpy()
    if lags:
        assert (abs_hours - max(lags) - 0).min() >= 0, "lag reaches before history"
    X, how, prof = build_block(cov, st, Y, abs_hours, grid[0], lags)

    states = ck["state_dict"]
    states = states if isinstance(states, list) else [states]
    models = []
    for sd in states:
        m = ResidualEncoder(n_feat=cfg["n_feat"], n_units=cfg["n_units"],
                            d_model=cfg["d_model"], layers=cfg["layers"],
                            nhead=cfg.get("nhead", 8),
                            ff=cfg.get("ff", 4 * cfg["d_model"]),
                            dropout=cfg["dropout"],
                            film=cfg.get("film", True),
                            level_head=cfg.get("level_head", False)).to(dev)
        m.load_state_dict(sd); models.append(m)
    pred = forecast(models, X, how, prof, st, dev, weights=ck.get("weights"))

    out = pd.DataFrame({
        SERIES: np.repeat(st["units"], len(stamps)),
        TIME: np.tile(stamps, len(st["units"])),
        "prediction": pred.reshape(-1)})
    sub = idx.merge(out, on=[SERIES, TIME], how="left")
    assert len(sub) == len(idx), "row count mismatch against forecast index"
    if sub["prediction"].isna().any():
        n = int(sub["prediction"].isna().sum())
        print(f"warning: {n} unmatched rows, filling with per-series median")
        sub["prediction"] = sub["prediction"].fillna(
            sub.groupby(SERIES)["prediction"].transform("median")).fillna(0.0)
    sub["prediction"] = sub["prediction"].clip(lower=0.0)
    sub[TIME] = sub[TIME].dt.strftime("%Y-%m-%d %H:%M:%S")

    d = os.path.dirname(os.path.abspath(a.output_file))
    os.makedirs(d, exist_ok=True)
    sub.to_csv(a.output_file, index=False)
    print(f"wrote {a.output_file}  rows={len(sub)}  kind={kind}  models={len(models)}")


if __name__ == "__main__":
    main()