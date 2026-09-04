import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_data
from src.data.tide_dataset import FeaturePipeline, STATIC_COLS
from src.evaluation.metrics import wape
from src.evaluation.tide_ensemble import predict, predict_ensemble
from src.training.tide_train import make_val_loader, train_model

SEED, HORIZON = 1337, 336
DROP_COLS = ["trend"]


CONFIGS = [
    (336, "series", 1337),
    (504, "series", 1337),
    (168, "series", 1337),
    (168, "none", 1337),
]

EPOCHS, PATIENCE = 60, 8
BATCH_SIZE, LR = 256, 1e-3
N_VAL_ORIGINS, VAL_STRIDE = 4, 168
REFIT_ON_FULL = True

CHECKPOINT_DIR = Path("checkpoints")


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    print("device:", device, "| torch:", torch.__version__)

    # 2. Daten laden
    train = load_data("train.csv")
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    val_in = load_data("validation_input.csv")
    val_in["timestamp"] = pd.to_datetime(val_in["timestamp"])
    val_idx = load_data("forecast_index_validation.csv")
    val_idx["timestamp"] = pd.to_datetime(val_idx["timestamp"])
    train = train.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    assert train.groupby("series_id").size().nunique() == 1, "Serien ungleich lang"
    assert (val_in.groupby("series_id").size() == HORIZON).all()
    assert train["target"].isna().sum() == 0, "NaN im Target"
    _d = train.groupby("series_id")["timestamp"].diff().dropna()
    assert (_d == pd.Timedelta("1h")).all(), "Luecken im Zeitraster"
    _gap = (
        val_in.groupby("series_id")["timestamp"].min()
        - train.groupby("series_id")["timestamp"].max()
    )
    assert (_gap == pd.Timedelta("1h")).all(), "val_in schliesst nicht an"

    dynamic_cols = [
        c
        for c in train.columns
        if c not in ("series_id", "timestamp", "target")
        and c not in STATIC_COLS
        and c not in DROP_COLS
    ]

    print("train", train.shape, "| val_in", val_in.shape)
    print(len(dynamic_cols), "dynamische Kovariaten, entfernt:", DROP_COLS)

    # 3. Dreiteilung
    n = train.groupby("series_id").cumcount(ascending=False)
    train_df = train[n >= 2 * HORIZON].copy()
    early_df = train[n >= HORIZON].copy()
    full_df = train.copy()

    for nm, d in [("train_df", train_df), ("early_df", early_df), ("full_df", full_df)]:
        print(
            f"{nm:9s} {d.shape}  je Serie {d.groupby('series_id').size().iloc[0]}"
            f"  bis {d['timestamp'].max().date()}"
        )

    assert (
        train_df["timestamp"].max()
        < early_df["timestamp"].max()
        < full_df["timestamp"].max()
    )
    print("Splits zeitlich getrennt.")

    # 4. Feature-Pipeline
    pipe = FeaturePipeline(train_df, dynamic_cols)
    train_t = pipe.transform(train_df)
    early_t = pipe.transform(early_df)
    full_t = pipe.transform(full_df)

    print(len(pipe.feature_cols), "Features (inkl. seasonal_profile)")

    # 7. Baselines
    ev = early_df.groupby("series_id").tail(HORIZON)
    y_ev = ev["target"].to_numpy()
    scores = {}

    last = train_df.groupby("series_id")["target"].last()
    scores["naive_last"] = wape(y_ev, ev["series_id"].map(last).to_numpy())

    for lag in (24, 168):
        p = []
        for _, g in early_df.groupby("series_id", sort=False):
            y = g["target"].to_numpy(np.float32)
            s = len(g) - HORIZON
            p.append(
                np.array([y[s + i - lag * (i // lag + 1)] for i in range(HORIZON)])
            )
        scores[f"lag{lag}_repeat"] = wape(y_ev, np.concatenate(p))

    _idx = pd.MultiIndex.from_arrays(
        [ev["series_id"], ev["timestamp"].dt.dayofweek, ev["timestamp"].dt.hour]
    )
    scores["seasonal_profile"] = wape(
        y_ev, np.nan_to_num(pipe.profile.reindex(_idx).to_numpy(), nan=pipe.global_mean)
    )

    print("\nBaselines (Early-Split):")
    for k, v in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {k:18s} {v:.4f}")

    # 8. Training
    t0 = time.time()
    ckpts, best_epochs = [], []

    for L, mode, seed in CONFIGS:
        print(f"\n=== L={L}, norm={mode}, seed={seed} ===")
        vl = make_val_loader(early_t, L, HORIZON, pipe, N_VAL_ORIGINS, VAL_STRIDE)
        ckpt, w, ep = train_model(
            L,
            mode,
            seed,
            train_t,
            pipe,
            HORIZON,
            device,
            BATCH_SIZE,
            LR,
            CHECKPOINT_DIR,
            val_loader=vl,
            epochs=EPOCHS,
            patience=PATIENCE,
        )
        ckpts.append((ckpt, L, mode))
        best_epochs.append(ep)
        print(f"  bestes val_wape {w:.4f} in Epoche {ep}")

    print(f"\nGesamt {(time.time()-t0)/60:.1f} min")

    # 9. Ensemble und Kalibrierung
    n_rows = early_t.groupby("series_id").size().iloc[0]
    parts, y_val = [], None
    for ckpt, L, mode in ckpts:
        st = [n_rows - L - HORIZON - k * VAL_STRIDE for k in range(N_VAL_ORIGINS)][::-1]
        p, y_val, _ = predict(ckpt, L, mode, early_t, pipe, HORIZON, device, starts=st)
        parts.append(p)
        print(f"  {mode}_L{L:3d} einzeln {wape(y_val, p):.4f}")

    ens_val = np.mean(parts, axis=0)
    print(f"  Ensemble roh    {wape(y_val, ens_val):.4f}")

    factors = np.arange(0.95, 1.0501, 0.005)
    sc = [wape(y_val, ens_val * f) for f in factors]
    cal = float(factors[int(np.argmin(sc))])
    print(f"  Faktor {cal:.3f} -> {min(sc):.4f}")

    print("\nFehlerkorrelation zwischen den Modellen:")
    res = {f"{m}_L{L}": (p - y_val).ravel() for (_, L, m), p in zip(ckpts, parts)}
    for a, b in itertools.combinations(res, 2):
        print(f"  {np.corrcoef(res[a], res[b])[0,1]:.3f}  {a} / {b}")

    # 10. Holdout
    hv = full_df.groupby("series_id").tail(HORIZON)
    _hidx = pd.MultiIndex.from_arrays(
        [hv["series_id"], hv["timestamp"].dt.dayofweek, hv["timestamp"].dt.hour]
    )
    w_base = wape(
        hv["target"].to_numpy(),
        np.nan_to_num(pipe.profile.reindex(_hidx).to_numpy(), nan=pipe.global_mean),
    )

    ens_ho, parts_ho, y_ho, _ = predict_ensemble(ckpts, full_t, pipe, HORIZON, device)
    print()
    for (_, L, mode), p in zip(ckpts, parts_ho):
        print(f"  {mode}_L{L:3d} einzeln {wape(y_ho, p):.4f}")

    print(f"\n  Baseline         {w_base:.4f}")
    print(f"  Ensemble roh     {wape(y_ho, ens_ho):.4f}")
    print(f"  Ensemble x{cal:.3f}   {wape(y_ho, ens_ho * cal):.4f}")

    # 11. Nachtraining auf allen Daten
    if REFIT_ON_FULL:
        pipe = FeaturePipeline(full_df, dynamic_cols)
        full_t = pipe.transform(full_df)
        ckpts = []
        for (L, mode, seed), ep in zip(CONFIGS, best_epochs):
            print(f"--- L={L}, norm={mode}, {ep} Epochen ---")
            ckpt, _, _ = train_model(
                L,
                mode,
                seed,
                full_t,
                pipe,
                HORIZON,
                device,
                BATCH_SIZE,
                LR,
                CHECKPOINT_DIR,
                val_loader=None,
                epochs=ep,
                tag="_refit",
            )
            ckpts.append((ckpt, L, mode))
    else:
        print("uebersprungen, es bleiben die auf train_df trainierten Modelle")

    # 12. Submission
    vi = val_in.copy()
    vi["target"] = np.nan
    combined = pd.concat([full_df, vi], ignore_index=True)
    combined = combined.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    assert not combined.duplicated(["series_id", "timestamp"]).any()
    combined_t = pipe.transform(combined)

    pred, _, _, meta = predict_ensemble(ckpts, combined_t, pipe, HORIZON, device)
    pred = pred * cal
    if float(train["target"].min()) >= 0:
        pred = np.clip(pred, 0, None)

    rows = pd.DataFrame(
        {
            "series_id": np.repeat([sid for sid, _ in meta], HORIZON),
            "timestamp": np.concatenate([ts for _, ts in meta]),
            "prediction": pred.ravel(),
        }
    )
    out = val_idx.merge(rows, on=["series_id", "timestamp"], how="left")

    assert out["prediction"].isna().sum() == 0, "fehlende Vorhersagen"
    assert len(out) == len(val_idx)
    out.to_csv("submission.csv", index=False)

    print(
        f"\n{len(out)} Zeilen | Mittel {out['prediction'].mean():.3f} "
        f"(train: {train['target'].mean():.3f})"
    )


if __name__ == "__main__":
    main()
