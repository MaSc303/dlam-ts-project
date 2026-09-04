# ForecastX (Group 37) — DLAM 2026 bonus project

Direct 336-step forecasting of 96 units from **known-future covariates only**.
The model is a residual Transformer on top of a per-unit hour-of-week median
profile: FiLM conditioning on the unit, a multiplicative per-window level head,
day-of-month harmonics and cross-unit covariate means as extra inputs, the
provided `trend` time index dropped, no target history in the inputs. The
submitted checkpoint is a six-member ensemble (three networks trained on the
L1/WAPE objective, three on a relative-MSE objective, blended 0.7 / 0.3).

Public validation leaderboard: **WAPE 12.89**, MAE 1.419, MSE 6.958, RMSE
2.638, MAPE 15.08, sMAPE 14.32 (naive last value 48.10, seasonal mean 34.41).

```
README.md            this file
requirements.txt     numpy, pandas, torch
predict.py           inference entrypoint (the graded command)
train.py             training: holdout / no-holdout, --loss l1|mse, ablation flags
make_ensemble.py     merges member checkpoints into one weighted checkpoint.pt
src/core.py          schema, features, statistics, dataset, model, metrics
diag.py              six leaderboard metrics + error-by-regime for a checkpoint (report Sec. 4)
diag2.py             level-cycle / covariate / spike diagnostics (report Sec. 4.1)
blend.py             L1-vs-MSE blend analysis on holdout predictions (report Table 1, last row)
fig_cycles.py        Figure 2 of the report
rossmann/            the additional-dataset study (report Sec. 5), same train.py, own schema
checkpoint.pt        NOT included here — see Section 3; it is in the model archive
```

## 1. Setup

Python 3.10+; a CUDA GPU is needed for training in reasonable time (the final
members were trained on an RTX 3050 4 GB), inference runs on CPU in seconds.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch: install the CUDA build for your driver if training
```

## 2. Data layout

Put the course files into `data/` (they are not redistributed here):

```
data/train.csv                       series_id, timestamp, target, 22 covariates   (2023-01-01 .. 2023-06-29, 4,320 h x 96 units)
data/validation_input.csv            covariates for 2023-06-30 .. 2023-07-13
data/forecast_index_validation.csv   rows to predict
```

For the private test the grader supplies `test_input.csv` and
`forecast_index_test.csv` instead; `predict.py` detects which one is present.

## 3. Inference with the submitted checkpoint

This is the graded command. `checkpoint.pt` is the ensemble from the model
archive (`final_submission.zip` = `predict.py`, `requirements.txt`,
`checkpoint.pt`, `src/`); Section 4 rebuilds it from scratch.

```bash
python predict.py --input_dir data --output_file out/predictions.csv --checkpoint checkpoint.pt
# -> wrote out/predictions.csv  rows=32256  kind=validation  models=6
```

The checkpoint contains only aggregate statistics (the 96 x 168 median profile,
per-unit residual scales, covariate means/stds, the feature list) and the six
state dicts with their blend weights — no training rows. The feature layout is
read from the checkpoint, so inference does not depend on any of the switches
in Section 5. Predictions are clipped at 0 and merged onto the forecast index;
the script asserts the row count matches.

## 4. Reproducing the final checkpoint

All six members use the same architecture and the same feature block:
`d_model 192, 3 layers, 8 heads, FF 768, dropout 0.3`, stride-6 windows,
`src/core.py` in its default configuration (`trend` dropped, day-of-month
harmonics on, cross-unit means on, FiLM on, level head on). Common flags:

```bash
COMMON="--data_dir data --stride 6 --d_model 192 --layers 3 --ff 768 --dropout 0.3"
```

**L1 (WAPE-objective) members**

```bash
# C10: local holdout (last 336 h), 10-epoch schedule, the epoch with the best pooled holdout WAPE is kept (epoch 5)
python train.py $COMMON --val_windows 1 --epochs 10 --seed 0 --out ck_C_10.pt
# two members on all 4,320 h, final weights kept
python train.py $COMMON --no_holdout --epochs 10 --seed 1 --out ck_full_L1_s1.pt
python train.py $COMMON --no_holdout --epochs 10 --seed 2 --out ck_full_L1_s2.pt
```

**MSE-objective members** (`--loss mse` = relative squared error; on holdout
runs the epoch with the best holdout MSE is kept)

```bash
python train.py $COMMON --val_windows 1 --epochs 6 --seed 0 --loss mse --out ck_D.pt      # best MSE at epoch 3
python train.py $COMMON --val_windows 1 --epochs 4 --seed 0 --loss mse --out ck_D_4.pt
python train.py $COMMON --no_holdout    --epochs 4 --seed 1 --loss mse --out ck_full_MSE_s1.pt
```

**Ensemble** — 0.7 total weight on the L1 group, 0.3 on the MSE group, equal
within a group (weights 0.2333 x 3 and 0.1 x 3):

```bash
python make_ensemble.py --l1 ck_C_10.pt ck_full_L1_s1.pt ck_full_L1_s2.pt \
                        --mse ck_D.pt ck_D_4.pt ck_full_MSE_s1.pt --w_l1 0.7 --out checkpoint.pt
# -> saved checkpoint.pt: 6 members, L1 group weight 0.70, weights sum 1.000
python predict.py --input_dir data --output_file out/predictions.csv --checkpoint checkpoint.pt
```

What the logs should show: every run prints `X (96, 4320, 55) | n_feat 55`
and `params 1,400,834`. Our runs: `ck_C_10` best pooled WAPE 0.1298 at epoch 5;
`ck_D` best MSE 6.18 at epoch 3; `ck_D_4` final MSE 6.30; the no-holdout runs
end with train loss 0.111 / 0.113 (L1) and 0.0415 (MSE). Runtime is about
90–190 s per epoch on a 4 GB GPU, 44 epochs in total, i.e. 1–2.5 h; the
ensemble and prediction take seconds. `--workers 0` if the DataLoader workers
cause trouble. Training is seeded but GPU kernels are not bit-deterministic:
expect member checkpoints to differ from ours by a few hundredths of a WAPE
point, and the ensemble on the validation leaderboard to land within about
+-0.1 of 12.89.

Note on statistics: on `--val_windows 1` runs the network trains against a
profile fitted before the holdout (so holdout numbers are honest), while every
checkpoint stores statistics fitted on all 4,320 h, which is what inference
uses; `make_ensemble.py` checks that all members carry the same statistics.

## 5. Reproducing the benchmark tables of the report (Table 1)

Local holdout = last 336 h of `train.csv` (16–29 Jun), statistics fitted before
it, six leaderboard metrics from `diag.py`:

```bash
python diag.py --data_dir data --checkpoint ck_X.pt --out_dir out/X     # CHECK 1 = the six metrics + bias
```

`diag.py` also prints the per-unit and per-regime (maintenance / spike /
ordinary) breakdown and the profile-only baselines (CHECK 4, first row is the
seasonal-profile row of the table); `python train.py --data_dir data --diagnose`
prints the naive, profile and lag-k baselines with the lag-safety flag.

The rows of Table 1 are the same command with different switches. Feature
switches are environment variables read by `src/core.py`; head/objective
switches are flags. `HOLD="$COMMON --val_windows 1 --epochs 6 --seed 0"`.

| Table 1 row | command |
|---|---|
| v2 residual encoder (with `trend`) | `DROP_FEATURES= USE_DOM=0 USE_GLOBAL=0 python train.py $HOLD --level_head 0 --out ck_v2.pt` (45 features) |
| v2 without `trend` | `USE_DOM=0 USE_GLOBAL=0 python train.py $HOLD --level_head 0 --out ck_v2_notrend.pt` (43) |
| A: v2 + calendar + cross-unit means + level head | `DROP_FEATURES= python train.py $HOLD --out ck_A.pt` (57) |
| B: A without level head | `DROP_FEATURES= python train.py $HOLD --level_head 0 --out ck_B.pt` |
| C: A without `trend` (final L1 member config) | `python train.py $HOLD --out ck_C.pt` (55, the default) |
| C, 10-epoch schedule | `python train.py $COMMON --val_windows 1 --epochs 10 --seed 0 --out ck_C_10.pt` |
| D: C with MSE objective | `python train.py $HOLD --loss mse --out ck_D.pt` |
| Blend 0.7 / 0.3 | `python blend.py --l1 out/C10/holdout_predictions.csv out/C/holdout_predictions.csv --mse out/D4/holdout_predictions.csv out/D/holdout_predictions.csv` (after `diag.py` on each) |

The v2 rows in the report were produced with the previous code version; the
commands above give the identical model (same feature layout, FiLM, no level
head) and match up to seed noise. `diag2.py --data_dir data --pred
out/C/holdout_predictions.csv` reproduces the diagnostics of Section 4.1
(level cycle and its autocorrelations, covariate correlations and R², spike
shares and Cohen's d). Other ablation flags: `--no_film`, `--lags 672,840`
(shorter lags are refused because they are hidden at test time), `--val_windows 3`
(three pooled fortnights, used for the noise estimate).

## 6. Rossmann Store Sales (report Section 5)

Download `train.csv` and `store.csv` of the Kaggle competition
(https://www.kaggle.com/c/rossmann-store-sales). `rossmann/train.py` is
byte-for-byte the benchmark `train.py`; only `rossmann/src/core.py` differs
(daily grid, 28-step horizon, store x day-of-week profile, 20 known-ahead
features, closed days as missing, cross-store means of the promo/holiday flags).

```bash
cd rossmann
python prep_rossmann.py --train /path/to/rossmann/train.csv --store /path/to/rossmann/store.csv \
                        --out_dir data --n_stores 100 --seed 0
R="--data_dir data --val_windows 2 --stride 7 --d_model 128 --layers 3 --ff 512 --dropout 0.3 --batch 64 --epochs 6 --seed 0"
python train.py $R --out ck_full.pt
python train.py $R --level_head 0 --out ck_nohead.pt
python train.py $R --no_film       --out ck_nofilm.pt
USE_DOM=0    python train.py $R    --out ck_nodom.pt
USE_GLOBAL=0 python train.py $R    --out ck_noglobal.pt
python train.py $R --loss mse      --out ck_mse.pt
python eval_ross.py --data_dir data --val_windows 2 \
    --checkpoints ck_full.pt ck_nohead.pt ck_nofilm.pt ck_nodom.pt ck_noglobal.pt ck_mse.pt
```

`eval_ross.py` prints Table 2 (naive, profile, and each checkpoint: WAPE per
28-day fold, pooled WAPE, RMSPE, MAE, bias; open days only). The replication
mentioned in the table caption used `--stride 1 --dropout 0.2 --epochs 8`
instead. Each run takes about a minute on a GPU.

## 7. Figure 2

```bash
python fig_cycles.py --dlam_train data/train.csv --ross_train /path/to/rossmann/train.csv --out fig_cycles
```

writes `fig_cycles.pdf` / `.png` (needs matplotlib).

## 8. Design notes that matter for grading

* **Lag safety.** The private test window (14–27 Jul) starts 336 h after the last
  observed target, so any lag below 672 h reads hidden data; `parse_lags`
  rejects such lags and the final model uses none. The model is therefore the
  same at validation and at test.
* **`trend` is dropped** in the final model: it is a global linear ramp that is
  out of its training range at validation and further out at test; the
  day-of-month harmonics and cross-unit means replaced it (report Sec. 4.1).
* **Two objectives.** L1 members target the conditional median (WAPE-optimal),
  MSE members the conditional mean; the 0.7/0.3 blend was chosen on the local
  holdout by a composite over all six leaderboard metrics (`blend.py`).
  
##9. Experimental models

The source code to all the experimental models can be found here: https://github.com/MaSc303/dlam-ts-project/tree/main
