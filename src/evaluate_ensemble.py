import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    apply_covariate_preprocessor,
    load_train_rolling_validation,
)
from src.evaluation.metrics import all_metrics, wape
from src.evaluation.rollout import rollout


VALIDATION_HORIZON=336


parser=argparse.ArgumentParser()
parser.add_argument("checkpoints",nargs="+",type=Path)
parser.add_argument("--num-val-windows",type=int,default=5)
parser.add_argument(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--output-dir",type=Path,default=None)
args=parser.parse_args()


device=torch.device(args.device)

checkpoints=[
    torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    for path in args.checkpoints
]

reference=checkpoints[0]

history_length=reference["history_length"]
horizon=reference.get(
    "horizon",
    reference["args"].get("horizon",24),
)

_,splits=load_train_rolling_validation(
    history_length,
    VALIDATION_HORIZON,
    args.num_val_windows,
)

models=[]

for path,checkpoint in zip(
    args.checkpoints,
    checkpoints,
):
    model_args=SimpleNamespace(**checkpoint["args"])

    model_args.num_static_features=len(
        checkpoint["static_feature_columns"],
    )

    module=importlib.import_module(
        f"src.models.{checkpoint['model']}",
    )

    model=module.create_model(
        model_args,
        checkpoint["history_length"],
        checkpoint.get(
            "horizon",
            getattr(model_args,"horizon",24),
        ),
        len(checkpoint["model_dynamic_columns"]),
        len(checkpoint["series_mapping"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["state_dict"],
    )

    model.eval()

    models.append(
        (
            path,
            checkpoint,
            model,
        )
    )


if args.output_dir is not None:
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


window_wapes=[]
window_metrics=[]

for window_idx,split in enumerate(
    splits,
    start=1,
):
    targets=(
        split["frame"]
        .groupby(
            "series_id",
            sort=False,
        )
        .tail(VALIDATION_HORIZON)
    )

    ensemble_frame=targets[
        ["series_id","timestamp","target"]
    ].copy()

    prediction_columns=[]

    for model_idx,(
        path,
        checkpoint,
        model,
    ) in enumerate(models):
        processed_frame,_=apply_covariate_preprocessor(
            split["frame"],
            checkpoint["dynamic_feature_columns"],
            checkpoint["static_feature_columns"],
            checkpoint["preprocessor"],
            standardize=checkpoint[
                "standardize_covariates"
            ],
            missing_masks=checkpoint[
                "missing_masks"
            ],
        )

        predictions=rollout(
            model,
            processed_frame,
            checkpoint["model_dynamic_columns"],
            checkpoint["static_feature_columns"],
            checkpoint["series_mapping"],
            VALIDATION_HORIZON,
            checkpoint["history_length"],
            checkpoint.get(
                "horizon",
                checkpoint["args"].get(
                    "horizon",
                    24,
                ),
            ),
            device,
            None,
        )

        column=f"prediction_{model_idx}"

        predictions=predictions.rename(
            columns={
                "prediction":column,
            },
        )

        ensemble_frame=ensemble_frame.merge(
            predictions[
                [
                    "series_id",
                    "timestamp",
                    column,
                ]
            ],
            on=[
                "series_id",
                "timestamp",
            ],
            how="left",
        )

        prediction_columns.append(column)

    ensemble_frame["prediction"]=ensemble_frame[
        prediction_columns
    ].mean(
        axis=1,
    )

    score=wape(
        ensemble_frame["target"],
        ensemble_frame["prediction"],
    )

    metrics=all_metrics(
        ensemble_frame["target"],
        ensemble_frame["prediction"],
    )

    window_wapes.append(score)
    window_metrics.append(metrics)

    print(
        f"Window {window_idx}: "
        f"WAPE={score*100:.2f}"
    )

    if args.output_dir is not None:
        ensemble_frame[
            [
                "series_id",
                "timestamp",
                "target",
                "prediction",
            ]
        ].to_csv(
            args.output_dir/f"window_{window_idx}.csv",
            index=False,
        )


mean_wape=float(
    np.mean(window_wapes),
)

mean_metrics={
    key:float(
        np.mean(
            [
                metrics[key]
                for metrics in window_metrics
            ]
        )
    )
    for key in window_metrics[0]
}


print("\n=== ENSEMBLE ===")
print(
    "Checkpoints:",
    len(checkpoints),
)
print(
    "Windows:",
    [
        round(score*100,2)
        for score in window_wapes
    ],
)
print(
    f"Mean WAPE: {mean_wape*100:.4f}",
)
print(
    f"MAE: {mean_metrics['mae']:.4f}",
)
print(
    f"MSE: {mean_metrics['mse']:.4f}",
)
print(
    f"RMSE: {mean_metrics['rmse']:.4f}",
)
print(
    f"MAPE: {mean_metrics['mape']:.2f}",
)
print(
    f"sMAPE: {mean_metrics['smape']:.2f}",
)