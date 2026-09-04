import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.data.dataset import apply_covariate_preprocessor


def load_data(input_dir):
    train_path=input_dir/"train.csv"
    input_path=next((p for p in [input_dir/"test_input.csv",input_dir/"validation_input.csv"] if p.exists()),None)
    index_path=next((p for p in [input_dir/"forecast_index_test.csv",input_dir/"forecast_index_validation.csv"] if p.exists()),None)

    if not train_path.exists(): raise FileNotFoundError("Missing train.csv")
    if input_path is None: raise FileNotFoundError("Missing test/validation input")
    if index_path is None: raise FileNotFoundError("Missing forecast index")

    return pd.read_csv(train_path),pd.read_csv(input_path),pd.read_csv(index_path)


def make_gap_frame(train_raw,series_id,gap_start,gap_hours,dynamic_columns,static_columns):
    series=train_raw[train_raw["series_id"].eq(series_id)].sort_values("timestamp")
    timestamps=pd.date_range(gap_start,periods=gap_hours,freq="h")
    gap=pd.DataFrame({"series_id":series_id,"timestamp":timestamps})

    for column in dynamic_columns:
        gap[column]=np.nan

    for column in static_columns:
        gap[column]=series.iloc[-1][column]

    hour=timestamps.hour.to_numpy()
    dow=timestamps.dayofweek.to_numpy()

    if "hour_sin" in gap: gap["hour_sin"]=np.sin(2*np.pi*hour/24)
    if "hour_cos" in gap: gap["hour_cos"]=np.cos(2*np.pi*hour/24)
    if "dow_sin" in gap: gap["dow_sin"]=np.sin(2*np.pi*dow/7)
    if "dow_cos" in gap: gap["dow_cos"]=np.cos(2*np.pi*dow/7)
    if "is_weekend" in gap: gap["is_weekend"]=(dow>=5).astype(float)

    if "trend" in gap:
        trend=pd.to_numeric(series["trend"],errors="coerce").dropna()
        if len(trend)>=2:
            diffs=trend.diff().dropna().tail(168)
            slope=float(diffs.median()) if len(diffs) else 0.0
            gap["trend"]=float(trend.iloc[-1])+slope*np.arange(1,gap_hours+1)

    return gap


def roll_forward(
    model,
    target_history,
    past_covariates,
    future_covariates,
    series_index,
    static_features,
    history_length,
    horizon,
    steps,
):
    all_covariates=torch.cat([past_covariates,future_covariates],dim=0)
    predictions=[]

    for offset in range(0,steps,horizon):
        block_steps=min(horizon,steps-offset)
        past_cov=all_covariates[offset:offset+history_length].unsqueeze(0)
        future_cov=all_covariates[offset+history_length:offset+history_length+horizon]

        if len(future_cov)<horizon:
            future_cov=torch.cat(
                [future_cov,future_cov[-1:].repeat(horizon-len(future_cov),1)],
                dim=0,
            )

        pred=model(
            target_history[-history_length:].unsqueeze(0),
            past_cov,
            future_cov.unsqueeze(0),
            series_index,
            static_features,
        ).squeeze(0)

        predictions.extend(pred[:block_steps].cpu().tolist())
        target_history=torch.cat([target_history,pred[:block_steps]])

    return target_history,predictions


def predict_one(checkpoint,train_raw,input_raw,forecast_index,device):
    module=importlib.import_module(f"src.models.{checkpoint['model']}")
    model_args=SimpleNamespace(**checkpoint["args"])

    series_mapping=checkpoint["series_mapping"]
    dynamic_columns=checkpoint["dynamic_feature_columns"]
    model_dynamic_columns=checkpoint["model_dynamic_columns"]
    static_columns=checkpoint.get("static_feature_columns",[])
    preprocessor=checkpoint["preprocessor"]
    standardize=checkpoint.get("standardize_covariates",False)
    missing_masks=checkpoint.get("missing_masks",False)
    history_length=checkpoint["history_length"]
    horizon=checkpoint.get("horizon",getattr(model_args,"horizon",24))

    model_args.num_static_features=len(static_columns)

    model=module.create_model(
        model_args,
        history_length,
        horizon,
        len(model_dynamic_columns),
        len(series_mapping),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    train_data,_=apply_covariate_preprocessor(
        train_raw,
        dynamic_columns,
        static_columns,
        preprocessor,
        standardize=standardize,
        missing_masks=missing_masks,
    )

    input_data,_=apply_covariate_preprocessor(
        input_raw,
        dynamic_columns,
        static_columns,
        preprocessor,
        standardize=standardize,
        missing_masks=missing_masks,
    )

    output=forecast_index[["series_id","timestamp"]].copy()
    output["prediction"]=float("nan")

    with torch.no_grad():
        for series_id,index_unsorted in forecast_index.groupby("series_id",sort=False):
            key=str(series_id)
            if key not in series_mapping: raise KeyError(f"Unknown series_id: {series_id}")

            index_part=index_unsorted.sort_values("timestamp")
            series_index=torch.tensor([series_mapping[key]],dtype=torch.long,device=device)

            series_train_raw=train_raw[train_raw["series_id"].eq(series_id)].sort_values("timestamp")
            series_train=train_data[train_data["series_id"].eq(series_id)].sort_values("timestamp")

            history_part=series_train.dropna(subset=["target"]).tail(history_length)
            if len(history_part)<history_length: raise ValueError(f"Not enough history for {series_id}")

            future_part=index_part[["series_id","timestamp"]].merge(
                input_data,
                on=["series_id","timestamp"],
                how="left",
            )

            if future_part[model_dynamic_columns].isna().any().any():
                raise ValueError(f"Missing future covariates for {series_id}")

            train_end=series_train_raw["timestamp"].max()
            test_start=index_part["timestamp"].min()
            gap_hours=int((test_start-train_end)/pd.Timedelta(hours=1))-1

            target_history=torch.tensor(
                history_part["target"].to_numpy(),
                dtype=torch.float32,
                device=device,
            )
            past_covariates=torch.tensor(
                history_part[model_dynamic_columns].to_numpy(),
                dtype=torch.float32,
                device=device,
            )
            future_covariates=torch.tensor(
                future_part[model_dynamic_columns].to_numpy(),
                dtype=torch.float32,
                device=device,
            )

            if static_columns:
                static_features=torch.tensor(
                    history_part.iloc[-1][static_columns].to_numpy(dtype="float32"),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
            else:
                static_features=torch.empty((1,0),dtype=torch.float32,device=device)

            if gap_hours>0:
                gap_raw=make_gap_frame(
                    train_raw,
                    series_id,
                    train_end+pd.Timedelta(hours=1),
                    gap_hours,
                    dynamic_columns,
                    static_columns,
                )

                gap_data,_=apply_covariate_preprocessor(
                    gap_raw,
                    dynamic_columns,
                    static_columns,
                    preprocessor,
                    standardize=standardize,
                    missing_masks=missing_masks,
                )

                gap_covariates=torch.tensor(
                    gap_data[model_dynamic_columns].to_numpy(),
                    dtype=torch.float32,
                    device=device,
                )

                target_history,_=roll_forward(
                    model,
                    target_history,
                    past_covariates,
                    gap_covariates,
                    series_index,
                    static_features,
                    history_length,
                    horizon,
                    gap_hours,
                )

                past_covariates=torch.cat(
                    [past_covariates,gap_covariates],
                    dim=0,
                )[-history_length:]

            _,predictions=roll_forward(
                model,
                target_history,
                past_covariates,
                future_covariates,
                series_index,
                static_features,
                history_length,
                horizon,
                len(index_part),
            )

            output.loc[index_part.index,"prediction"]=predictions

    del model
    if device.type=="cuda": torch.cuda.empty_cache()

    return output["prediction"].to_numpy()


parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",required=True,type=Path)
parser.add_argument("--output_file",required=True,type=Path)
parser.add_argument("--input_dir",required=True,type=Path)
parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
args=parser.parse_args()

device=torch.device(args.device)
ensemble=torch.load(args.checkpoint,map_location="cpu",weights_only=False)

train_raw,input_raw,forecast_index=load_data(args.input_dir)

for frame in [train_raw,input_raw,forecast_index]:
    frame["timestamp"]=pd.to_datetime(frame["timestamp"])

if ensemble.get("type")!="weighted_ensemble":
    raise ValueError("Expected weighted_ensemble checkpoint")

cv_predictions=[]
for i,checkpoint in enumerate(ensemble["cv_models"],1):
    print(f"CV model {i}/{len(ensemble['cv_models'])}")
    cv_predictions.append(
        predict_one(checkpoint,train_raw,input_raw,forecast_index,device)
    )

full_predictions=[]
for i,checkpoint in enumerate(ensemble["full_models"],1):
    print(f"Full model {i}/{len(ensemble['full_models'])}")
    full_predictions.append(
        predict_one(checkpoint,train_raw,input_raw,forecast_index,device)
    )

cv_mean=np.mean(cv_predictions,axis=0)
full_mean=np.mean(full_predictions,axis=0)

predictions=(
    ensemble["cv_weight"]*cv_mean
    +ensemble["full_weight"]*full_mean
)

output=forecast_index[["series_id","timestamp"]].copy()
output["prediction"]=predictions

if output["prediction"].isna().any():
    raise ValueError("Missing predictions")

args.output_file.parent.mkdir(parents=True,exist_ok=True)
output.to_csv(args.output_file,index=False)

print(f"Wrote {len(output)} predictions to {args.output_file}")