import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.data.dataset import apply_covariate_preprocessor
from src.evaluation.rollout import rollout

HF_PATH="hf://datasets/AIML-TUDA/dlam-ts-project-data-2026/"
VALIDATION_HORIZON=336

parser=argparse.ArgumentParser()
parser.add_argument("checkpoints",nargs="+",type=Path)
parser.add_argument("--output",default="prediction_ensemble.csv")
parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
args=parser.parse_args()

device=torch.device(args.device)
train=pd.read_csv(HF_PATH+"train.csv")
validation_input=pd.read_csv(HF_PATH+"validation_input.csv")
forecast_index=pd.read_csv(HF_PATH+"forecast_index_validation.csv")
keys=["series_id","timestamp"]
all_predictions=[]

for checkpoint_path in args.checkpoints:
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
    history_length=checkpoint["history_length"]
    horizon=checkpoint.get("horizon",checkpoint["args"].get("horizon",24))

    history=train.sort_values(keys).groupby("series_id",sort=False,group_keys=False).tail(history_length).copy()
    future=forecast_index.merge(validation_input,on=keys,how="left",sort=False,validate="one_to_one")

    if len(future)!=len(forecast_index):
        raise RuntimeError("Forecast input size mismatch")

    future["target"]=np.nan
    frame=pd.concat([history,future],ignore_index=True,sort=False)

    frame,_=apply_covariate_preprocessor(
        frame,
        checkpoint["dynamic_feature_columns"],
        checkpoint["static_feature_columns"],
        checkpoint["preprocessor"],
        standardize=checkpoint["standardize_covariates"],
        missing_masks=checkpoint["missing_masks"],
    )

    model_args=SimpleNamespace(**checkpoint["args"])
    model_args.num_static_features=len(checkpoint["static_feature_columns"])
    module=importlib.import_module(f"src.models.{checkpoint['model']}")
    model=module.create_model(model_args,history_length,horizon,len(checkpoint["model_dynamic_columns"]),len(checkpoint["series_mapping"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    prediction=rollout(
        model,
        frame,
        checkpoint["model_dynamic_columns"],
        checkpoint["static_feature_columns"],
        checkpoint["series_mapping"],
        VALIDATION_HORIZON,
        history_length,
        horizon,
        device,
        None,
    )

    prediction=forecast_index.merge(prediction,on=keys,how="left",sort=False,validate="one_to_one")

    if prediction["prediction"].isna().any():
        raise RuntimeError(f"Missing predictions from {checkpoint_path}")

    all_predictions.append(prediction["prediction"].to_numpy())

result=forecast_index.copy()
result["prediction"]=np.mean(np.stack(all_predictions,axis=0),axis=0)
result=result[["series_id","timestamp","prediction"]]
result.to_csv(args.output,index=False)

print(f"Saved: {args.output}")
print(f"Rows: {len(result)}")
print(result.head())