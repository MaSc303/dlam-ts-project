from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

HF_PATH="hf://datasets/AIML-TUDA/dlam-ts-project-data-2026/"
CACHE=Path(".cache")
STATIC_COLUMNS=["nominal_capacity","zone_sin","zone_cos"]

def load_data(name):
    CACHE.mkdir(exist_ok=True)
    path=CACHE/name
    if not path.exists(): pd.read_csv(HF_PATH+name).to_csv(path,index=False)
    return pd.read_csv(path)

def load_train_rolling_validation(history_length=168,validation_horizon=336,num_windows=3):
    frame=load_data("train.csv").copy()
    frame["timestamp"]=pd.to_datetime(frame["timestamp"])
    train_parts=[]
    windows=[{"history":[],"frame":[]} for _ in range(num_windows)]

    for _,group in frame.groupby("series_id",sort=False):
        group=group.sort_values("timestamp").reset_index(drop=True)
        train_end=len(group)-num_windows*validation_horizon
        if train_end<history_length: raise ValueError("Not enough data for rolling validation")

        train_parts.append(group.iloc[:train_end])

        for i in range(num_windows):
            start=train_end+i*validation_horizon
            end=start+validation_horizon
            windows[i]["history"].append(group.iloc[:start])
            windows[i]["frame"].append(group.iloc[start-history_length:end])

    train=pd.concat(train_parts,ignore_index=True)
    splits=[{"history":pd.concat(w["history"],ignore_index=True),"frame":pd.concat(w["frame"],ignore_index=True)} for w in windows]
    return train,splits

def get_feature_columns(frame):
    return [c for c in frame.columns if c not in ["series_id","timestamp","target"]]

def fit_covariate_preprocessor(frame,dynamic_columns,static_columns):
    columns=dynamic_columns+static_columns
    medians=frame[columns].median()
    filled=frame[columns].fillna(medians)
    means=filled.mean()
    stds=filled.std().replace(0,1).fillna(1)
    return {"medians":medians.to_dict(),"means":means.to_dict(),"stds":stds.to_dict()}

def apply_covariate_preprocessor(frame,dynamic_columns,static_columns,preprocessor,standardize=False,missing_masks=False):
    frame=frame.copy()
    columns=dynamic_columns+static_columns
    missing=frame[dynamic_columns].isna()

    for column in columns:
        frame[column]=frame[column].fillna(preprocessor["medians"][column])
        if standardize: frame[column]=(frame[column]-preprocessor["means"][column])/preprocessor["stds"][column]

    model_dynamic_columns=list(dynamic_columns)

    if missing_masks:
        for column in dynamic_columns:
            mask_column=f"{column}__missing"
            frame[mask_column]=missing[column].astype("float32")
            model_dynamic_columns.append(mask_column)

    return frame,model_dynamic_columns

class ForecastDataset(Dataset):
    def __init__(self,frame,dynamic_feature_columns,static_feature_columns,series_mapping,history_length=168,horizon=24):
        self.history_length=history_length
        self.horizon=horizon
        self.series_mapping=series_mapping
        self.series={}
        self.windows=[]

        for series_id,group in frame.groupby("series_id"):
            group=group.sort_values("timestamp")
            target=torch.tensor(group["target"].to_numpy(),dtype=torch.float32)
            covariates=torch.tensor(group[dynamic_feature_columns].to_numpy(),dtype=torch.float32)
            static_features=torch.tensor(group.iloc[0][static_feature_columns].to_numpy(dtype="float32"),dtype=torch.float32)

            self.series[series_id]=(target,covariates,static_features)

            for start in range(len(group)-history_length-horizon+1):
                self.windows.append((series_id,start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        series_id,start=self.windows[index]
        target,covariates,static_features=self.series[series_id]
        end=start+self.history_length
        future_end=end+self.horizon

        return {
            "series_index":torch.tensor(self.series_mapping[str(series_id)],dtype=torch.long),
            "static_features":static_features,
            "past_target":target[start:end],
            "past_covariates":covariates[start:end],
            "future_covariates":covariates[end:future_end],
            "target":target[end:future_end],
        }

def load_train_validation(history_length=168,validation_hours=24):
    df=load_data("train.csv")
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    train=[]
    val=[]

    for _,group in df.groupby("series_id"):
        group=group.sort_values("timestamp")
        split=len(group)-validation_hours
        train.append(group.iloc[:split])
        val.append(group.iloc[split-history_length:])

    return pd.concat(train,ignore_index=True),pd.concat(val,ignore_index=True)