import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset


HISTORY_LENGTH=168
HORIZON=24
ROLLOUT_LENGTH=336


class JenaDataset(Dataset):
    def __init__(self,target,covariates,start,end,stride=6):
        self.target=target
        self.covariates=covariates
        self.indices=list(range(start,end-HORIZON+1,stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        t=self.indices[index]
        return (
            self.target[t-HISTORY_LENGTH:t],
            self.covariates[t-HISTORY_LENGTH:t],
            self.covariates[t:t+HORIZON],
            self.target[t:t+HORIZON],
        )


def rollout(model,target,covariates,start,device):
    history=target[start-HISTORY_LENGTH:start].clone()
    predictions=[]

    for offset in range(0,ROLLOUT_LENGTH,HORIZON):
        t=start+offset

        past_cov=covariates[t-HISTORY_LENGTH:t]
        future_cov=covariates[t:t+HORIZON]

        pred=model(
            history[-HISTORY_LENGTH:].unsqueeze(0).to(device),
            past_cov.unsqueeze(0).to(device),
            future_cov.unsqueeze(0).to(device),
            torch.zeros(1,dtype=torch.long,device=device),
            torch.empty((1,0),dtype=torch.float32,device=device),
        ).squeeze(0).cpu()

        predictions.append(pred)
        history=torch.cat([history,pred])

    return torch.cat(predictions)[:ROLLOUT_LENGTH]


parser=argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=Path,
    default=Path("data/jena/jena_climate_2009_2016.csv"),
)
parser.add_argument(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
)
args=parser.parse_args()

device=torch.device(args.device)

df=pd.read_csv(args.data)
df["timestamp"]=pd.to_datetime(
    df["Date Time"],
    format="%d.%m.%Y %H:%M:%S",
)
df=df.set_index("timestamp")

# 10-minute data -> hourly
df=df.select_dtypes(include=[np.number]).resample("1h").mean().dropna()
df=df.reset_index()

timestamps=df["timestamp"]
hour=timestamps.dt.hour.to_numpy()
dow=timestamps.dt.dayofweek.to_numpy()
doy=timestamps.dt.dayofyear.to_numpy()

covariates=np.column_stack([
    np.sin(2*np.pi*hour/24),
    np.cos(2*np.pi*hour/24),
    np.sin(2*np.pi*dow/7),
    np.cos(2*np.pi*dow/7),
    np.sin(2*np.pi*doy/365.25),
    np.cos(2*np.pi*doy/365.25),
]).astype("float32")

target=torch.tensor(
    df["T (degC)"].to_numpy(dtype="float32"),
)
covariates=torch.tensor(covariates)

n=len(df)
train_end=int(n*0.70)
val_end=int(n*0.85)

train_dataset=JenaDataset(
    target,
    covariates,
    HISTORY_LENGTH,
    train_end,
    stride=6,
)

train_loader=DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
)

model_args=SimpleNamespace(
    hidden_size=512,
    num_blocks=6,
    num_layers=3,
    covariate_hidden_size=16,
    series_embedding_size=32,
    static_hidden_size=8,
    dropout=0.11545133998174503,
    feature_gating=False,
    film=False,
    past_tcn=True,
    future_tcn=True,
    target_tcn=False,
    multiscale=False,
    static_branch=False,
    capacity_normalize=False,
    target_revin=False,
    seasonal_residual=False,
    num_static_features=0,
)

module=importlib.import_module("src.models.nbeatsx_sep_v2")

model=module.create_model(
    model_args,
    HISTORY_LENGTH,
    HORIZON,
    covariates.shape[1],
    1,
).to(device)

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.0011232544738615938,
)

scheduler=torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[4],
    gamma=0.2,
)

loss_fn=torch.nn.HuberLoss(delta=0.5)

for epoch in range(5):
    model.train()
    losses=[]

    for target_history,past_cov,future_cov,y in train_loader:
        target_history=target_history.to(device)
        past_cov=past_cov.to(device)
        future_cov=future_cov.to(device)
        y=y.to(device)

        batch_size=len(y)

        pred=model(
            target_history,
            past_cov,
            future_cov,
            torch.zeros(batch_size,dtype=torch.long,device=device),
            torch.empty((batch_size,0),dtype=torch.float32,device=device),
        )

        loss=loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    scheduler.step()
    print(f"Epoch {epoch+1}: loss={np.mean(losses):.6f}")


model.eval()

available=val_end-train_end
num_windows=min(5,available//ROLLOUT_LENGTH)

starts=np.linspace(
    train_end,
    val_end-ROLLOUT_LENGTH,
    num_windows,
    dtype=int,
)

all_true=[]
all_pred=[]
all_naive=[]

with torch.no_grad():
    for start in starts:
        pred=rollout(
            model,
            target,
            covariates,
            start,
            device,
        )

        true=target[start:start+ROLLOUT_LENGTH]
        naive=torch.full_like(
            true,
            target[start-1].item(),
        )

        all_true.append(true)
        all_pred.append(pred)
        all_naive.append(naive)

true=torch.cat(all_true).numpy()
pred=torch.cat(all_pred).numpy()
naive=torch.cat(all_naive).numpy()


def metrics(y,p):
    mae=np.mean(np.abs(y-p))
    rmse=np.sqrt(np.mean((y-p)**2))
    wape=np.sum(np.abs(y-p))/np.sum(np.abs(y))
    return mae,rmse,wape


model_mae,model_rmse,model_wape=metrics(true,pred)
naive_mae,naive_rmse,naive_wape=metrics(true,naive)

print()
print("Jena Climate results")
print("--------------------")
print(
    f"Model: MAE={model_mae:.4f}, "
    f"RMSE={model_rmse:.4f}, "
    f"WAPE={model_wape*100:.2f}%"
)
print(
    f"Naive: MAE={naive_mae:.4f}, "
    f"RMSE={naive_rmse:.4f}, "
    f"WAPE={naive_wape*100:.2f}%"
)