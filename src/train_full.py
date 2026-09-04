import argparse
import importlib
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.dataset import ForecastDataset,STATIC_COLUMNS,load_data,get_feature_columns,fit_covariate_preprocessor,apply_covariate_preprocessor

parser=argparse.ArgumentParser()
parser.add_argument("--model",default="nbeatsx")
parser.add_argument("--epochs",type=int,default=15)
parser.add_argument("--horizon",type=int,default=24)
parser.add_argument("--batch-size",type=int,default=128)
parser.add_argument("--lr",type=float,required=True)
parser.add_argument("--weight-decay",type=float,default=0)
parser.add_argument("--hidden-size",type=int,required=True)
parser.add_argument("--num-blocks",type=int,required=True)
parser.add_argument("--num-layers",type=int,required=True)
parser.add_argument("--dropout",type=float,required=True)
parser.add_argument("--covariate-hidden-size",type=int,required=True)
parser.add_argument("--series-embedding-size",type=int,required=True)
parser.add_argument("--static-hidden-size",type=int,default=4)
parser.add_argument("--loss",choices=["mse","huber","l1"],default="huber")
parser.add_argument("--huber-delta",type=float,default=0.5)
parser.add_argument("--standardize-covariates",action="store_true")
parser.add_argument("--missing-masks",action="store_true")
parser.add_argument("--static-branch",action="store_true")
parser.add_argument("--seasonal-residual",action="store_true")
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
args=parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
device=torch.device(args.device)
history_length=168

raw=load_data("train.csv")
features=get_feature_columns(raw)
static=[c for c in STATIC_COLUMNS if c in features] if args.static_branch else []
dynamic=[c for c in features if c not in static]
preprocessor=fit_covariate_preprocessor(raw,dynamic,static)
frame,model_dynamic=apply_covariate_preprocessor(raw,dynamic,static,preprocessor,standardize=args.standardize_covariates,missing_masks=args.missing_masks)

series_ids=sorted(raw["series_id"].astype(str).unique())
series_mapping={sid:i for i,sid in enumerate(series_ids)}
args.num_static_features=len(static)

dataset=ForecastDataset(frame,model_dynamic,static,series_mapping,history_length,args.horizon)
loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

module=importlib.import_module(f"src.models.{args.model}")
model=module.create_model(args,history_length,args.horizon,len(model_dynamic),len(series_mapping)).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
criterion={"mse":nn.MSELoss(),"huber":nn.HuberLoss(delta=args.huber_delta),"l1":nn.L1Loss()}[args.loss]

for epoch in range(args.epochs):
    model.train()
    total=0
    for batch in loader:
        optimizer.zero_grad()
        pred=model(batch["past_target"].to(device),batch["past_covariates"].to(device),batch["future_covariates"].to(device),batch["series_index"].to(device),batch["static_features"].to(device))
        loss=criterion(pred,batch["target"].to(device))
        loss.backward()
        optimizer.step()
        total+=loss.item()
    print(f"{epoch+1}: train={total/len(loader):.4f}")

Path("checkpoints").mkdir(exist_ok=True)
name=f"FULL_{args.model}_hz{args.horizon}_lr{args.lr:g}_h{args.hidden_size}_b{args.num_blocks}_l{args.num_layers}_c{args.covariate_hidden_size}_e{args.series_embedding_size}_sh{args.static_hidden_size}_d{args.dropout:g}_delta{args.huber_delta:g}.pt"
path=Path("checkpoints")/name

torch.save({
    "model":args.model,"args":vars(args),"epoch":args.epochs-1,
    "history_length":history_length,"horizon":args.horizon,
    "dynamic_feature_columns":dynamic,"model_dynamic_columns":model_dynamic,
    "static_feature_columns":static,"preprocessor":preprocessor,
    "standardize_covariates":args.standardize_covariates,
    "missing_masks":args.missing_masks,
    "series_mapping":series_mapping,
    "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
    "optimizer_state_dict":optimizer.state_dict()
},path)

print(f"Checkpoint: {path}")