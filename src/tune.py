import argparse
from types import SimpleNamespace

import numpy as np
import optuna
import torch

from src.training.train_model import train_model

parser=argparse.ArgumentParser()
parser.add_argument("--trials",type=int,default=20)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--study-name",required=True)
parser.add_argument("--model",default="nbeatsx_sep_v2")
parser.add_argument("--horizon",type=int,default=24)
parser.add_argument("--num-val-windows",type=int,default=5)
parser.add_argument("--verify-top-k",type=int,default=3)
parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
cli=parser.parse_args()

def make_args(params,seed):
    return SimpleNamespace(
        model=cli.model,epochs=cli.epochs,horizon=cli.horizon,num_val_windows=cli.num_val_windows,
        batch_size=128,device=cli.device,seed=seed,
        loss="huber",huber_delta=0.5,
        hidden_size=params["hidden_size"],num_blocks=params["num_blocks"],num_layers=3,
        covariate_hidden_size=16,series_embedding_size=32,static_hidden_size=8,
        dropout=params["dropout"],lr=params["lr"],
        optimizer=params["optimizer"],
        weight_decay=params["weight_decay"] if params["optimizer"]=="adamw" else 0.0,
        lr_decay_milestone=params["lr_decay_milestone"],lr_decay_gamma=params["lr_decay_gamma"],
        standardize_covariates=True,missing_masks=True,static_branch=True,seasonal_residual=False,
        past_tcn=True,future_tcn=True,feature_gating=False,film_conditioning=False,
        multiscale_target=False,capacity_normalize_target=False,target_tcn=False,target_revin=False,
        history_length=168,resume_checkpoint=None,reserve_holdout=False,
        train_rollout_blocks=1,ema_decay=0.0,
    )

def objective(trial):
    params={
        "hidden_size":trial.suggest_categorical("hidden_size",[384,512]),
        "num_blocks":trial.suggest_categorical("num_blocks",[5,6]),
        "dropout":trial.suggest_float("dropout",0.04,0.18),
        "lr":trial.suggest_float("lr",6e-4,1.3e-3,log=True),
        "optimizer":trial.suggest_categorical("optimizer",["adam","adamw"]),
        "lr_decay_milestone":trial.suggest_categorical("lr_decay_milestone",[0,3,4]),
        "lr_decay_gamma":trial.suggest_categorical("lr_decay_gamma",[0.2,0.3,0.5]),
    }

    params["weight_decay"]=trial.suggest_float("weight_decay",1e-4,3e-2,log=True) if params["optimizer"]=="adamw" else 0.0
    return train_model(make_args(params,seed=42),trial)

study=optuna.create_study(
    study_name=cli.study_name,
    storage="sqlite:///optuna.db",
    load_if_exists=True,
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=4,interval_steps=1,n_min_trials=3),
)

study.optimize(objective,n_trials=cli.trials)

print("\n=== OPTUNA BEST ===")
print(f"Best WAPE: {study.best_value:.6f}")
print(f"Best trial: {study.best_trial.number}")
for key,value in study.best_params.items(): print(f"{key}: {value}")

completed=[trial for trial in study.trials if trial.state==optuna.trial.TrialState.COMPLETE]
completed.sort(key=lambda trial:trial.value)
top_trials=completed[:cli.verify_top_k]
verification=[]

print("\n=== MULTI-SEED VERIFICATION ===")

for rank,trial in enumerate(top_trials,start=1):
    params=dict(trial.params)
    if params["optimizer"]!="adamw": params["weight_decay"]=0.0

    scores=[]
    print(f"\nCandidate {rank}: trial={trial.number}, optuna_WAPE={trial.value:.6f}")

    for seed in [42,123,777]:
        print(f"\n--- seed {seed} ---")
        score=train_model(make_args(params,seed=seed),trial=None)
        scores.append(score)

    mean_score=float(np.mean(scores))
    std_score=float(np.std(scores))
    verification.append({"trial":trial.number,"params":params,"scores":scores,"mean":mean_score,"std":std_score})

    print(f"\nCandidate {rank}: scores={[round(x,6) for x in scores]}, mean={mean_score:.6f}, std={std_score:.6f}")

verification.sort(key=lambda result:result["mean"])

if verification:
    best=verification[0]

    print("\n=== FINAL BEST CONFIGURATION ===")
    print(f"Trial: {best['trial']}")
    print(f"Mean WAPE: {best['mean']:.6f}")
    print(f"Std WAPE: {best['std']:.6f}")
    print("Seed WAPEs:",", ".join(f"{score:.6f}" for score in best["scores"]))

    for key,value in best["params"].items(): print(f"{key}: {value}")