import copy
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ForecastDataset,STATIC_COLUMNS,load_train_rolling_validation,get_feature_columns,fit_covariate_preprocessor,apply_covariate_preprocessor
from src.evaluation.baselines import make_all_baselines
from src.evaluation.metrics import wape,all_metrics
from src.evaluation.rollout import rollout

HF_TRAIN_PATH="hf://datasets/AIML-TUDA/dlam-ts-project-data-2026/train.csv"
HISTORY_LENGTH=168
VALIDATION_HORIZON=336

def train_model(args,trial=None):
    history_length=getattr(args,"history_length",HISTORY_LENGTH)
    seed=getattr(args,"seed",42)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    device=torch.device(args.device)
    horizon=getattr(args,"horizon",24)
    train_rollout_blocks=getattr(args,"train_rollout_blocks",1)
    train_horizon=horizon*train_rollout_blocks
    num_val_windows=getattr(args,"num_val_windows",3)
    module=importlib.import_module(f"src.models.{args.model}")
    Path("checkpoints").mkdir(exist_ok=True)

    train_full=getattr(args,"train_full",False)
    if train_full:
        if trial is not None: raise ValueError("train_full must not be used during Optuna tuning")
        train_raw=pd.read_csv(HF_TRAIN_PATH)
        val_splits=[]
        holdout_split=None
    else:
        reserve_holdout=getattr(args,"reserve_holdout",False)
        load_windows=num_val_windows+1 if reserve_holdout else num_val_windows
        train_raw,all_splits=load_train_rolling_validation(history_length,VALIDATION_HORIZON,load_windows)
        if reserve_holdout:
            val_splits=all_splits[:-1]
            holdout_split=all_splits[-1]
        else:
            val_splits=all_splits
            holdout_split=None

    metric_scale=max(float(train_raw["target"].dropna().abs().mean()),1e-6)
    metric_eps=max(metric_scale*1e-3,1e-6)

    all_features=get_feature_columns(train_raw)
    static_columns=[c for c in STATIC_COLUMNS if c in all_features] if getattr(args,"static_branch",False) else []
    dynamic_columns=[c for c in all_features if c not in static_columns]

    preprocessor=fit_covariate_preprocessor(train_raw,dynamic_columns,static_columns)
    train_frame,model_dynamic_columns=apply_covariate_preprocessor(train_raw,dynamic_columns,static_columns,preprocessor,standardize=getattr(args,"standardize_covariates",False),missing_masks=getattr(args,"missing_masks",False))

    processed_splits=[]
    for split in val_splits:
        frame,_=apply_covariate_preprocessor(split["frame"],dynamic_columns,static_columns,preprocessor,standardize=getattr(args,"standardize_covariates",False),missing_masks=getattr(args,"missing_masks",False))
        processed_splits.append({"frame":frame,"raw":split["frame"],"history":split["history"]})

    series_ids=sorted(train_raw["series_id"].astype(str).unique())
    series_mapping={sid:i for i,sid in enumerate(series_ids)}
    args.num_static_features=len(static_columns)

    capacity_by_series={str(series_id):float(capacity) for series_id,capacity in train_raw.groupby("series_id")["nominal_capacity"].median().items()}
    series_scales=torch.ones(len(series_mapping),dtype=torch.float32,device=device)
    for series_id,index in series_mapping.items():
        series_scales[index]=max(capacity_by_series[str(series_id)],1e-6)

    baseline_values={}
    for split in processed_splits:
        targets=split["raw"].groupby("series_id",sort=False).tail(VALIDATION_HORIZON)
        index=targets[["series_id","timestamp"]]
        for name,pred in make_all_baselines(split["history"],index).items():
            result=targets.merge(pred,on=["series_id","timestamp"])
            baseline_values.setdefault(name,[]).append(wape(result["target"],result["prediction"]))
    baseline_scores={k:float(np.mean(v)) for k,v in baseline_values.items()}

    dataset=ForecastDataset(train_frame,model_dynamic_columns,static_columns,series_mapping,history_length,train_horizon)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    model=module.create_model(args,history_length,horizon,len(model_dynamic_columns),len(series_mapping)).to(device)

    if getattr(args,"resume_checkpoint",None):
        resume=torch.load(args.resume_checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(resume["state_dict"])
        print(f"Loaded checkpoint: {args.resume_checkpoint}")

    ema_decay=getattr(args,"ema_decay",0.0)
    ema_model=copy.deepcopy(model) if ema_decay>0 else None
    if ema_model is not None:
        ema_model.eval()
        for parameter in ema_model.parameters(): parameter.requires_grad_(False)

    optimizer_cls=torch.optim.AdamW if getattr(args,"optimizer","adam")=="adamw" else torch.optim.Adam
    optimizer=optimizer_cls(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    lr_decay_milestone=getattr(args,"lr_decay_milestone",0)
    lr_decay_gamma=getattr(args,"lr_decay_gamma",0.3)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[lr_decay_milestone],gamma=lr_decay_gamma) if lr_decay_milestone>0 else None

    loss_type=getattr(args,"loss","mse")
    delta=getattr(args,"huber_delta",1.0)

    if loss_type=="metric_mix":
        def criterion(pred,target):
            error=pred-target
            absolute_error=error.abs()
            squared_error=error.square()
            mae=absolute_error.mean()/metric_scale
            wape_loss=absolute_error.sum()/target.abs().sum().clamp_min(metric_eps)
            mse=squared_error.mean()/(metric_scale**2)
            rmse=torch.sqrt(squared_error.mean()+1e-8)/metric_scale
            mape=(absolute_error/target.abs().clamp_min(metric_eps)).mean()
            smape=(2.0*absolute_error/(target.abs()+pred.abs()).clamp_min(metric_eps)).mean()
            return (mae+wape_loss+mse+rmse+mape+smape)/6.0
        loss_name="MetricMix"
    elif loss_type=="huber_mse":
        huber=nn.HuberLoss(delta=args.huber_delta)
        mse=nn.MSELoss()
        criterion=lambda pred,target:(1-args.mse_weight)*huber(pred,target)+args.mse_weight*mse(pred,target)
        loss_name="HuberMSE"
    elif loss_type == "huber_smape":
        huber = nn.HuberLoss(delta=args.huber_delta)
        relative_weight = float(args.relative_loss_weight)
        relative_eps = max(metric_scale * 1e-3, 1e-6)
        relative_scale = args.huber_delta * metric_scale

        def criterion(pred, target):
            huber_loss = huber(pred, target)
            absolute_error = (pred - target).abs()
            smape_loss = (2.0 * absolute_error / (target.abs() + pred.abs()).clamp_min(relative_eps)).mean()
            return (1.0 - relative_weight) * huber_loss + relative_weight * relative_scale * smape_loss

        loss_name = "HuberSMAPE"
    else:
        criterion={"mse":nn.MSELoss(),"huber":nn.HuberLoss(delta=args.huber_delta),"l1":nn.L1Loss()}[loss_type]
        loss_name={"mse":"MSELoss","huber":"HuberLoss","l1":"L1Loss"}[loss_type]

    best_val=float("inf")
    best_epoch=-1
    best_state=best_optimizer_state=best_metrics=best_window_metrics=None

    print(f"Device: {device}")
    print(f"Horizon: {horizon}")
    print(f"Validation windows: {num_val_windows}")
    print(f"Dynamic covariates: {len(model_dynamic_columns)}")
    print(f"Static covariates: {len(static_columns)}")

    patience=4
    epochs_without_improvement=0
    train_loss_history=[]
    val_wape_history=[]
    lr_history=[]
    val_huber_history=[]

    for epoch in range(args.epochs):
        lr_history.append(optimizer.param_groups[0]["lr"])
        model.train()
        train_loss=0.0

        for batch in loader:
            series_index=batch["series_index"].to(device)
            static_features=batch["static_features"].to(device)
            past_target=batch["past_target"].to(device)
            past_cov=batch["past_covariates"].to(device)
            future_cov=batch["future_covariates"].to(device)
            target=batch["target"].to(device)

            optimizer.zero_grad()
            scale=series_scales[series_index].unsqueeze(1)
            history=past_target
            all_cov=torch.cat([past_cov,future_cov],dim=1)
            loss=0.0

            for block_idx in range(train_rollout_blocks):
                offset=block_idx*horizon
                block_past_cov=all_cov[:,offset:offset+history_length]
                block_future_cov=all_cov[:,history_length+offset:history_length+offset+horizon]
                block_target=target[:,offset:offset+horizon]
                model_history=history[:,-history_length:]/scale if getattr(args,"capacity_normalize_target",False) else history[:,-history_length:]

                prediction=model(model_history,block_past_cov,block_future_cov,series_index,static_features)
                if getattr(args,"capacity_normalize_target",False): prediction=prediction*scale

                loss+=criterion(prediction,block_target)
                history=torch.cat([history,prediction],dim=1)

            loss/=train_rollout_blocks
            loss.backward()
            optimizer.step()

            if ema_model is not None:
                with torch.no_grad():
                    for ema_parameter,parameter in zip(ema_model.parameters(),model.parameters()):
                        ema_parameter.mul_(ema_decay).add_(parameter,alpha=1-ema_decay)
                    for ema_buffer,buffer in zip(ema_model.buffers(),model.buffers()):
                        ema_buffer.copy_(buffer)

            train_loss+=loss.item()

        train_loss/=len(loader)
        train_loss_history.append(train_loss)

        if train_full:
            eval_model=ema_model if ema_model is not None else model
            best_epoch=epoch
            best_state={key:value.detach().cpu().clone() for key,value in eval_model.state_dict().items()}
            best_optimizer_state=copy.deepcopy(optimizer.state_dict())
            print(f"{epoch+1}: train_{loss_name}={train_loss:.4f}")
            if scheduler is not None: scheduler.step()
            continue

        window_metrics=[]
        window_wapes=[]
        window_hubers=[]
        eval_model=ema_model if ema_model is not None else model

        for split in processed_splits:
            targets=split["raw"].groupby("series_id",sort=False).tail(VALIDATION_HORIZON)
            predictions=rollout(eval_model,split["frame"],model_dynamic_columns,static_columns,series_mapping,VALIDATION_HORIZON,history_length,horizon,device,series_scales if getattr(args,"capacity_normalize_target",False) else None)
            result=targets.merge(predictions,on=["series_id","timestamp"])

            target_tensor=torch.tensor(result["target"].to_numpy(),dtype=torch.float32)
            prediction_tensor=torch.tensor(result["prediction"].to_numpy(),dtype=torch.float32)
            window_hubers.append(nn.functional.huber_loss(prediction_tensor,target_tensor,delta=delta).item())
            window_metrics.append(all_metrics(result["target"],result["prediction"]))
            window_wapes.append(wape(result["target"],result["prediction"]))

        metrics={k:float(np.mean([m[k] for m in window_metrics])) for k in window_metrics[0]}
        val=float(np.mean(window_wapes))
        val_huber=float(np.mean(window_hubers))
        val_wape_history.append(val)
        val_huber_history.append(val_huber)

        if val<best_val:
            best_val,best_epoch=val,epoch
            best_metrics=metrics.copy()
            best_window_metrics=copy.deepcopy(window_metrics)
            best_state={k:v.detach().cpu().clone() for k,v in eval_model.state_dict().items()}
            best_optimizer_state=copy.deepcopy(optimizer.state_dict())
            epochs_without_improvement=0
        else:
            epochs_without_improvement+=1

        ws=",".join(f"{x*100:.2f}" for x in window_wapes)
        print(f"{epoch+1}: train_{loss_name}={train_loss:.4f}, meanWAPE={val*100:.2f}, windows=[{ws}], MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}, sMAPE={metrics['smape']:.2f}")

        if trial is not None:
            trial.report(best_val,epoch)
            if trial.should_prune(): raise __import__("optuna").TrialPruned()

        if trial is None and epochs_without_improvement>=patience:
            print(f"Early stopping after epoch {epoch+1}")
            break

        if scheduler is not None: scheduler.step()

    mapping={"hidden_size":"h","num_blocks":"b","num_layers":"l","dropout":"d","covariate_hidden_size":"c","series_embedding_size":"e","static_hidden_size":"sh"}
    model_args=[f"{short}{getattr(args,key)}" for key,short in mapping.items() if hasattr(args,key)]

    if getattr(args,"standardize_covariates",False): model_args.append("std")
    if getattr(args,"missing_masks",False): model_args.append("mask")
    if getattr(args,"static_branch",False): model_args.append("static")
    if getattr(args,"past_tcn",False): model_args.append("ptcn")
    if getattr(args,"future_tcn",False): model_args.append("ftcn")
    if getattr(args,"feature_gating",False): model_args.append("gate")
    if getattr(args,"film_conditioning",False): model_args.append("film")
    if getattr(args,"multiscale_target",False): model_args.append("multiscale")
    if getattr(args,"capacity_normalize_target",False): model_args.append("capnorm")
    if getattr(args,"target_tcn",False): model_args.append("ttcn")
    if getattr(args,"target_revin",False): model_args.append("revin")
    if getattr(args,"optimizer","adam")!="adam": model_args.append(getattr(args,"optimizer"))
    if getattr(args,"weight_decay",0.0)>0: model_args.append(f"wd{args.weight_decay:g}")
    if loss_type=="huber": model_args.append(f"delta{delta:g}")
    if loss_type == "huber_smape": model_args.append(f"rel{args.relative_loss_weight:g}")
    if train_rollout_blocks>1: model_args.append(f"roll{train_rollout_blocks}")
    model_args.append(f"seed{seed}")
    if ema_decay>0: model_args.append(f"ema{ema_decay:g}")
    if lr_decay_milestone>0: model_args.append(f"sched{lr_decay_milestone}x{lr_decay_gamma:g}")

    prefix=f"trial{trial.number}_" if trial is not None else ""
    split_tag="fulltrain" if train_full else f"cv{num_val_windows}"
    score_parts=[] if train_full else [f"wape{best_val:.4f}"]
    name="_".join([prefix+args.model,f"hz{horizon}",split_tag,f"loss{loss_type}",f"lr{args.lr:g}",f"bs{args.batch_size}",*model_args,*score_parts])
    path=Path("checkpoints")/f"{name}.pt"

    torch.save({
        "model":args.model,
        "args":vars(args),
        "epoch":best_epoch,
        "best_val":best_val,
        "best_metrics":best_metrics,
        "best_window_metrics":best_window_metrics,
        "history_length":history_length,
        "dynamic_feature_columns":dynamic_columns,
        "model_dynamic_columns":model_dynamic_columns,
        "static_feature_columns":static_columns,
        "preprocessor":preprocessor,
        "standardize_covariates":getattr(args,"standardize_covariates",False),
        "missing_masks":getattr(args,"missing_masks",False),
        "series_mapping":series_mapping,
        "state_dict":best_state,
        "optimizer_state_dict":best_optimizer_state,
        "horizon":horizon,
    },path)

    if train_full:
        print(f"\n{args.model}: full-data training complete")
        print(f"Final epoch: {best_epoch+1}")
    else:
        print(f"\n{args.model}: mean WAPE={best_val:.4f}")
        print(f"Best epoch: {best_epoch+1}")

    print(f"Checkpoint: {path}")

    if trial is None and not train_full:
        for name,score in baseline_scores.items(): print(f"{name}: {score:.4f}")

        import matplotlib.pyplot as plt

        epochs_axis=range(1,len(train_loss_history)+1)

        fig,ax1=plt.subplots()
        ax1.plot(epochs_axis,train_loss_history,marker="o",label="Train Huber")
        ax1.plot(epochs_axis,val_huber_history,marker="o",label="Validation Huber")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Huber loss")
        ax1.legend(loc="upper left")

        ax2=ax1.twinx()
        ax2.plot(epochs_axis,val_wape_history,marker="o",linestyle="--",label="Validation WAPE")
        ax2.set_ylabel("WAPE")
        ax2.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig("training_curve.png",dpi=150)
        plt.close(fig)

        fig,ax=plt.subplots()
        ax.plot(epochs_axis,lr_history,marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig("learning_rate.png",dpi=150)
        plt.close(fig)

    return best_val