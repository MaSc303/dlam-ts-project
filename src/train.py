import argparse
from pathlib import Path

import torch

from src.training.train_model import train_model

parser=argparse.ArgumentParser()

parser.add_argument("--model",required=True)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--batch-size",type=int,default=128)
parser.add_argument("--lr",type=float,default=5e-4)
parser.add_argument("--weight-decay",type=float,default=0.0)
parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed",type=int,default=42)

parser.add_argument("--history-length",type=int,default=168)
parser.add_argument("--horizon",type=int,default=24)
parser.add_argument("--num-val-windows",type=int,default=5)

parser.add_argument("--loss",choices=["mse", "huber", "l1", "huber_mse", "metric_mix", "huber_smape"],default="mse")
parser.add_argument("--huber-delta",type=float,default=1.0)
parser.add_argument("--mse-weight",type=float,default=0.2)

parser.add_argument("--hidden-size",type=int,default=128)
parser.add_argument("--num-blocks",type=int,default=4)
parser.add_argument("--num-layers",type=int,default=3)
parser.add_argument("--dropout",type=float,default=0.0)
parser.add_argument("--covariate-hidden-size",type=int,default=8)
parser.add_argument("--series-embedding-size",type=int,default=16)
parser.add_argument("--static-hidden-size",type=int,default=8)

parser.add_argument("--standardize-covariates",action="store_true")
parser.add_argument("--missing-masks",action="store_true")
parser.add_argument("--static-branch",action="store_true")
parser.add_argument("--seasonal-residual",action="store_true")

parser.add_argument("--tcn-covariates",action="store_true")
parser.add_argument("--past-tcn",action="store_true")
parser.add_argument("--future-tcn",action="store_true")
parser.add_argument("--target-tcn",action="store_true")
parser.add_argument("--multiscale-target",action="store_true")
parser.add_argument("--capacity-normalize-target",action="store_true")
parser.add_argument("--target-revin",action="store_true")
parser.add_argument("--feature-gating",action="store_true")
parser.add_argument("--film-conditioning",action="store_true")

parser.add_argument("--train-rollout-blocks",type=int,default=1)
parser.add_argument("--resume-checkpoint",type=Path)
parser.add_argument("--train-full",action="store_true")
parser.add_argument("--reserve-holdout",action="store_true")

parser.add_argument("--optimizer",choices=["adam","adamw"],default="adam")
parser.add_argument("--ema-decay",type=float,default=0.0)
parser.add_argument("--lr-decay-milestone",type=int,default=0)
parser.add_argument("--lr-decay-gamma",type=float,default=0.3)
parser.add_argument("--relative-loss-weight", type=float, default=0.2)

args=parser.parse_args()
train_model(args)