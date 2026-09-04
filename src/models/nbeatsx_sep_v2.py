import torch
from torch import nn

class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation,dropout):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels,channels,1),
        )

    def forward(self,x):
        return x+self.net(x)

class CovariateEncoder(nn.Module):
    def __init__(self,length,num_covariates,covariate_hidden_size,hidden_size,dropout,use_tcn,feature_gating=False):
        super().__init__()
        self.feature_gates=nn.Parameter(torch.full((num_covariates,),4.0)) if feature_gating else None
        self.proj=nn.Sequential(nn.Linear(num_covariates,covariate_hidden_size),nn.ReLU())

        if use_tcn:
            dilations=[1,2,4,8,16,32] if length>=168 else [1,2,4,8]
            self.temporal=nn.Sequential(*[TemporalBlock(covariate_hidden_size,dilation,dropout) for dilation in dilations])
        else:
            self.temporal=nn.Identity()

        self.out=nn.Sequential(nn.Linear(length*covariate_hidden_size,hidden_size),nn.ReLU())

    def forward(self,x):
        if self.feature_gates is not None: x=x*torch.sigmoid(self.feature_gates)
        x=self.proj(x)
        x=self.temporal(x.transpose(1,2)).transpose(1,2)
        return self.out(x.flatten(1))

class TargetEncoder(nn.Module):
    def __init__(self,history_length,hidden_size,use_multiscale,use_tcn=False,dropout=0.0):
        super().__init__()
        self.history_length=history_length
        self.use_multiscale=use_multiscale
        self.use_tcn=use_tcn
        self.full=nn.Sequential(nn.Linear(history_length,hidden_size),nn.ReLU())

        if use_tcn:
            channels=16
            self.target_proj=nn.Conv1d(1,channels,1)
            self.target_temporal=nn.Sequential(*[TemporalBlock(channels,dilation,dropout) for dilation in [1,2,4,8,16,32]])
            self.target_out=nn.Sequential(nn.Linear(history_length*channels,hidden_size),nn.ReLU())

        if use_multiscale:
            self.daily_length=min(24,history_length)
            lags=list(range(24,min(history_length,168)+1,24))
            indices=[history_length-lag for lag in lags]
            self.register_buffer("weekly_indices",torch.tensor(indices,dtype=torch.long))
            self.daily=nn.Sequential(nn.Linear(self.daily_length,hidden_size),nn.ReLU())
            self.weekly=nn.Sequential(nn.Linear(len(indices),hidden_size),nn.ReLU())
            self.fusion=nn.Sequential(nn.Linear(hidden_size*3,hidden_size),nn.ReLU())

    def forward(self,x):
        if self.use_tcn:
            h=self.target_proj(x.unsqueeze(1))
            h=self.target_temporal(h)
            return self.target_out(h.flatten(1))

        full=self.full(x)
        if not self.use_multiscale: return full

        daily=self.daily(x[:,-self.daily_length:])
        weekly=self.weekly(x.index_select(1,self.weekly_indices))
        return self.fusion(torch.cat([full,daily,weekly],dim=-1))

class Block(nn.Module):
    def __init__(self,history_length,horizon,hidden_size,context_size,num_layers,dropout,use_multiscale,use_target_tcn):
        super().__init__()
        self.target_encoder=TargetEncoder(history_length,hidden_size,use_multiscale,use_target_tcn,dropout)
        self.input=nn.Sequential(nn.Linear(hidden_size*3+context_size,hidden_size),nn.ReLU(),nn.Dropout(dropout))
        self.layers=nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Dropout(dropout))
            for _ in range(max(0,num_layers-1))
        ])
        self.backcast=nn.Linear(hidden_size,history_length)
        self.forecast=nn.Linear(hidden_size,horizon)

    def forward(self,residual,past_encoded,future_encoded,context):
        target=self.target_encoder(residual)
        h=self.input(torch.cat([target,past_encoded,future_encoded,context],dim=-1))
        for layer in self.layers: h=h+layer(h)
        return self.backcast(h),self.forecast(h)

class NBeatsXSeparateV2(nn.Module):
    def __init__(
        self,history_length,horizon,num_covariates,num_series,
        hidden_size=128,num_blocks=4,num_layers=3,dropout=0.05,
        covariate_hidden_size=8,series_embedding_size=16,
        num_static_features=0,static_hidden_size=4,
        seasonal_residual=False,past_tcn=False,future_tcn=False,
        feature_gating=False,film_conditioning=False,
        multiscale_target=False,target_tcn=False,target_revin=False,
    ):
        super().__init__()
        self.horizon=horizon
        self.seasonal_residual=seasonal_residual
        self.target_revin=target_revin

        self.past_encoder=CovariateEncoder(history_length,num_covariates,covariate_hidden_size,hidden_size,dropout,past_tcn,feature_gating)
        self.future_encoder=CovariateEncoder(horizon,num_covariates,covariate_hidden_size,hidden_size,dropout,future_tcn,feature_gating)
        self.series_embedding=nn.Embedding(num_series,series_embedding_size)

        self.static_encoder=nn.Sequential(nn.Linear(num_static_features,static_hidden_size),nn.ReLU()) if num_static_features>0 else None
        context_size=series_embedding_size+(static_hidden_size if self.static_encoder is not None else 0)

        self.film=nn.Linear(context_size,hidden_size*4) if film_conditioning else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        self.blocks=nn.ModuleList([
            Block(history_length,horizon,hidden_size,context_size,num_layers,dropout,multiscale_target,target_tcn)
            for _ in range(num_blocks)
        ])

    def forward(self,past_target,past_covariates,future_covariates,series_index,static_features=None):
        past=self.past_encoder(past_covariates)
        future=self.future_encoder(future_covariates)

        context=[self.series_embedding(series_index)]
        if self.static_encoder is not None: context.append(self.static_encoder(static_features))
        context=torch.cat(context,dim=-1)

        if self.film is not None:
            gamma_p,beta_p,gamma_f,beta_f=self.film(context).chunk(4,dim=-1)
            past=past*(1+gamma_p)+beta_p
            future=future*(1+gamma_f)+beta_f

        if self.target_revin:
            target_mean=past_target.mean(dim=1,keepdim=True)
            target_std=past_target.std(dim=1,keepdim=True,unbiased=False).clamp_min(1e-5)
            residual=(past_target-target_mean)/target_std
        else:
            target_mean=None
            target_std=None
            residual=past_target

        forecast=torch.zeros(past_target.size(0),self.horizon,device=past_target.device,dtype=past_target.dtype)

        for block in self.blocks:
            backcast,block_forecast=block(residual,past,future,context)
            residual=residual-backcast
            forecast=forecast+block_forecast

        if self.target_revin: forecast=forecast*target_std+target_mean
        if self.seasonal_residual: forecast=forecast+past_target[:,-self.horizon:]

        return forecast

def create_model(args,history_length,horizon,num_covariates,num_series):
    return NBeatsXSeparateV2(
        history_length=history_length,
        horizon=horizon,
        num_covariates=num_covariates,
        num_series=num_series,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        num_layers=args.num_layers,
        dropout=args.dropout,
        covariate_hidden_size=args.covariate_hidden_size,
        series_embedding_size=args.series_embedding_size,
        num_static_features=getattr(args,"num_static_features",0),
        static_hidden_size=getattr(args,"static_hidden_size",4),
        seasonal_residual=getattr(args,"seasonal_residual",False),
        multiscale_target=getattr(args,"multiscale_target",False),
        past_tcn=getattr(args,"past_tcn",False),
        future_tcn=getattr(args,"future_tcn",False),
        feature_gating=getattr(args,"feature_gating",False),
        film_conditioning=getattr(args,"film_conditioning",False),
        target_tcn=getattr(args,"target_tcn",False),
        target_revin=getattr(args,"target_revin",False),
    )