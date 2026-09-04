import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,history_length,horizon,hidden_size,input_size,num_layers,dropout):
        super().__init__()
        self.target_encoder=nn.Sequential(nn.Linear(history_length,hidden_size),nn.ReLU())
        self.input=nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),nn.Dropout(dropout))
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Dropout(dropout)) for _ in range(max(0,num_layers-1))])
        self.backcast=nn.Linear(hidden_size,history_length)
        self.forecast=nn.Linear(hidden_size,horizon)

    def forward(self,residual,past_encoded,future_encoded,context):
        target=self.target_encoder(residual)
        h=self.input(torch.cat([target,past_encoded,future_encoded,context],dim=-1))
        for layer in self.layers: h=h+layer(h)
        return self.backcast(h),self.forecast(h)

class NBeatsXSeparate(nn.Module):
    def __init__(self,history_length,horizon,num_covariates,num_series,hidden_size=128,num_blocks=4,num_layers=3,dropout=0.05,covariate_hidden_size=8,series_embedding_size=16,num_static_features=0,static_hidden_size=4,seasonal_residual=False):
        super().__init__()
        self.horizon=horizon
        self.seasonal_residual=seasonal_residual

        self.past_proj=nn.Sequential(nn.Linear(num_covariates,covariate_hidden_size),nn.ReLU())
        self.future_proj=nn.Sequential(nn.Linear(num_covariates,covariate_hidden_size),nn.ReLU())
        self.past_encoder=nn.Sequential(nn.Linear(history_length*covariate_hidden_size,hidden_size),nn.ReLU())
        self.future_encoder=nn.Sequential(nn.Linear(horizon*covariate_hidden_size,hidden_size),nn.ReLU())

        self.series_embedding=nn.Embedding(num_series,series_embedding_size)
        self.static_encoder=nn.Sequential(nn.Linear(num_static_features,static_hidden_size),nn.ReLU()) if num_static_features>0 else None

        context_size=series_embedding_size+(static_hidden_size if self.static_encoder is not None else 0)
        input_size=hidden_size*3+context_size
        self.blocks=nn.ModuleList([Block(history_length,horizon,hidden_size,input_size,num_layers,dropout) for _ in range(num_blocks)])

    def forward(self,past_target,past_covariates,future_covariates,series_index,static_features=None):
        past=self.past_encoder(self.past_proj(past_covariates).flatten(1))
        future=self.future_encoder(self.future_proj(future_covariates).flatten(1))

        context=[self.series_embedding(series_index)]
        if self.static_encoder is not None: context.append(self.static_encoder(static_features))
        context=torch.cat(context,dim=-1)

        residual=past_target
        forecast=torch.zeros(past_target.size(0),self.horizon,device=past_target.device,dtype=past_target.dtype)

        for block in self.blocks:
            backcast,block_forecast=block(residual,past,future,context)
            residual=residual-backcast
            forecast=forecast+block_forecast

        if self.seasonal_residual: forecast=forecast+past_target[:,-self.horizon:]
        return forecast

def create_model(args,history_length,horizon,num_covariates,num_series):
    return NBeatsXSeparate(
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
        seasonal_residual=getattr(args,"seasonal_residual",False)
    )