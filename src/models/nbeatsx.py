import torch
from torch import nn

def add_model_args(parser):
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--covariate-hidden-size", type=int, default=16)
    parser.add_argument("--series-embedding-size", type=int, default=8)
    parser.add_argument("--static-hidden-size", type=int, default=8)
    parser.add_argument("--static-branch", action="store_true")
    parser.add_argument("--seasonal-residual", action="store_true")

def create_model(args, history_length, horizon, num_covariates, num_series):
    return ForecastModel(
        history_length,
        horizon,
        num_covariates,
        num_series,
        getattr(args, "num_static_features", 0),
        args.hidden_size,
        args.num_blocks,
        args.num_layers,
        args.dropout,
        args.covariate_hidden_size,
        args.series_embedding_size,
        getattr(args, "static_hidden_size", 8),
        getattr(args, "static_branch", False),
        getattr(args, "seasonal_residual", False)
    )

class NBeatsXBlock(nn.Module):
    def __init__(
        self,
        history_length,
        horizon,
        covariate_size,
        context_size,
        hidden_size,
        num_layers,
        dropout
    ):
        super().__init__()

        input_size = (
            history_length
            + history_length * covariate_size
            + horizon * covariate_size
            + context_size
        )

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        self.mlp = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_size, history_length)
        self.forecast = nn.Linear(hidden_size, horizon)

    def forward(self, residual, past_covariates, future_covariates, context):
        x = torch.cat([
            residual,
            past_covariates.flatten(1),
            future_covariates.flatten(1),
            context
        ], dim=1)

        x = self.mlp(x)
        return self.backcast(x), self.forecast(x)

class ForecastModel(nn.Module):
    def __init__(
        self,
        history_length,
        horizon,
        num_covariates,
        num_series,
        num_static_features,
        hidden_size=256,
        num_blocks=4,
        num_layers=3,
        dropout=0.1,
        covariate_hidden_size=16,
        series_embedding_size=8,
        static_hidden_size=8,
        static_branch=False,
        seasonal_residual=False
    ):
        super().__init__()

        self.horizon = horizon
        self.seasonal_residual = seasonal_residual

        self.covariate_projection = nn.Sequential(
            nn.Linear(num_covariates, covariate_hidden_size),
            nn.ReLU()
        )

        self.series_embedding = nn.Embedding(
            num_series,
            series_embedding_size
        )

        if static_branch and num_static_features > 0:
            self.static_projection = nn.Sequential(
                nn.Linear(num_static_features, static_hidden_size),
                nn.ReLU()
            )
            context_size = series_embedding_size + static_hidden_size
        else:
            self.static_projection = None
            context_size = series_embedding_size

        self.blocks = nn.ModuleList([
            NBeatsXBlock(
                history_length,
                horizon,
                covariate_hidden_size,
                context_size,
                hidden_size,
                num_layers,
                dropout
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        past_target,
        past_covariates,
        future_covariates,
        series_index,
        static_features=None
    ):
        past_covariates = self.covariate_projection(past_covariates)
        future_covariates = self.covariate_projection(future_covariates)

        context = [self.series_embedding(series_index)]

        if self.static_projection is not None:
            context.append(self.static_projection(static_features))

        context = torch.cat(context, dim=1)

        residual = past_target
        forecast = torch.zeros(
            past_target.shape[0],
            future_covariates.shape[1],
            device=past_target.device
        )

        for block in self.blocks:
            backcast, block_forecast = block(
                residual,
                past_covariates,
                future_covariates,
                context
            )
            residual = residual - backcast
            forecast = forecast + block_forecast

        if self.seasonal_residual:
            forecast = forecast + past_target[:, -forecast.shape[1]:]

        return forecast