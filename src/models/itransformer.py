import torch
from torch import nn


def add_model_args(parser):
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)


def create_model(args, history_length, horizon):
    return ForecastModel(
        history_length=history_length,
        horizon=horizon,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
    )


class ForecastModel(nn.Module):
    def __init__(
        self,
        history_length,
        horizon,
        d_model,
        num_layers,
        num_heads,
        ff_size,
        dropout,
    ):
        super().__init__()

        # Die komplette Historie einer Variable wird ein Embedding.
        self.variable_embedding = nn.Linear(history_length, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.forecast_head = nn.Linear(d_model, horizon)

    def forward(self, x):
        # Dein Dataset liefert [B, L].
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [B, L, 1]

        # Normalisierung jeder Variable entlang der Zeitachse.
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        x = (x - mean) / std

        # Inversion: [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)

        # Jeder Variablenverlauf wird ein Token.
        x = self.variable_embedding(x)  # [B, C, d_model]
        x = self.encoder(x)             # Attention über C
        x = self.forecast_head(x)       # [B, C, horizon]

        # [B, C, horizon] -> [B, horizon, C]
        prediction = x.transpose(1, 2)

        # Rückskalieren.
        prediction = prediction * std + mean

        # Dein Trainingsloop erwartet [B, horizon].
        if prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)

        return prediction