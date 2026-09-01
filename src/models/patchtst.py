from torch import nn
from transformers import PatchTSTConfig, PatchTSTForPrediction


def add_model_args(parser):
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-length", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)


def create_model(args, history_length, horizon):
    return ForecastModel(args, history_length, horizon)


class ForecastModel(nn.Module):
    def __init__(self, args, history_length, horizon):
        super().__init__()

        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=history_length,
            prediction_length=horizon,
            patch_length=args.patch_length,
            patch_stride=args.patch_stride,
            d_model=args.d_model,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            dropout=args.dropout,
            loss="mse",
        )

        # Erstellt ein neues, zufällig initialisiertes Modell.
        self.model = PatchTSTForPrediction(config)

    def forward(self, x):
        # [B, 168] -> [B, 168, 1]
        output = self.model(past_values=x.unsqueeze(-1))

        prediction = output.prediction_outputs

        # Je nach Transformers-Version: [B, 24, 1] oder [B, 1, 24]
        if prediction.ndim == 3:
            if prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            elif prediction.shape[1] == 1:
                prediction = prediction.squeeze(1)

        return prediction