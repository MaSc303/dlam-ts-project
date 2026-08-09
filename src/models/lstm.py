from torch import nn

def add_model_args(parser):
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)

def create_model(args, history_length, horizon):
    return ForecastModel(horizon, args.hidden_size, args.num_layers)

class ForecastModel(nn.Module):
    def __init__(self, horizon, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(-1))
        return self.output(x[:, -1])