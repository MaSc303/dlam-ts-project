from torch import nn

def add_model_args(parser):
    pass

def create_model(args, history_length, horizon):
    return ForecastModel(history_length, horizon)

class ForecastModel(nn.Module):
    def __init__(self, history_length, horizon):
        super().__init__()
        self.linear = nn.Linear(history_length, horizon)

    def forward(self, x):
        return self.linear(x)