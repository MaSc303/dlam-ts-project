from torch import nn

def add_model_args(parser):
    return

def create_model(args, history_length, horizon, num_covariates, num_series):
    return ForecastModel(history_length, horizon)

class ForecastModel(nn.Module):
    def __init__(self, history_length=168, horizon=24):
        super().__init__()
        self.linear = nn.Linear(history_length, horizon)

    def forward(self, past_target, past_covariates, future_covariates, series_index, static_features=None):
        return self.linear(past_target)