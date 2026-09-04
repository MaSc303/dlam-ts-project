import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout, layer_norm=True):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.skip = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out) if layer_norm else None

    def forward(self, x):
        out = self.fc2(self.drop(torch.relu(self.fc1(x)))) + self.skip(x)
        return self.norm(out) if self.norm is not None else out


class TiDE(nn.Module):
    def __init__(self, L, H, n_dynamic, n_static,
                 hidden_size=256, encoder_layers=2, decoder_layers=2,
                 decoder_output_size=16, temporal_hidden_size=64,
                 feature_proj_size=4, dropout=0.1, norm_mode="series"):
        super().__init__()
        assert norm_mode in ("series", "window", "none")
        self.L, self.H, self.norm_mode = L, H, norm_mode
        self.d_out = decoder_output_size

        self.cov_proj = ResidualBlock(n_dynamic, feature_proj_size,
                                      feature_proj_size, dropout)
        enc_in = L + (L + H) * feature_proj_size + n_static + 2

        enc = [ResidualBlock(enc_in, hidden_size, hidden_size, dropout)]
        enc += [ResidualBlock(hidden_size, hidden_size, hidden_size, dropout)
                for _ in range(encoder_layers - 1)]
        self.encoder = nn.Sequential(*enc)

        dec = [ResidualBlock(hidden_size, hidden_size, hidden_size, dropout)
               for _ in range(decoder_layers - 1)]
        dec += [ResidualBlock(hidden_size, hidden_size, H * decoder_output_size, dropout)]
        self.decoder = nn.Sequential(*dec)

        self.temporal_decoder = ResidualBlock(
            decoder_output_size + feature_proj_size,
            temporal_hidden_size, 1, dropout, layer_norm=False)
        self.global_skip = nn.Linear(L, H)

    def _stats(self, x, stats):
        if self.norm_mode == "series":
            m, s = stats[:, :1], stats[:, 1:2]
        elif self.norm_mode == "window":
            m = x.mean(dim=1, keepdim=True)
            s = x.std(dim=1, keepdim=True)
        else:
            m = torch.zeros_like(x[:, :1]); s = torch.ones_like(x[:, :1])
        s = torch.where(s < 1e-3, torch.ones_like(s), s)
        return m, s

    def forward(self, x, past_cov, future_cov, static, stats):
        m, s = self._stats(x, stats)
        xn = (x - m) / s

        p_past = self.cov_proj(past_cov).flatten(1)
        p_future = self.cov_proj(future_cov)

        w_mean = x.mean(dim=1, keepdim=True)
        w_std = x.std(dim=1, keepdim=True).clamp(min=1e-3)
        level = torch.cat([(w_mean - m) / s, torch.log1p(w_std / s)], dim=1)

        enc_in = torch.cat([xn, p_past, p_future.flatten(1), static, level], dim=1)
        decoded = self.decoder(self.encoder(enc_in)).view(-1, self.H, self.d_out)
        decoded = torch.cat([decoded, p_future], dim=-1)

        out = self.temporal_decoder(decoded).squeeze(-1) + self.global_skip(xn)
        return out * s + m
