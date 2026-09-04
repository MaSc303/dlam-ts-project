"""DLAM 2026 — residual encoder, Rossmann (daily) instantiation of the v3 core.

Shared by train.py, predict.py and diag.py.

What changed from v2, and why (diagnostics of 3 Sep):

*   The per-unit level swings by +-15-20% from one fortnight to the next, in a
    calendar-month cycle (second half of the month high), synchronised across
    units, and it is almost entirely explained by the covariates (weekly means
    of queue_pressure / network_pressure correlate 0.98 with it).  The v2 model
    could only locate the phase through `trend` -- a linear time index that is
    out of its training range at validation and further out at test.  So:

    - DAY-OF-MONTH harmonics (4 features): deterministic, in range everywhere.
    - GLOBAL covariate means (8 features): the cross-unit mean of each partly
      per-unit covariate at every hour -- the clean system-wide signal a
      per-unit window cannot see.
    - a multiplicative LEVEL HEAD: one scalar per window that scales the whole
      profile, so a level shift is one number instead of 336 residuals.
    - DROP_FEATURES = ("trend",) removes the time index for the ablation.

*   The switches are module constants and are recorded in the checkpoint stats,
    so predict.py rebuilds the exact feature block that was trained on.

Lag features are kept behind a flag exactly as in v2 (k >= 672 only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ------------------------------------------------------------------ constants
import os

SERIES, TIME, TARGET = "series_id", "timestamp", "target"

# ---- schema block: the ONLY thing that changes between the benchmark (hourly,
# 336-step, hour-of-week profile) and Rossmann (daily, 28-step, day-of-week
# profile).  Everything below this block is identical to the benchmark core.
FREQ = "D"                          # grid frequency
STEP = pd.Timedelta(days=1)         # one grid step
H = 28                              # prediction horizon, in steps
HOW = 7                             # length of the seasonal profile (day-of-week)
ROLL = 7                            # rolling block length for lag features
MIN_LAG = 56                        # lag floor (lags unused here)
VALID_MIN = 0.7                     # min share of observed targets per window
                                    # (closed days are missing; Sundays alone
                                    # are 14% of a 28-day window)

FEATURES = [
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "promo", "promo2_active", "school_holiday", "open",
    "state_holiday_a", "state_holiday_b", "state_holiday_c",
    "comp_dist_log", "comp_open",
    "store_type_a", "store_type_b", "store_type_c", "store_type_d",
    "assortment_a", "assortment_b", "assortment_c",
]

# switches (env vars so ablations need no code edits); recorded in the checkpoint
DROP_FEATURES = tuple(f for f in os.environ.get("DROP_FEATURES", "").split(",") if f)
USE_DOM = os.environ.get("USE_DOM", "1") == "1"            # day-of-month harmonics
GLOBAL_OF = ["promo", "school_holiday", "open",             # cross-store daily means
             "state_holiday_a", "state_holiday_b", "state_holiday_c"]
USE_GLOBAL = os.environ.get("USE_GLOBAL", "1") == "1"
# -----------------------------------------------------------------------------

ACTIVE = [f for f in FEATURES if f not in DROP_FEATURES]
N_DOM = 4


def n_feat(lags) -> int:
    """covariates + masks + profile (+ lags + 2 rolling) + global means + dom."""
    n = 2 * len(ACTIVE) + 1 + (len(lags) + 2 if lags else 0)
    n += len(GLOBAL_OF) if USE_GLOBAL else 0
    n += N_DOM if USE_DOM else 0
    return n


def min_hist(lags) -> int:
    """Earliest window start for which every lag reference is observed."""
    return max(lags) + ROLL if lags else 0


def parse_lags(spec: str):
    """'none' -> (); '672,840' -> (672, 840). Rejects anything below MIN_LAG."""
    if not spec or spec.strip().lower() in {"none", "off", "0", ""}:
        return ()
    out = tuple(int(x) for x in spec.replace(" ", "").split(","))
    bad = [k for k in out if k < MIN_LAG]
    if bad:
        raise ValueError(
            f"lags {bad} are below {MIN_LAG}h. They land inside the hidden "
            f"validation window at private test time, which inflates the "
            f"leaderboard and collapses on the private split.")
    return out


def _how_of(index: pd.DatetimeIndex) -> np.ndarray:
    """Position in the seasonal profile: day-of-week for daily data."""
    return index.dayofweek.to_numpy(np.int64)


def _dom_of(index: pd.DatetimeIndex) -> np.ndarray:
    """(L, 4) first two harmonics of the day-of-month phase."""
    d = index.day.to_numpy() - 1
    n = index.days_in_month.to_numpy()
    ph = 2.0 * np.pi * d / n
    return np.stack([np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)],
                    axis=1).astype(np.float32)


# ------------------------------------------------------------------ statistics
def fit_stats(train: pd.DataFrame) -> dict:
    """Aggregate statistics only -- no raw rows are stored in the checkpoint."""
    units = sorted(train[SERIES].unique())
    uid = {u: i for i, u in enumerate(units)}
    df = train.copy()
    df[TIME] = pd.to_datetime(df[TIME])
    df["_how"] = _how_of(pd.DatetimeIndex(df[TIME]))

    # per-unit x hour-of-week MEDIAN profile (96 x 168). Median, not mean:
    # WAPE is L1 and these series carry spiky level shifts.
    prof = np.zeros((len(units), HOW), np.float32)
    piv = df.groupby([SERIES, "_how"])[TARGET].median()
    gmed = df.groupby(SERIES)[TARGET].median()
    for u in units:
        prof[uid[u]] = (piv.loc[u].reindex(range(HOW)).fillna(gmed[u])
                        .to_numpy(np.float32))

    res_scale = np.zeros(len(units), np.float32)
    lag_center = np.zeros(len(units), np.float32)
    lag_scale = np.zeros(len(units), np.float32)
    for u in units:
        i, d = uid[u], df[df[SERIES] == u]
        t = d[TARGET].to_numpy(np.float32)
        r = t - prof[i][d["_how"].to_numpy()]
        q = np.nanpercentile(r, [25, 75])
        res_scale[i] = max(float(q[1] - q[0]), 1e-3)
        qt = np.nanpercentile(t, [25, 50, 75])
        lag_center[i], lag_scale[i] = float(qt[1]), max(float(qt[2] - qt[0]), 1e-3)

    mu = train[ACTIVE].mean().astype(np.float32).values
    sd = train[ACTIVE].std().replace(0.0, 1.0).astype(np.float32).values
    return {"units": units, "uid": uid, "profile": prof, "res_scale": res_scale,
            "lag_center": lag_center, "lag_scale": lag_scale, "mu": mu, "sd": sd,
            # v3: the feature block is self-describing
            "features": list(ACTIVE), "use_dom": USE_DOM,
            "global_of": list(GLOBAL_OF) if USE_GLOBAL else []}


# ------------------------------------------------------------------ arrays
def target_matrix(train: pd.DataFrame, st: dict):
    """(U, T) observed targets on a dense hourly grid, plus the grid."""
    df = train.copy()
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.drop_duplicates([SERIES, TIME]).sort_values([SERIES, TIME])
    grid = pd.date_range(df[TIME].min(), df[TIME].max(), freq=FREQ)
    Y = np.full((len(st["units"]), len(grid)), np.nan, np.float32)
    for u in st["units"]:
        Y[st["uid"][u]] = (df[df[SERIES] == u].set_index(TIME)
                           .reindex(grid)[TARGET].to_numpy(np.float32))
    return Y, grid


def _lag_block(Y, st, abs_hours, lags):
    """(U, L, len(lags)+2), normalised per unit."""
    U, L = Y.shape[0], len(abs_hours)
    out = np.zeros((U, L, len(lags) + 2), np.float32)
    c, s = st["lag_center"][:, None], st["lag_scale"][:, None]
    for j, k in enumerate(lags):
        out[:, :, j] = np.nan_to_num((Y[:, abs_hours - k] - c) / s)
    k0 = min(lags)
    med = np.zeros((U, L), np.float32)
    iqr = np.zeros((U, L), np.float32)
    for i, s0 in enumerate(abs_hours - k0 - ROLL + 1):
        q = np.nanpercentile(Y[:, s0:s0 + ROLL], [25, 50, 75], axis=1)
        med[:, i], iqr[:, i] = q[1], q[2] - q[0]
    out[:, :, len(lags)] = np.nan_to_num((med - c) / s)
    out[:, :, len(lags) + 1] = np.nan_to_num(iqr / s)
    return out


def build_block(cov, st, Y, abs_hours, grid_start, lags=()):
    """-> X (U, L, n_feat), how (U, L), prof (U, L) for the given absolute hours.

    The feature layout is read from `st` (recorded at training time), so
    inference rebuilds exactly what the checkpoint was trained on:
        [z-scored covariates | missing masks | normalised profile | (lags) |
         global means | day-of-month harmonics]
    """
    feats = st.get("features", ACTIVE)
    use_dom = st.get("use_dom", USE_DOM)
    gcols = [c for c in st.get("global_of", GLOBAL_OF if USE_GLOBAL else [])
             if c in feats]

    df = cov.copy()
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.drop_duplicates([SERIES, TIME]).sort_values([SERIES, TIME])
    stamps = pd.DatetimeIndex([grid_start + int(h) * STEP for h in abs_hours])
    how = _how_of(stamps)

    # shared-across-units blocks
    shared = []
    if gcols:
        g = df.groupby(TIME)[gcols].mean().reindex(stamps)      # NaN-aware
        idx = [feats.index(c) for c in gcols]
        gz = (g.to_numpy(np.float32) - st["mu"][idx]) / st["sd"][idx]
        shared.append(np.nan_to_num(gz, posinf=0.0, neginf=0.0))
    if use_dom:
        shared.append(_dom_of(stamps))
    shared = np.concatenate(shared, axis=1) if shared else None

    U, L = len(st["units"]), len(abs_hours)
    nf = 2 * len(feats) + 1 + (len(lags) + 2 if lags else 0) \
        + (shared.shape[1] if shared is not None else 0)
    X = np.zeros((U, L, nf), np.float32)
    prof = np.zeros((U, L), np.float32)
    lagb = _lag_block(Y, st, abs_hours, lags) if lags else None

    for u in st["units"]:
        i = st["uid"][u]
        s = df[df[SERIES] == u].set_index(TIME).reindex(stamps)
        raw = s[feats].to_numpy(np.float32)
        mask = np.isnan(raw).astype(np.float32)
        z = (np.nan_to_num(raw) - st["mu"]) / st["sd"]
        z = np.nan_to_num(z, posinf=0.0, neginf=0.0) * (1.0 - mask)
        pr = st["profile"][i][how]
        pz = ((pr - pr.mean()) / (pr.std() + 1e-6)).astype(np.float32)[:, None]
        parts = [z, mask, pz] + ([lagb[i]] if lags else []) \
            + ([shared] if shared is not None else [])
        X[i] = np.concatenate(parts, axis=1)
        prof[i] = pr
    return X, np.tile(how, (U, 1)), prof


# ------------------------------------------------------------------ dataset
class WindowDS(Dataset):
    """Sliding H-hour windows. `max_start` keeps the local holdout untouched."""

    def __init__(self, X, how, prof, y, valid, stride=6, min_start=0,
                 max_start=None):
        self.X, self.how, self.prof, self.y, self.valid = X, how, prof, y, valid
        T = X.shape[1]
        last = T - H if max_start is None else min(max_start, T - H)
        self.index = [(u, int(s)) for u in range(X.shape[0])
                      for s in np.arange(min_start, last + 1, stride)
                      if valid[u, s:s + H].mean() >= VALID_MIN]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        u, s = self.index[i]
        sl = slice(s, s + H)
        t = torch.from_numpy
        return (t(self.X[u, sl]), torch.tensor(u), t(self.how[u, sl]),
                t(self.prof[u, sl]), t(self.y[u, sl]), t(self.valid[u, sl]))


# ------------------------------------------------------------------ model
class ResidualEncoder(nn.Module):
    """Covariate sequence -> 336 normalised residuals (+ one window level),
    conditioned on the unit.

    With `film=True` the unit embedding emits a per-unit scale and shift applied
    to the projected covariates.  With `level_head=True` the mean-pooled encoder
    output emits one scalar per window; `compose` multiplies the seasonal
    profile by exp(level), so a level shift costs the network one number rather
    than 336 coherent residuals.  Both are initialised to identity.
    """

    def __init__(self, n_feat, n_units=96, d_model=128, nhead=8, layers=3,
                 ff=512, dropout=0.2, emb_dim=32, emb_drop=0.1, film=True,
                 level_head=False):
        super().__init__()
        self.emb_drop, self.film, self.level_head = emb_drop, film, level_head
        self.proj = nn.Linear(n_feat, d_model)
        self.unit = nn.Embedding(n_units, emb_dim)
        self.unit_proj = nn.Linear(emb_dim, d_model)
        if film:
            self.film_proj = nn.Linear(emb_dim, 2 * d_model)
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)
        self.how_emb = nn.Embedding(HOW, d_model)
        pe = torch.zeros(H, d_model)
        pos = torch.arange(H).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        if level_head:
            self.level = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
            nn.init.zeros_(self.level[1].weight)
            nn.init.zeros_(self.level[1].bias)
        nn.init.normal_(self.unit.weight, std=0.02)

    def forward(self, x, uid, how):
        e = self.unit(uid)
        if self.training and self.emb_drop > 0:
            keep = (torch.rand(x.size(0), 1, device=x.device) > self.emb_drop)
            e = e * keep.to(e.dtype)
        h = self.proj(x)
        if self.film:
            g, b = self.film_proj(e).chunk(2, dim=-1)
            h = h * (1.0 + g.unsqueeze(1)) + b.unsqueeze(1)
        h = h + self.unit_proj(e).unsqueeze(1) + self.how_emb(how) \
            + self.pe.unsqueeze(0)
        z = self.encoder(h)
        res = self.head(z).squeeze(-1)
        if self.level_head:
            lvl = self.level(z.mean(dim=1)).squeeze(-1)
            return res, lvl
        return res


# ------------------------------------------------------------------ metrics
def wape(y_true, y_pred, valid=None):
    m = np.ones_like(y_true) if valid is None else valid
    return float((np.abs(y_true - y_pred) * m).sum() / (np.abs(y_true) * m).sum())


def compose(out, prof, uid, res_scale):
    """model output (+ seasonal profile) -> real load units.

    `out` is either the residual tensor (B, H) or a (residual, level) pair
    when the model has a level head; level scales the profile multiplicatively.
    """
    if isinstance(out, (tuple, list)):
        res_n, lvl = out
        prof = prof * torch.exp(lvl.clamp(-2.0, 2.0)).unsqueeze(1).to(prof.dtype)
    else:
        res_n = out
    return (prof + res_n * res_scale[uid].unsqueeze(1)).clamp(min=0.0)


# ------------------------------------------------------------------ inference
@torch.no_grad()
def forecast(models, X, how, prof, st, device, batch=16, weights=None):
    """Weighted average of predictions over one or more models. -> (U, L)."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    w = np.ones(len(models)) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    rs = torch.tensor(st["res_scale"], device=device)
    U = X.shape[0]
    acc = np.zeros((U, X.shape[1]), np.float32)
    for m, wm in zip(models, w):
        m.eval()
        for s in range(0, U, batch):
            sl = slice(s, min(s + batch, U))
            u = torch.arange(sl.start, sl.stop, device=device)
            out = compose(m(torch.from_numpy(X[sl]).to(device), u,
                            torch.from_numpy(how[sl]).to(device)),
                          torch.from_numpy(prof[sl]).to(device), u, rs)
            acc[sl] += wm * out.float().cpu().numpy()
    return acc
