import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.tide_dataset import ForecastDataset, STATIC_COLS
from src.models.tide import TiDE


def make_val_loader(frame, L, horizon, pipe, n_origins, stride):
    n_rows = frame.groupby("series_id").size().iloc[0]
    last = n_rows - L - horizon
    starts = [last - k * stride for k in range(n_origins)][::-1]
    assert min(starts) >= 0, f"zu wenig Daten fuer L={L}"
    ds = ForecastDataset(frame, L, horizon, pipe.feature_cols,
                         pipe.series_stats, starts=starts)
    return DataLoader(ds, batch_size=128, shuffle=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); num = den = 0.0
    for batch in loader:
        *inp, y = [t.to(device, non_blocking=True) for t in batch]
        p = model(*inp)
        num += (p - y).abs().sum().item(); den += y.abs().sum().item()
    return num / den


def train_model(L, norm_mode, seed, train_frame, pipe, horizon, device,
                batch_size, lr, checkpoint_dir, val_loader=None,
                epochs=60, patience=8, tag=""):
    """Ohne val_loader laeuft die volle Epochenzahl durch und das letzte Modell
    wird behalten - fuer das Nachtraining ohne unabhaengigen Validierungssatz."""
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    ds = ForecastDataset(train_frame, L, horizon, pipe.feature_cols, pipe.series_stats)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                    pin_memory=(device.type == "cuda"), drop_last=True,
                    generator=torch.Generator().manual_seed(seed))

    model = TiDE(L, horizon, len(pipe.feature_cols), len(STATIC_COLS),
                 norm_mode=norm_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                 total_steps=epochs * len(dl), pct_start=0.3)
    crit = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ckpt = checkpoint_dir / f"tide_L{L}_{norm_mode}_s{seed}{tag}.pt"
    best, best_ep, bad = float("inf"), epochs, 0

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for batch in dl:
            *inp, y = [t.to(device, non_blocking=True) for t in batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=(device.type == "cuda")):
                loss = crit(model(*inp), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()

        if val_loader is None:
            print(f"  ep {ep:2d}  train_mae {tot/len(dl):.4f}")
            continue

        w = evaluate(model, val_loader, device); mark = ""
        if w < best - 1e-4:
            best, best_ep, bad = w, ep, 0
            torch.save(model.state_dict(), ckpt); mark = " *"
        else:
            bad += 1
        print(f"  ep {ep:2d}  train_mae {tot/len(dl):.4f}  val_wape {w:.4f}{mark}")
        if bad >= patience:
            print(f"  early stop nach Epoche {ep}"); break

    if val_loader is None:
        torch.save(model.state_dict(), ckpt)
    return ckpt, best, best_ep
