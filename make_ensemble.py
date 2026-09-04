"""Assemble one weighted ensemble checkpoint from member checkpoints.

    python make_ensemble.py --l1 ck_C_10.pt ck_full_L1_s1.pt ck_full_L1_s2.pt \
                            --mse ck_D.pt ck_D_4.pt ck_full_MSE_s1.pt \
                            --w_l1 0.6 --out checkpoint.pt

Members within a group share the group weight equally.  All members must have
the same architecture and the same feature layout; the stats block (profile,
scales, feature list) is taken from the first member after checking that every
member's profile matches it.  predict.py reads `state_dict` as a list and
`weights` as the per-member blend.
"""
from __future__ import annotations

import argparse
import numpy as np, torch

ARCH = ["d_model", "layers", "nhead", "ff", "n_feat", "n_units", "film", "level_head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", nargs="+", required=True)
    ap.add_argument("--mse", nargs="*", default=[])
    ap.add_argument("--w_l1", type=float, default=0.6, help="total weight of the L1 group")
    ap.add_argument("--out", default="checkpoint.pt")
    a = ap.parse_args()

    members = [(p, "l1") for p in a.l1] + [(p, "mse") for p in a.mse]
    n1, n2 = len(a.l1), len(a.mse)
    w1 = a.w_l1 if n2 else 1.0
    weights = [w1 / n1] * n1 + ([(1 - w1) / n2] * n2 if n2 else [])

    states, ref, ref_stats = [], None, None
    for path, kind in members:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        assert not isinstance(sd, list), f"{path} is already an ensemble"
        cfg = ck["cfg"]
        if ref is None:
            ref, ref_stats = cfg, ck["stats"]
        for k in ARCH:
            assert cfg.get(k) == ref.get(k), f"{path}: cfg[{k}]={cfg.get(k)} != {ref.get(k)}"
        st = ck["stats"]
        assert st.get("features") == ref_stats.get("features"), f"{path}: feature list differs"
        assert np.allclose(st["profile"], ref_stats["profile"]), f"{path}: profile differs"
        assert np.allclose(st["mu"], ref_stats["mu"]), f"{path}: covariate stats differ"
        states.append(sd)
        print(f"  {path:28s} {kind:3s}  weight {weights[len(states) - 1]:.3f}  "
              f"holdout {ck.get('holdout_wape')}  best_epoch {ck.get('best_epoch')}  "
              f"val_windows {ck.get('val_windows')}  seed {ck.get('seed')}")

    out = {"state_dict": states, "weights": weights, "stats": ref_stats,
           "cfg": {**ref, "loss": "ensemble"}, "members": [p for p, _ in members],
           "member_kind": [k for _, k in members]}
    torch.save(out, a.out)
    print(f"saved {a.out}: {len(states)} members, L1 group weight {w1:.2f}, "
          f"weights sum {sum(weights):.3f}")


if __name__ == "__main__":
    main()
