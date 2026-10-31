"""Can present-day held-out data choose blend weights, when it cannot choose models?

Measured on the leaderboard: the anchor alone scores 0.094, the neural emulator alone
0.166, and 0.7*anchor + 0.3*emulator scores 0.076. Fitting a parabola through those
three points puts the leaderboard optimum at w = 0.309, i.e. the submitted weight was
already essentially optimal.

The question this script answers is whether the *same* weight comes out of present-day
held-out data. Recalibrating the anchor on present-day data hurt it (0.094 -> 0.137, the
C01 submission), but a blend weight is a different kind of quantity: one number
describing the relative error structure of two predictors, not a per-age correction.
If present-day agrees with the leaderboard here, three-way weights can be fitted
locally and spent on a single submission instead of three.

Writes experiments/blend_weights.json.
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from common import WORK, SUB_SCALE, N_SITE, stats, SCENARIOS
import dataset as D
from dataset import age_shift_anchor
from hrc_onestep import (forcing_norm, build_multi, readout_feats, readout_dim,
                         rollout as hrc_rollout, SQUARE, AGE_COL, DEV)
from nn_onestep import Step, unroll, state_bounds

CHUNK = 120


def main():
    prep, _ = forcing_norm(True)
    _, _, y_mean, y_std = stats()
    ymf = y_mean.astype(np.float32); ysf = (y_std + 1e-10).astype(np.float32)
    ym = torch.as_tensor(ymf, device=DEV); ys = torch.as_tensor(ysf, device=DEV)
    ages_col = torch.as_tensor(AGE_COL, device=DEV)
    xa = np.load(WORK / "train_xann.npy", mmap_mode="r")
    yd = np.load(WORK / "train_ydec.npy", mmap_mode="r")
    y0 = np.load(WORK / "train_y0.npy")
    fold, _, _ = D.spatial_folds(5, 10.0, seed=0)
    val = np.arange(N_SITE)[fold == 0]

    # ---- the neural emulator, exactly the run that produced the 0.0451 rollout
    model = Step(144, 512, 3).to(DEV)
    model.load_state_dict(torch.load(WORK / "runs" / "IM_clamp_e24_h512.pt", weights_only=True))
    model.eval()
    bounds = state_bounds()

    # ---- the reservoir emulator, its best configuration
    SQUARE[0] = True
    Wi, Wr, leak = build_multi(801, 0, 0.9, [0.1, 0.3, 0.7])
    Wout = torch.as_tensor(np.load(WORK / "runs" / "AB_hrc_s0_wout.npy"), device=DEV)

    nn_p, hrc_p, truth = [], [], []
    with torch.no_grad():
        for i in range(0, len(val), CHUNK):
            s = val[i:i + CHUNK]
            xz = torch.as_tensor(prep(np.asarray(xa[s], dtype=np.float32)), device=DEV)
            yv = np.asarray(yd[s], dtype=np.float32)
            yz = torch.as_tensor((yv - ymf) / ysf, device=DEV)
            pn = unroll(model, xz, yz, ages_col, 0.0, None, bounds)
            nn_p.append((pn.view(len(s), 15, 40, 7)[:, :, -1, :2] * ys[:2]
                         + ym[:2]).cpu().numpy())
            ph = hrc_rollout(xz, yz[:, :, 0], Wout, Wi, Wr, leak, ages_col)
            hrc_p.append((ph[:, :, -1, :2] * ys[:2] + ym[:2]).cpu().numpy())
            truth.append(yv[:, :, 40, :2])
            del xz, yz, pn, ph
    nn_p = np.concatenate(nn_p) / SUB_SCALE
    hrc_p = np.concatenate(hrc_p) / SUB_SCALE
    truth = np.concatenate(truth) / SUB_SCALE
    anc = age_shift_anchor(y0[val].astype(np.float64))[:, :, :2] / SUB_SCALE

    def mse(p):
        return float(np.mean((p - truth) ** 2))

    base = dict(anchor=mse(anc), nn=mse(nn_p), hrc=mse(hrc_p))
    print("local held-out, competition units:",
          {k: round(v, 5) for k, v in base.items()})

    # two-way optimum, closed form over the single weight
    d = (nn_p - anc).ravel(); r = (truth - anc).ravel()
    w2 = float(d @ r / (d @ d))
    print(f"local optimal anchor/nn weight       w = {w2:.3f}  -> mse {mse((1-w2)*anc + w2*nn_p):.5f}")
    print(f"leaderboard parabola through 0.094 / 0.076 / 0.166 puts it at w = 0.309")
    print(f"local mse at the submitted w=0.30    {mse(0.7*anc + 0.3*nn_p):.5f}")

    # three-way, unconstrained least squares on the three predictors
    A = np.stack([anc.ravel(), nn_p.ravel(), hrc_p.ravel()], 1)
    coef, *_ = np.linalg.lstsq(A, truth.ravel(), rcond=None)
    p3 = (A @ coef).reshape(truth.shape)
    print(f"local optimal three-way weights      anchor {coef[0]:.3f} nn {coef[1]:.3f} "
          f"hrc {coef[2]:.3f}  -> mse {mse(p3):.5f}")

    out = dict(local=base, w2_local=w2, w2_leaderboard_parabola=0.309,
               mse_local_at_w030=mse(0.7 * anc + 0.3 * nn_p),
               mse_local_at_w2=mse((1 - w2) * anc + w2 * nn_p),
               three_way=dict(anchor=float(coef[0]), nn=float(coef[1]),
                              hrc=float(coef[2]), mse_local=mse(p3)),
               n_val_sites=int(len(val)))
    with open("experiments/blend_weights.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    np.save(WORK / "blend3_coef.npy", coef)
    print("wrote experiments/blend_weights.json and blend3_coef.npy")


if __name__ == "__main__":
    main()
