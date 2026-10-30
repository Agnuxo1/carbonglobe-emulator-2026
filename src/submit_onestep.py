"""Fit the HRC one-step emulator on all 54152 sites and roll it out on the SSP test."""
import argparse, json, sys, time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, "src")
from common import WORK, COMP, SCENARIOS, AGES, SUB_SCALE, N_SITE, stats
from hrc_onestep import (forcing_norm, build_reservoir, teacher_forced, rollout,
                         AGE_COL, DEV)

OUT = WORK / "subs"; OUT.mkdir(parents=True, exist_ok=True)


def main(a):
    prep, bad = forcing_norm(not a.no_anomaly)
    _, _, y_mean, y_std = stats()
    ymf = y_mean.astype(np.float32); ysf = (y_std + 1e-10).astype(np.float32)
    ym = torch.as_tensor(ymf, device=DEV); ys = torch.as_tensor(ysf, device=DEV)
    Wi, Wr, leak = build_reservoir(a.size, a.seed, a.spectral_radius, a.leak)
    d = a.size + 145
    ages_col = torch.as_tensor(AGE_COL, device=DEV)

    xa = np.load(WORK / "train_xann.npy", mmap_mode="r")
    yd = np.load(WORK / "train_ydec.npy", mmap_mode="r")
    accum = [torch.zeros((d, d), device=DEV, dtype=torch.float64),
             torch.zeros((d, 7), device=DEV, dtype=torch.float64), 0]
    t0 = time.time()
    for i in range(0, N_SITE, a.chunk):
        s = slice(i, min(i + a.chunk, N_SITE))
        xz = torch.as_tensor(prep(np.asarray(xa[s], dtype=np.float32)), device=DEV)
        yv = np.asarray(yd[s], dtype=np.float32)
        yz = torch.as_tensor((yv - ymf) / ysf, device=DEV)
        teacher_forced(xz, yz, Wi, Wr, leak, ages_col, 1, 40, accum)
        del xz, yz
    A_, B_, n = accum
    Wout = torch.linalg.solve(A_ + a.ridge * torch.eye(d, device=DEV, dtype=torch.float64),
                              B_).float()
    print(f"fitted on all sites, {n:,} samples, {time.time()-t0:.0f}s", flush=True)

    rows = []
    for scn in SCENARIOS:
        xt = np.load(WORK / f"test_xann_{scn}.npy", mmap_mode="r")
        y0 = np.load(WORK / f"{scn}_test_y0.npy")
        preds = []
        for i in range(0, y0.shape[0], a.chunk):
            s = slice(i, min(i + a.chunk, y0.shape[0]))
            xz = torch.as_tensor(prep(np.asarray(xt[s], dtype=np.float32)), device=DEV)
            y0z = torch.as_tensor((y0[s] - ymf) / ysf, device=DEV)
            pr = rollout(xz, y0z, Wout, Wi, Wr, leak, ages_col)
            p40 = (pr[:, :, -1, :2] * ys[:2] + ym[:2]).cpu().numpy()
            preds.append(p40)
            del xz, y0z, pr
        p = np.concatenate(preds, 0) / SUB_SCALE
        nst = y0.shape[0]
        rows.append(pd.DataFrame({
            "id": [f"{scn}_{i}_{ag}" for i in range(nst) for ag in AGES],
            "height": p[:, :, 0].ravel(), "agb": p[:, :, 1].ravel()}))
        print(f"  {scn}: height mean {p[:,:,0].mean():.3f} agb mean {p[:,:,1].mean():.3f}",
              flush=True)

    sub = pd.concat(rows, ignore_index=True)
    tmpl = pd.read_csv(COMP / "sample_submission.csv")
    sub = tmpl[["id"]].merge(sub, on="id", how="left")
    assert sub[["height", "agb"]].notna().all().all(), "unmatched ids"
    sub[["height", "agb"]] = sub[["height", "agb"]].clip(lower=0.0)
    path = OUT / f"{a.name}.csv"
    sub.to_csv(path, index=False)
    (OUT / f"{a.name}.meta.json").write_text(json.dumps(vars(a), indent=1))
    print("wrote", path, sub.shape, sub[["height", "agb"]].mean().to_dict())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--size", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spectral-radius", type=float, default=0.9)
    p.add_argument("--leak", type=float, default=0.3)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--chunk", type=int, default=120)
    p.add_argument("--no-anomaly", action="store_true")
    main(p.parse_args())
