"""Fit the neural one-step emulator on all 54152 sites and roll it out on the SSP test.

Also writes the per-(scenario,site,age) year-40 prediction as a raw .npy so that it
can be blended with the age-shifted anchor without refitting.
"""
import argparse, json, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, "src")
from common import WORK, COMP, SCENARIOS, AGES, SUB_SCALE, N_SITE, stats
from hrc_onestep import forcing_norm, build_multi, AGE_COL, DEV
from nn_onestep import Step, Batcher, unroll, year_weights

BAD = np.array(json.loads((WORK / "bad_channels.json").read_text()), dtype=np.int64)

OUT = WORK / "subs"; OUT.mkdir(parents=True, exist_ok=True)


def main(a):
    torch.manual_seed(a.seed)
    prep, _ = forcing_norm(True)
    _, _, y_mean, y_std = stats()
    ymf = y_mean.astype(np.float32); ysf = (y_std + 1e-10).astype(np.float32)
    ys_t = torch.as_tensor(ysf, device=DEV); ym_t = torch.as_tensor(ymf, device=DEV)
    ages_col = torch.as_tensor(AGE_COL, device=DEV)

    res = None; din = 144
    if a.hrc_dim:
        res = build_multi(a.hrc_dim, a.seed, 0.9, [float(v) for v in a.leaks.split(",")])
        din = 144 + res[1].shape[0]
    model = Step(din, a.hidden, a.depth).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    bat = Batcher(prep, ymf, ysf, a.chunk)
    wy = year_weights(0, DEV)
    sites = np.arange(N_SITE)
    spe = len(sites) // a.chunk
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * spe,
                                                pct_start=0.15)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(sites)
        teacher = max(0.0, 1.0 - ep / max(a.tf_epochs, 1))
        model.train(); run = 0.0
        for i in range(spe):
            s = np.sort(perm[i * a.chunk:(i + 1) * a.chunk])
            xz, yz = bat.get(s)
            pred = unroll(model, xz, yz, ages_col, teacher, res)
            loss = ((pred - yz.reshape(-1, 41, 7)[:, 1:]) ** 2).mean(dim=(0, 2)) @ wy / 40.0
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += float(loss)
            del xz, yz, pred
        print(f"  epoch {ep} teacher {teacher:.2f} loss {run/spe:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # Two rollouts per scenario: one under the SSP forcing as given, one under a
    # stationary-climate counterfactual in which the within-record anomaly of the
    # 25 shifted drivers is set to zero. Their difference is the emulator's estimate
    # of the climate effect alone, in which its systematic bias largely cancels --
    # the "delta change" construction used in climate-impact studies. Combined with
    # the age-shifted anchor for the level, it gives the C02_delta candidate.
    model.eval(); rows = []; raw = {}; raw_stat = {}
    with torch.no_grad():
        for scn in SCENARIOS:
            xt = np.load(WORK / f"test_xann_{scn}.npy", mmap_mode="r")
            y0 = np.load(WORK / f"{scn}_test_y0.npy")
            outs, outs_stat = [], []
            for i in range(0, y0.shape[0], a.chunk):
                s = slice(i, min(i + a.chunk, y0.shape[0]))
                xraw = np.asarray(xt[s], dtype=np.float32)
                xzn = prep(xraw)
                xz = torch.as_tensor(xzn, device=DEV)
                xzs = xzn.copy(); xzs[:, :, BAD] = 0.0
                xz_stat = torch.as_tensor(xzs, device=DEV)
                y0b = y0[s]
                yz = torch.as_tensor(
                    (np.concatenate([y0b[:, :, None, :],
                                     np.zeros((y0b.shape[0], 15, 40, 7), np.float32)],
                                    axis=2) - ymf) / ysf, device=DEV)
                pred = unroll(model, xz, yz, ages_col, 0.0, res)
                p40 = pred.view(y0b.shape[0], 15, 40, 7)[:, :, -1, :2] * ys_t[:2] + ym_t[:2]
                outs.append(p40.cpu().numpy())
                prs = unroll(model, xz_stat, yz, ages_col, 0.0, res)
                p40s = prs.view(y0b.shape[0], 15, 40, 7)[:, :, -1, :2] * ys_t[:2] + ym_t[:2]
                outs_stat.append(p40s.cpu().numpy())
                del xz, xz_stat, yz, pred, prs
            p = np.concatenate(outs, 0) / SUB_SCALE               # (6171,15,2) scaled
            raw[scn] = p
            raw_stat[scn] = np.concatenate(outs_stat, 0) / SUB_SCALE
            n = y0.shape[0]
            rows.append(pd.DataFrame({
                "id": [f"{scn}_{i}_{ag}" for i in range(n) for ag in AGES],
                "height": p[:, :, 0].ravel(), "agb": p[:, :, 1].ravel()}))
            print(f"  {scn}: height {p[:,:,0].mean():.3f} agb {p[:,:,1].mean():.3f}",
                  flush=True)

    np.save(OUT / f"{a.name}_raw.npy",
            np.stack([raw[s] for s in SCENARIOS]))                # (2,6171,15,2)
    np.save(OUT / f"{a.name}_stat.npy",
            np.stack([raw_stat[s] for s in SCENARIOS]))
    sub = pd.concat(rows, ignore_index=True)
    tmpl = pd.read_csv(COMP / "sample_submission.csv")
    sub = tmpl[["id"]].merge(sub, on="id", how="left")
    assert sub[["height", "agb"]].notna().all().all()
    sub[["height", "agb"]] = sub[["height", "agb"]].clip(lower=0.0)
    sub.to_csv(OUT / f"{a.name}.csv", index=False)
    torch.save(model.state_dict(), OUT / f"{a.name}.pt")
    (OUT / f"{a.name}.meta.json").write_text(json.dumps(vars(a), indent=1))
    print("wrote", OUT / f"{a.name}.csv", sub.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--tf-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--chunk", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hrc-dim", type=int, default=0)
    p.add_argument("--leaks", default="0.1,0.3,0.7")
    main(p.parse_args())
