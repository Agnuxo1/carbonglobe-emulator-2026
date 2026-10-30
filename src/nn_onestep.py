"""Professional control: a residual one-step ecosystem emulator trained through
the 40-year rollout itself.

Same interface to the data and the same rollout protocol as the Holographic-Reservoir
emulator in src/hrc_onestep.py, so the two are directly comparable on the same
held-out sites with the same metric.

Two differences from the organizers' Random-Forest baseline, both standard practice:

* Residual parameterisation. The map predicts the annual *increment*
  dy = y_t - y_{t-1} rather than y_t. Ecosystem state is strongly autocorrelated
  from year to year, so the increment is the part that actually has to be learned,
  and this keeps the 40-step composition stable.
* Training on the rollout. The baseline trains one step ahead with teacher forcing
  and is then evaluated after 40 autoregressive steps, which is a different problem:
  the tutorial itself warns that long-horizon drift is what separates emulators.
  Here we warm up with teacher forcing and then backpropagate through the whole
  40-step unroll, with the loss weighted towards the scored final year.

Optionally (--hrc-dim) the reservoir state of the Holographic-Reservoir is appended
to the network input, which is the hybrid arm of the H1 ablation.
"""
import argparse, json, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "src")
from common import WORK, SUB_SCALE, N_SITE, stats
import dataset as D
from hrc_onestep import forcing_norm, build_multi, AGE_COL, DEV

RUNS = WORK / "runs"; RUNS.mkdir(parents=True, exist_ok=True)


class Step(nn.Module):
    def __init__(self, din=144, hidden=512, depth=3, dout=7):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, dout)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, u):
        return self.head(self.body(u))


def year_weights(tf_frac, device):
    """Loss weight per forecast year: every year counts, the scored one counts most."""
    w = torch.ones(40, device=device)
    w[-1] = 8.0
    w[30:39] = 2.0
    return w / w.mean()


class Batcher:
    def __init__(self, prep, ymf, ysf, chunk):
        self.xa = np.load(WORK / "train_xann.npy", mmap_mode="r")
        self.yd = np.load(WORK / "train_ydec.npy", mmap_mode="r")
        self.prep, self.ymf, self.ysf, self.chunk = prep, ymf, ysf, chunk

    def get(self, sites):
        xz = torch.as_tensor(self.prep(np.asarray(self.xa[sites], dtype=np.float32)),
                             device=DEV)
        yv = np.asarray(self.yd[sites], dtype=np.float32)
        yz = torch.as_tensor((yv - self.ymf) / self.ysf, device=DEV)
        return xz, yz


def unroll(model, xz, yz, ages_col, teacher, res=None):
    """Return predicted normalised states for years 1..40, shape (B,40,7).

    teacher=1.0 feeds the truth every step, 0.0 feeds the model's own prediction.
    """
    S, A = xz.shape[0], yz.shape[1]
    B = S * A
    xf = xz.unsqueeze(1).expand(S, A, 40, 136).reshape(B, 40, 136)
    yy = yz.reshape(B, 41, 7)
    ac = ages_col.view(1, A, 1).expand(S, A, 1).reshape(B, 1)
    v = yy[:, 0]
    x = None
    if res is not None:
        Wi, Wr, leak = res
        x = torch.zeros((B, Wr.shape[0]), device=DEV)
    outs = []
    for t in range(1, 41):
        u = torch.cat([xf[:, t - 1], v, ac], dim=1)
        if res is not None:
            x = (1.0 - leak) * x + leak * torch.tanh(u @ Wi + x @ Wr)
            inp = torch.cat([u, x], dim=1)
        else:
            inp = u
        v_next = v + model(inp)
        outs.append(v_next)
        v = yy[:, t] if teacher >= 1.0 else (
            v_next if teacher <= 0.0 else
            torch.where(torch.rand((B, 1), device=DEV) < teacher, yy[:, t], v_next))
    return torch.stack(outs, 1)


def main(a):
    torch.manual_seed(a.seed)
    prep, bad = forcing_norm(not a.no_anomaly)
    _, _, y_mean, y_std = stats()
    ymf = y_mean.astype(np.float32); ysf = (y_std + 1e-10).astype(np.float32)
    ages_col = torch.as_tensor(AGE_COL, device=DEV)
    ys_t = torch.as_tensor(ysf, device=DEV); ym_t = torch.as_tensor(ymf, device=DEV)
    scale = torch.as_tensor(SUB_SCALE.astype(np.float32), device=DEV)

    res = None; din = 144
    if a.hrc_dim:
        leaks = [float(v) for v in a.leaks.split(",")]
        res = build_multi(a.hrc_dim, a.seed, a.spectral_radius, leaks)
        din = 144 + res[1].shape[0]
    model = Step(din, a.hidden, a.depth).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    fold, _, _ = D.spatial_folds(5, 10.0, seed=0)
    fit_sites = np.arange(N_SITE)[fold != 0]
    val_sites = np.arange(N_SITE)[fold == 0]
    if a.val_sites:
        val_sites = np.sort(np.random.default_rng(1).choice(val_sites, a.val_sites, False))
    bat = Batcher(prep, ymf, ysf, a.chunk)
    wy = year_weights(0, DEV)
    steps_per_epoch = len(fit_sites) // a.chunk
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(total, 2),
                                                pct_start=0.15)
    rng = np.random.default_rng(a.seed)
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        perm = rng.permutation(fit_sites)
        # teacher forcing is annealed to zero, so the model ends up trained on its
        # own rollout, which is how it is scored
        teacher = max(0.0, 1.0 - ep / max(a.tf_epochs, 1))
        model.train()
        run = 0.0
        for i in range(steps_per_epoch):
            s = np.sort(perm[i * a.chunk:(i + 1) * a.chunk])
            xz, yz = bat.get(s)
            pred = unroll(model, xz, yz, ages_col, teacher, res)
            true = yz.reshape(-1, 41, 7)[:, 1:]
            loss = ((pred - true) ** 2).mean(dim=(0, 2)) @ wy / 40.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run += float(loss)
            del xz, yz, pred, true
        print(f"  epoch {ep} teacher {teacher:.2f} loss {run/steps_per_epoch:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # -------- rollout scoring, competition units, year 40
    model.eval()
    sq = np.zeros(2); cnt = 0; sqw = np.zeros(2); cntw = 0
    temp = np.asarray(bat.xa[val_sites][:, :, 0], dtype=np.float32).mean(axis=1)
    thr = np.percentile(temp, 80)
    with torch.no_grad():
        for i in range(0, len(val_sites), a.chunk):
            s = val_sites[i:i + a.chunk]
            xz, yz = bat.get(s)
            pred = unroll(model, xz, yz, ages_col, 0.0, res)
            p40 = pred.view(len(s), 15, 40, 7)[:, :, -1, :2] * ys_t[:2] + ym_t[:2]
            t40 = yz[:, :, 40, :2] * ys_t[:2] + ym_t[:2]
            e = ((p40 - t40) / scale) ** 2
            sq += e.sum(dim=(0, 1)).double().cpu().numpy(); cnt += e.shape[0] * 15
            w = torch.as_tensor(temp[i:i + len(s)] >= thr, device=DEV)
            if w.any():
                ew = e[w]; sqw += ew.sum(dim=(0, 1)).double().cpu().numpy()
                cntw += ew.shape[0] * 15
            del xz, yz, pred
    res_d = dict(exp=a.exp, model="nn_onestep" if not a.hrc_dim else "nn_plus_hrc",
                 mse_rollout_y40=float(sq.sum() / (2 * cnt)),
                 mse_height=float(sq[0] / cnt), mse_agb=float(sq[1] / cnt),
                 mse_warm_sites=float(sqw.sum() / (2 * max(cntw, 1))),
                 n_val_pairs=int(cnt), n_params=int(nparam), din=int(din),
                 hrc_dim=a.hrc_dim, minutes=round((time.time() - t0) / 60, 2),
                 args=vars(a))
    (RUNS / f"{a.exp}.json").write_text(json.dumps(res_d, indent=1))
    torch.save(model.state_dict(), RUNS / f"{a.exp}.pt")
    print(json.dumps({k: v for k, v in res_d.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--tf-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--chunk", type=int, default=96)
    p.add_argument("--val-sites", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-anomaly", action="store_true")
    p.add_argument("--hrc-dim", type=int, default=0)
    p.add_argument("--leaks", default="0.1,0.3,0.7")
    p.add_argument("--spectral-radius", type=float, default=0.9)
    main(p.parse_args())
