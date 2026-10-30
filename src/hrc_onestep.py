"""One-step autoregressive ecosystem emulator with the Holographic-Reservoir as
its recurrent core (hypothesis H1, the role JEV selected with confidence 0.91).

Design, following the organizers' own baseline protocol:
  per site s, seed age a, year t = 1..40
    u_t = [ annual-mean forcing of year t (136) , December state of year t-1 (7) ,
            age/500 (1) ]                                            -> 144 inputs
    x_t = (1-leak) x_{t-1} + leak * tanh(W_in u_t + W x_{t-1})        reservoir, fixed
    y_t = W_out [ x_t ; u_t ; 1 ]                                     readout, trained
  trained with teacher forcing (true y_{t-1}); evaluated by rolling the model out
  for 40 years from y_0 only, feeding its own prediction forward.

W_in and W come from holographic_reservoir's own HolographicReservoirModel, so the
reservoir is Francisco Angulo's technology unchanged -- the only port is batching the
identical update rule over sequences on the GPU (verified to 8e-8 in
src/test_hrc_port.py). The readout is a closed-form ridge solution accumulated in
float64 over all 32.5 million teacher-forced samples: because only the readout is
trained, the whole 54152-site training set can be used, which is the practical
advantage reservoir computing is supposed to deliver here.

Why one-step rather than the direct 40-year jump: measured on the leaderboard, the
direct regression generalises 6x worse than its local validation promised, while
untrained deterministic predictors transfer exactly. A one-step map learns local
physics that does not depend on which emission scenario produced the forcing.
"""
import os
import argparse, json, sys, time
import numpy as np
import torch

sys.path.insert(0, "src")
# pip install holographic-reservoir; or point HRC_SRC at a local checkout
if os.environ.get("HRC_SRC"):
    sys.path.insert(0, os.environ["HRC_SRC"])
from common import WORK, AGES, SUB_SCALE, N_SITE, stats
import dataset as D
from holographic_reservoir.model import HolographicReservoirModel  # noqa: E402

RUNS = WORK / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
AGE_COL = np.array(AGES, dtype=np.float32) / 500.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------- forcing prep
def forcing_norm(bad_as_anomaly=True):
    """Return a callable mapping raw annual forcing (S,40,136) -> z-scored inputs.

    For the 24 drivers whose test values lie almost entirely outside the training
    range (CO2 and the warming signal), the absolute level is useless and harmful:
    we keep only the within-record anomaly relative to the site's own 40-year mean,
    which stays in range while still carrying the scenario difference. JEV chose
    this treatment (`shifted_drivers = anomaly`).
    """
    x_mean, x_std, _, _ = stats()
    xm = x_mean.astype(np.float32)
    xs = (x_std + 1e-10).astype(np.float32)
    bad = np.array(json.loads((WORK / "bad_channels.json").read_text()), dtype=np.int64)
    # anomaly scale: sd of the within-record anomaly, measured on a training sample
    xa = np.load(WORK / "train_xann.npy", mmap_mode="r")
    smp = np.asarray(xa[::97], dtype=np.float32)
    anom_sd = (smp - smp.mean(axis=1, keepdims=True)).std(axis=(0, 1)) + 1e-6

    def prep(x):                                  # x: (S,40,136) float32 raw
        z = (x - xm) / xs
        if bad_as_anomaly and len(bad):
            a = (x[:, :, bad] - x[:, :, bad].mean(axis=1, keepdims=True)) / anom_sd[bad]
            z[:, :, bad] = a
        return z
    return prep, bad


def build_reservoir(size, seed, rho, leak, input_dim=144):
    """Backward-compatible single-reservoir build."""
    return build_multi(size, seed, rho, [leak], input_dim)


def build_multi(total_size, seed, rho, leaks, input_dim=144):
    """Several Holographic-Reservoir blocks with different leak rates, stacked.

    Forest dynamics act on very different timescales -- annual productivity swings
    against century-scale biomass accumulation -- and a single leak rate can only
    resolve one of them. Each block is still built by the library's own
    __post_init__, so the weights, the SHA-256 diagonal entropy and the spectral
    rescaling are unchanged; we simply run several of them side by side and let the
    readout see all their states. Returns block-diagonal W, stacked W_in and a
    per-unit leak vector, so the single batched update rule still applies.
    """
    per = total_size // len(leaks)
    Wins, Ws, lk = [], [], []
    for b, lv in enumerate(leaks):
        m = HolographicReservoirModel(size=per, input_dim=input_dim,
                                      spectral_radius=rho, leak=lv,
                                      random_seed=seed * 100 + b)
        Wins.append(m.W_in.astype(np.float32))
        Ws.append(m.W.astype(np.float32))
        lk.append(np.full(per, float(m.leak), dtype=np.float32))
    N = per * len(leaks)
    W = np.zeros((N, N), dtype=np.float32)
    for b in range(len(leaks)):
        W[b * per:(b + 1) * per, b * per:(b + 1) * per] = Ws[b]
    Win = np.concatenate(Wins, axis=0)
    return (torch.as_tensor(Win.T.copy(), device=DEV),
            torch.as_tensor(W.T.copy(), device=DEV),
            torch.as_tensor(np.concatenate(lk), device=DEV))


SQUARE = [False]


def readout_feats(x, u):
    """z = [x ; x^2 (optional) ; u ; 1]. The quadratic term is the standard ESN
    trick for giving a linear readout access to products of reservoir states."""
    parts = [x]
    if SQUARE[0]:
        parts.append(x * x)
    parts.append(u)
    parts.append(torch.ones((x.shape[0], 1), device=x.device))
    return torch.cat(parts, dim=1)


def readout_dim(size):
    return (2 * size if SQUARE[0] else size) + 145


# --------------------------------------------------------------- core dynamics
@torch.no_grad()
def teacher_forced(xz, yz, Wi, Wr, leak, ages_col, y_lo, y_hi, accum):
    """Accumulate ridge normal equations over one site chunk, all ages, all years.

    xz: (S,40,136) normalised forcing. yz: (S,A,41,7) normalised December states.
    """
    S, A = xz.shape[0], yz.shape[1]
    B = S * A
    N = Wr.shape[0]
    xf = xz.unsqueeze(1).expand(S, A, 40, 136).reshape(B, 40, 136)
    yy = yz.reshape(B, 41, 7)
    ac = ages_col.view(1, A, 1).expand(S, A, 1).reshape(B, 1)
    x = torch.zeros((B, N), device=DEV)
    for t in range(1, 41):
        u = torch.cat([xf[:, t - 1], yy[:, t - 1], ac], dim=1)        # (B,144)
        x = (1.0 - leak) * x + leak * torch.tanh(u @ Wi + x @ Wr)
        if t < y_lo or t > y_hi:
            continue
        z = readout_feats(x, u).double()
        accum[0] += z.T @ z
        accum[1] += z.T @ yy[:, t].double()
        accum[2] += B


@torch.no_grad()
def rollout(xz, y0z, Wout, Wi, Wr, leak, ages_col, start=0, x_init=None, yz_true=None):
    """Roll the model out from year `start` to year 40, feeding its own prediction.

    Returns normalised predicted states for years start+1..40, shape (S,A,40-start,7).
    """
    S, A = xz.shape[0], y0z.shape[1]
    B = S * A
    N = Wr.shape[0]
    xf = xz.unsqueeze(1).expand(S, A, 40, 136).reshape(B, 40, 136)
    ac = ages_col.view(1, A, 1).expand(S, A, 1).reshape(B, 1)
    x = torch.zeros((B, N), device=DEV) if x_init is None else x_init
    v = y0z.reshape(B, 7)
    outs = []
    for t in range(1, 41):
        u = torch.cat([xf[:, t - 1], v, ac], dim=1)
        x = (1.0 - leak) * x + leak * torch.tanh(u @ Wi + x @ Wr)
        if t <= start:
            # warm-up with the truth, used only by the temporal-holdout protocol
            v = yz_true.reshape(B, 41, 7)[:, t]
            continue
        z = readout_feats(x, u)
        v = z @ Wout
        outs.append(v)
    return torch.stack(outs, 1).view(S, A, -1, 7)


# --------------------------------------------------------------------- driver
def fit_and_score(a):
    prep, bad = forcing_norm(not a.no_anomaly)
    _, _, y_mean, y_std = stats()
    ym = torch.as_tensor(y_mean.astype(np.float32), device=DEV)
    ys = torch.as_tensor((y_std + 1e-10).astype(np.float32), device=DEV)
    SQUARE[0] = a.square
    leaks = [float(v) for v in a.leaks.split(",")] if a.leaks else [a.leak]
    Wi, Wr, leak = build_multi(a.size, a.seed, a.spectral_radius, leaks)
    N = Wr.shape[0]
    d = readout_dim(N)
    ages_col = torch.as_tensor(AGE_COL, device=DEV)

    xa = np.load(WORK / "train_xann.npy", mmap_mode="r")
    yd = np.load(WORK / "train_ydec.npy", mmap_mode="r")
    fold, lon, lat = D.spatial_folds(5, 10.0, seed=0)
    fit_sites = np.arange(N_SITE)[fold != 0]
    val_sites = np.arange(N_SITE)[fold == 0]
    if a.fit_sites:
        fit_sites = np.sort(np.random.default_rng(0).choice(fit_sites, a.fit_sites, False))
    if a.val_sites:
        val_sites = np.sort(np.random.default_rng(1).choice(val_sites, a.val_sites, False))

    accum = [torch.zeros((d, d), device=DEV, dtype=torch.float64),
             torch.zeros((d, 7), device=DEV, dtype=torch.float64), 0]
    t0 = time.time()
    for i in range(0, len(fit_sites), a.chunk):
        s = fit_sites[i:i + a.chunk]
        xz = torch.as_tensor(prep(np.asarray(xa[s], dtype=np.float32)), device=DEV)
        yv = np.asarray(yd[s], dtype=np.float32)
        yz = torch.as_tensor((yv - y_mean.astype(np.float32)) /
                             (y_std + 1e-10).astype(np.float32), device=DEV)
        teacher_forced(xz, yz, Wi, Wr, leak, ages_col, a.year_lo, a.year_hi, accum)
        if (i // a.chunk) % 25 == 0:
            print(f"  fit {i+len(s)}/{len(fit_sites)} {time.time()-t0:.0f}s", flush=True)
        del xz, yz
    A_, B_, n = accum
    Wout = torch.linalg.solve(A_ + a.ridge * torch.eye(d, device=DEV, dtype=torch.float64),
                              B_).float()
    print(f"readout solved from {n:,} teacher-forced samples, d={d}, "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- rollout scoring on held-out sites, competition units, year 40 only
    sq = np.zeros(2); cnt = 0
    sq_warm = np.zeros(2); cnt_w = 0
    temp = np.asarray(xa[val_sites][:, :, a.warm_channel], dtype=np.float32).mean(axis=1)
    warm_thr = np.percentile(temp, 80)
    scale = torch.as_tensor(SUB_SCALE.astype(np.float32), device=DEV)
    for i in range(0, len(val_sites), a.chunk):
        s = val_sites[i:i + a.chunk]
        xz = torch.as_tensor(prep(np.asarray(xa[s], dtype=np.float32)), device=DEV)
        yv = np.asarray(yd[s], dtype=np.float32)
        yz = torch.as_tensor((yv - y_mean.astype(np.float32)) /
                             (y_std + 1e-10).astype(np.float32), device=DEV)
        pr = rollout(xz, yz[:, :, 0], Wout, Wi, Wr, leak, ages_col)
        p40 = pr[:, :, -1, :2] * ys[:2] + ym[:2]
        t40 = torch.as_tensor(yv[:, :, 40, :2], device=DEV)
        e = ((p40 / scale) - (t40 / scale)) ** 2
        sq += e.sum(dim=(0, 1)).double().cpu().numpy(); cnt += e.shape[0] * e.shape[1]
        w = torch.as_tensor(temp[i:i + len(s)] >= warm_thr, device=DEV)
        if w.any():
            ew = e[w]
            sq_warm += ew.sum(dim=(0, 1)).double().cpu().numpy()
            cnt_w += ew.shape[0] * ew.shape[1]
        del xz, yz, pr
    mse = float(sq.sum() / (2 * cnt))
    res = dict(exp=a.exp, model="hrc_onestep", mse_rollout_y40=mse,
               mse_height=float(sq[0] / cnt), mse_agb=float(sq[1] / cnt),
               mse_warm_sites=float(sq_warm.sum() / (2 * max(cnt_w, 1))),
               n_val_pairs=int(cnt), n_fit_samples=int(n), d=int(d),
               reservoir=dict(size=int(N), seed=a.seed, rho=a.spectral_radius,
                              leaks=leaks, square=a.square, ridge=a.ridge),
               anomaly_drivers=(not a.no_anomaly), n_bad_channels=int(len(bad)),
               minutes=round((time.time() - t0) / 60, 2), args=vars(a))
    (RUNS / f"{a.exp}.json").write_text(json.dumps(res, indent=1))
    np.save(RUNS / f"{a.exp}_wout.npy", Wout.cpu().numpy())
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("args", "reservoir")}, indent=1))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--size", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spectral-radius", type=float, default=0.9)
    p.add_argument("--leak", type=float, default=0.3)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--chunk", type=int, default=150)
    p.add_argument("--fit-sites", type=int, default=0)
    p.add_argument("--val-sites", type=int, default=2500)
    p.add_argument("--year-lo", type=int, default=1)
    p.add_argument("--year-hi", type=int, default=40)
    p.add_argument("--no-anomaly", action="store_true")
    p.add_argument("--warm-channel", type=int, default=0)
    p.add_argument("--leaks", default=None,
                   help="comma-separated leak rates; the reservoir is split evenly")
    p.add_argument("--square", action="store_true")
    fit_and_score(p.parse_args())
